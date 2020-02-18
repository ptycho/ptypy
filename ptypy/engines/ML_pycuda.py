# -*- coding: utf-8 -*-
"""
Maximum Likelihood reconstruction engine.

TODO.

  * Implement other regularizers

This file is part of the PTYPY package.

    :copyright: Copyright 2014 by the PTYPY team, see AUTHORS.
    :license: GPLv2, see LICENSE for details.
"""
import numpy as np
import time

from .ML import ML, BaseModel, prepare_smoothing_preconditioner, Regul_del2
from .ML_serial import ML_serial
from .. import utils as u
from ..utils.verbose import logger
from ..utils import parallel
from .utils import Cnorm2, Cdot
from ..accelerate import py_cuda as gpu
from ..accelerate.py_cuda.kernels import GradientDescentKernel, AuxiliaryWaveKernel, PoUpdateKernel, PositionCorrectionKernel
from ..accelerate.py_cuda.array_utils import ArrayUtilsKernel
from ..accelerate.array_based import address_manglers

__all__ = ['ML_pycuda']


@register()
class ML_pycuda(ML_serial):

    def __init__(self, ptycho_parent, pars=None):
        """
        Maximum likelihood reconstruction engine.
        """
        super(ML_pycuda, self).__init__(ptycho_parent, pars)

        self.context, self.queue = gpu.get_context()

    def engine_initialize(self):
        """
        Prepare for ML reconstruction.
        """
        super(ML_serial, self).engine_initialize()
        self._setup_kernels()

    def _setup_kernels(self):
        """
        Setup kernels, one for each scan. Derive scans from ptycho class
        """
        # get the scans
        for label, scan in self.ptycho.model.scans.items():

            kern = u.Param()
            self.kernels[label] = kern

            # TODO: needs to be adapted for broad bandwidth
            geo = scan.geometries[0]

            # Get info to shape buffer arrays
            # TODO: make this part of the engine rather than scan
            fpc = self.ptycho.frames_per_block

            # TODO : make this more foolproof
            try:
                nmodes = scan.p.coherence.num_probe_modes * \
                         scan.p.coherence.num_object_modes
            except:
                nmodes = 1

            # create buffer arrays
            ash = (fpc * nmodes,) + tuple(geo.shape)
            aux = np.zeros(ash, dtype=np.complex64)
            kern.aux = aux
            kern.a = np.zeros(ash, dtype=np.complex64)
            kern.a = np.zeros(ash, dtype=np.complex64)

            # setup kernels, one for each SCAN.
            kern.GDK = GradientDescentKernel(aux, nmodes, queue_thread=self.queue)
            kern.GDK.allocate()

            kern.POK = PoUpdateKernel(queue_thread=self.queue, denom_type=np.float32)
            kern.POK.allocate()

            kern.AWK = AuxiliaryWaveKernel(queue_thread=self.queue)
            kern.AWK.allocate()

            try:
                from ptypy.accelerate.py_cuda.cufft import FFT
            except:
                logger.warning('Unable to import cuFFT version - using Reikna instead')
                from ptypy.accelerate.py_cuda.fft import FFT

            kern.FW = FFT(aux, self.queue,
                          pre_fft=geo.propagator.pre_fft,
                          post_fft=geo.propagator.post_fft,
                          inplace=True,
                          symmetric=True,
                          forward=True).ft
            kern.BW = FFT(aux, self.queue,
                          pre_fft=geo.propagator.pre_ifft,
                          post_fft=geo.propagator.post_ifft,
                          inplace=True,
                          symmetric=True,
                          forward=False).ift

            if self.do_position_refinement:
                addr_mangler = address_manglers.RandomIntMangle(int(self.p.position_refinement.amplitude // geo.resolution[0]),
                                                                self.p.position_refinement.start,
                                                                self.p.position_refinement.stop,
                                                                max_bound=int(self.p.position_refinement.max_shift // geo.resolution[0]),
                                                                randomseed=0)
                logger.warning("amplitude is %s " % (self.p.position_refinement.amplitude // geo.resolution[0]))
                logger.warning("max bound is %s " % (self.p.position_refinement.max_shift // geo.resolution[0]))

                kern.PCK = PositionCorrectionKernel(aux, nmodes, queue_thread=self.queue)
                kern.PCK.allocate()
                kern.PCK.address_mangler = addr_mangler

    def engine_prepare(self):

        ## Serialize new data ##

        for label, d in self.ptycho.new_data:
            prep = u.Param()
            prep.label = label
            self.diff_info[d.ID] = prep
            prep.err_phot = np.zeros_like((d.data.shape[0],), dtype=np.float32)

        # Unfortunately this needs to be done for all pods, since
        # the shape of the probe / object was modified.
        # TODO: possible scaling issue
        for label, d in self.di.storages.items():
            prep = self.diff_info[d.ID]
            prep.view_IDs, prep.poe_IDs, prep.addr = serialize_array_access(d)
            if self.do_position_refinement:
                prep.original_addr = np.zeros_like(prep.addr)
                prep.original_addr[:] = prep.addr

        self.ML_model.prepare()

    def engine_iterate(self, num=1):
        """
        Compute `num` iterations.
        """
        ########################
        # Compute new gradient
        ########################
        tg = 0.
        tc = 0.
        for it in range(num):
            t1 = time.time()
            error_dct = self.ML_model.new_grad()
            new_ob_grad = self.ob_grad_new
            new_pr_grad = self.pr_grad_new

            tg += time.time() - t1

            if self.p.probe_update_start <= self.curiter:
                # Apply probe support if needed
                for name, s in new_pr_grad.storages.items():
                    self.support_constraint(s)
            else:
                new_pr_grad.fill(0.)

            # Smoothing preconditioner
            if self.smooth_gradient:
                self.smooth_gradient.sigma *= (1. - self.p.smooth_gradient_decay)
                for name, s in new_ob_grad.storages.items():
                    s.data[:] = self.smooth_gradient(s.data)

            cn2_new_pr_grad = Cnorm2(new_pr_grad)
            cn2_new_ob_grad = Cnorm2(new_ob_grad)

            # probe/object rescaling
            if self.p.scale_precond:
                cn2_new_pr_grad = cn2_new_pr_grad
                if cn2_new_pr_grad > 1e-5:
                    scale_p_o = (self.p.scale_probe_object * cn2_new_ob_grad
                                 / cn2_new_pr_grad)
                else:
                    scale_p_o = self.p.scale_probe_object
                if self.scale_p_o is None:
                    self.scale_p_o = scale_p_o
                else:
                    self.scale_p_o = self.scale_p_o ** self.scale_p_o_memory
                    self.scale_p_o *= scale_p_o ** (1-self.scale_p_o_memory)
                logger.debug('Scale P/O: %6.3g' % scale_p_o)
            else:
                self.scale_p_o = self.p.scale_probe_object

            ############################
            # Compute next conjugate
            ############################
            if self.curiter == 0:
                bt = 0.
            else:
                bt_num = (self.scale_p_o
                          * (cn2_new_pr_grad
                             - np.real(Cdot(new_pr_grad, self.pr_grad)))
                          + (cn2_new_ob_grad
                             - np.real(Cdot(new_ob_grad, self.ob_grad))))

                bt_denom = self.scale_p_o * self.cn2_pr_grad + self.cn2_ob_grad

                bt = max(0, bt_num/bt_denom)

            # verbose(3,'Polak-Ribiere coefficient: %f ' % bt)

            self.ob_grad << new_ob_grad
            self.pr_grad << new_pr_grad
            self.cn2_ob_grad = cn2_new_ob_grad
            self.cn2_pr_grad = cn2_new_pr_grad

            # 3. Next conjugate
            self.ob_h *= bt / self.tmin

            # Smoothing preconditioner
            if self.smooth_gradient:
                for name, s in self.ob_h.storages.items():
                    s.data[:] -= self.smooth_gradient(self.ob_grad.storages[name].data)
            else:
                self.ob_h -= self.ob_grad

            self.pr_h *= bt / self.tmin
            self.pr_grad *= self.scale_p_o
            self.pr_h -= self.pr_grad

            # In principle, the way things are now programmed this part
            # could be iterated over in a real Newton-Raphson style.
            t2 = time.time()
            B = self.ML_model.poly_line_coeffs(self.ob_h, self.pr_h)
            tc += time.time() - t2

            if np.isinf(B).any() or np.isnan(B).any():
                logger.warning(
                    'Warning! inf or nan found! Trying to continue...')
                B[np.isinf(B)] = 0.
                B[np.isnan(B)] = 0.

            self.tmin = -.5 * B[1] / B[2]
            self.ob_h *= self.tmin
            self.pr_h *= self.tmin
            self.ob += self.ob_h
            self.pr += self.pr_h
            # Newton-Raphson loop would end here

            # increase iteration counter
            self.curiter +=1

        logger.info('Time spent in gradient calculation: %.2f' % tg)
        logger.info('  ....  in coefficient calculation: %.2f' % tc)
        return error_dct  # np.array([[self.ML_model.LL[0]] * 3])

class BaseModelSerial(BaseModel):
    """
    Base class for log-likelihood models.
    """

    def __del__(self):
        """
        Clean up routine
        """
        pass


class GaussianModel(BaseModelSerial):
    """
    Gaussian noise model.
    TODO: feed actual statistical weights instead of using the Poisson statistic heuristic.
    """

    def __init__(self, MLengine):
        """
        Core functions for ML computation using a Gaussian model.
        """
        super(GaussianModel, self).__init__(MLengine)

    def prepare(self):

        super(GaussianModel, self).prepare()

        for label, d in self.engine.ptycho.new_data:
            prep = self.engine.diff_info[d.ID]
            prep.weights = (self.Irenorm * self.engine.ma.S[d.ID].data
                            / (1. / self.Irenorm + d.data))

    def __del__(self):
        """
        Clean up routine
        """
        super(GaussianModel, self).__del__()

    def new_grad(self):
        """
        Compute a new gradient direction according to a Gaussian noise model.

        Note: The negative log-likelihood and local errors are also computed
        here.
        """
        ob_grad = self.engine.ob_grad_new
        pr_grad = self.engine.pr_grad_new
        ob_grad.fill(0.)
        pr_grad.fill(0.)

        # We need an array for MPI
        LL = np.array([0.])
        error_dct = {}

        for dID in self.di.S.keys():
            prep = self.engine.diff_info[dID]
            # find probe, object in exit ID in dependence of dID
            pID, oID, eID = prep.poe_IDs

            # references for kernels
            kern = self.engine.kernels[prep.label]
            GDK = kern.GDK
            AWK = kern.AWK
            POK = kern.POK

            aux = kern.aux

            FW = kern.FW
            BW = kern.BW

            # get addresses and auxilliary array
            addr = prep.addr
            w = prep.weights
            err_phot = prep.err_phot

            # local references
            ob = self.engine.ob.S[oID].data
            obg = ob_grad.S[oID].data
            pr = self.engine.pr.S[pID].data
            prg = pr_grad.S[pID].data
            I = self.engine.di.S[dID].data

            # make propagated exit (to buffer)
            AWK.build_aux_no_ex(aux, addr, ob, pr, add=False)

            # forward prop
            aux[:] = FW(aux)
            GDK.make_model(aux, addr)

            """
            # for later
            if self.p.floating_intensities:
                tmp = np.zeros_like(Imodel)
                tmp = w * Imodel * I
                GDK.error_reduce(err_num, w * Imodel * I)
                GDK.error_reduce(err_den, w * Imodel ** 2)
                Imodel *= (err_num / err_den).reshape(Imodel.shape[0], 1, 1)
            """

            GDK.main(aux, addr, w, I)
            GDK.error_reduce(addr, err_phot)
            aux[:] = BW(aux)

            POK.ob_update_ML(addr, obg, pr, aux)
            POK.pr_update_ML(addr, prg, ob, aux)

        for dID, prep in self.engine.diff_info.items():
            err_phot = prep.err_phot / np.prod(prep.weights.shape)
            err_fourier = np.zeros_like(err_phot)
            err_exit = np.zeros_like(err_phot)
            errs = np.ascontiguousarray(np.vstack([err_fourier, err_phot, err_exit]).T)
            error_dct.update(zip(prep.view_IDs, errs))
            LL += err_phot.sum()

        # MPI reduction of gradients
        ob_grad.allreduce()
        pr_grad.allreduce()
        parallel.allreduce(LL)

        # Object regularizer
        if self.regularizer:
            for name, s in self.engine.ob.storages.items():
                ob_grad.storages[name].data += self.regularizer.grad(s.data)
                LL += self.regularizer.LL

        self.LL = LL / self.tot_measpts

        return error_dct

    def poly_line_coeffs(self, c_ob_h, c_pr_h):
        """
        Compute the coefficients of the polynomial for line minimization
        in direction h
        """

        B = np.zeros((3,), dtype=np.longdouble)
        Brenorm = 1. / self.LL[0] ** 2

        # Outer loop: through diffraction patterns
        for dID in self.di.S.keys():
            prep = self.engine.diff_info[dID]

            # find probe, object in exit ID in dependence of dID
            pID, oID, eID = prep.poe_IDs

            # references for kernels
            kern = self.engine.kernels[prep.label]
            GDK = kern.GDK
            AWK = kern.AWK

            f = kern.aux
            a = kern.a
            b = kern.b

            FW = kern.FW

            # get addresses and auxilliary array
            addr = prep.addr
            w = prep.weights

            # local references
            ob = self.ob.S[oID].data
            ob_h = c_ob_h.S[oID].data
            pr = self.pr.S[pID].data
            pr_h = c_pr_h.S[pID].data
            I = self.di.S[dID].data

            # make propagated exit (to buffer)
            AWK.build_aux_no_ex(f, addr, ob, pr, add=False)
            AWK.build_aux_no_ex(a, addr, ob_h, pr, add=False)
            AWK.build_aux_no_ex(a, addr, ob, pr_h, add=True)
            AWK.build_aux_no_ex(b, addr, ob_h, pr_h, add=False)

            # forward prop
            f[:] = FW(f)
            a[:] = FW(a)
            b[:] = FW(b)

            GDK.make_a012(f, a, b, addr, I)

            """
            if self.p.floating_intensities:
                A0 *= self.float_intens_coeff[dname]
                A1 *= self.float_intens_coeff[dname]
                A2 *= self.float_intens_coeff[dname]
            """
            GDK.fill_b(addr, Brenorm, w, B)

        parallel.allreduce(B)

        # Object regularizer
        if self.regularizer:
            for name, s in self.ob.storages.items():
                B += Brenorm * self.regularizer.poly_line_coeffs(
                    ob_h.storages[name].data, s.data)

        self.B = B

        return B