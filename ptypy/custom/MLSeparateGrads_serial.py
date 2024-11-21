# -*- coding: utf-8 -*-
"""
Maximum Likelihood reconstruction engine.

TODO.

  * Implement other regularizers

This file is part of the PTYPY package.

    :copyright: Copyright 2024 by the PTYPY team, see AUTHORS.
    :license: see LICENSE for details.
"""
import numpy as np
import time

from .MLSeparateGrads import MLSeparateGrads, BaseModel
from ..accelerate.base.engines.projectional_serial import serialize_array_access
from .. import utils as u
from ..utils.verbose import logger, log
from ..utils import parallel
from ..engines.utils import Cnorm2, Cdot
from ..engines import register
from ..accelerate.base.kernels import GradientDescentKernel, AuxiliaryWaveKernel, PoUpdateKernel, PositionCorrectionKernel
from ..accelerate.base.array_utils import complex_gaussian_filter, complex_gaussian_filter_fft


__all__ = ['MLSeparateGrads_serial']

@register()
class MLSeparateGrads_serial(MLSeparateGrads):

    """
    Defaults:

    [smooth_gradient_method]
    default = convolution
    type = str
    help = Method to be used for smoothing the gradient, choose between ```convolution``` or ```fft```.
    """

    def __init__(self, ptycho_parent, pars=None):
        """
        Maximum likelihood reconstruction engine.
        """
        super(MLSeparateGrads_serial, self).__init__(ptycho_parent, pars)

        self.kernels = {}
        self.diff_info = {}
        self.cn2_ob_grad = 0.
        self.cn2_pr_grad = 0.

    def engine_initialize(self):
        """
        Prepare for ML reconstruction.
        """
        super(MLSeparateGrads_serial, self).engine_initialize()
        self._setup_kernels()

    def _initialize_model(self):

        # Create noise model
        if self.p.ML_type.lower() == "gaussian":
            self.ML_model = GaussianModel(self)
        elif self.p.ML_type.lower() == "poisson":
            raise NotImplementedError('Poisson norm model not yet implemented')
        elif self.p.ML_type.lower() == "euclid":
            raise NotImplementedError('Euclid norm model not yet implemented')
        else:
            raise RuntimeError("Unsupported ML_type: '%s'" % self.p.ML_type)

    def _setup_kernels(self):
        """
        Setup kernels, one for each scan. Derive scans from ptycho class
        """
        # get the scans
        for label, scan in self.ptycho.model.scans.items():

            kern = u.Param()
            kern.scanmodel = type(scan).__name__
            self.kernels[label] = kern

            # TODO: needs to be adapted for broad bandwidth
            geo = scan.geometries[0]

            # Get info to shape buffer arrays
            fpc = scan.max_frames_per_block

            # TODO : make this more foolproof
            try:
                nmodes = scan.p.coherence.num_probe_modes
            except:
                nmodes = 1

            # create buffer arrays
            ash = (fpc * nmodes,) + tuple(geo.shape)
            aux = np.zeros(ash, dtype=np.complex64)
            kern.aux = aux
            kern.a = np.zeros(ash, dtype=np.complex64)
            kern.b = np.zeros(ash, dtype=np.complex64)

            # setup kernels, one for each SCAN.
            kern.GDK = GradientDescentKernel(aux, nmodes)
            kern.GDK.allocate()

            kern.POK = PoUpdateKernel()
            kern.POK.allocate()

            kern.AWK = AuxiliaryWaveKernel()
            kern.AWK.allocate()

            kern.FW = geo.propagator.fw
            kern.BW = geo.propagator.bw
            kern.resolution = geo.resolution[0]

            if self.do_position_refinement:
                kern.PCK = PositionCorrectionKernel(aux, nmodes, self.p.position_refinement, geo.resolution)
                kern.PCK.allocate()

    def engine_prepare(self):

        ## Serialize new data ##

        for label, d in self.ptycho.new_data:
            prep = u.Param()
            prep.label = label
            self.diff_info[d.ID] = prep
            prep.err_phot = np.zeros((d.data.shape[0],), dtype=np.float32)
            # set floating intensity coefficients to 1.0
            # they get overridden if self.p.floating_intensities=True
            prep.float_intens_coeff = np.ones((d.data.shape[0],), dtype=np.float32)

        # Unfortunately this needs to be done for all pods, since
        # the shape of the probe / object was modified.
        # TODO: possible scaling issue
        for label, d in self.di.storages.items():
            prep = self.diff_info[d.ID]
            prep.view_IDs, prep.poe_IDs, prep.addr = serialize_array_access(d)
            # Re-create exit addresses when gradient models (single exit buffer per view) are used
            # TODO: this should not be necessary, kernels should not use exit wave information
            if self.kernels[prep.label].scanmodel in ("GradFull", "BlockGradFull"):
                for i,addr in enumerate(prep.addr):
                    nmodes = len(addr[:,2,0])
                    for j,ea in enumerate(addr[:,2,0]):
                        prep.addr[i,j,2,0] = i*nmodes+j
            prep.I = d.data
            if self.do_position_refinement:
                prep.original_addr = np.zeros_like(prep.addr)
                prep.original_addr[:] = prep.addr
                prep.ma = self.ma.S[d.ID].data

        self.ML_model.prepare()

    def _get_smooth_gradient(self, data, sigma):
        if self.p.smooth_gradient_method == "convolution":
            return complex_gaussian_filter(data, sigma)
        elif self.p.smooth_gradient_method == "fft":
            return complex_gaussian_filter_fft(data, sigma)
        else:
            raise NotImplementedError("smooth_gradient_method should be ```convolution``` or ```fft```.")

    def _replace_ob_grad(self):
        new_ob_grad = self.ob_grad_new
        # Smoothing preconditioner
        if self.smooth_gradient:
            self.smooth_gradient.sigma *= (1. - self.p.smooth_gradient_decay)
            for name, s in new_ob_grad.storages.items():
                s.data[:] = self._get_smooth_gradient(s.data, self.smooth_gradient.sigma)

        norm = Cnorm2(new_ob_grad)
        dot = np.real(Cdot(new_ob_grad, self.ob_grad))
        self.ob_grad << new_ob_grad
        return norm, dot

    def _replace_pr_grad(self):
        new_pr_grad = self.pr_grad_new
        # probe support
        if self.p.probe_update_start <= self.curiter:
            # Apply probe support if needed
            for name, s in new_pr_grad.storages.items():
                self.support_constraint(s)
        # FIXME: this hack doesn't work here as we step in probe and object separately
        # FIXME: really it's the probe step that should be zeroed out not the gradient
        else:
            new_pr_grad.fill(0.)

        norm = Cnorm2(new_pr_grad)
        dot = np.real(Cdot(new_pr_grad, self.pr_grad))
        self.pr_grad << new_pr_grad
        return norm, dot

    def engine_iterate(self, num=1):
        """
        Compute `num` iterations.
        """
        ########################
        # Compute new gradient
        ########################
        tg = 0.
        tc = 0.
        ta = time.time()
        for it in range(num):

            ########################
            # Compute new gradients
            # ob: new_ob_grad
            # pr: new_pr_grad
            ########################
            t1 = time.time()
            error_dct = self.ML_model.new_grad()
            tg += time.time() - t1

            cn2_new_pr_grad, cdotr_pr_grad = self._replace_pr_grad()
            cn2_new_ob_grad, cdotr_ob_grad = self._replace_ob_grad()

            ############################
            # Compute Polak-Ribiere betas
            # bt_ob = bt_num_ob/bt_denom_ob
            # bt_pr = bt_num_pr/bt_denom_pr
            ############################
            if self.curiter == 0:
                bt_ob = 0.
                bt_pr = 0.
            else:
                # For the object
                bt_num_ob = cn2_new_ob_grad - cdotr_ob_grad
                bt_denom_ob = self.cn2_ob_grad
                bt_ob = max(0, bt_num_ob/bt_denom_ob)

                # For the probe
                bt_num_pr = cn2_new_pr_grad - cdotr_pr_grad
                bt_denom_pr = self.cn2_pr_grad
                bt_pr = max(0, bt_num_pr/bt_denom_pr)

            self.cn2_ob_grad = cn2_new_ob_grad
            self.cn2_pr_grad = cn2_new_pr_grad

            ############################
            # Compute next conjugates
            # ob_h -= bt_ob * ob_grad
            # pr_h -= bt_pr * pr_grad
            # NB: in the below need to do h/tmin
            # as did h*tmin when taking steps
            # (don't you just love containers?)
            ############################
            dt = self.ptycho.FType
            self.ob_h *= dt(bt_ob / self.tmin_ob)

            # Smoothing preconditioner
            if self.smooth_gradient:
                for name, s in self.ob_h.storages.items():
                    s.data[:] -= self._get_smooth_gradient(self.ob_grad.storages[name].data, self.smooth_gradient.sigma)
            else:
                self.ob_h -= self.ob_grad

            self.pr_h *= dt(bt_pr / self.tmin_pr)
            self.pr_h -= self.pr_grad

            ########################
            # Compute step-sizes
            # ob: tmin_ob
            # pr: tmin_pr
            ########################
            t2 = time.time()
            B_ob = self.ML_model.poly_line_coeffs_ob(self.ob_h)
            B_pr = self.ML_model.poly_line_coeffs_pr(self.pr_h)
            tc += time.time() - t2

            self.tmin_ob = dt(-0.5 * B_ob[1] / B_ob[2])
            self.tmin_pr = dt(-0.5 * B_pr[1] / B_pr[2])

            ########################
            # Take conjugate steps
            # ob += tmin_ob * ob_h
            # pr += tmin_pr * pr_h
            ########################
            self.ob_h *= self.tmin_ob
            self.pr_h *= self.tmin_pr
            self.ob += self.ob_h
            self.pr += self.pr_h

            # Position correction
            self.position_update()

            # Allow for customized modifications at the end of each iteration
            self._post_iterate_update()

            # increase iteration counter
            self.curiter += 1

        logger.info('Time spent in gradient calculation: %.2f' % tg)
        logger.info('  ....  in coefficient calculation: %.2f' % tc)
        return error_dct  # np.array([[self.ML_model.LL[0]] * 3])

    def position_update(self):
        """
        Position refinement
        """
        if not self.do_position_refinement:
            return
        do_update_pos = (self.p.position_refinement.stop > self.curiter >= self.p.position_refinement.start)
        do_update_pos &= (self.curiter % self.p.position_refinement.interval) == 0

        # Update positions
        if do_update_pos:
            """
            Iterates through all positions and refines them by a given algorithm.
            """
            log(4, "----------- START POS REF -------------")
            for dID in self.di.S.keys():

                prep = self.diff_info[dID]
                pID, oID, eID = prep.poe_IDs
                ob = self.ob.S[oID].data
                pr = self.pr.S[pID].data
                kern = self.kernels[prep.label]
                aux = kern.aux
                addr = prep.addr
                original_addr = prep.original_addr
                mangled_addr = addr.copy()
                w = prep.weights
                I = prep.I
                err_phot = prep.err_phot

                PCK = kern.PCK
                FW = kern.FW

                # Keep track of object boundaries
                max_oby = ob.shape[-2] - aux.shape[-2] - 1
                max_obx = ob.shape[-1] - aux.shape[-1] - 1

                # We need to re-calculate the current error
                PCK.build_aux(aux, addr, ob, pr)
                aux[:] = FW(aux)
                PCK.log_likelihood_ml(aux, addr, I, w, err_phot)
                error_state = np.zeros_like(err_phot)
                error_state[:] = err_phot
                PCK.mangler.setup_shifts(self.curiter, nframes=addr.shape[0])

                log(4, 'Position refinement trial: iteration %s' % (self.curiter))
                for i in range(PCK.mangler.nshifts):
                    PCK.mangler.get_address(i, addr, mangled_addr, max_oby, max_obx)
                    PCK.build_aux(aux, mangled_addr, ob, pr)
                    aux[:] = FW(aux)
                    PCK.log_likelihood_ml(aux, mangled_addr, I, w, err_phot)
                    PCK.update_addr_and_error_state(addr, error_state, mangled_addr, err_phot)

                prep.err_phot = error_state
                prep.addr = addr

    def engine_finalize(self):
        """
        try deleting ever helper contianer
        """
        if self.do_position_refinement and self.p.position_refinement.record:
            for label, d in self.di.storages.items():
                prep = self.diff_info[d.ID]
                res = self.kernels[prep.label].resolution
                for i,view in enumerate(d.views):
                    for j,(pname, pod) in enumerate(view.pods.items()):
                        delta = (prep.addr[i][j][1][1:] - prep.original_addr[i][j][1][1:]) * res
                        pod.ob_view.coord += delta
                        pod.ob_view.storage.update_views(pod.ob_view)
            self.ptycho.record_positions = True

        # Save floating intensities into runtime
        float_intens_coeff = {}
        for label, d in self.di.storages.items():
            prep = self.diff_info[d.ID]
            float_intens_coeff[label] = prep.float_intens_coeff
        self.ptycho.runtime["float_intens"] = parallel.gather_dict(float_intens_coeff)
        super().engine_finalize()


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
                            / (1. / self.Irenorm + d.data)).astype(d.data.dtype)

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
        ob_grad << 0.
        pr_grad << 0.

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
            fic = prep.float_intens_coeff

            # local references
            ob = self.engine.ob.S[oID].data
            obg = ob_grad.S[oID].data
            pr = self.engine.pr.S[pID].data
            prg = pr_grad.S[pID].data
            I = prep.I

            # make propagated exit (to buffer)
            AWK.build_aux_no_ex(aux, addr, ob, pr, add=False)

            # forward prop
            aux[:] = FW(aux)

            GDK.make_model(aux, addr)

            if self.p.floating_intensities:
                GDK.floating_intensity(addr, w, I, fic)

            GDK.main(aux, addr, w, I)
            GDK.error_reduce(addr, err_phot)
            aux[:] = BW(aux)

            POK.ob_update_ML(addr, obg, pr, aux)
            POK.pr_update_ML(addr, prg, ob, aux)

        for dID, prep in self.engine.diff_info.items():
            err_phot = prep.err_phot
            LL += err_phot.sum()
            err_phot /= np.prod(prep.weights.shape[-2:])
            err_fourier = np.zeros_like(err_phot)
            err_exit = np.zeros_like(err_phot)
            errs = np.ascontiguousarray(np.vstack([err_fourier, err_phot, err_exit]).T)
            error_dct.update(zip(prep.view_IDs, errs))

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

    def poly_line_coeffs_ob(self, c_ob_h):
        """
        Compute the coefficients of the polynomial for line minimization
        in direction h for the object
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

            FW = kern.FW

            # get addresses and auxilliary array
            addr = prep.addr
            w = prep.weights
            fic = prep.float_intens_coeff

            # local references
            ob = self.ob.S[oID].data
            ob_h = c_ob_h.S[oID].data
            pr = self.pr.S[pID].data
            I = self.di.S[dID].data

            # make propagated exit (to buffer)
            AWK.build_aux_no_ex(f, addr, ob, pr, add=False)
            AWK.build_aux_no_ex(a, addr, ob_h, pr, add=False)

            # forward prop
            f[:] = FW(f)
            a[:] = FW(a)

            GDK.make_a012(f, a, 0, addr, I, fic) # FIXME: need new kernel
            GDK.fill_b(addr, Brenorm, w, B)

        parallel.allreduce(B)

        # Object regularizer
        if self.regularizer:
            for name, s in self.ob.storages.items():
                B += Brenorm * self.regularizer.poly_line_coeffs(
                    c_ob_h.storages[name].data, s.data)

        self.B = B

        return B

def poly_line_coeffs_pr(self, c_pr_h):
        """
        Compute the coefficients of the polynomial for line minimization
        in direction h for the probe
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

            FW = kern.FW

            # get addresses and auxilliary array
            addr = prep.addr
            w = prep.weights
            fic = prep.float_intens_coeff

            # local references
            ob = self.ob.S[oID].data
            pr = self.pr.S[pID].data
            pr_h = c_pr_h.S[pID].data
            I = self.di.S[dID].data

            # make propagated exit (to buffer)
            AWK.build_aux_no_ex(f, addr, ob, pr, add=False)
            AWK.build_aux_no_ex(a, addr, ob, pr_h, add=True)

            # forward prop
            f[:] = FW(f)
            a[:] = FW(a)

            GDK.make_a012(f, a, 0, addr, I, fic) # FIXME: need new kernel
            GDK.fill_b(addr, Brenorm, w, B)

        parallel.allreduce(B)

        self.B = B

        return B
