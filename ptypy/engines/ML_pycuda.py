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
from pycuda import gpuarray
import pycuda.driver as cuda

from . import register
from .ML import ML, BaseModel, prepare_smoothing_preconditioner, Regul_del2
from .ML_serial import ML_serial, BaseModelSerial
from .. import utils as u
from ..utils.verbose import logger
from ..utils import parallel
from .utils import Cnorm2, Cdot
from ..accelerate import py_cuda as gpu
from ..accelerate.py_cuda.kernels import GradientDescentKernel, AuxiliaryWaveKernel, PoUpdateKernel, PositionCorrectionKernel
from ..accelerate.array_based.kernels import GradientDescentKernel as GDK_serial
from ..accelerate.py_cuda.array_utils import ArrayUtilsKernel
from ..accelerate.array_based import address_manglers

__all__ = ['ML_pycuda']



@register()
class ML_pycuda(ML_serial):

    """
    Defaults:

    [probe_update_cuda_atomics]
    default = False
    type = bool
    help = For GPU, use the atomics version for probe update kernel

    [object_update_cuda_atomics]
    default = True
    type = bool
    help = For GPU, use the atomics version for object update kernel

    """

    def __init__(self, ptycho_parent, pars=None):
        """
        Maximum likelihood reconstruction engine.
        """
        super().__init__(ptycho_parent, pars)

        self.context, self.queue = gpu.get_context()

        self.dmp = DeviceMemoryPool()
        self.queue_transfer = cuda.Stream()

    def engine_initialize(self):
        """
        Prepare for ML reconstruction.
        """
        super().engine_initialize()
        self._setup_kernels()

    def _setup_kernels(self):
        """
        Setup kernels, one for each scan. Derive scans from ptycho class
        """
        AUK = ArrayUtilsKernel(queue=self.queue)
        self._dot_kernel = AUK.dot
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
            aux = gpuarray.zeros(ash, dtype=np.complex64)
            kern.aux = aux
            kern.a = gpuarray.zeros(ash, dtype=np.complex64)
            kern.b = gpuarray.zeros(ash, dtype=np.complex64)

            # setup kernels, one for each SCAN.
            kern.GDK = GradientDescentKernel(aux, nmodes, queue=self.queue)
            kern.GDK.allocate()

            #kern.GDKs = GDK_serial(aux.get(), nmodes)
            #kern.GDKs.allocate()

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

    def _set_pr_ob_ref_for_data(self, dev='gpu', container=None):
        """
        Overloading the context of Storage.data here, to allow for in-place math on Container instances:
        """
        if container is not None and container.original != self.pr and container.original != self.ob:
            for s in container.S.values():
                # convert data here
                if dev == 'gpu':
                    s.data = s.gpu
                elif dev == 'cpu':
                    s.data = s.cpu
        else:
            for container in self.ptycho.containers.values():
                self._set_pr_ob_ref_for_data(dev=dev, container=container)

    def _replace_ob_grad(self):
        new_ob_grad = self.ob_grad_new
        # Smoothing preconditioner
        if self.smooth_gradient:
            self.smooth_gradient.sigma *= (1. - self.p.smooth_gradient_decay)
            for name, s in new_ob_grad.storages.items():
                s.data[:] = self.smooth_gradient(s.data)

        return self._replace_grad(self.ob_grad, new_ob_grad)

    def _replace_pr_grad(self):
        new_pr_grad = self.pr_grad_new
        # probe support
        if self.p.probe_update_start <= self.curiter:
            # Apply probe support if needed
            for name, s in new_pr_grad.storages.items():
                self.support_constraint(s)
        else:
            new_pr_grad.fill(0.)

        return self._replace_grad(self.pr_grad , new_pr_grad)

    def _replace_grad(self, grad, new_grad):
        norm = np.double(0.)
        dot = np.double(0.)
        for name, new in new_grad.storages.items():
            old = grad.storages[name]
            norm += self._dot_kernel(new.gpu,new.gpu)
            dot += self._dot_kernel(new.gpu,old.gpu)
            old.gpu[:] = new.gpu
        return norm, dot

    def engine_prepare(self):

        super().engine_prepare()
        ## Serialize new data ##
        use_tiles = (not self.p.probe_update_cuda_atomics) or (not self.p.object_update_cuda_atomics)

        # recursive copy to gpu for probe and object
        for _cname, c in self.ptycho.containers.items():
            if c.original != self.pr and c.original != self.ob:
                continue
            for _sname, s in c.S.items():
                # convert data here
                s.gpu = gpuarray.to_gpu(s.data)
                s.cpu = cuda.pagelocked_empty(s.data.shape, s.data.dtype, order="C")
                s.cpu[:] = s.data

        for label, d in self.ptycho.new_data:
            prep = self.diff_info[d.ID]
            prep.err_phot_gpu = gpuarray.to_gpu(prep.err_phot)

            if use_tiles:
                prep.addr2 = np.ascontiguousarray(np.transpose(prep.addr, (2, 3, 0, 1)))

            prep.addr_gpu = gpuarray.to_gpu(prep.addr)

            # Todo: Which address to pick?
            if use_tiles:
                prep.addr2_gpu = gpuarray.to_gpu(prep.addr2)

            prep.I = cuda.pagelocked_empty(d.data.shape, d.data.dtype, order="C", mem_flags=4)
            prep.I[:] = d.data

    def engine_finalize(self):
        """
        try deleting ever helper contianer
        """

        #self.queue.synchronize()
        self.context.detach()
        super().engine_finalize()

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
            w = (self.Irenorm * self.engine.ma.S[d.ID].data
                       / (1. / self.Irenorm + d.data)).astype(d.data.dtype)
            prep.weights = cuda.pagelocked_empty(w.shape, w.dtype, order="C", mem_flags=4)
            prep.weights[:] = w

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

        self.engine._set_pr_ob_ref_for_data('gpu')
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
            addr = prep.addr_gpu

            err_phot = prep.err_phot_gpu
            # local references
            # ob = gpuarray.to_gpu(self.engine.ob.S[oID].data)
            # obg = gpuarray.to_gpu(ob_grad.S[oID].data)
            # pr = gpuarray.to_gpu(self.engine.pr.S[pID].data)
            # prg = gpuarray.to_gpu(pr_grad.S[pID].data)
            ob = self.engine.ob.S[oID].data
            obg = ob_grad.S[oID].data
            pr = self.engine.pr.S[pID].data
            prg = pr_grad.S[pID].data

            # TODO streaming?
            #w = gpuarray.to_gpu(prep.weights)
            #I = gpuarray.to_gpu(prep.I)
            stream = self.engine.queue_transfer
            w = gpuarray.to_gpu_async(prep.weights, allocator=self.engine.dmp.allocate, stream=stream)
            I = gpuarray.to_gpu_async(prep.I, allocator=self.engine.dmp.allocate, stream=stream)
            ev = cuda.Event()
            ev.record(stream)

            # make propagated exit (to buffer)
            AWK.build_aux_no_ex(aux, addr, ob, pr, add=False)

            # forward prop
            FW(aux, aux)
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
            #LLerr = GDK.gpu.LLerr.get()
            #print(np.allclose(GDK.gpu.Imodel.get(), kern.GDKs.npy.Imodel))
            #print(np.isnan(LLerr).any())
            #print(np.isnan(GDK.gpu.Imodel.get()).any())
            #print(np.isnan(I.get()).any())
            #print(np.isnan(aux.get()).any())
            #aux2 = aux.get()
            GDK.queue.wait_for_event(ev)
            GDK.main(aux, addr, w, I)
            #kern.GDKs.main(aux2, addr.get(), w.get(), I.get())
            #LLerr = GDK.gpu.LLerr.get()
            #LLerrs = kern.GDKs.npy.LLerr
            #na = np.isnan(LLerr)
            #print('main cpu made nan', np.isnan(LLerrs).any())
            #print('main gpu made nan', na.any(), na.sum())
            #print('diff', LLerr-LLerrs)
            #print('LLerr', LLerrs)
            #print(I.get()[na])
            #print(w.get()[na])
            #print(GDK.gpu.LLerr.get()[0])
            GDK.error_reduce(addr, err_phot)
            BW(aux, aux)
            #print(err_phot.get())
            use_atomics = self.p.object_update_cuda_atomics
            addr = prep.addr_gpu if use_atomics else prep.addr2_gpu
            POK.ob_update_ML(addr, obg, pr, aux, atomics=use_atomics)

            use_atomics = self.p.probe_update_cuda_atomics
            addr = prep.addr_gpu if use_atomics else prep.addr2_gpu
            POK.pr_update_ML(addr, prg, ob, aux, atomics=use_atomics)

        for dID, prep in self.engine.diff_info.items():
            err_phot = prep.err_phot_gpu.get() / np.prod(prep.weights.shape)
            err_fourier = np.zeros_like(err_phot)
            err_exit = np.zeros_like(err_phot)
            errs = np.ascontiguousarray(np.vstack([err_fourier, err_phot, err_exit]).T)
            error_dct.update(zip(prep.view_IDs, errs))
            LL += err_phot.sum()

        # MPI reduction of gradients

        # DtoH copies
        for s in ob_grad.S.values():
            s.gpu.get(s.cpu)
        for s in pr_grad.S.values():
            s.gpu.get(s.cpu)
        self.engine._set_pr_ob_ref_for_data('cpu')

        ob_grad.allreduce()
        pr_grad.allreduce()
        parallel.allreduce(LL)

        # HtoD cause we continue on gpu
        for s in ob_grad.S.values():
            s.gpu.set(s.cpu)
        for s in pr_grad.S.values():
            s.gpu.set(s.cpu)
        self.engine._set_pr_ob_ref_for_data('gpu')

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
        self.engine._set_pr_ob_ref_for_data('gpu')

        B = gpuarray.zeros((3,), dtype=np.float32) # does not accept np.longdouble
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
            addr = prep.addr_gpu

            # TODO streaming?
            #w = gpuarray.to_gpu(prep.weights)
            #I = gpuarray.to_gpu(prep.I)
            stream = self.engine.queue_transfer
            w = gpuarray.to_gpu_async(prep.weights, allocator=self.engine.dmp.allocate, stream=stream)
            I = gpuarray.to_gpu_async(prep.I, allocator=self.engine.dmp.allocate, stream=stream)
            ev = cuda.Event()
            ev.record(stream)

            # local references
            ob = self.ob.S[oID].data
            ob_h = c_ob_h.S[oID].data
            pr = self.pr.S[pID].data
            pr_h = c_pr_h.S[pID].data
            # ob = gpuarray.to_gpu(self.ob.S[oID].data)
            # ob_h = gpuarray.to_gpu(c_ob_h.S[oID].data)
            # pr = gpuarray.to_gpu(self.pr.S[pID].data)
            # pr_h = gpuarray.to_gpu(c_pr_h.S[pID].data)


            # make propagated exit (to buffer)
            AWK.build_aux_no_ex(f, addr, ob, pr, add=False)
            AWK.build_aux_no_ex(a, addr, ob_h, pr, add=False)
            AWK.build_aux_no_ex(a, addr, ob, pr_h, add=True)
            AWK.build_aux_no_ex(b, addr, ob_h, pr_h, add=False)

            # forward prop
            FW(f,f)
            FW(a,a)
            FW(b,b)

            GDK.make_a012(f, a, b, addr, I)

            """
            if self.p.floating_intensities:
                A0 *= self.float_intens_coeff[dname]
                A1 *= self.float_intens_coeff[dname]
                A2 *= self.float_intens_coeff[dname]
            """
            GDK.fill_b(addr, Brenorm, w, B)

        B = B.get()
        parallel.allreduce(B)

        # Object regularizer
        if self.regularizer:
            for name, s in self.ob.storages.items():
                B += Brenorm * self.regularizer.poly_line_coeffs(
                    ob_h.storages[name].data, s.data)

        self.B = B

        return B