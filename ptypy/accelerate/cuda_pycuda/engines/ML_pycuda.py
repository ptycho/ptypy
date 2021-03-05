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
from pycuda import gpuarray
import pycuda.driver as cuda
from pycuda.tools import DeviceMemoryPool
from collections import deque

from ptypy.engines import register
from ptypy.accelerate.base.engines.ML_serial import ML_serial, BaseModelSerial
from ptypy import utils as u
from ptypy.utils.verbose import logger
from ptypy.utils import parallel
from .. import get_context
from ..kernels import GradientDescentKernel, AuxiliaryWaveKernel, PoUpdateKernel, \
    PositionCorrectionKernel, PropagationKernel
from ..array_utils import ArrayUtilsKernel, DerivativesKernel, GaussianSmoothingKernel

from ptypy.accelerate.base import address_manglers

# for debugging
import h5py

__all__ = ['ML_pycuda']


class MemoryManager:

    def __init__(self, fraction=0.7):
        self.fraction = fraction
        self.dmp = DeviceMemoryPool()
        self.queue_in = cuda.Stream()
        self.queue_out = cuda.Stream()
        self.mem_avail = None
        self.mem_total = None
        self.get_free_memory()
        self.on_device = {}
        self.on_device_inv = {}
        self.out_events = deque()
        self.bytes = 0

    def get_free_memory(self):
        self.mem_avail, self.mem_total = cuda.mem_get_info()

    def device_is_full(self, nbytes = 0):
        return (nbytes + self.bytes) > self.mem_avail

    def to_gpu(self, ar, ev=None):
        """
        Issues asynchronous copy to device. Waits for optional event ev
        Emits event for other streams to synchronize with
        """
        stream = self.queue_in
        id_cpu = id(ar)
        gpu_ar = self.on_device.get(id_cpu)

        if gpu_ar is None:
            if ev is not None:
                stream.wait_for_event(ev)
            if self.device_is_full(ar.nbytes):
                self.wait_for_freeing_events(ar.nbytes)

            # TOD0: try /except with garbage collection to make sure there is space
            gpu_ar = gpuarray.to_gpu_async(ar, allocator=self.dmp.allocate, stream=stream)

            # keeps gpuarray alive
            self.on_device[id_cpu] = gpu_ar

            # for deleting later
            self.on_device_inv[id(gpu_ar)] = ar

            self.bytes += gpu_ar.mem_size * gpu_ar.dtype.itemsize


        ev = cuda.Event()
        ev.record(stream)
        return ev, gpu_ar


    def wait_for_freeing_events(self, nbytes):
        """
        Wait until at least nbytes have been copied back to the host. Or marked for deletion
        """
        freed = 0
        if not self.out_events:
            #print('Waiting for memory to be released on device failed as no release event was scheduled')
            self.queue_out.synchronize()
        while self.out_events and freed < nbytes:
            ev, id_cpu, id_gpu = self.out_events.popleft()
            gpu_ar = self.on_device.pop(id_cpu)
            cpu_ar = self.on_device_inv.pop(id_gpu)
            ev.synchronize()
            freed += cpu_ar.nbytes
            self.bytes -= gpu_ar.mem_size * gpu_ar.dtype.itemsize

    def mark_release_from_gpu(self, gpu_ar, to_cpu=False, ev=None):
        stream = self.queue_out
        if ev is not None:
            stream.wait_for_event(ev)
        if to_cpu:
            cpu_ar = self.on_device_inv[id(gpu_ar)]
            gpu_ar.get_asynch(stream, host_array)

        ev_out = cuda.Event()
        ev_out.record(stream)
        self.out_events.append((ev_out, id(cpu_ar), id(gpu_ar)))
        return ev_out


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

    [use_cuda_device_memory_pool]
    default = True
    type = bool
    help = For GPU, use a device memory pool

    """

    def __init__(self, ptycho_parent, pars=None):
        """
        Maximum likelihood reconstruction engine.
        """
        super().__init__(ptycho_parent, pars)

    def engine_initialize(self):
        """
        Prepare for ML reconstruction.
        """
        self.context, self.queue = get_context(new_context=True, new_queue=True)

        if self.p.use_cuda_device_memory_pool:
            self._dmp = DeviceMemoryPool()
            self.allocate = self._dmp.allocate
        else:
            self._dmp = None
            self.allocate = cuda.mem_alloc

        self.queue_transfer = cuda.Stream()
        
        self.GSK = GaussianSmoothingKernel(queue=self.queue)
        
        super().engine_initialize()
        #self._setup_kernels()

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
            ash = (fpc * nmodes,) + tuple([int(s) for s in geo.shape])
            aux = gpuarray.zeros(ash, dtype=np.complex64)
            kern.aux = aux
            kern.a = gpuarray.zeros(ash, dtype=np.complex64)
            kern.b = gpuarray.zeros(ash, dtype=np.complex64)

            # setup kernels, one for each SCAN.
            kern.GDK = GradientDescentKernel(aux, nmodes, queue=self.queue, math_type='double')
            kern.GDK.allocate()

            kern.POK = PoUpdateKernel(queue_thread=self.queue, denom_type='float')
            kern.POK.allocate()

            kern.AWK = AuxiliaryWaveKernel(queue_thread=self.queue)
            kern.AWK.allocate()

            kern.PROP = PropagationKernel(aux, geo.propagator, queue_thread=self.queue)
            kern.PROP.allocate()


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

    def _set_pr_ob_ref_for_data(self, dev='gpu', container=None, sync_copy=False):
        """
        Overloading the context of Storage.data here, to allow for in-place math on Container instances:
        """
        if container is not None:
            if container.original==self.pr or container.original==self.ob:
                for s in container.S.values():
                    # convert data here
                    if dev == 'gpu':
                        s.data = s.gpu
                        if sync_copy: s.gpu.set(s.cpu)
                    elif dev == 'cpu':
                        s.data = s.cpu
                        if sync_copy:
                            s.gpu.get(s.cpu)
                            #print('%s to cpu' % s.ID)
        else:
            for container in self.ptycho.containers.values():
                self._set_pr_ob_ref_for_data(dev=dev, container=container, sync_copy=sync_copy)

    def _get_smooth_gradient(self, data, sigma):
        tmp = gpuarray.empty(data.shape, dtype=np.complex64)
        self.GSK.convolution(data, tmp, [sigma, sigma])
        return tmp

    def _replace_ob_grad(self):
        new_ob_grad = self.ob_grad_new
        # Smoothing preconditioner
        if self.smooth_gradient:
            self.smooth_gradient.sigma *= (1. - self.p.smooth_gradient_decay)
            for name, s in new_ob_grad.storages.items():
                s.gpu = self._get_smooth_gradient(s.gpu, self.smooth_gradient.sigma)

        return self._replace_grad(self.ob_grad, new_ob_grad)

    def _replace_pr_grad(self):
        new_pr_grad = self.pr_grad_new
        # probe support
        if self.p.probe_update_start <= self.curiter:
            # Apply probe support if needed
            for name, s in new_pr_grad.storages.items():

                # DtoH copies
                s.gpu.get(s.cpu)
                self._set_pr_ob_ref_for_data('cpu')

                # TODO this needs to be implemented on GPU
                self.support_constraint(s)

                # HtoD cause we continue on gpu
                s.gpu.set(s.cpu)
                self._set_pr_ob_ref_for_data('gpu')

        else:
            new_pr_grad.fill(0.)

        return self._replace_grad(self.pr_grad , new_pr_grad)

    def _replace_grad(self, grad, new_grad):
        norm = np.double(0.)
        dot = np.double(0.)
        for name, new in new_grad.storages.items():
            old = grad.storages[name]
            norm += self._dot_kernel(new.gpu,new.gpu).get()[0]
            dot += self._dot_kernel(new.gpu,old.gpu).get()[0]
            old.gpu[:] = new.gpu
        return norm, dot

    def engine_iterate(self, num=1):
        err = super().engine_iterate(num)
        # copy all data back to cpu
        self._set_pr_ob_ref_for_data(dev='cpu', container=None, sync_copy=True)
        return err

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
            prep.fic_gpu = gpuarray.ones_like(prep.err_phot_gpu)

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
        for name, s in self.pr.S.items():
            s.data = s.gpu.get() # need this, otherwise getting segfault once context is detached
            # no longer need those
            del s.gpu
            del s.cpu
        for name, s in self.ob.S.items():
            s.data = s.gpu.get() # need this, otherwise getting segfault once context is detached
            # no longer need those
            del s.gpu
            del s.cpu

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

        if self.p.reg_del2:
            self.regularizer = Regul_del2_pycuda(
                self.p.reg_del2_amplitude,
                queue=self.engine.queue,
                allocator=self.engine.allocate
            )
        else:
            self.regularizer = None

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

            FW = kern.PROP.fw
            BW = kern.PROP.bw

            # get addresses and auxilliary array
            addr = prep.addr_gpu
            fic = prep.fic_gpu

            err_phot = prep.err_phot_gpu
            # local references
            ob = self.engine.ob.S[oID].data
            obg = ob_grad.S[oID].data
            pr = self.engine.pr.S[pID].data
            prg = pr_grad.S[pID].data

            # TODO streaming?
            #w = gpuarray.to_gpu(prep.weights)
            #I = gpuarray.to_gpu(prep.I)
            stream = self.engine.queue_transfer
            # TODO keep alive
            w = gpuarray.to_gpu_async(prep.weights, allocator=self.engine.allocate, stream=stream)
            I = gpuarray.to_gpu_async(prep.I, allocator=self.engine.allocate, stream=stream)
            ev = cuda.Event()
            ev.record(stream)

            # make propagated exit (to buffer)
            AWK.build_aux_no_ex(aux, addr, ob, pr, add=False)

            # debugging
            if self.p.debug and parallel.master and (self.engine.curiter == self.p.debug_iter):
                with h5py.File(self.p.debug + "/ml_pycuda_before_fft_%04d.h5" %self.engine.curiter, "w") as f:
                    f["aux"] = aux.get()[0]
                    f["addr"] = addr.get()

            # forward prop
            FW(aux, aux)

            # debugging
            if self.p.debug and parallel.master and (self.engine.curiter == self.p.debug_iter):
                with h5py.File(self.p.debug + "/ml_pycuda_after_fft_%04d.h5" %self.engine.curiter, "w") as f:
                    f["aux"] = aux.get()[0]
                    f["addr"] = addr.get()
            GDK.make_model(aux, addr)

            GDK.queue.wait_for_event(ev)

            # debugging
            if self.p.debug and parallel.master and (self.engine.curiter == self.p.debug_iter):
                with h5py.File(self.p.debug + "/ml_pycuda_before_floating_%04d.h5" %self.engine.curiter, "w") as f:
                    f["Imodel"] = GDK.gpu.Imodel.get()
                    f["fic"] = fic.get()

            if self.p.floating_intensities:
                GDK.floating_intensity(addr, w, I, fic)

            # debugging
            if self.p.debug and parallel.master and (self.engine.curiter == self.p.debug_iter):
                with h5py.File(self.p.debug + "/ml_pycuda_after_floating_%04d.h5" %self.engine.curiter, "w") as f:
                    f["Imodel"] = GDK.gpu.Imodel.get()
                    f["fic"] = fic.get()
            
            GDK.main(aux, addr, w, I)
            ev = cuda.Event()
            ev.record(GDK.queue)

            GDK.error_reduce(addr, err_phot)

            # debugging
            if self.p.debug and parallel.master and (self.engine.curiter == self.p.debug_iter):
                with h5py.File(self.p.debug + "/ml_pycuda_after_error_reduce_%04d.h5" %self.engine.curiter, "w") as f:
                    f["LLerr"] = GDK.gpu.LLerr.get()
                    f["err_phot"] = err_phot.get()

            BW(aux, aux)

            use_atomics = self.p.object_update_cuda_atomics
            addr = prep.addr_gpu if use_atomics else prep.addr2_gpu
            POK.ob_update_ML(addr, obg, pr, aux, atomics=use_atomics)

            use_atomics = self.p.probe_update_cuda_atomics
            addr = prep.addr_gpu if use_atomics else prep.addr2_gpu
            POK.pr_update_ML(addr, prg, ob, aux, atomics=use_atomics)

            GDK.queue.synchronize()

        # TODO we err_phot.sum, but not necessarily this error_dct until the end of contiguous iteration
        for dID, prep in self.engine.diff_info.items():
            err_phot = prep.err_phot_gpu.get()
            LL += err_phot.sum()
            err_phot /= np.prod(prep.weights.shape[-2:])
            err_fourier = np.zeros_like(err_phot)
            err_exit = np.zeros_like(err_phot)
            errs = np.ascontiguousarray(np.vstack([err_fourier, err_phot, err_exit]).T)
            error_dct.update(zip(prep.view_IDs, errs))

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

            FW = kern.PROP.fw

            # get addresses and auxiliary arrays
            addr = prep.addr_gpu
            fic = prep.fic_gpu

            # TODO streaming?
            #w = gpuarray.to_gpu(prep.weights)
            #I = gpuarray.to_gpu(prep.I)
            stream = self.engine.queue_transfer
            w = gpuarray.to_gpu_async(prep.weights, allocator=self.engine.allocate, stream=stream)
            I = gpuarray.to_gpu_async(prep.I, allocator=self.engine.allocate, stream=stream)
            ev = cuda.Event()
            ev.record(stream)

            # local references
            ob = self.ob.S[oID].data
            ob_h = c_ob_h.S[oID].data
            pr = self.pr.S[pID].data
            pr_h = c_pr_h.S[pID].data

            # make propagated exit (to buffer)
            AWK.build_aux_no_ex(f, addr, ob, pr, add=False)
            AWK.build_aux_no_ex(a, addr, ob_h, pr, add=False)
            AWK.build_aux_no_ex(a, addr, ob, pr_h, add=True)
            AWK.build_aux_no_ex(b, addr, ob_h, pr_h, add=False)

            # forward prop
            FW(f,f)
            FW(a,a)
            FW(b,b)

            GDK.queue.wait_for_event(ev)
            GDK.make_a012(f, a, b, addr, I, fic)

            """
            if self.p.floating_intensities:
                A0 *= self.float_intens_coeff[dname]
                A1 *= self.float_intens_coeff[dname]
                A2 *= self.float_intens_coeff[dname]
            """
            GDK.fill_b(addr, Brenorm, w, B)
            GDK.queue.synchronize()

        B = B.get()
        parallel.allreduce(B)

        # Object regularizer
        if self.regularizer:
            for name, s in self.ob.storages.items():
                B += Brenorm * self.regularizer.poly_line_coeffs(
                    c_ob_h.storages[name].data, s.data)

        self.B = B

        return B

class Regul_del2_pycuda(object):
    """\
    Squared gradient regularizer (Gaussian prior).

    This class applies to any numpy array.
    """
    def __init__(self, amplitude, axes=[-2, -1], queue=None, allocator=None):
        # Regul.__init__(self, axes)
        self.axes = axes
        self.amplitude = amplitude
        self.delxy = None
        self.g = None
        self.LL = None
        self.queue = queue
        self.AUK = ArrayUtilsKernel(queue=queue)
        self.DELK_c = DerivativesKernel(np.complex64, queue=queue)
        self.DELK_f = DerivativesKernel(np.float32, queue=queue)

        if allocator is None:
            self._dmp = DeviceMemoryPool()
            self.allocator=self._dmp.allocate
        else:
            self.allocator = allocator
            self._dmp= None

        empty = lambda x: gpuarray.empty(x.shape, x.dtype, allocator=self.allocator)

        def delxb(x, axis=-1):
            out = empty(x)
            if x.dtype == np.float32:
                self.DELK_f.delxb(x, out, axis)
            elif x.dtype == np.complex64:
                self.DELK_c.delxb(x, out, axis)
            else:
                raise TypeError("Type %s invalid for derivatives" % x.dtype)
            return out

        self.delxb = delxb

        def delxf(x, axis=-1):
            out = empty(x)
            if x.dtype == np.float32:
                self.DELK_f.delxf(x, out, axis)
            elif x.dtype == np.complex64:
                self.DELK_c.delxf(x, out, axis)
            else:
                raise TypeError("Type %s invalid for derivatives" % x.dtype)
            return out

        self.delxf = delxf
        self.norm =  lambda x : self.AUK.norm2(x).get().item()
        self.dot = lambda x, y : self.AUK.dot(x,y).get().item()

        from pycuda.elementwise import ElementwiseKernel
        self._grad_reg_kernel = ElementwiseKernel(
            "pycuda::complex<float> *g, float fac, \
            pycuda::complex<float> *py, pycuda::complex<float> *px, \
            pycuda::complex<float> *my, pycuda::complex<float> *mx",
            "g[i] = (px[i]+py[i]-my[i]-mx[i]) * fac",
            "grad_reg",
        )
        def grad(amp, px,py, mx, my):
            out = empty(px)
            self._grad_reg_kernel(out, amp, py, px, mx, my, stream=self.queue)
            return out
        self.reg_grad = grad

    def grad(self, x):
        """
        Compute and return the regularizer gradient given the array x.
        """
        ax0, ax1 = self.axes
        del_xf = self.delxf(x, axis=ax0)
        del_yf = self.delxf(x, axis=ax1)
        del_xb = self.delxb(x, axis=ax0)
        del_yb = self.delxb(x, axis=ax1)

        self.delxy = [del_xf, del_yf, del_xb, del_yb]

        # TODO this one might be slow, maybe try with elementwise kernel
        #self.g = (del_xb + del_yb - del_xf - del_yf) * 2. * self.amplitude
        self.g = self.reg_grad(2. * self.amplitude, del_xb, del_yb, del_xf, del_yf)


        self.LL = self.amplitude * (self.norm(del_xf)
                               + self.norm(del_yf)
                               + self.norm(del_xb)
                               + self.norm(del_yb))

        return self.g

    def poly_line_coeffs(self, h, x=None):
        ax0, ax1 = self.axes
        if x is None:
            del_xf, del_yf, del_xb, del_yb = self.delxy
        else:
            del_xf = self.delxf(x, axis=ax0)
            del_yf = self.delxf(x, axis=ax1)
            del_xb = self.delxb(x, axis=ax0)
            del_yb = self.delxb(x, axis=ax1)

        hdel_xf = self.delxf(h, axis=ax0)
        hdel_yf = self.delxf(h, axis=ax1)
        hdel_xb = self.delxb(h, axis=ax0)
        hdel_yb = self.delxb(h, axis=ax1)

        c0 = self.amplitude * (self.norm(del_xf)
                               + self.norm(del_yf)
                               + self.norm(del_xb)
                               + self.norm(del_yb))

        c1 = 2 * self.amplitude * (self.dot(del_xf, hdel_xf)
                                 + self.dot(del_yf, hdel_yf)
                                 + self.dot(del_xb, hdel_xb)
                                 + self.dot(del_yb, hdel_yb))

        c2 = self.amplitude * (self.norm(hdel_xf)
                               + self.norm(hdel_yf)
                               + self.norm(hdel_xb)
                               + self.norm(hdel_yb))

        self.coeff = np.array([c0, c1, c2])
        return self.coeff
