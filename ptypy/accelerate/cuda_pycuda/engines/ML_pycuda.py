# -*- coding: utf-8 -*-
"""
Maximum Likelihood reconstruction engine.

TODO.

  * Implement other regularizers

This file is part of the PTYPY package.

    :copyright: Copyright 2014 by the PTYPY team, see AUTHORS.
    :license: see LICENSE for details.
"""
import numpy as np
from pycuda import gpuarray
import pycuda.driver as cuda
import pycuda.cumath
from pycuda.tools import DeviceMemoryPool

from ptypy.engines import register
from ptypy.accelerate.base.engines.ML_serial import ML_serial, BaseModelSerial
from ptypy import utils as u
from ptypy.utils.verbose import logger, log
from ptypy.utils import parallel
from .. import get_context
from ..kernels import PropagationKernel, RealSupportKernel, FourierSupportKernel
from ..kernels import GradientDescentKernel, AuxiliaryWaveKernel, PoUpdateKernel, PositionCorrectionKernel
from ..array_utils import ArrayUtilsKernel, DerivativesKernel, GaussianSmoothingKernel, TransposeKernel

from ..mem_utils import GpuDataManager
from ptypy.accelerate.base import address_manglers

__all__ = ['ML_pycuda']

MAX_BLOCKS = 99999  # can be used to limit the number of blocks, simulating that they don't fit
#MAX_BLOCKS = 3  # can be used to limit the number of blocks, simulating that they don't fit

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

    [fft_lib]
    default = reikna
    type = str
    help = Choose the pycuda-compatible FFT module.
    doc = One of:
      - ``'reikna'`` : the reikna packaga (fast load, competitive compute for streaming)
      - ``'cuda'`` : ptypy's cuda wrapper (delayed load, but fastest compute if all data is on GPU)
      - ``'skcuda'`` : scikit-cuda (fast load, slowest compute due to additional store/load stages)
    choices = 'reikna','cuda','skcuda'
    userlevel = 2

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

        self.qu_htod = cuda.Stream()
        self.qu_dtoh = cuda.Stream()

        self.GSK = GaussianSmoothingKernel(queue=self.queue)
        self.GSK.tmp = None

        # Real/Fourier Support Kernel
        self.RSK = {}
        self.FSK = {}

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
            kern.scanmodel = type(scan).__name__
            self.kernels[label] = kern

            # TODO: needs to be adapted for broad bandwidth
            geo = scan.geometries[0]

            # Get info to shape buffer arrays
            fpc = scan.max_frames_per_block

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
            kern.GDK = GradientDescentKernel(aux, nmodes, queue=self.queue, math_type="double")
            kern.GDK.allocate()

            kern.POK = PoUpdateKernel(queue_thread=self.queue)
            kern.POK.allocate()

            kern.AWK = AuxiliaryWaveKernel(queue_thread=self.queue)
            kern.AWK.allocate()

            kern.TK = TransposeKernel(queue=self.queue)

            kern.PROP = PropagationKernel(aux, geo.propagator, queue_thread=self.queue, fft=self.p.fft_lib)
            kern.PROP.allocate()
            kern.resolution = geo.resolution[0]

            if self.do_position_refinement:
                kern.PCK = PositionCorrectionKernel(aux, nmodes, self.p.position_refinement, geo.resolution, queue_thread=self.queue)
                kern.PCK.allocate()

        mag_mem = 0
        for scan, kern in self.kernels.items():
            mag_mem = max(kern.aux.nbytes // 2, mag_mem)
        ma_mem = mag_mem
        mem = cuda.mem_get_info()[0]
        blk = ma_mem + mag_mem
        fit = int(mem - 200 * 1024 * 1024) // blk  # leave 200MB room for safety
        if not fit:
            log(1,"Cannot fit memory into device, if possible reduce frames per block. Exiting...")
            self.context.pop()
            self.context.detach()
            raise SystemExit("ptypy has been exited.")

        # TODO grow blocks dynamically
        nma = min(fit, MAX_BLOCKS)
        log(4, 'Free memory on device: %.2f GB' % (float(mem)/1e9))
        log(4, 'PyCUDA max blocks fitting on GPU: ma_arrays={}'.format(nma))
        # reset memory or create new
        self.w_data = GpuDataManager(ma_mem, 0, nma, False)
        self.I_data = GpuDataManager(mag_mem, 0, nma, False)

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
            if self.do_position_refinement:
                prep.original_addr_gpu = gpuarray.to_gpu(prep.original_addr)
                prep.error_state_gpu = gpuarray.empty_like(prep.err_phot_gpu)
                prep.mangled_addr_gpu = prep.addr_gpu.copy()

            # Todo: Which address to pick?
            if use_tiles:
                prep.addr2_gpu = gpuarray.to_gpu(prep.addr2)

            prep.I = cuda.pagelocked_empty(d.data.shape, d.data.dtype, order="C", mem_flags=4)
            prep.I[:] = d.data

            # Todo: avoid that extra copy of data
            if self.do_position_refinement:
                ma = self.ma.S[d.ID].data.astype(np.float32)
                prep.ma = cuda.pagelocked_empty(ma.shape, ma.dtype, order="C", mem_flags=4)
                prep.ma[:] = ma

            log(4, 'Free memory on device: %.2f GB' % (float(cuda.mem_get_info()[0])/1e9))
            self.w_data.add_data_block()
            self.I_data.add_data_block()

        self.dID_list = list(self.di.S.keys())

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
        if self.GSK.tmp is None:
            self.GSK.tmp = gpuarray.empty(data.shape, dtype=np.complex64)
        self.GSK.convolution(data, [sigma, sigma], tmp=self.GSK.tmp)
        return data

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
                self.support_constraint(s)
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

    def position_update(self):
        """ 
        Position refinement
        """
        if not self.do_position_refinement or (not self.curiter):
            return
        do_update_pos = (self.p.position_refinement.stop > self.curiter >= self.p.position_refinement.start)
        do_update_pos &= (self.curiter % self.p.position_refinement.interval) == 0
        use_tiles = (not self.p.probe_update_cuda_atomics) or (not self.p.object_update_cuda_atomics)

        # Update positions
        if do_update_pos:
            """
            Iterates through all positions and refines them by a given algorithm.
            """
            log(4, "----------- START POS REF -------------")
            for dID in self.dID_list:

                prep = self.diff_info[dID]
                pID, oID, eID = prep.poe_IDs
                ob = self.ob.S[oID].gpu
                pr = self.pr.S[pID].gpu
                kern = self.kernels[prep.label]
                aux = kern.aux
                addr = prep.addr_gpu
                original_addr = prep.original_addr
                mangled_addr = prep.mangled_addr_gpu
                err_phot = prep.err_phot_gpu
                error_state = prep.error_state_gpu

                # copy intensities and weights to GPU
                ev_w, w, data_w = self.w_data.to_gpu(prep.weights, dID, self.qu_htod)
                ev, I, data_I = self.I_data.to_gpu(prep.I, dID, self.qu_htod)

                PCK = kern.PCK
                TK  = kern.TK
                PROP = kern.PROP

                # Keep track of object boundaries
                max_oby = ob.shape[-2] - aux.shape[-2] - 1
                max_obx = ob.shape[-1] - aux.shape[-1] - 1

                # We need to re-calculate the current error 
                PCK.build_aux(aux, addr, ob, pr)
                PROP.fw(aux, aux)
                PCK.queue.wait_for_event(ev)
                # w & I now on device
                PCK.log_likelihood_ml(aux, addr, I, w, err_phot)
                cuda.memcpy_dtod(dest=error_state.ptr,
                                    src=err_phot.ptr,
                                    size=err_phot.nbytes)
                
                PCK.mangler.setup_shifts(self.curiter, nframes=addr.shape[0])
                                
                log(4, 'Position refinement trial: iteration %s' % (self.curiter))
                for i in range(PCK.mangler.nshifts):
                    PCK.mangler.get_address(i, addr, mangled_addr, max_oby, max_obx)
                    PCK.build_aux(aux, mangled_addr, ob, pr)
                    PROP.fw(aux, aux)
                    PCK.log_likelihood_ml(aux, mangled_addr, I, w, err_phot)
                    PCK.update_addr_and_error_state(addr, error_state, mangled_addr, err_phot)

                data_w.record_done(self.queue, 'compute')
                data_I.record_done(self.queue, 'compute')
                cuda.memcpy_dtod(dest=err_phot.ptr,
                                 src=error_state.ptr,
                                 size=err_phot.nbytes)
                if use_tiles:
                    s1 = addr.shape[0] * addr.shape[1]
                    s2 = addr.shape[2] * addr.shape[3]
                    TK.transpose(addr.reshape(s1, s2), prep.addr2_gpu.reshape(s2, s1))

            self.dID_list.reverse()

    def support_constraint(self, storage=None):
        """
        Enforces 2D support constraint on probe.
        """
        if storage is None:
            for s in self.pr.storages.values():
                self.support_constraint(s)

        # Fourier space
        support = self._probe_fourier_support.get(storage.ID)
        if support is not None:
            if storage.ID not in self.FSK:
                supp = support.astype(np.complex64)
                self.FSK[storage.ID] = FourierSupportKernel(supp, self.queue, self.p.fft_lib)
                self.FSK[storage.ID].allocate()
            self.FSK[storage.ID].apply_fourier_support(storage.gpu)

        # Real space
        support = self._probe_support.get(storage.ID)
        if support is not None:
            if storage.ID not in self.RSK:
                self.RSK[storage.ID] = RealSupportKernel(support.astype(np.complex64))
                self.RSK[storage.ID].allocate()
            self.RSK[storage.ID].apply_real_support(storage.gpu)

    def engine_finalize(self):
        """
        Clear all GPU data, pinned memory, etc
        """
        self.w_data = None
        self.I_data = None

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
        for dID, prep in self.diff_info.items():
            prep.addr = prep.addr_gpu.get()
            prep.float_intens_coeff = prep.fic_gpu.get()


        #self.queue.synchronize()
        self.context.pop()
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
        qu_htod = self.engine.qu_htod
        queue = self.engine.queue

        self.engine._set_pr_ob_ref_for_data('gpu')
        ob_grad << 0.
        pr_grad << 0.

        # We need an array for MPI
        LL = np.array([0.])
        error_dct = {}

        for dID in self.engine.dID_list:
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

            # Schedule w & I to device
            ev_w, w, data_w = self.engine.w_data.to_gpu(prep.weights, dID, qu_htod)
            ev, I, data_I = self.engine.I_data.to_gpu(prep.I, dID, qu_htod)

            # make propagated exit (to buffer)
            AWK.build_aux_no_ex(aux, addr, ob, pr, add=False)

            # forward prop
            FW(aux, aux)
            GDK.make_model(aux, addr)

            queue.wait_for_event(ev)

            if self.p.floating_intensities:
                GDK.floating_intensity(addr, w, I, fic)

            GDK.main(aux, addr, w, I)
            data_w.record_done(queue, 'compute')
            data_I.record_done(queue, 'compute')

            GDK.error_reduce(addr, err_phot)

            BW(aux, aux)

            use_atomics = self.p.object_update_cuda_atomics
            addr = prep.addr_gpu if use_atomics else prep.addr2_gpu
            POK.ob_update_ML(addr, obg, pr, aux, atomics=use_atomics)

            use_atomics = self.p.probe_update_cuda_atomics
            addr = prep.addr_gpu if use_atomics else prep.addr2_gpu
            POK.pr_update_ML(addr, prg, ob, aux, atomics=use_atomics)

        queue.synchronize()
        self.engine.dID_list.reverse()

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
        qu_htod = self.engine.qu_htod
        queue = self.engine.queue

        B = gpuarray.zeros((3,), dtype=np.float32) # does not accept np.longdouble
        Brenorm = 1. / self.LL[0] ** 2

        # Outer loop: through diffraction patterns
        for dID in self.engine.dID_list:
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

            # Schedule w & I to device
            ev_w, w, data_w = self.engine.w_data.to_gpu(prep.weights, dID, qu_htod)
            ev, I, data_I = self.engine.I_data.to_gpu(prep.I, dID, qu_htod)

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

            queue.wait_for_event(ev)

            GDK.make_a012(f, a, b, addr, I, fic)
            GDK.fill_b(addr, Brenorm, w, B)

            data_w.record_done(queue, 'compute')
            data_I.record_done(queue, 'compute')

        queue.synchronize()
        self.engine.dID_list.reverse()

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
