# -*- coding: utf-8 -*-
"""
Difference Map reconstruction engine.

This file is part of the PTYPY package.

    :copyright: Copyright 2014 by the PTYPY team, see AUTHORS.
    :license: GPLv2, see LICENSE for details.
"""

import numpy as np
import time
from pycuda import gpuarray
import pycuda.driver as cuda

from ptypy import utils as u
from ptypy.utils.verbose import logger, log
from ptypy.utils import parallel
from ptypy.engines import register
from ptypy.accelerate.base.engines import DM_serial
from ptypy.accelerate.base import address_manglers
from .. import get_context
from ..kernels import FourierUpdateKernel, AuxiliaryWaveKernel, PoUpdateKernel, PositionCorrectionKernel, PropagationKernel
from ..array_utils import ArrayUtilsKernel, GaussianSmoothingKernel, TransposeKernel
from ..mem_utils import make_pagelocked_paired_arrays as mppa
from ..multi_gpu import MultiGpuCommunicator

MPI = parallel.size > 1
MPI = True

__all__ = ['DM_pycuda']

@register()
class DM_pycuda(DM_serial.DM_serial):

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
        Difference map reconstruction engine.
        """
        super(DM_pycuda, self).__init__(ptycho_parent, pars)
        self.multigpu = None

    def engine_initialize(self):
        """
        Prepare for reconstruction.
        """
        self.context, self.queue = get_context(new_context=True, new_queue=True)
        # allocator for READ only buffers
        # self.const_allocator = cl.tools.ImmediateAllocator(queue, cl.mem_flags.READ_ONLY)

        # Gaussian Smoothing Kernel
        self.GSK = GaussianSmoothingKernel(queue=self.queue)

        self.multigpu = MultiGpuCommunicator()

        super(DM_pycuda, self).engine_initialize()

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
            kern.aux = gpuarray.to_gpu(aux)

            # setup kernels, one for each SCAN.
            logger.info("Setting up FourierUpdateKernel")
            kern.FUK = FourierUpdateKernel(aux, nmodes, queue_thread=self.queue)
            kern.FUK.allocate()

            logger.info("Setting up PoUpdateKernel")
            kern.POK = PoUpdateKernel(queue_thread=self.queue)
            kern.POK.allocate()

            logger.info("Setting up AuxiliaryWaveKernel")
            kern.AWK = AuxiliaryWaveKernel(queue_thread=self.queue)
            kern.AWK.allocate()

            logger.info("Setting up ArrayUtilsKernel")
            kern.AUK = ArrayUtilsKernel(queue=self.queue)

            logger.info("Setting up TransposeKernel")
            kern.TK = TransposeKernel(queue=self.queue)

            logger.info("Setting up PropagationKernel")
            kern.PROP = PropagationKernel(aux, geo.propagator, self.queue, self.p.fft_lib)
            kern.PROP.allocate()
            kern.resolution = geo.resolution[0]

            if self.do_position_refinement:
                logger.info("Setting up position correction")
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
            logger.info("Kernel setup completed")

    def engine_prepare(self):

        super(DM_pycuda, self).engine_prepare()

        for name, s in self.ob.S.items():
            s.gpu = gpuarray.to_gpu(s.data)
        for name, s in self.ob_buf.S.items():
            s.gpu, s.data = mppa(s.data)
        for name, s in self.ob_nrm.S.items():
            s.gpu, s.data = mppa(s.data)
        for name, s in self.pr.S.items():
            s.gpu, s.data = mppa(s.data)
        for name, s in self.pr_buf.S.items():
            s.gpu, s.data = mppa(s.data)
        for name, s in self.pr_nrm.S.items():
            s.gpu, s.data = mppa(s.data)

        use_tiles = (not self.p.probe_update_cuda_atomics) or (not self.p.object_update_cuda_atomics)

        # TODO : like the serialization this one is needed due to object reformatting
        for label, d in self.di.storages.items():
            prep = self.diff_info[d.ID]
            prep.addr_gpu = gpuarray.to_gpu(prep.addr)
            if use_tiles:
                prep.addr2 = np.ascontiguousarray(np.transpose(prep.addr, (2, 3, 0, 1)))
                prep.addr2_gpu = gpuarray.to_gpu(prep.addr2)

        for label, d in self.ptycho.new_data:
            prep = self.diff_info[d.ID]
            pID, oID, eID = prep.poe_IDs
            s = self.ex.S[eID]
            s.gpu = gpuarray.to_gpu(s.data)
            s = self.ma.S[d.ID]
            s.gpu = gpuarray.to_gpu(s.data.astype(np.float32))

            prep.mag = gpuarray.to_gpu(prep.mag)
            prep.ma_sum = gpuarray.to_gpu(prep.ma_sum)
            prep.err_fourier_gpu = gpuarray.to_gpu(prep.err_fourier)
            prep.err_phot_gpu = gpuarray.to_gpu(prep.err_phot)
            prep.err_exit_gpu = gpuarray.to_gpu(prep.err_exit)
            if self.do_position_refinement:
                prep.error_state_gpu = gpuarray.empty_like(prep.err_fourier_gpu)

    def engine_iterate(self, num=1):
        """
        Compute one iteration.
        """
        queue = self.queue
        use_tiles = (not self.p.probe_update_cuda_atomics) or (not self.p.object_update_cuda_atomics)

        for it in range(num):
            error = {}
            for dID in self.di.S.keys():

                # find probe, object and exit ID in dependence of dID
                prep = self.diff_info[dID]
                pID, oID, eID = prep.poe_IDs

                # references for kernels
                kern = self.kernels[prep.label]
                FUK = kern.FUK
                AWK = kern.AWK
                PROP = kern.PROP

                # get addresses and buffers
                addr = prep.addr_gpu
                mag = prep.mag
                ma_sum = prep.ma_sum
                err_fourier = prep.err_fourier_gpu
                err_phot = prep.err_phot_gpu
                err_exit = prep.err_exit_gpu
                pbound = self.pbound_scan[prep.label]
                aux = kern.aux

                # local references
                ma = self.ma.S[dID].gpu
                ob = self.ob.S[oID].gpu
                pr = self.pr.S[pID].gpu
                ex = self.ex.S[eID].gpu

                ## compute log-likelihood
                if self.p.compute_log_likelihood:
                    t1 = time.time()
                    AWK.build_aux_no_ex(aux, addr, ob, pr)
                    PROP.fw(aux, aux)
                    FUK.log_likelihood(aux, addr, mag, ma, err_phot)
                    self.benchmark.F_LLerror += time.time() - t1

                ## build auxilliary wave
                t1 = time.time()
                AWK.build_aux(aux, addr, ob, pr, ex, alpha=self.p.alpha)
                self.benchmark.A_Build_aux += time.time() - t1

                ## forward FFT
                t1 = time.time()
                PROP.fw(aux, aux)
                self.benchmark.B_Prop += time.time() - t1

                ## Deviation from measured data
                t1 = time.time()
                FUK.fourier_error(aux, addr, mag, ma, ma_sum)
                FUK.error_reduce(addr, err_fourier)
                FUK.fmag_all_update(aux, addr, mag, ma, err_fourier, pbound)
                self.benchmark.C_Fourier_update += time.time() - t1

                ## backward FFT
                t1 = time.time()
                PROP.bw(aux, aux)
                self.benchmark.D_iProp += time.time() - t1

                ## build exit wave
                t1 = time.time()
                AWK.build_exit(aux, addr, ob, pr, ex)
                FUK.exit_error(aux, addr)
                FUK.error_reduce(addr, err_exit)
                self.benchmark.E_Build_exit += time.time() - t1

                self.benchmark.calls_fourier += 1

            parallel.barrier()

            sync = (self.curiter % 1 == 0)
            self.overlap_update(MPI=MPI)

            parallel.barrier()
            if self.do_position_refinement and (self.curiter):
                do_update_pos = (self.p.position_refinement.stop > self.curiter >= self.p.position_refinement.start)
                do_update_pos &= (self.curiter % self.p.position_refinement.interval) == 0

                # Update positions
                if do_update_pos:
                    """
                    Iterates through all positions and refines them by a given algorithm.
                    """
                    log(3, "----------- START POS REF -------------")
                    for dID in self.di.S.keys():

                        prep = self.diff_info[dID]
                        pID, oID, eID = prep.poe_IDs
                        ma = self.ma.S[dID].gpu
                        ob = self.ob.S[oID].gpu
                        pr = self.pr.S[pID].gpu
                        kern = self.kernels[prep.label]
                        aux = kern.aux
                        addr = prep.addr_gpu
                        original_addr = prep.original_addr
                        mag = prep.mag
                        ma_sum = prep.ma_sum
                        err_fourier = prep.err_fourier_gpu

                        PCK = kern.PCK
                        AUK = kern.AUK

                        #error_state = np.zeros(err_fourier.shape, dtype=np.float32)
                        #error_state[:] = err_fourier.get()
                        cuda.memcpy_dtod(dest=prep.error_state_gpu.ptr,
                                         src=err_fourier.ptr,
                                         size=err_fourier.nbytes)
                        log(4, 'Position refinement trial: iteration %s' % (self.curiter))
                        for i in range(self.p.position_refinement.nshifts):
                            mangled_addr = PCK.address_mangler.mangle_address(addr.get(), original_addr, self.curiter)
                            mangled_addr_gpu = gpuarray.to_gpu(mangled_addr)
                            PCK.build_aux(aux, mangled_addr_gpu, ob, pr)
                            PROP.fw(aux, aux)
                            PCK.fourier_error(aux, mangled_addr_gpu, mag, ma, ma_sum)
                            PCK.error_reduce(mangled_addr_gpu, err_fourier)
                            PCK.update_addr_and_error_state(addr,
                                prep.error_state_gpu,
                                mangled_addr_gpu,
                                err_fourier)
                        
                        # prep.err_fourier_gpu.set(error_state)
                        cuda.memcpy_dtod(dest=prep.err_fourier_gpu.ptr,
                            src=prep.error_state_gpu.ptr,
                            size=prep.err_fourier_gpu.nbytes)
                        if use_tiles:
                            s1 = addr.shape[0] * addr.shape[1]
                            s2 = addr.shape[2] * addr.shape[3]
                            kern.TK.transpose(addr.reshape(s1, s2), prep.addr2_gpu.reshape(s2, s1))

            self.curiter += 1
            queue.synchronize()

        for name, s in self.ob.S.items():
            s.data[:] = s.gpu.get()
        for name, s in self.pr.S.items():
            s.data[:] = s.gpu.get()

        # costly but needed to sync back with
        # for name, s in self.ex.S.items():
        #     s.data[:] = s.gpu.get()
        for dID, prep in self.diff_info.items():
            err_fourier = prep.err_fourier_gpu.get()
            err_phot = prep.err_phot_gpu.get()
            err_exit = prep.err_exit_gpu.get()
            errs = np.ascontiguousarray(np.vstack([err_fourier, err_phot, err_exit]).T)
            error.update(zip(prep.view_IDs, errs))

        self.error = error
        return error

    ## object update
    def object_update(self, MPI=False):
        t1 = time.time()
        use_atomics = self.p.object_update_cuda_atomics
        queue = self.queue
        queue.synchronize()
        for oID, ob in self.ob.storages.items():
            obn = self.ob_nrm.S[oID]
            cfact = self.ob_cfact[oID]

            if self.p.obj_smooth_std is not None:
                logger.info('Smoothing object, cfact is %.2f' % cfact)
                smooth_mfs = [self.p.obj_smooth_std, self.p.obj_smooth_std]
                ob_gpu_tmp = gpuarray.empty(ob.shape, dtype=np.complex64)
                self.GSK.convolution(ob.gpu, ob_gpu_tmp, smooth_mfs)
                ob.gpu = ob_gpu_tmp

            ob.gpu *= cfact
            obn.gpu.fill(cfact)
            queue.synchronize()

        # storage for-loop
        for dID in self.di.S.keys():
            prep = self.diff_info[dID]

            POK = self.kernels[prep.label].POK
            # find probe, object in exit ID in dependence of dID
            pID, oID, eID = prep.poe_IDs

            # scan for loop
            addr = prep.addr_gpu if use_atomics else prep.addr2_gpu
            ev = POK.ob_update(addr,
                               self.ob.S[oID].gpu,
                               self.ob_nrm.S[oID].gpu,
                               self.pr.S[pID].gpu,
                               self.ex.S[eID].gpu,
                               atomics = use_atomics)
            queue.synchronize()

        for oID, ob in self.ob.storages.items():
            obn = self.ob_nrm.S[oID]
            # MPI test
            if MPI:
                ob.data[:] = ob.gpu.get()
                obn.data[:] = obn.gpu.get()
                queue.synchronize()
                parallel.allreduce(ob.data)
                parallel.allreduce(obn.data)
                ob.data /= obn.data

                self.clip_object(ob)
                ob.gpu.set(ob.data)
            else:
                ob.gpu /= obn.gpu

            queue.synchronize()

        # print 'object update: ' + str(time.time()-t1)
        self.benchmark.object_update += time.time() - t1
        self.benchmark.calls_object += 1

    ## probe update
    def probe_update(self, MPI=False):
        t1 = time.time()
        queue = self.queue

        # storage for-loop
        change = 0
        cfact = self.p.probe_inertia
        use_atomics = self.p.probe_update_cuda_atomics
        for pID, pr in self.pr.storages.items():
            prn = self.pr_nrm.S[pID]
            cfact = self.pr_cfact[pID]
            pr.gpu *= cfact
            prn.gpu.fill(cfact)

        for dID in self.di.S.keys():
            prep = self.diff_info[dID]

            POK = self.kernels[prep.label].POK
            # find probe, object in exit ID in dependence of dID
            pID, oID, eID = prep.poe_IDs

            # scan for-loop
            addr = prep.addr_gpu if use_atomics else prep.addr2_gpu
            ev = POK.pr_update(addr,
                               self.pr.S[pID].gpu,
                               self.pr_nrm.S[pID].gpu,
                               self.ob.S[oID].gpu,
                               self.ex.S[eID].gpu,
                               atomics=use_atomics)
            queue.synchronize()

        for pID, pr in self.pr.storages.items():

            buf = self.pr_buf.S[pID]
            prn = self.pr_nrm.S[pID]

            self.multigpu.allReduceSum(pr.gpu)
            self.multigpu.allReduceSum(prn.gpu)
            pr.gpu /= prn.gpu
            # TODO: self.support_constraint(pr)

            ## calculate change on GPU
            #queue.synchronize()
            AUK = self.kernels[list(self.kernels)[0]].AUK # this is very ugly, any better idea?
            buf.gpu -= pr.gpu
            change += (AUK.norm2(buf.gpu) / AUK.norm2(pr.gpu)).get().item()
            cuda.memcpy_dtod(dest=buf.gpu.ptr,
                    src=pr.gpu.ptr,
                    size=pr.gpu.nbytes)
            if MPI:
                change = parallel.allreduce(change) / parallel.size

        # print 'probe update: ' + str(time.time()-t1)
        self.benchmark.probe_update += time.time() - t1
        self.benchmark.calls_probe += 1

        return np.sqrt(change)

    def engine_finalize(self):
        """
        clear GPU data and destroy context.
        """
        for name, s in self.ob.S.items():
            del s.gpu
        for name, s in self.ob_buf.S.items():
            del s.gpu
        for name, s in self.ob_nrm.S.items():
            del s.gpu
        for name, s in self.pr.S.items():
            del s.gpu
        for name, s in self.pr_nrm.S.items():
            del s.gpu
        for dID, prep in self.diff_info.items():
            prep.addr = prep.addr_gpu.get()

        # copy data to cpu 
        # this kills the pagelock memory (otherwise we get segfaults in h5py)
        for name, s in self.pr.S.items():
            s.data = np.copy(s.data)

        self.context.detach()
        super(DM_pycuda, self).engine_finalize()