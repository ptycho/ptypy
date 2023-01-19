# -*- coding: utf-8 -*-
"""
Difference Map reconstruction engine for NVIDIA GPUs.

This engine uses three streams, one for the compute queue and one for each I/O queue.
Events are used to synchronize download / compute/ upload. we cannot manipulate memory
for each loop over the state vector, a certain number of memory sections is preallocated
and reused.

This file is part of the PTYPY package.

    :copyright: Copyright 2014 by the PTYPY team, see AUTHORS.
    :license: see LICENSE for details.
"""

import numpy as np
import time
from pycuda import gpuarray
import pycuda.driver as cuda
from pycuda.tools import DeviceMemoryPool

from ptypy import utils as u
from ptypy.utils.verbose import log, logger
from ptypy.utils import parallel
from ptypy.engines import register
from ptypy.engines.projectional import DMMixin, RAARMixin
from . import projectional_pycuda

from ..mem_utils import make_pagelocked_paired_arrays as mppa
from ..mem_utils import GpuDataManager

EX_MA_BLOCKS_RATIO = 2
MAX_BLOCKS = 99999  # can be used to limit the number of blocks, simulating that they don't fit
#MAX_BLOCKS = 3  # can be used to limit the number of blocks, simulating that they don't fit

__all__ = ['DM_pycuda_stream', 'RAAR_pycuda_stream']


class _ProjectionEngine_pycuda_stream(projectional_pycuda._ProjectionEngine_pycuda):

    def __init__(self, ptycho_parent, pars=None):

        super().__init__(ptycho_parent, pars)
        self.ma_data = None
        self.mag_data = None
        self.ex_data = None

    def engine_initialize(self):
        super().engine_initialize()
        self.qu_htod = cuda.Stream()
        self.qu_dtoh = cuda.Stream()

    def _setup_kernels(self):

        super()._setup_kernels()
        ex_mem = 0
        mag_mem = 0
        for scan, kern in self.kernels.items():
            ex_mem = max(kern.aux.nbytes, ex_mem)
            mag_mem = max(kern.FUK.gpu.fdev.nbytes, mag_mem)
        ma_mem = mag_mem
        mem = cuda.mem_get_info()[0]
        blk = ex_mem * EX_MA_BLOCKS_RATIO + ma_mem + mag_mem
        fit = int(mem - 200 * 1024 * 1024) // blk  # leave 200MB room for safety
        if not fit:
            log(1,"Cannot fit memory into device, if possible reduce frames per block. Exiting...")
            self.context.pop()
            self.context.detach()
            raise SystemExit("ptypy has been exited.")

        # TODO grow blocks dynamically
        nex = min(fit * EX_MA_BLOCKS_RATIO, MAX_BLOCKS)
        nma = min(fit, MAX_BLOCKS)
        log(4, 'Free memory on device: %.2f GB' % (float(mem)/1e9))
        log(4, 'PyCUDA max blocks fitting on GPU: exit arrays={}, ma_arrays={}'.format(nex, nma))
        # reset memory or create new
        self.ex_data = GpuDataManager(ex_mem, 0, nex, True)
        self.ma_data = GpuDataManager(ma_mem, 0, nma, False)
        self.mag_data = GpuDataManager(mag_mem, 0, nma, False)

    def engine_prepare(self):

        super(projectional_pycuda._ProjectionEngine_pycuda, self).engine_prepare()

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

        # Extra object buffer for smoothing kernel
        if self.p.obj_smooth_std is not None:
            for name, s in self.ob_buf.S.items():
                s.tmp = gpuarray.empty(s.gpu.shape, s.gpu.dtype)

        # TODO : like the serialization this one is needed due to object reformatting
        for label, d in self.di.storages.items():
            prep = self.diff_info[d.ID]
            prep.addr_gpu = gpuarray.to_gpu(prep.addr)
            if use_tiles:
                prep.addr2 = np.ascontiguousarray(np.transpose(prep.addr, (2, 3, 0, 1)))
                prep.addr2_gpu = gpuarray.to_gpu(prep.addr2)
            if self.do_position_refinement:
                prep.mangled_addr_gpu = prep.addr_gpu.copy()

        for label, d in self.ptycho.new_data:
            dID = d.ID
            prep = self.diff_info[dID]
            pID, oID, eID = prep.poe_IDs

            prep.ma_sum_gpu = gpuarray.to_gpu(prep.ma_sum)
            # prepare page-locked mems:
            prep.err_fourier_gpu = gpuarray.to_gpu(prep.err_fourier)
            prep.err_phot_gpu = gpuarray.to_gpu(prep.err_phot)
            prep.err_exit_gpu = gpuarray.to_gpu(prep.err_exit)
            if self.do_position_refinement:
                prep.error_state_gpu = gpuarray.empty_like(prep.err_fourier_gpu)
            ma = self.ma.S[dID].data.astype(np.float32)
            prep.ma = cuda.pagelocked_empty(ma.shape, ma.dtype, order="C", mem_flags=4)
            prep.ma[:] = ma
            ex = self.ex.S[eID].data
            prep.ex = cuda.pagelocked_empty(ex.shape, ex.dtype, order="C", mem_flags=4)
            prep.ex[:] = ex
            mag = prep.mag
            prep.mag = cuda.pagelocked_empty(mag.shape, mag.dtype, order="C", mem_flags=4)
            prep.mag[:] = mag

            log(4, 'Free memory on device: %.2f GB' % (float(cuda.mem_get_info()[0])/1e9))
            self.ex_data.add_data_block()
            self.ma_data.add_data_block()
            self.mag_data.add_data_block()
        
    def engine_iterate(self, num=1):
        """
        Compute one iteration.
        """
        # ma_buf = ma_c = np.zeros(FUK.fshape, dtype=np.float32)
        self.dID_list = list(self.di.S.keys())
        atomics_probe = self.p.probe_update_cuda_atomics
        atomics_object = self.p.object_update_cuda_atomics
        use_tiles = (not atomics_object) or (not atomics_probe)
        
        for it in range(num):

            error = {}

            for inner in range(self.p.overlap_max_iterations):

                change = 0

                do_update_probe = (self.curiter >= self.p.probe_update_start)
                do_update_object = (self.p.update_object_first or (inner > 0) or not do_update_probe)
                do_update_fourier = (inner == 0)

                # initialize probe and object buffer to receive an update
                if do_update_object:
                    for oID, ob in self.ob.storages.items():
                        cfact = self.ob_cfact[oID]
                        obn = self.ob_nrm.S[oID]
                        obb = self.ob_buf.S[oID]

                        if self.p.obj_smooth_std is not None:
                            log(4, 'Smoothing object, cfact is %.2f' % cfact)
                            smooth_mfs = [self.p.obj_smooth_std, self.p.obj_smooth_std]
                            # We need a third copy, because we still need ob.gpu for the fourier update
                            obb.gpu[:] = ob.gpu[:]
                            self.GSK.convolution(obb.gpu, smooth_mfs, tmp=obb.tmp)
                            obb.gpu *= np.complex64(cfact)
                        else:
                            # obb.gpu[:] = ob.gpu * np.complex64(cfact)
                            ob.gpu._axpbz(np.complex64(cfact), 0, obb.gpu)
                        obn.gpu.fill(np.float32(cfact))

                # First cycle: Fourier + object update
                for iblock, dID in enumerate(self.dID_list):
                    prep = self.diff_info[dID]

                    # find probe, object in exit ID in dependence of dID
                    pID, oID, eID = prep.poe_IDs

                    # references for kernels
                    kern = self.kernels[prep.label]
                    FUK = kern.FUK
                    AWK = kern.AWK
                    POK = kern.POK

                    pbound = self.pbound_scan[prep.label]
                    aux = kern.aux
                    PROP = kern.PROP

                    # get addresses and auxilliary array
                    addr = prep.addr_gpu
                    addr2 = prep.addr2_gpu if use_tiles else None
                    err_fourier = prep.err_fourier_gpu
                    err_phot = prep.err_phot_gpu
                    err_exit = prep.err_exit_gpu
                    ma_sum = prep.ma_sum_gpu

                    # local references
                    ob = self.ob.S[oID].gpu
                    obn = self.ob_nrm.S[oID].gpu
                    obb = self.ob_buf.S[oID].gpu
                    pr = self.pr.S[pID].gpu

                    # Schedule ex to device
                    ev_ex, ex, data_ex = self.ex_data.to_gpu(prep.ex, dID, self.qu_htod)

                    # Fourier update.
                    if do_update_fourier:
                        self.ex_data.syncback = True
                        log(4, '----- Fourier update -----', True)

                        # Schedule ma & mag to device
                        ev_ma, ma, data_ma = self.ma_data.to_gpu(prep.ma, dID, self.qu_htod)
                        ev_mag, mag, data_mag = self.mag_data.to_gpu(prep.mag, dID, self.qu_htod)

                        ## compute log-likelihood
                        if self.p.compute_log_likelihood:
                            AWK.build_aux_no_ex(aux, addr, ob, pr)
                            PROP.fw(aux, aux)
                            # synchronize h2d stream with compute stream
                            self.queue.wait_for_event(ev_mag)
                            FUK.log_likelihood(aux, addr, mag, ma, err_phot)

                        # synchronize h2d stream with compute stream
                        self.queue.wait_for_event(ev_ex)
                        #AWK.build_aux(aux, addr, ob, pr, ex, alpha=self.p.alpha)
                        AWK.make_aux(aux, addr, ob, pr, ex, c_po=self._c, c_e=1-self._c)

                        ## FFT
                        PROP.fw(aux, aux)

                        ## Deviation from measured data
                        # synchronize h2d stream with compute stream
                        self.queue.wait_for_event(ev_mag)
                        FUK.fourier_error(aux, addr, mag, ma, ma_sum)
                        FUK.error_reduce(addr, err_fourier)
                        FUK.fmag_all_update(aux, addr, mag, ma, err_fourier, pbound)

                        data_mag.record_done(self.queue, 'compute')
                        data_ma.record_done(self.queue, 'compute')

                        PROP.bw(aux, aux)
                        ## apply changes
                        #AWK.build_exit(aux, addr, ob, pr, ex, alpha=self.p.alpha)
                        AWK.make_exit(aux, addr, ob, pr, ex, c_a=self._b, c_po=self._a, c_e=-(self._a + self._b))
                        FUK.exit_error(aux, addr)
                        FUK.error_reduce(addr, err_exit)

                    prestr = '%d Iteration (Overlap) #%02d:  ' % (parallel.rank, inner)

                    # Update object
                    if do_update_object:
                        log(4, prestr + '----- object update -----', True)
                        addrt = addr if atomics_object else addr2
                        self.queue.wait_for_event(ev_ex)
                        POK.ob_update(addrt, obb, obn, pr, ex, atomics=atomics_object)

                    data_ex.record_done(self.queue, 'compute')
                    if iblock + len(self.ex_data) < len(self.dID_list):
                        data_ex.from_gpu(self.qu_dtoh)

                # swap direction
                if do_update_fourier or do_update_object:
                    self.dID_list.reverse()

                # wait for compute stream to finish
                self.queue.synchronize()

                if do_update_object:

                    for oID, ob in self.ob.storages.items():
                        obn = self.ob_nrm.S[oID]
                        obb = self.ob_buf.S[oID]
                        self.multigpu.allReduceSum(obb.gpu)
                        self.multigpu.allReduceSum(obn.gpu)
                        obb.gpu /= obn.gpu

                        self.clip_object(obb.gpu)
                        ob.gpu[:] = obb.gpu

                # Exit if probe should not yet be updated
                if not do_update_probe:
                    break
                self.ex_data.syncback = False

                # Update probe
                log(4, prestr + '----- probe update -----', True)
                change = self.probe_update()
                log(4, prestr + 'change in probe is %.3f' % change, True)

                # stop iteration if probe change is small
                if change < self.p.overlap_converge_factor: break

            self.queue.synchronize()
            parallel.barrier()

            if self.do_position_refinement and (self.curiter):
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
                        ob = self.ob.S[oID].gpu
                        pr = self.pr.S[pID].gpu
                        kern = self.kernels[prep.label]
                        aux = kern.aux
                        addr = prep.addr_gpu
                        original_addr = prep.original_addr
                        mangled_addr = prep.mangled_addr_gpu
                        ma_sum = prep.ma_sum_gpu
                        err_fourier = prep.err_fourier_gpu
                        error_state = prep.error_state_gpu

                        PCK = kern.PCK
                        TK = kern.TK
                        PROP = kern.PROP

                        # Make sure our data arrays are on device
                        ev_ma, ma, data_ma = self.ma_data.to_gpu(prep.ma, dID, self.qu_htod)
                        ev_mag, mag, data_mag = self.mag_data.to_gpu(prep.mag, dID, self.qu_htod)

                        # Keep track of object boundaries
                        max_oby = ob.shape[-2] - aux.shape[-2] - 1
                        max_obx = ob.shape[-1] - aux.shape[-1] - 1

                        # We need to re-calculate the current error
                        PCK.build_aux(aux, addr, ob, pr)
                        PROP.fw(aux, aux)
                        # wait for data to arrive
                        self.queue.wait_for_event(ev_mag)

                        # We need to re-calculate the current error 
                        if self.p.position_refinement.metric == "fourier":
                            PCK.fourier_error(aux, addr, mag, ma, ma_sum)
                            PCK.error_reduce(addr, err_fourier)
                        if self.p.position_refinement.metric == "photon":
                            PCK.log_likelihood(aux, addr, mag, ma, err_fourier)
                        cuda.memcpy_dtod_async(dest=error_state.ptr,
                                               src=err_fourier.ptr,
                                               size=err_fourier.nbytes, stream=self.queue)

                        log(4, 'Position refinement trial: iteration %s' % (self.curiter))
                        PCK.mangler.setup_shifts(self.curiter, nframes=addr.shape[0])
                        for i in range(PCK.mangler.nshifts):
                            PCK.mangler.get_address(i, addr, mangled_addr, max_oby, max_obx)
                            PCK.build_aux(aux, mangled_addr, ob, pr)
                            PROP.fw(aux, aux)
                            if self.p.position_refinement.metric == "fourier":
                                PCK.fourier_error(aux, mangled_addr, mag, ma, ma_sum)
                                PCK.error_reduce(mangled_addr, err_fourier)
                            if self.p.position_refinement.metric == "photon":
                                PCK.log_likelihood(aux, mangled_addr, mag, ma, err_fourier)
                            PCK.update_addr_and_error_state(addr, error_state, mangled_addr, err_fourier)

                        data_mag.record_done(self.queue, 'compute')
                        data_ma.record_done(self.queue, 'compute')
                        cuda.memcpy_dtod_async(dest=err_fourier.ptr,
                                               src=error_state.ptr,
                                               size=err_fourier.nbytes, stream=self.queue)
                        if use_tiles:
                            s1 = prep.addr_gpu.shape[0] * prep.addr_gpu.shape[1]
                            s2 = prep.addr_gpu.shape[2] * prep.addr_gpu.shape[3]
                            TK.transpose(prep.addr_gpu.reshape(s1, s2), prep.addr2_gpu.reshape(s2, s1))

            self.curiter += 1
            self.queue.synchronize()

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

    ## probe update
    def probe_update(self, MPI=False):
        queue = self.queue
        use_atomics = self.p.probe_update_cuda_atomics
        # storage for-loop
        change_gpu = gpuarray.zeros((1,), dtype=np.float32)
        for pID, pr in self.pr.storages.items():
            prn = self.pr_nrm.S[pID]
            cfact = self.pr_cfact[pID]
            # pr.gpu *= np.float64(cfact)
            pr.gpu._axpbz(np.complex64(cfact), 0, pr.gpu, stream=queue)
            prn.gpu.fill(np.float32(cfact), stream=self.queue)

        for iblock, dID in enumerate(self.dID_list):
            prep = self.diff_info[dID]

            POK = self.kernels[prep.label].POK
            # find probe, object in exit ID in dependence of dID
            pID, oID, eID = prep.poe_IDs

            ev, ex, data_ex = self.ex_data.to_gpu(prep.ex, dID, self.qu_htod)
            self.queue.wait_for_event(ev)

            addrt = prep.addr_gpu if use_atomics else prep.addr2_gpu
            ev = POK.pr_update(addrt,
                               self.pr.S[pID].gpu,
                               self.pr_nrm.S[pID].gpu,
                               self.ob.S[oID].gpu,
                               ex,
                               atomics=use_atomics)

            data_ex.record_done(self.queue, 'compute')
            if iblock + len(self.ex_data) < len(self.dID_list):
                data_ex.from_gpu(self.qu_dtoh)

        self.dID_list.reverse()

        self.queue.synchronize()
        for pID, pr in self.pr.storages.items():

            buf = self.pr_buf.S[pID]
            prn = self.pr_nrm.S[pID]

            self.multigpu.allReduceSum(pr.gpu)
            self.multigpu.allReduceSum(prn.gpu)
            pr.gpu /= prn.gpu
            self.support_constraint(pr)

            ## calculate change on GPU
            AUK = self.kernels[list(self.kernels)[0]].AUK
            buf.gpu -= pr.gpu
            change_gpu += (AUK.norm2(buf.gpu) / AUK.norm2(pr.gpu))
            buf.gpu[:] = pr.gpu
            self.multigpu.allReduceSum(change_gpu)
            change = change_gpu.get().item() / parallel.size

        return np.sqrt(change)

    def engine_finalize(self):
        """
        Clear all GPU data, pinned memory, etc
        """
        self.ex_data = None
        self.ma_data = None
        self.mag_data = None

        super().engine_finalize()


@register(name="DM_pycuda")
class DM_pycuda_stream(_ProjectionEngine_pycuda_stream, DMMixin):
    """
    A full-fledged Difference Map engine accelerated with pycuda.

    Defaults:

    [name]
    default = DM_pycuda
    type = str
    help =
    doc =

    """

    def __init__(self, ptycho_parent, pars=None):
        _ProjectionEngine_pycuda_stream.__init__(self, ptycho_parent, pars)
        DMMixin.__init__(self, self.p.alpha)
        ptycho_parent.citations.add_article(**self.article)


@register(name="RAAR_pycuda")
class RAAR_pycuda_stream(_ProjectionEngine_pycuda_stream, RAARMixin):
    """
    A RAAR engine in accelerated with pycuda.

    Defaults:

    [name]
    default = RAAR_pycuda
    type = str
    help =
    doc =

    """

    def __init__(self, ptycho_parent, pars=None):

        _ProjectionEngine_pycuda_stream.__init__(self, ptycho_parent, pars)
        RAARMixin.__init__(self, self.p.beta)
