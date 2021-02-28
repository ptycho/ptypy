# -*- coding: utf-8 -*-
"""
Difference Map reconstruction engine for NVIDIA GPUs.

This engine uses three streams, one for the compute queue and one for each I/O queue.
Events are used to synchronize download / compute/ upload. we cannot manipulate memory
for each loop over the state vector, a certain number of memory sections is preallocated
and reused.

This file is part of the PTYPY package.

    :copyright: Copyright 2014 by the PTYPY team, see AUTHORS.
    :license: GPLv2, see LICENSE for details.
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
from . import DM_pycuda

from ..mem_utils import make_pagelocked_paired_arrays as mppa
from ..mem_utils import GpuDataManager2

MPI = parallel.size > 1
MPI = True

EX_MA_BLOCKS_RATIO = 2
MAX_BLOCKS = 99999  # can be used to limit the number of blocks, simulating that they don't fit
MAX_BLOCKS = 3  # can be used to limit the number of blocks, simulating that they don't fit


__all__ = ['DM_pycuda_stream']

@register()
class DM_pycuda_stream(DM_pycuda.DM_pycuda):

    def __init__(self, ptycho_parent, pars = None):

        super(DM_pycuda_stream, self).__init__(ptycho_parent, pars)
        self.dmp = DeviceMemoryPool()
        self.qu_htod = cuda.Stream()
        self.qu_dtoh = cuda.Stream()

        self.ma_data = None
        self.mag_data = None
        self.ex_data = None

        self._ex_blocks_on_device = {}
        self._data_blocks_on_device = {}


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
        fit = int(mem - 200*1024*1024) // blk  # leave 200MB room for safety
        fit = min(MAX_BLOCKS, fit)
        blocks = MAX_BLOCKS
        # TODO grow blocks dynamically
        nex = min(fit * EX_MA_BLOCKS_RATIO, blocks)
        nma = min(fit, blocks)

        log(3, 'PyCUDA blocks fitting on GPU: exit arrays={}, ma_arrays={}, totalblocks={}'.format(nex, nma, blocks))
        # reset memory or create new
        self.ex_data = GpuDataManager2(ex_mem, nex, True)
        self.ma_data = GpuDataManager2(ma_mem, nma, False)
        self.mag_data = GpuDataManager2(mag_mem, nma, False)


    def engine_prepare(self):

        super(DM_pycuda.DM_pycuda, self).engine_prepare()

        for name, s in self.ob.S.items():
            s.gpu = gpuarray.to_gpu(s.data)
        for name, s in self.ob_buf.S.items():
            s.gpu, s.data = mppa(s.data)
        for name, s in self.ob_nrm.S.items():
            s.gpu, s.data = mppa(s.data)
        for name, s in self.pr.S.items():
            s.gpu, s.data = mppa(s.data)
        for name, s in self.pr_nrm.S.items():
            s.gpu, s.data = mppa(s.data)

        use_atomics = self.p.probe_update_cuda_atomics or self.p.object_update_cuda_atomics
        use_tiles = (not self.p.probe_update_cuda_atomics) or (not self.p.object_update_cuda_atomics)

        for label, d in self.ptycho.new_data:
            dID = d.ID
            prep = self.diff_info[dID]
            pID, oID, eID = prep.poe_IDs

            prep.addr_gpu = gpuarray.to_gpu(prep.addr)
            if use_tiles:
                prep.addr2 = np.ascontiguousarray(np.transpose(prep.addr, (2, 3, 0, 1)))
                prep.addr2_gpu = gpuarray.to_gpu(prep.addr2)

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

    
    # @property
    # def ex_is_full(self):
    #     exl = self._ex_blocks_on_device
    #     return len([e for e in exl.values() if e > 1]) > BLOCKS_ON_DEVICE
    #
    # @property
    # def data_is_full(self):
    #     exl = self._data_blocks_on_device
    #     return len([e for e in exl.values() if e > 1]) > BLOCKS_ON_DEVICE
    #
    # def gpu_swap_ex(self, swaps=1, upload=True):
    #     """
    #     Find an exit wave block to transfer until. Delete block on device if full
    #     """
    #     s = 0
    #     for tID in self.dID_list:
    #         stat = self._ex_blocks_on_device[tID]
    #         prep = self.diff_info[tID]
    #         if stat == 3 and self.ex_is_full:
    #             # release data if already used and device full
    #             #print('Ex Free : ' + str(tID))
    #             self.qu3.wait_for_event(prep.ev_ex_d2h)
    #             if upload:
    #                 prep.ex_gpu.get_async(self.qu3, prep.ex)
    #             del prep.ex_gpu
    #             del prep.ev_ex_h2d
    #             self._ex_blocks_on_device[tID] = 0
    #         elif stat == 1 and not self.ex_is_full and s<=swaps:
    #             #print('Ex H2D : ' + str(tID))
    #             # not on device but there is space -> queue for stream
    #             prep.ex_gpu = gpuarray.to_gpu_async(prep.ex, allocator=self.dmp.allocate, stream=self.qu2)
    #             prep.ev_ex_h2d = cuda.Event()
    #             prep.ev_ex_h2d.record(self.qu2)
    #             # mark transfer
    #             self._ex_blocks_on_device[tID] = 2
    #             s+=1
    #         else:
    #             continue
    #
    # def gpu_swap_data(self, swaps=1):
    #     """
    #     Find an exit wave block to transfer until. Delete block on device if full
    #     """
    #     s = 0
    #     for tID in self.dID_list:
    #         stat = self._data_blocks_on_device[tID]
    #         if stat == 3 and self.data_is_full:
    #             # release data if already used and device full
    #             #rint('Data Free : ' + str(tID))
    #             del self.diff_info[tID].ma_gpu
    #             del self.diff_info[tID].mag_gpu
    #             del self.diff_info[tID].ev_data_h2d
    #             self._data_blocks_on_device[tID] = 0
    #         elif stat == 1 and not self.data_is_full and s<=swaps:
    #             #print('Data H2D : ' + str(tID))
    #             # not on device but there is space -> queue for stream
    #             prep = self.diff_info[tID]
    #             prep.mag_gpu = gpuarray.to_gpu_async(prep.mag, allocator=self.dmp.allocate, stream=self.qu2)
    #             prep.ma_gpu = gpuarray.to_gpu_async(prep.ma, allocator=self.dmp.allocate, stream=self.qu2)
    #             prep.ev_data_h2d = cuda.Event()
    #             prep.ev_data_h2d.record(self.qu2)
    #             # mark transfer
    #             self._data_blocks_on_device[tID] = 2
    #             s+=1
    #         else:
    #             continue

    def swap_gpu_data(self, datamanager, what, block):
        self.queue.synchronize()
        if block + len(datamanager) < len(self.dID_list):
            pop_dID = self.dID_list[block]
            dID = self.dID_list[block + len(datamanager)]
            cpu = self.diff_info[dID][what]
            datamanager.to_gpu(cpu, dID, self.qu_htod, self.qu_dtoh, pop_dID)

    def engine_iterate(self, num=1):
        """
        Compute one iteration.
        """
        #ma_buf = ma_c = np.zeros(FUK.fshape, dtype=np.float32)
        self.dID_list = list(self.di.S.keys())

        self._ex_blocks_on_device = dict.fromkeys(self.dID_list,1)
        self._data_blocks_on_device = dict.fromkeys(self.dID_list,1)
        # 0: used, freed
        # 1: unused, not on device
        # 2: transfer to or on device
        # 3: used, on device

        # atomics or tiled version for probe / object update kernels
        atomics_probe = self.p.probe_update_cuda_atomics
        atomics_object = self.p.object_update_cuda_atomics
        use_atomics = atomics_object or atomics_probe
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
                            logger.info('Smoothing object, cfact is %.2f' % cfact)
                            smooth_mfs = [self.p.obj_smooth_std, self.p.obj_smooth_std]
                            self.GSK.convolution(ob.gpu, obb.gpu, smooth_mfs)
                        #obb.gpu[:] = ob.gpu * cfactf32
                        ob.gpu._axpbz(np.complex64(cfact), 0, obb.gpu, stream=self.queue)

                        obn.gpu.fill(np.float32(cfact), stream=self.queue)

                # First cycle: Fourier + object update
                for iblock, dID in enumerate(self.dID_list):
                    t1 = time.time()
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

                    # # get me the next exit wave
                    # ev_ex_set, ex = ep.set_array(prep.ex, synchback=True)
                    # #self.gpu_swap_ex()
                    # #prep.ev_ex_h2d.synchronize()
                    # #ex = prep.ex_gpu
                    
                    # Make sure the ex array is on GPU
                    ex = self.ex_data.to_gpu(prep.ex, dID, self.qu_htod, self.qu_dtoh)

                    # Fourier update.
                    if do_update_fourier:
                        log(4, '----- Fourier update -----', True)

                        # Make sure our data arrays are on device
                        ma = self.ma_data.to_gpu(prep.ma, dID, self.qu_htod, self.qu_dtoh)
                        mag = self.mag_data.to_gpu(prep.mag, dID, self.qu_htod, self.qu_dtoh)

                        #ev_ma_set, ma = dp.set_array(prep.ma)
                        #ev_mag_set, mag = dp.set_array(prep.mag)
                        #self.gpu_swap_data()

                        ## compute log-likelihood
                        if self.p.compute_log_likelihood:
                            t1 = time.time()
                            AWK.build_aux_no_ex(aux, addr, ob, pr)
                            PROP.fw(aux, aux)
                            # synchronize h2d stream
                            FUK.log_likelihood(aux, addr, mag, ma, err_phot)
                            self.benchmark.F_LLerror += time.time() - t1

                        t1 = time.time()
                        AWK.build_aux(aux, addr, ob, pr, ex, alpha=self.p.alpha)
                        self.benchmark.A_Build_aux += time.time() - t1


                        ## FFT
                        t1 = time.time()
                        PROP.fw(aux, aux)
                        self.benchmark.B_Prop += time.time() - t1

                        ## Deviation from measured data
                        #self.queue.wait_for_event(ev_ma_set) # this is technically not needed as we cross cuda-queues
                        #self.queue.wait_for_event(ev_mag_set)
                        FUK.fourier_error(aux, addr, mag, ma, ma_sum)
                        FUK.error_reduce(addr, err_fourier)
                        FUK.fmag_all_update(aux, addr, mag, ma, err_fourier, pbound)

                        # We are partially synchronous with compute
                        self.queue.synchronize()
                        self.benchmark.C_Fourier_update += time.time() - t1

                        # Mark ma and mag release for release
                        self.swap_gpu_data(self.ma_data,'ma', iblock)
                        self.swap_gpu_data(self.mag_data,'mag', iblock)

                        # # Suggest the block which will replace this one
                        # if iblock+len(self.ma_data)<len(self.dID_list):
                        #     dID = self.dID_list[iblock+len(self.ma_data)]
                        #     ma = self.diff_info[dID].ma
                        #     self.ma_data.to_gpu(ma, dID, self.qu_htod, self.qu_dtoh)
                        #
                        # if iblock+len(self.mag_data) < len(self.dID_list):
                        #     dID = self.dID_list[iblock + len(self.mag_data)]
                        #     mag = self.diff_info[dID].mag
                        #     self.mag_data.to_gpu(mag, dID, self.qu_htod, self.qu_dtoh)

                        #ev = cuda.Event()
                        #ev.record(self.queue)
                        # Mark computed
                        #dp.computed(prep.ma, ev)
                        #dp.computed(prep.mag, ev)

                        t1 = time.time()
                        PROP.bw(aux, aux)
                        ## apply changes
                        AWK.build_exit(aux, addr, ob, pr, ex)
                        FUK.exit_error(aux, addr)
                        FUK.error_reduce(addr, err_exit)

                        self.queue.synchronize()
                        self.benchmark.E_Build_exit += time.time() - t1
                        self.benchmark.calls_fourier += 1

                    prestr = '%d Iteration (Overlap) #%02d:  ' % (parallel.rank, inner)

                    # Update object
                    if do_update_object:
                        log(4, prestr + '----- object update -----', True)
                        t1 = time.time()

                        addrt = addr if atomics_object else addr2
                        POK.ob_update(addrt, obb, obn, pr, ex, atomics=atomics_object)
                        self.benchmark.object_update += time.time() - t1
                        self.benchmark.calls_object += 1

                    self.swap_gpu_data(self.ex_data,'ex', iblock)

                    # ev = cuda.Event()
                    # ev.record(self.queue)
                    # ep.computed(prep.ex, ev)
                    # mark as computed
                    #prep.ev_ex_d2h = cuda.Event()
                    #prep.ev_ex_d2h.record(self.queue)
                    #self._ex_blocks_on_device[dID] = 3

                # for _dID, stat in self._ex_blocks_on_device.items():
                #     if stat == 3: self._ex_blocks_on_device[_dID] = 2
                #     elif stat == 0: self._ex_blocks_on_device[_dID] = 1
                #
                # for _dID, stat in self._data_blocks_on_device.items():
                #     if stat == 3: self._data_blocks_on_device[_dID] = 2
                #     elif stat == 0: self._data_blocks_on_device[_dID] = 1

                # swap direction
                if do_update_fourier:
                    self.dID_list.reverse()

                # wait for compute stream to finish
                self.queue.synchronize()

                if do_update_object:

                    for oID, ob in self.ob.storages.items():
                        obn = self.ob_nrm.S[oID]
                        obb = self.ob_buf.S[oID]
                        # MPI test
                        if MPI:
                            obb.data[:] = obb.gpu.get()
                            obn.data[:] = obn.gpu.get()
                            parallel.allreduce(obb.data)
                            parallel.allreduce(obn.data)
                            obb.data /= obn.data
                            self.clip_object(obb)
                            ob.gpu.set(obb.data)
                        else:
                            obb.gpu /= obn.gpu
                            ob.gpu[:] = obb.gpu

                    #queue.synchronize()
                # Exit if probe should not yet be updated
                if not do_update_probe:
                    break

                # Update probe
                log(4, prestr + '----- probe update -----', True)
                change = self.probe_update(MPI=MPI)
                # change = self.probe_update(MPI=(parallel.size>1 and MPI))

                log(4, prestr + 'change in probe is %.3f' % change, True)

                # stop iteration if probe change is small
                if change < self.p.overlap_converge_factor: break

            #queue.synchronize()
            parallel.barrier()
            

            self.curiter += 1

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
        t1 = time.time()
        queue = self.queue
        use_atomics = self.p.probe_update_cuda_atomics
        # storage for-loop
        change = 0
        for pID, pr in self.pr.storages.items():
            prn = self.pr_nrm.S[pID]
            cfact = self.pr_cfact[pID]
            #pr.gpu *= np.float64(cfact)
            pr.gpu._axpbz(np.complex64(cfact), 0, pr.gpu, stream=queue)
            prn.gpu.fill(np.float32(cfact), stream=self.queue)

        for iblock, dID in enumerate(self.dID_list):
            prep = self.diff_info[dID]

            POK = self.kernels[prep.label].POK
            # find probe, object in exit ID in dependence of dID
            pID, oID, eID = prep.poe_IDs

            ex = self.ex_data.to_gpu(prep.ex, dID, self.qu_htod, self.qu_dtoh)
            # self.gpu_swap_ex(upload=True)
            # prep.ev_ex_h2d.synchronize()
            # scan for-loop
            addrt = prep.addr_gpu if use_atomics else prep.addr2_gpu
            ev = POK.pr_update(addrt,
                               self.pr.S[pID].gpu,
                               self.pr_nrm.S[pID].gpu,
                               self.ob.S[oID].gpu,
                               ex,
                               atomics=use_atomics)

            self.swap_gpu_data(self.ex_data, 'ex', iblock)

        #     # mark as computed
        #     prep.ev_ex_d2h = cuda.Event()
        #     prep.ev_ex_d2h.record(self.queue)
        #     self._ex_blocks_on_device[dID] = 3
        # 
        # for _dID, stat in self._ex_blocks_on_device.items():
        #     if stat == 3:
        #         self._ex_blocks_on_device[_dID] = 2
        #     elif stat == 0:
        #         self._ex_blocks_on_device[_dID] = 1

        self.dID_list.reverse()

        for pID, pr in self.pr.storages.items():

            buf = self.pr_buf.S[pID]
            prn = self.pr_nrm.S[pID]

            # MPI test
            if MPI:
                # if False:
                pr.data[:] = pr.gpu.get()
                prn.data[:] = prn.gpu.get()
                #queue.synchronize()
                parallel.allreduce(pr.data)
                parallel.allreduce(prn.data)
                pr.data /= prn.data

                self.support_constraint(pr)

                pr.gpu.set(pr.data)
            else:
                pr.gpu /= prn.gpu
                # ca. 0.3 ms
                # self.pr.S[pID].gpu = probe_gpu
                pr.data[:] = pr.gpu.get()

            ## this should be done on GPU

            #queue.synchronize()
            change += u.norm2(pr.data - buf.data) / u.norm2(pr.data)
            buf.data[:] = pr.data
            if MPI:
                change = parallel.allreduce(change) / parallel.size

        # print 'probe update: ' + str(time.time()-t1)
        self.benchmark.probe_update += time.time() - t1
        self.benchmark.calls_probe += 1

        return np.sqrt(change)

