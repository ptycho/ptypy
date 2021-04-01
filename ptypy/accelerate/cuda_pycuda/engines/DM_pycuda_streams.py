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
from . import DM_pycuda
from ..mem_utils import GpuDataManager

MPI = parallel.size > 1
MPI = True

# factor how many more exit waves we wanna keep on GPU compared to 
# ma / mag data
EX_MA_BLOCKS_RATIO = 2
MAX_STREAMS = 500   # max number of streams to use
MAX_BLOCKS = 99999  # can be used to limit the number of blocks, simulating that they don't fit

__all__ = ['DM_pycuda_streams']

class GpuStreamData:
    def __init__(self, ex_data, ma_data, mag_data):
        self.queue = cuda.Stream()
        self.ex_data = ex_data
        self.ma_data = ma_data
        self.mag_data = mag_data
        self.ev_compute = None

    def end_compute(self):
        """
        called at end of kernels using shared data (aux or probe),
        to mark when computing is done and it can be re-used
        """
        self.ev_compute = cuda.Event()
        self.ev_compute.record(self.queue)
        return self.ev_compute

    def start_compute(self, prev_event):
        """
        called at start of kernels using shared data (aux or probe),
        to wait for previous use of this data is finished
        """

        if prev_event is not None:
            self.queue.wait_for_event(prev_event)

    def ex_to_gpu(self, dID, ex):
        """
        copy exit wave to GPU, but check first if it's already there
        If not, but a previous block is on GPU, sync it back to the host
        before overwriting it
        """

        return self.ex_data.to_gpu(ex, dID, self.queue)

    def ma_to_gpu(self, dID, ma, mag):
        """
        Copy MA array to GPU
        """
        # wait for previous work on memory to complete
        ma_gpu = self.ma_data.to_gpu(ma, dID, self.queue)
        mag_gpu = self.mag_data.to_gpu(mag, dID, self.queue)
        return ma_gpu, mag_gpu

    def record_done_ex(self, dID):
        """
        Record when we're done with this stream, so that it can be re-used
        """
        self.ex_data.record_done(dID, self.queue)

    def record_done_ma(self, dID):
        self.ma_data.record_done(dID, self.queue)
        self.mag_data.record_done(dID, self.queue)

    def synchronize(self):
        """
        Wait for stream to finish its work
        """
        self.queue.synchronize()


@register()
class DM_pycuda_streams(DM_pycuda.DM_pycuda):

    """
    Defaults:

    [fft_lib]
    default = cuda
    type = str
    help = Choose the pycuda-compatible FFT module.
    doc = One of:
      - ``'reikna'`` : the reikna packaga (fast load, competitive compute for streaming)
      - ``'cuda'`` : ptypy's cuda wrapper (delayed load, but fastest compute if all data is on GPU)
      - ``'skcuda'`` : scikit-cuda (fast load, slowest compute due to additional store/load stages)
    choices = 'reikna','cuda','skcuda'
    userlevel = 2

    """

    def __init__(self, ptycho_parent, pars = None):

        super(DM_pycuda_streams, self).__init__(ptycho_parent, pars)
        self.streams = None 
        self.ma_data = None
        self.mag_data = None
        self.ex_data = None
        self.cur_stream = 0
        self.stream_direction = 1

    def engine_prepare(self):

        super(DM_pycuda.DM_pycuda, self).engine_prepare()

        for name, s in self.ob.S.items():
            s.gpu = gpuarray.to_gpu(s.data)
        # we use default mem_flags for ob/obn/pr/prn page-locking, as we are 
        # operating on them on CPU as well after each iteration.
        # Write-Combined memory (flags=4) is for write-only on CPU side,
        # reads are really slow.
        for name, s in self.ob_buf.S.items():
            # obb
            d = s.data
            s.data = cuda.pagelocked_empty(d.shape, d.dtype, order="C", mem_flags=0)
            s.data[:] = d
            s.gpu = gpuarray.to_gpu(s.data)
        for name, s in self.ob_nrm.S.items():
            # obn
            d = s.data
            s.data = cuda.pagelocked_empty(d.shape, d.dtype, order="c", mem_flags=0)
            s.data[:] = d
            s.gpu = gpuarray.to_gpu(s.data)
        for name, s in self.pr.S.items():
            # pr
            d = s.data
            s.data = cuda.pagelocked_empty(d.shape, d.dtype, order="C", mem_flags=0)
            s.data[:] = d
            s.gpu = gpuarray.to_gpu(s.data)
        for name, s in self.pr_nrm.S.items():
            # prn
            d = s.data
            s.data = cuda.pagelocked_empty(d.shape, d.dtype, order="C", mem_flags=0)
            s.data[:] = d
            s.gpu = gpuarray.to_gpu(s.data)

        use_tiles = (not self.p.probe_update_cuda_atomics) or (not self.p.object_update_cuda_atomics)

        ex_mem = ma_mem = mag_mem = 0
        idlist = list(self.di.S.keys())
        blocks = len(idlist)
        for dID in idlist:
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
            ex_mem = max(ex_mem, prep.ex.nbytes)
            ma_mem = max(ma_mem, prep.ma.nbytes)
            mag_mem = max(mag_mem, prep.mag.nbytes)
        
        # now check remaining memory and allocate as many blocks as would fit
        mem = cuda.mem_get_info()[0]
        if self.ex_data is not None:
            mem += self.ex_data.memory    # as we realloc these, consider as free memory
        if self.ma_data is not None:
            mem += self.ma_data.memory
        if self.mag_data is not None:
            mem += self.mag_data.memory

        blk = ex_mem * EX_MA_BLOCKS_RATIO + ma_mem + mag_mem
        fit = int(mem - 200*1024*1024) // blk  # leave 200MB room for safety
        fit = min(MAX_BLOCKS, fit)
        nex = min(fit * EX_MA_BLOCKS_RATIO, blocks)
        nma = min(fit, blocks)
        nstreams = min(MAX_STREAMS, blocks)

        log(3, 'PyCUDA blocks fitting on GPU: exit arrays={}, ma_arrays={}, streams={}, totalblocks={}'.format(nex, nma, nstreams, blocks))
        # reset memory or create new
        if self.ex_data is not None:
            self.ex_data.reset(ex_mem, nex)
        else:
            self.ex_data = GpuDataManager(ex_mem, nex, True)
        if self.ma_data is not None:
            self.ma_data.reset(ma_mem, nma)
        else:
            self.ma_data = GpuDataManager(ma_mem, nma, False)
        if self.mag_data is not None:
            self.mag_data.reset(mag_mem, nma)
        else:
            self.mag_data = GpuDataManager(mag_mem, nma, False)
        self.streams = [GpuStreamData(self.ex_data, self.ma_data, self.mag_data) for _ in range(nstreams)]

    def engine_iterate(self, num=1):
        """
        Compute one iteration.
        """
        self.dID_list = list(self.di.S.keys())

        prev_event = None

        # atomics or tiled version for probe / object update kernels
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
                # we do this on the first stream we work on
                streamdata = self.streams[self.cur_stream]
                streamdata.start_compute(prev_event)
                if do_update_object:
                    for oID, ob in self.ob.storages.items():
                        cfact = self.ob_cfact[oID]
                        obn = self.ob_nrm.S[oID]

                        if self.p.obj_smooth_std is not None:
                            logger.info('Smoothing object, cfact is %.2f' % cfact)
                            obb = self.ob_buf.S[oID]
                            smooth_mfs = [self.p.obj_smooth_std, self.p.obj_smooth_std]
                            self.GSK.convolution(ob.gpu, smooth_mfs, tmp=obb.gpu)
                        
                        ob.gpu._axpbz(np.complex64(cfact), 0, ob.gpu, stream=streamdata.queue)
                        obn.gpu.fill(np.float32(cfact), stream=streamdata.queue)
                
                self.ex_data.syncback = True

                # First cycle: Fourier + object update
                for dID in self.dID_list:
                    t1 = time.time()
                    prep = self.diff_info[dID]
                    streamdata = self.streams[self.cur_stream]
                    
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

                    # set streams
                    queue = streamdata.queue
                    FUK.queue = queue
                    AWK.queue = queue
                    POK.queue = queue
                    PROP.queue = queue

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

                    # transfer exit wave to gpu
                    ex = streamdata.ex_to_gpu(dID, prep.ex)
                    
                    # transfer ma/mag data to gpu if needed
                    if do_update_fourier:
                        # transfer other input data in
                        ma, mag = streamdata.ma_to_gpu(dID, prep.ma, prep.mag)
                    # waits for compute on previous stream to finish before continuing
                    streamdata.start_compute(prev_event)

                    # Fourier update
                    if do_update_fourier:
                        log(4, '------ Fourier update -----', True)

                        ## compute log-likelihood
                        if self.p.compute_log_likelihood:
                            t1 = time.time()
                            AWK.build_aux_no_ex(aux, addr, ob, pr)
                            PROP.fw(aux, aux)
                            FUK.log_likelihood(aux, addr, mag, ma, err_phot)                    
                            self.benchmark.F_LLerror += time.time() - t1

                        ## prep + forward FFT
                        t1 = time.time()
                        AWK.build_aux(aux, addr, ob, pr, ex, alpha=self.p.alpha)
                        self.benchmark.A_Build_aux += time.time() - t1

                        t1 = time.time()
                        PROP.fw(aux, aux)
                        self.benchmark.B_Prop += time.time() - t1

                        ## Deviation from measured data
                        t1 = time.time()
                        FUK.fourier_error(aux, addr, mag, ma, ma_sum)
                        FUK.error_reduce(addr, err_fourier)
                        FUK.fmag_all_update(aux, addr, mag, ma, err_fourier, pbound)
                        self.benchmark.C_Fourier_update += time.time() - t1
                        streamdata.record_done_ma(dID)

                        ## Backward FFT
                        t1 = time.time()
                        PROP.bw(aux, aux)
                        ## apply changes
                        AWK.build_exit(aux, addr, ob, pr, ex)
                        FUK.exit_error(aux, addr)
                        FUK.error_reduce(addr, err_exit)
                        self.benchmark.E_Build_exit += time.time() - t1
                        
                        self.benchmark.calls_fourier += 1

                    prestr = '%d Iteration (Overlap) #%02d:  ' % (parallel.rank, inner)

                    # Object update
                    if do_update_object:
                        log(4, prestr + '----- object update -----', True)
                        t1 = time.time()

                        addrt = addr if atomics_object else addr2
                        POK.ob_update(addrt, obb, obn, pr, ex, atomics=atomics_object)
                        self.benchmark.object_update += time.time() - t1
                        self.benchmark.calls_object += 1

                    # end_compute is to allow aux + ob re-use, so we can mark it here
                    prev_event = streamdata.end_compute()
                    streamdata.record_done_ex(dID)
                    self.cur_stream = (self.cur_stream + self.stream_direction) % len(self.streams)

                # swap direction for next time
                if do_update_fourier:
                    self.dID_list.reverse()
                    self.stream_direction = -self.stream_direction
                    # make sure we start with the same stream were we stopped
                    self.cur_stream = (self.cur_stream + self.stream_direction) % len(self.streams)

                if do_update_object:
                    self._object_allreduce()

                # Exit if probe should fnot yet be updated
                if not do_update_probe:
                    break

                # Update probe
                log(4, prestr + '----- probe update -----', True)
                self.ex_data.syncback = False
                change = self.probe_update(MPI=MPI)
                # change = self.probe_update(MPI=(parallel.size>1 and MPI))
                
                # swap direction for next time
                self.dID_list.reverse()
                self.stream_direction = -self.stream_direction
                # make sure we start with the same stream were we stopped
                self.cur_stream = (self.cur_stream + self.stream_direction) % len(self.streams)

                log(4, prestr + 'change in probe is %.3f' % change, True)

                # stop iteration if probe change is small
                if change < self.p.overlap_converge_factor: break

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
                    prev_event = None
                    for dID in self.di.S.keys():
                        streamdata = self.streams[self.cur_stream]

                        prep = self.diff_info[dID]
                        pID, oID, eID = prep.poe_IDs
                        ob = self.ob.S[oID].gpu
                        pr = self.pr.S[pID].gpu
                        kern = self.kernels[prep.label]
                        aux = kern.aux
                        addr = prep.addr_gpu
                        original_addr = prep.original_addr
                        ma_sum = prep.ma_sum_gpu
                        ma, mag = streamdata.ma_to_gpu(dID, prep.ma, prep.mag)

                        PCK = kern.PCK
                        AUK = kern.AUK
                        PCK.queue = streamdata.queue
                        PROP.queue = streamdata.queue
                        AUK.queue = streamdata.queue

                        #error_state = np.zeros(err_fourier.shape, dtype=np.float32)
                        #err_fourier.get_async(streamdata.queue, error_state)
                        cuda.memcpy_dtod_async(dest=prep.error_state_gpu.ptr,
                                               src=prep.err_fourier_gpu.ptr,
                                               size=prep.err_fourier_gpu.nbytes,
                                               stream=streamdata.queue)
                        streamdata.start_compute(prev_event)

                        log(4, 'Position refinement trial: iteration %s' % (self.curiter))
                        for i in range(self.p.position_refinement.nshifts):
                            addr_cpu = addr.get_async(streamdata.queue)
                            streamdata.queue.synchronize()
                            mangled_addr = PCK.address_mangler.mangle_address(addr_cpu, original_addr, self.curiter)
                            mangled_addr_gpu = gpuarray.to_gpu_async(mangled_addr, stream=streamdata.queue)
                            PCK.build_aux(aux, mangled_addr_gpu, ob, pr)
                            PROP.fw(aux, aux)
                            PCK.fourier_error(aux, mangled_addr_gpu, mag, ma, ma_sum)
                            PCK.error_reduce(mangled_addr_gpu, prep.err_fourier_gpu)
                            # err_fourier_cpu = err_fourier.get_async(streamdata.queue)
                            PCK.update_addr_and_error_state(addr, 
                                prep.error_state_gpu, 
                                mangled_addr_gpu, 
                                prep.err_fourier_gpu)
                        cuda.memcpy_dtod_async(dest=prep.err_fourier_gpu.ptr,
                                               src=prep.error_state_gpu.ptr,
                                               size=prep.err_fourier_gpu.nbytes,
                                               stream=streamdata.queue)
                        if use_tiles:
                            s1 = prep.addr_gpu.shape[0] * prep.addr_gpu.shape[1]
                            s2 = prep.addr_gpu.shape[2] * prep.addr_gpu.shape[3]
                            kern.TK.transpose(prep.addr_gpu.reshape(s1, s2), prep.addr2_gpu.reshape(s2, s1))

                        prev_event = streamdata.end_compute()
                        
                        # next stream
                        self.cur_stream = (self.cur_stream + self.stream_direction) % len(self.streams)

            self.curiter += 1

        #print('end loop, syncall and copy back')
        for sd in self.streams:
            sd.synchronize()
        
        for name, s in self.ob.S.items():
            s.gpu.get(s.data)
        for name, s in self.pr.S.items():
            s.gpu.get(s.data)


        # FIXXME: copy to pinned memory
        for dID, prep in self.diff_info.items():
            err_fourier = prep.err_fourier_gpu.get()
            err_phot = prep.err_phot_gpu.get()
            err_exit = prep.err_exit_gpu.get()
            errs = np.ascontiguousarray(np.vstack([err_fourier, err_phot, err_exit]).T)
            error.update(zip(prep.view_IDs, errs))

        self.error = error
        return error
    

    def _object_allreduce(self):
        # make sure that all transfers etc are finished
        for sd in self.streams:
            sd.synchronize()
        # sync all
        for oID, ob in self.ob.storages.items():
            obn = self.ob_nrm.S[oID]
            obb = self.ob_buf.S[oID]
            if MPI:
                obb.gpu.get(obb.data)
                obn.gpu.get(obn.data)
                parallel.allreduce(obb.data)
                parallel.allreduce(obn.data)
                obb.data /= obn.data
                self.clip_object(obb)
                tt1 = time.time()
                ob.gpu.set(obb.data)  # async tx on same stream?
                
            else:
                obb.gpu /= obn.gpu
                ob.gpu[:] = obb.gpu


    ## probe update
    def probe_update(self, MPI=False):
        t1 = time.time()
        streamdata = self.streams[self.cur_stream]
        use_atomics = self.p.probe_update_cuda_atomics
        # storage for-loop
        change = 0
        prev_event = None
        for pID, pr in self.pr.storages.items():
            prn = self.pr_nrm.S[pID]
            cfact = self.pr_cfact[pID]
            pr.gpu._axpbz(np.complex64(cfact), 0, pr.gpu, stream=streamdata.queue)
            prn.gpu.fill(np.float32(cfact), stream=streamdata.queue)

        for dID in self.dID_list:
            prep = self.diff_info[dID]
            streamdata = self.streams[self.cur_stream]

            POK = self.kernels[prep.label].POK
            POK.queue = streamdata.queue
            # find probe, object in exit ID in dependence of dID
            pID, oID, eID = prep.poe_IDs

            ex = streamdata.ex_to_gpu(dID, prep.ex)

            # scan for-loop
            addrt = prep.addr_gpu if use_atomics else prep.addr2_gpu
            streamdata.start_compute(prev_event)
            ev = POK.pr_update(addrt,
                               self.pr.S[pID].gpu,
                               self.pr_nrm.S[pID].gpu,
                               self.ob.S[oID].gpu,
                               ex,
                               atomics=use_atomics)
            streamdata.record_done_ex(dID)
            prev_event = streamdata.end_compute()
            self.cur_stream = (self.cur_stream + self.stream_direction) % len(self.streams)

            
        # sync all streams first
        for sd in self.streams:
            sd.synchronize()

        for pID, pr in self.pr.storages.items():

            buf = self.pr_buf.S[pID]
            prn = self.pr_nrm.S[pID]

            # MPI test
            if MPI:
                # if False:
                pr.gpu.get(pr.data)
                prn.gpu.get(prn.data)
                parallel.allreduce(pr.data)
                parallel.allreduce(prn.data)
                pr.data /= prn.data
                self.support_constraint(pr)
                pr.gpu.set(pr.data)
            else:
                pr.gpu /= prn.gpu
                # ca. 0.3 ms
                # self.pr.S[pID].gpu = probe_gpu
                pr.gpu.get(pr.data)

            ## this should be done on GPU
            tt1 = time.time()
            change += u.norm2(pr.data - buf.data) / u.norm2(pr.data)
            buf.data[:] = pr.data
            if MPI:
                change = parallel.allreduce(change) / parallel.size
            tt2 = time.time()
            #print('time for pr change: {}s'.format(tt2-tt1))

        # print 'probe update: ' + str(time.time()-t1)
        self.benchmark.probe_update += time.time() - t1
        self.benchmark.calls_probe += 1

        return np.sqrt(change)

    def engine_finalize(self):
        """
        Clear all GPU data, pinned memory, etc
        """ 
        self.streams = None
        self.ex_data = None
        self.ma_data = None
        self.mag_data = None

        super().engine_finalize()
