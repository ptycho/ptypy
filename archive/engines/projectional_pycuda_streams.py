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
from ptypy.engines.projectional import DMMixin, RAARMixin
from . import projectional_pycuda

# factor how many more exit waves we wanna keep on GPU compared to 
# ma / mag data
EX_MA_BLOCKS_RATIO = 2
MAX_STREAMS = 500   # max number of streams to use
MAX_BLOCKS = 99999  # can be used to limit the number of blocks, simulating that they don't fit

__all__ = ['DM_pycuda_streams']

class GpuData:
    """
    Manages one block of GPU data with corresponding CPU data.
    Keeps track of which cpu array is currently on GPU by its id,
    and transfers if it's not already there.

    To be used for the exit wave, ma, and mag arrays.
    Note: Allocator should be pooled for best performance
    """

    def __init__(self, nbytes, syncback=False):
        """
        New instance of GpuData. Allocates the GPU-side array.

        :param nbytes: Number of bytes held by this instance.
        :param syncback: Should the data be synced back to CPU any time it's swapped out
        """

        self.gpu = None
        self.gpuraw = cuda.mem_alloc(nbytes)
        self.nbytes = nbytes
        self.nbytes_buffer = nbytes
        self.gpuId = None
        self.cpu = None
        self.syncback = syncback
        self.ev_done = None

    def _allocator(self, nbytes):
        if nbytes > self.nbytes:
            raise Exception('requested more bytes than maximum given before: {} vs {}'.format(nbytes, self.nbytes))
        return self.gpuraw

    def record_done(self, stream):
        self.ev_done = cuda.Event()
        self.ev_done.record(stream)

    def to_gpu(self, cpu, id, stream):
        """
        Transfer cpu array to GPU on stream (async), keeping track of its id
        """
        if self.gpuId != id:
            if self.syncback:
                self.from_gpu(stream)
            self.gpuId = id
            self.cpu = cpu
            if self.ev_done is not None:
                self.ev_done.synchronize()
            self.gpu = gpuarray.to_gpu_async(cpu, allocator=self._allocator, stream=stream)
        return self.gpu

    def from_gpu(self, stream):
        """
        Transfer data back to CPU, into same data handle it was copied from
        before.
        """
        if self.cpu is not None and self.gpuId is not None and self.gpu is not None:
            if self.ev_done is not None:
                stream.wait_for_event(self.ev_done)
            self.gpu.get_async(stream, self.cpu)
            self.ev_done = cuda.Event()
            self.ev_done.record(stream)

    def resize(self, nbytes):
        """
        Resize the size of the underlying buffer, to allow re-use in different contexts.
        Note that memory will only be freed/reallocated if the new number of bytes are
        either larger than before, or if they are less than 90% of the original size -
        otherwise it reuses the existing buffer
        """
        if nbytes > self.nbytes_buffer or nbytes < self.nbytes_buffer * .9:
            self.nbytes_buffer = nbytes
            self.gpuraw.free()
            self.gpuraw = cuda.mem_alloc(nbytes)
        self.nbytes = nbytes
        self.reset()

    def reset(self):
        """
        Resets handles of cpu references and ids, so that all data will be transfered
        again even if IDs match.
        """
        self.gpuId = None
        self.cpu = None
        self.ev_done = None

    def free(self):
        """
        Free the underlying buffer on GPU - this object should not be used afterwards
        """
        self.gpuraw.free()
        self.gpuraw = None

class GpuDataManager:
    """
    Manages a set of GpuData instances, to keep several blocks on device.

    Note that the syncback property is used so that during fourier updates,
    the exit wave array is synced bck to cpu (it is updated),
    while during probe update, it's not.
    """

    def __init__(self, nbytes, num, syncback=False):
        """
        Create an instance of GpuDataManager.
        Parameters are the same as for GpuData, and num is the number of
        GpuData instances to create (blocks on device).
        """
        self.data = [GpuData(nbytes, syncback) for _ in range(num)]

    @property
    def syncback(self):
        """
        Get if syncback of data to CPU on swapout is enabled.
        """
        return self.data[0].syncback

    @syncback.setter
    def syncback(self, whether):
        """
        Adjust the syncback setting
        """
        for d in self.data:
            d.syncback = whether

    @property
    def nbytes(self):
        """
        Get the number of bytes in each block
        """
        return self.data[0].nbytes

    @property
    def memory(self):
        """
        Get all memory occupied by all blocks
        """
        m = 0
        for d in self.data:
            m += d.nbytes_buffer
        return m

    def __len__(self):
        return len(self.data)

    def reset(self, nbytes, num):
        """
        Reset this object as if these parameters were given to the constructor.
        The syncback property is untouched.
        """
        sync = self.syncback
        # remove if too many, explictly freeing memory
        for i in range(num, len(self.data)):
            self.data[i].free()
        # cut short if too many
        self.data = self.data[:num]
        # reset existing
        for d in self.data:
            d.resize(nbytes)
        # append new ones
        for i in range(len(self.data), num):
            self.data.append(GpuData(nbytes, sync))

    def free(self):
        """
        Explicitly clear all data blocks - same as resetting to 0 blocks
        """
        self.reset(0, 0)


    def to_gpu(self, cpu, id, stream):
        """
        Transfer a block to the GPU, given its ID and CPU data array
        """
        idx = 0
        for x in self.data:
            if x.gpuId == id:
                break
            idx += 1
        if idx == len(self.data):
            idx = 0
        else:
            pass
        m = self.data.pop(idx)
        self.data.append(m)
        return m.to_gpu(cpu, id, stream)

    def record_done(self, id, stream):
        for x in self.data:
            if x.gpuId == id:
                x.record_done(stream)
                return
        raise Exception('recording done for id not in pool')


    def sync_to_cpu(self, stream):
        """
        Sync back all data to CPU
        """
        for x in self.data:
            x.from_gpu(stream)


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


class _ProjectionEngine_pycuda_streams(projectional_pycuda._ProjectionEngine_pycuda):

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

        super().__init__(ptycho_parent, pars)
        self.streams = None 
        self.ma_data = None
        self.mag_data = None
        self.ex_data = None
        self.cur_stream = 0
        self.stream_direction = 1

    def engine_prepare(self):

        super(projectional_pycuda._ProjectionEngine_pycuda, self).engine_prepare()

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
        for name, s in self.pr_buf.S.items():
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

        # Extra object buffer for smoothing kernel
        if self.p.obj_smooth_std is not None:
            for name, s in self.ob_buf.S.items():
                s.tmp = gpuarray.empty(s.gpu.shape, s.gpu.dtype)

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
            if self.do_position_refinement:
                prep.mangled_addr_gpu = prep.addr_gpu.copy()

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

        log(4, 'PyCUDA blocks fitting on GPU: exit arrays={}, ma_arrays={}, streams={}, totalblocks={}'.format(nex, nma, nstreams, blocks))
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
                        obb = self.ob_buf.S[oID]

                        if self.p.obj_smooth_std is not None:
                            log(4,'Smoothing object, cfact is %.2f' % cfact)
                            smooth_mfs = [self.p.obj_smooth_std, self.p.obj_smooth_std]
                            # We need a third copy, because we still need ob.gpu for the fourier update
                            # obb.gpu[:] = ob.gpu[:]
                            cuda.memcpy_dtod_async(dest=obb.gpu.ptr,
                                                   src=ob.gpu.ptr,
                                                   size=ob.gpu.nbytes,
                                                   stream=streamdata.queue)
                            streamdata.queue.synchronize()
                            self.GSK.queue = streamdata.queue
                            self.GSK.convolution(obb.gpu, smooth_mfs, tmp=obb.tmp)
                            obb.gpu._axpbz(np.complex64(cfact), 0, obb.gpu, stream=streamdata.queue)
                        else:
                            ob.gpu._axpbz(np.complex64(cfact), 0, obb.gpu, stream=streamdata.queue)
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
                        #AWK.build_aux(aux, addr, ob, pr, ex, alpha=self.p.alpha)
                        AWK.make_aux(aux, addr, ob, pr, ex, c_po=self._c, c_e=1-self._c)
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
                        #AWK.build_exit(aux, addr, ob, pr, ex, alpha=self.p.alpha)
                        AWK.make_exit(aux, addr, ob, pr, ex, c_a=self._b, c_po=self._a, c_e=-(self._a + self._b))
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
                change = self.probe_update()
                
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
                    log(4, "----------- START POS REF -------------")
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
                        mangled_addr = prep.mangled_addr_gpu
                        ma_sum = prep.ma_sum_gpu
                        ma, mag = streamdata.ma_to_gpu(dID, prep.ma, prep.mag)
                        err_fourier = prep.err_fourier_gpu
                        error_state = prep.error_state_gpu

                        PCK = kern.PCK
                        TK = kern.TK
                        PROP = kern.PROP
                        PCK.queue = streamdata.queue
                        TK.queue = streamdata.queue
                        PROP.queue = streamdata.queue

                        # Keep track of object boundaries
                        max_oby = ob.shape[-2] - aux.shape[-2] - 1
                        max_obx = ob.shape[-1] - aux.shape[-1] - 1

                        # We need to re-calculate the current error
                        PCK.build_aux(aux, addr, ob, pr)
                        PROP.fw(aux, aux)
                        if self.p.position_refinement.metric == "fourier":
                            PCK.fourier_error(aux, addr, mag, ma, ma_sum)
                            PCK.error_reduce(addr, err_fourier)
                        if self.p.position_refinement.metric == "photon":
                            PCK.log_likelihood(aux, addr, mag, ma, err_fourier)
                        cuda.memcpy_dtod_async(dest=error_state.ptr,
                                               src=err_fourier.ptr,
                                               size=err_fourier.nbytes,
                                               stream=streamdata.queue)
                        streamdata.start_compute(prev_event)

                        log(4, 'Position refinement trial: iteration %s' % (self.curiter))
                        PCK.mangler.setup_shifts(self.curiter, nframes=addr.shape[0])
                        for i in range(PCK.mangler.nshifts):
                            streamdata.queue.synchronize()
                            PCK.mangler.get_address(i, addr, mangled_addr, max_oby, max_obx)
                            PCK.build_aux(aux, mangled_addr, ob, pr)
                            PROP.fw(aux, aux)
                            if self.p.position_refinement.metric == "fourier":
                                PCK.fourier_error(aux, mangled_addr, mag, ma, ma_sum)
                                PCK.error_reduce(mangled_addr, err_fourier)
                            if self.p.position_refinement.metric == "photon":
                                PCK.log_likelihood(aux, mangled_addr, mag, ma, err_fourier)
                            PCK.update_addr_and_error_state(addr, error_state, mangled_addr, err_fourier)

                        cuda.memcpy_dtod_async(dest=err_fourier.ptr,
                                               src=error_state.ptr,
                                               size=err_fourier.nbytes,
                                               stream=streamdata.queue)
                        if use_tiles:
                            s1 = prep.addr_gpu.shape[0] * prep.addr_gpu.shape[1]
                            s2 = prep.addr_gpu.shape[2] * prep.addr_gpu.shape[3]
                            TK.transpose(prep.addr_gpu.reshape(s1, s2), prep.addr2_gpu.reshape(s2, s1))

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
            self.multigpu.allReduceSum(obb.gpu)
            self.multigpu.allReduceSum(obn.gpu)
            obb.gpu /= obn.gpu
            
            self.clip_object(obb.gpu)
            ob.gpu[:] = obb.gpu

    ## probe update
    def probe_update(self, MPI=False):
        t1 = time.time()
        streamdata = self.streams[self.cur_stream]
        use_atomics = self.p.probe_update_cuda_atomics
        # storage for-loop
        change_gpu = gpuarray.zeros((1,), dtype=np.float32)
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

        # print 'probe update: ' + str(time.time()-t1)
        self.benchmark.probe_update += time.time() - t1
        self.benchmark.calls_probe += 1

        return np.sqrt(change)

    def engine_finalize(self, benchmark=False):
        """
        Clear all GPU data, pinned memory, etc
        """ 
        self.streams = None
        self.ex_data = None
        self.ma_data = None
        self.mag_data = None

        super().engine_finalize(benchmark)


@register()
class DM_pycuda_streams(_ProjectionEngine_pycuda_streams, DMMixin):
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
        _ProjectionEngine_pycuda_streams.__init__(self, ptycho_parent, pars)
        DMMixin.__init__(self, self.p.alpha)
        ptycho_parent.citations.add_article(**self.article)


@register()
class RAAR_pycuda_streams(_ProjectionEngine_pycuda_streams, RAARMixin):
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

        _ProjectionEngine_pycuda_streams.__init__(self, ptycho_parent, pars)
        RAARMixin.__init__(self, self.p.beta)
