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

from .. import utils as u
from ..utils.verbose import logger, log
from ..utils import parallel
from . import register, DM_pycuda
from ..accelerate import py_cuda as gpu
from ..accelerate.py_cuda.kernels import FourierUpdateKernel, AuxiliaryWaveKernel, PoUpdateKernel

from pycuda.tools import DeviceMemoryPool

MPI = parallel.size > 1
MPI = True

BLOCKS_ON_DEVICE = 4

__all__ = ['DM_pycuda_stream']

class GpuStreamData:
    def __init__(self, allocator):
        self.queue = cuda.Stream()
        self.ex = None
        self.ma = None
        self.mag = None
        self.ma_dID = None
        self.ev_done = None
        self.ex_dID = None
        self.allocator = allocator

    def ex_to_gpu(self, dID, ex):
        # we have that block already on device
        if self.ex_dID == dID:
            return self.ex
        # wait for previous work on same memory to complete
        if self.ev_done is not None:
            self.ev_done.synchronize()
            self.ev_done = None  
        self.ex_dID = dID
        # transfer async
        self.ex = gpuarray.to_gpu_async(ex, allocator=self.allocator, stream=self.queue)
        return self.ex

    def ex_from_gpu(self, dID, ex):
        self.ex.get_async(self.qeue, ex)

    def ma_to_gpu(self, dID, ma, mag):
        # we have that block already on device
        if self.ma_dID == dID:
            return self.ma, self.mag
        # wait for previous work on memory to complete
        if self.ev_done is not None:
            self.ev_done.synchronize()
            self.ev_done = None
        self.ma_dID = dID
        # transfer async
        self.ma = gpuarray.to_gpu_async(ma, allocator=self.allocator, stream=self.queue)
        self.mag = gpuarray.to_gpu_async(mag, allocator=self.allocator, stream=self.queue)
        return self.ma, self.mag
    
    def record_done(self):
        self.ev_done = cuda.Event()
        self.ev_done.record(self.queue)

    def synchronize(self):
        self.queue.synchronize()
        self.ev_done = None



@register()
class DM_pycuda_stream(DM_pycuda.DM_pycuda):

    def __init__(self, ptycho_parent, pars = None):

        super(DM_pycuda_stream, self).__init__(ptycho_parent, pars)
        self.dmp = DeviceMemoryPool()
        self.streams = [GpuStreamData() for _ in range(BLOCKS_ON_DEVICE)]
        self.cur_stream = 0
        self.stream_direction = 1

    def engine_prepare(self):

        super(DM_pycuda.DM_pycuda, self).engine_prepare()

        for name, s in self.ob.S.items():
            s.gpu = gpuarray.to_gpu(s.data)
        for name, s in self.ob_buf.S.items():
            # obb
            d = s.data
            s.data = cuda.pagelocked_empty(d.shape, d.dtype, order="C", mem_flags=4)
            s.data[:] = d
            s.gpu = gpuarray.to_gpu(s.data)
        for name, s in self.ob_nrm.S.items():
            # obn
            d = s.data
            s.data = cuda.pagelocked_empty(d.shape, d.dtype, order="c", mem_flags=4)
            s.data[:] = d
            s.gpu = gpuarray.to_gpu(s.data)
        for name, s in self.pr.S.items():
            # pr
            d = s.data
            s.data = cuda.pagelocked_empty(d.shape, d.type, order="C", mem_flags=4)
            s.data[:] = d
            s.gpu = gpuarray.to_gpu(s.data)
        for name, s in self.pr_nrm.S.items():
            # prn
            d = s.data
            s.data = cuda.pagelocked_empty(d.shape, d.type, order="C", mem_flags=4)
            s.data[:] = d
            s.gpu = gpuarray.to_gpu(s.data)

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
            ma = self.ma.S[dID].data.astype(np.float32)
            prep.ma = cuda.pagelocked_empty(ma.shape, ma.dtype, order="C", mem_flags=4)
            prep.ma[:] = ma            
            ex = self.ex.S[eID].data
            prep.ex = cuda.pagelocked_empty(ex.shape, ex.dtype, order="C", mem_flags=4)
            prep.ex[:] = ex
            mag = prep.mag
            prep.mag = cuda.pagelocked_empty(mag.shape, mag.dtype, order="C", mem_flags=4)
            prep.mag[:] = mag

    def engine_iterate(self, num=1):
        """
        Compute one iteration.
        """
        self.dID_list = list(self.di.S.keys())

        # atomics or tiled version for probe / object update kernels
        atomics_probe = self.p.probe_update_cuda_atomics
        atomics_object = self.p.object_update_cuda_atomics
        use_atomics = atomics_object or atomics_probe
        use_tiles = (not atomics_object) or (not atomics_probe)
        
        for it in range(num):

            error = {}

            for inner in rnage(self.p.overlap_max_iterations):

                change = 0

                do_update_probe = (self.curiter >= self.p.probe_update_start)
                do_update_object = (self.p.update_object_first or (inner > 0) or not do_update_probe)
                do_update_fourier = (inner == 0)

                # initialize probe and object buffer to receive an update
                # we do this on the first stream we work on
                streamdata = self.streams[self.cur_stream]
                if do_update_object:
                    for oID, ob in self.ob.storages.items():
                        cfact = self.ob_cfact[oID]
                        obn = self.ob_nrm.S[oID]
                        obb = self.ob_buf.S[oID]
                        """
                        if self.p.obj_smooth_std is not None:
                            logger.info('Smoothing object, cfact is %.2f' % cfact)
                            t2 = time.time()
                            self.prg.gaussian_filter(queue, (info[3],info[4]), None, obj_gpu.data, self.gauss_kernel_gpu.data)
                            queue.finish()
                            obj_gpu *= cfact
                            print 'gauss: '  + str(time.time()-t2)
                        else:
                            obj_gpu *= cfact
                        """
                        ob.gpu._axpbz(np.complex64(cfact), 0, obb.gpu, stream=streamdata.queue)
                        obn.gpu.fill(np.float32(cfact), stream=streamdata.queue)
                
                # First cycle: Fourier + object update
                for dID in self.dID_list:
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
                    FW = kern.FW
                    BW = kern.BW

                    # set streams
                    queue = streamdata.queue
                    FUK.queue = queue
                    AWK.queue = queue
                    POK.queue = queue
                    FW.queue = queue
                    BW.queue = queue

                    # get addresses and auxilliary array
                    addr = prep.addr_gpu
                    addr2 = prep.addr2_gpu if use_tiles else None
                    err_fourier = prep.err_fourier_gpu
                    ma_sum = prep.ma_sum_gpu

                    # local references
                    ob = self.ob.S[oID].gpu
                    obn = self.ob_nrm.S[oID].gpu
                    obb = self.ob_buf.S[oID].gpu
                    pr = self.pr.S[pID].gpu

                    # transfer exit wave to gpu
                    prep.ex_gpu = streamdata.ex_to_gpu(dId, prep.ex)
                    ex = prep.ex_gpu 

                    # Fourier update
                    if do_update_fourier:
                        log(4, '------ Fourier update -----', True)

                        # transfer other input data in
                        prep.ma_gpu, prep.mag_gpu = streamdata.ma_to_gpu(dID, prep.ma, prep.mag)
                        ma = prep.ma_gpu
                        mag = prep.mag_gpu

                        ## prep + forward FFT
                        AWK.build_aux(aux, addr, ob, pr, ex, alpha=self.p.alpha)
                        FW.ft(aux, aux)
                        ## Deviation from measured data
                        FUK.fourier_error(aux, addr, mag, ma, ma_sum)
                        FUK.error_reduce(addr, err_fourier)
                        FUK.fmag_all_update(aux, addr, mag, ma, err_fourier, pbound)
                        ## Backward FFT
                        BW.ift(aux, aux)
                        ## apply changes
                        AWK.build_exit(aux, addr, ob, pr, ex)

                    prestr = '%d Iteration (Overlap) #%02d:  ' % (parallel.rank, inner)

                    # Object update
                    if do_update_object:
                        log(4, prestr + '----- object update -----', True)

                        addrt = addr if atomics_object else addr2
                        ev = POK.ob_update(addrt, obb, obn, pr, ex, atomics=atomics_object)

                    streamdata.record_done()
                    self.cur_stream = (self.cur_stream + self.stream_direction) % BLOCKS_ON_DEVICE

                # swap direction for next time
                if do_update_fourier:
                    self.dID_list.reverse()
                    self.stream_direction = -self.stream_direction
                    # make sure we start with the same stream were we stopped
                    self.cur_stream = (self.cur_stream + self.stream_direction) % BLOCKS_ON_DEVICE

                if do_update_object:
                    self._object_allreduce()

                # Exit if probe should not yet be updated
                if not do_update_probe:
                    return

                # Update probe
                log(4, prestr + '----- probe update -----', True)
                change = self.probe_update(MPI=MPI)
                # change = self.probe_update(MPI=(parallel.size>1 and MPI))

                log(4, prestr + 'change in probe is %.3f' % change, True)

                # stop iteration if probe change is small
                if change < self.p.overlap_converge_factor: break

            parallel.barrier()
            self.curiter += 1

        for name, s in self.ob.S.items():
            s.gpu.get(s.data)
        for name, s in self.pr.S.items():
            s.gpu.get(s.data)

        # FIXXME: copy to pinned memory
        for dID, prep in self.diff_info.items():
            err_fourier = prep.err_fourier_gpu.get()
            err_phot = np.zeros_like(err_fourier)
            err_exit = np.zeros_like(err_fourier)
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
                ## FIXXME: make obb/obn data pinned memory + schedule on
                ## last stream that was used
                obb.gpu.get(obb.data)
                obn.gpu.get(obn.data)
                parallel.allreduce(obb.data)
                parallel.allreduce(obn.data)
                obb.data /= obn.data
                self.clip_object(obb)
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

            prep.ex_gpu = streamdata.ex_to_gpu(dID, prep.ex)

            # scan for-loop
            addrt = prep.addr_gpu if use_atomics else prep.addr2_gpu
            ev = POK.pr_update(addrt,
                               self.pr.S[pID].gpu,
                               self.pr_nrm.S[pID].gpu,
                               self.ob.S[oID].gpu,
                               prep.ex_gpu,
                               atomics=use_atomics)
            self.cur_stream = (self.cur_stream + self.stream_direction) % BLOCKS_ON_DEVICE

            
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
            print('time for pr change: {}s'.format(tt2-tt1))

        # print 'probe update: ' + str(time.time()-t1)
        self.benchmark.probe_update += time.time() - t1
        self.benchmark.calls_probe += 1

        return np.sqrt(change)

