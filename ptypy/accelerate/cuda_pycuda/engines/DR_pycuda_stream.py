# -*- coding: utf-8 -*-
"""
Local Douglas-Rachford reconstruction engine for NVIDIA GPUs.

This engine uses three streams, one for the compute queue and one for each I/O queue.
Events are used to synchronize download / compute/ upload. we cannot manipulate memory
for each loop over the state vector, a certain number of memory sections is preallocated
and reused.

This file is part of the PTYPY package.

    :copyright: Copyright 2014 by the PTYPY team, see AUTHORS.
    :license: GPLv2, see LICENSE for details.
"""

from ptypy.accelerate.cuda_pycuda.engines.DM_pycuda_stream import DM_pycuda_stream
import numpy as np
from pycuda import gpuarray
import pycuda.driver as cuda

from ptypy import utils as u
from ptypy.utils.verbose import logger, log
from ptypy.utils import parallel
from ptypy.engines import register
from . import DR_pycuda

from ..mem_utils import make_pagelocked_paired_arrays as mppa
from ..mem_utils import GpuDataManager2

MPI = False

EX_MA_BLOCKS_RATIO = 2
MAX_BLOCKS = 99999  # can be used to limit the number of blocks, simulating that they don't fit
#MAX_BLOCKS = 4  # can be used to limit the number of blocks, simulating that they don't fit

__all__ = ['DR_pycuda_stream']

@register()
class DR_pycuda_stream(DR_pycuda.DR_pycuda):

    def __init__(self, ptycho_parent, pars=None):

        super(DR_pycuda_stream, self).__init__(ptycho_parent, pars)
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
        fpc = self.ptycho.frames_per_block
        for scan, kern in self.kernels.items():
            ex_mem = max(kern.aux.nbytes * fpc, ex_mem)
            mag_mem = max(kern.FUK.gpu.fdev.nbytes * fpc, mag_mem)
        ma_mem = mag_mem
        mem = cuda.mem_get_info()[0]
        blk = ex_mem * EX_MA_BLOCKS_RATIO + ma_mem + mag_mem
        fit = int(mem - 200 * 1024 * 1024) // blk  # leave 200MB room for safety

        # TODO grow blocks dynamically
        nex = min(fit * EX_MA_BLOCKS_RATIO, MAX_BLOCKS)
        nma = min(fit, MAX_BLOCKS)

        log(3, 'PyCUDA max blocks fitting on GPU: exit arrays={}, ma_arrays={}'.format(nex, nma))
        # reset memory or create new
        self.ex_data = GpuDataManager2(ex_mem, 0, nex, True)
        self.ma_data = GpuDataManager2(ma_mem, 0, nma, False)
        self.mag_data = GpuDataManager2(mag_mem, 0, nma, False)

    def engine_prepare(self):

        super(DR_pycuda.DR_pycuda, self).engine_prepare()

        for name, s in self.ob.S.items():
            s.gpu, s.data = mppa(s.data)
        for name, s in self.pr.S.items():
            s.gpu, s.data = mppa(s.data)

        for label, d in self.di.storages.items():
            prep = self.diff_info[d.ID]
            prep.addr_gpu = gpuarray.to_gpu(prep.addr)

        for label, d in self.ptycho.new_data:
            dID = d.ID
            prep = self.diff_info[dID]
            pID, oID, eID = prep.poe_IDs

            prep.ma_sum_gpu = gpuarray.to_gpu(prep.ma_sum)
            # prepare page-locked mems:
            prep.err_fourier_gpu = gpuarray.to_gpu(prep.err_fourier)
            prep.err_phot_gpu = gpuarray.to_gpu(prep.err_phot)
            prep.err_exit_gpu = gpuarray.to_gpu(prep.err_exit)
            ma = self.ma.S[dID].data.astype(np.float32)
            prep.ma = cuda.pagelocked_empty(ma.shape, ma.dtype, order="C", mem_flags=4)
            prep.ma[:] = ma
            ex = self.ex.S[eID].data
            prep.ex = cuda.pagelocked_empty(ex.shape, ex.dtype, order="C", mem_flags=4)
            prep.ex[:] = ex
            mag = prep.mag
            prep.mag = cuda.pagelocked_empty(mag.shape, mag.dtype, order="C", mem_flags=4)
            prep.mag[:] = mag

            self.ex_data.add_data_block()
            self.ma_data.add_data_block()
            self.mag_data.add_data_block()

    def engine_iterate(self, num=1):
        """
        Compute one iteration.
        """
        self.dID_list = list(self.di.S.keys())
        error = {}

        for it in range(num):

            for iblock, dID in enumerate(self.dID_list):

                # find probe, object and exit ID in dependence of dID
                prep = self.diff_info[dID]
                pID, oID, eID = prep.poe_IDs

                # references for kernels
                kern = self.kernels[prep.label]
                FUK = kern.FUK
                AWK = kern.AWK
                POK = kern.POK
                PROP = kern.PROP

                # get aux buffer
                aux = kern.aux

                # local references
                ob = self.ob.S[oID].gpu
                pr = self.pr.S[pID].gpu

                # shuffle view order
                vieworder = prep.vieworder
                prep.rng.shuffle(vieworder)

                # Schedule ex, ma, mag to device
                ev_ex, ex_full, data_ex = self.ex_data.to_gpu(prep.ex, dID, self.qu_htod)
                ev_mag, mag_full, data_mag = self.mag_data.to_gpu(prep.mag, dID, self.qu_htod)
                ev_ma, ma_full, data_ma = self.ma_data.to_gpu(prep.ma, dID, self.qu_htod)

                ## synchronize h2d stream with compute stream
                self.queue.wait_for_event(ev_ex)

                # Iterate through views
                for i in vieworder:

                    # Get local adress and arrays
                    addr = prep.addr_gpu[i,None]
                    ex = ex_full[i,None]
                    mag = mag_full[i,None]
                    ma = ma_full[i,None]
                    ma_sum = prep.ma_sum[i,None]
                    err_phot = prep.err_phot_gpu[i,None]
                    err_fourier = prep.err_fourier_gpu[i,None]
                    err_exit = prep.err_exit_gpu[i,None]

                    ## build auxilliary wave
                    AWK.build_aux2(aux, addr, ob, pr, ex, alpha=self.p.alpha)

                    ## forward FFT
                    PROP.fw(aux, aux)

                    ## Deviation from measured data
                    self.queue.wait_for_event(ev_mag)
                    if self.p.compute_fourier_error:
                        self.queue.wait_for_event(ev_ma)
                        FUK.fourier_error(aux, addr, mag, ma, ma_sum)
                        FUK.error_reduce(addr, err_fourier)
                    else:
                        FUK.fourier_deviation(aux, addr, mag)
                        self.queue.wait_for_event(ev_ma)
                    FUK.fmag_update_nopbound(aux, addr, mag, ma)

                    ## backward FFT
                    PROP.bw(aux, aux)

                    ## build exit wave
                    AWK.build_exit_alpha_tau(aux, addr, ob, pr, ex, alpha=self.p.alpha, tau=self.p.tau)
                    if self.p.compute_exit_error:
                        FUK.exit_error(aux,addr)
                        FUK.error_reduce(addr, err_exit)

                    ## probe/object rescale
                    #if self.p.rescale_probe:
                    #    pr *= np.sqrt(self.mean_power / (np.abs(pr)**2).mean())

                    ## build auxilliary wave (ob * pr product)
                    AWK.build_aux2_no_ex(aux, addr, ob, pr)

                    # object update
                    POK.ob_update_local(addr, ob, pr, ex, aux)

                    # probe update
                    POK.pr_update_local(addr, pr, ob, ex, aux)

                    ## compute log-likelihood
                    if self.p.compute_log_likelihood:
                        PROP.fw(aux, aux)
                        FUK.log_likelihood2(aux, addr, mag, ma, err_phot)

                data_ex.record_done(self.queue, 'compute')
                if iblock + len(self.ex_data) < len(self.dID_list):
                    data_ex.from_gpu(self.qu_dtoh)
            
            # swap direction
            self.dID_list.reverse()
            
            self.curiter += 1
            self.ex_data.syncback = False

        # finish all the compute
        self.queue.synchronize()
        
        for name, s in self.ob.S.items():
            s.gpu.get_async(stream=self.qu_dtoh, ary=s.data)
        for name, s in self.pr.S.items():
            s.gpu.get_async(stream=self.qu_dtoh, ary=s.data)

        for dID, prep in self.diff_info.items():
            prep.err_fourier_gpu.get(prep.err_fourier)
            prep.err_phot_gpu.get(prep.err_phot)
            prep.err_exit_gpu.get(prep.err_exit)
            errs = np.ascontiguousarray(np.vstack([
                prep.err_fourier, prep.err_phot, prep.err_exit
                ]).T)
            error.update(zip(prep.view_IDs, errs))

        # wait for the async transfers
        self.qu_dtoh.synchronize()

        self.error = error
        return error

    def engine_finalize(self):
        """
        Clear all GPU data, pinned memory, etc
        """
        self.ex_data = None
        self.ma_data = None
        self.mag_data = None

        # replacing page-locked data with normal npy to avoid
        # crash on context destroy
        for name, s in self.pr.S.items():
            s.data = np.copy(s.data)
        for name, s in self.ob.S.items():
            s.data = np.copy(s.data)

        super().engine_finalize()
        