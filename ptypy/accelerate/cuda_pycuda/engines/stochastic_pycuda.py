# -*- coding: utf-8 -*-
"""
Accelerated stochastic reconstruction engine.

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
from ptypy.accelerate.base.engines import stochastic_serial
from ptypy.accelerate.base import address_manglers
from .. import get_context
from ..kernels import FourierUpdateKernel, AuxiliaryWaveKernel, PoUpdateKernel, PositionCorrectionKernel, PropagationKernel
from ..array_utils import ArrayUtilsKernel, GaussianSmoothingKernel, TransposeKernel
from ..mem_utils import make_pagelocked_paired_arrays as mppa
from ..mem_utils import GpuDataManager2

MPI = False

EX_MA_BLOCKS_RATIO = 2
MAX_BLOCKS = 99999  # can be used to limit the number of blocks, simulating that they don't fit
#AX_BLOCKS = 10  # can be used to limit the number of blocks, simulating that they don't fit

class StochasticBaseEnginePycuda(stochastic_serial.StochasticBaseEngineSerial):

    """
    An accelerated implementation of a stochastic algorithm for ptychography

    Defaults:

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
        Accelerated base engine for stochastic algorithms.
        """
        super().__init__(ptycho_parent, pars)
        self.ma_data = None
        self.mag_data = None
        self.ex_data = None
    
    def engine_initialize(self):
        """
        Prepare for reconstruction.
        """
        self.context, self.queue = get_context(new_context=True, new_queue=True)
        super().engine_initialize()
        self.qu_htod = cuda.Stream()
        self.qu_dtoh = cuda.Stream()

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
            fpc = 1
            ash = (fpc * nmodes,) + tuple(geo.shape)
            aux = np.zeros(ash, dtype=np.complex64)
            kern.aux = gpuarray.to_gpu(aux)

            # setup kernels, one for each SCAN.
            log(4, "Setting up FourierUpdateKernel")
            kern.FUK = FourierUpdateKernel(aux, nmodes, queue_thread=self.queue)
            kern.FUK.fshape = (1,) + kern.FUK.fshape[1:]
            kern.FUK.allocate()

            log(4, "Setting up PoUpdateKernel")
            kern.POK = PoUpdateKernel(queue_thread=self.queue)
            kern.POK.allocate()

            log(4, "Setting up AuxiliaryWaveKernel")
            kern.AWK = AuxiliaryWaveKernel(queue_thread=self.queue)
            kern.AWK.allocate()

            log(4, "Setting up ArrayUtilsKernel")
            kern.AUK = ArrayUtilsKernel(queue=self.queue)

            #log(4, "Setting up TransposeKernel")
            #kern.TK = TransposeKernel(queue=self.queue)

            log(4, "Setting up PropagationKernel")
            kern.PROP = PropagationKernel(aux, geo.propagator, self.queue, self.p.fft_lib)
            kern.PROP.allocate()
            kern.resolution = geo.resolution[0]

            if self.do_position_refinement:
                log(4, "Setting up position correction")
                kern.PCK = PositionCorrectionKernel(aux, nmodes, self.p.position_refinement, geo.resolution, queue_thread=self.queue)
                kern.PCK.allocate()
            #     addr_mangler = address_manglers.RandomIntMangle(int(self.p.position_refinement.amplitude // geo.resolution[0]),
            #                                                     self.p.position_refinement.start,
            #                                                     self.p.position_refinement.stop,
            #                                                     max_bound=int(self.p.position_refinement.max_shift // geo.resolution[0]),
            #                                                     randomseed=0)
            #     log(5, "amplitude is %s " % (self.p.position_refinement.amplitude // geo.resolution[0]))
            #     log(5, "max bound is %s " % (self.p.position_refinement.max_shift // geo.resolution[0]))

            #     kern.PCK = PositionCorrectionKernel(aux, nmodes, queue_thread=self.queue)
            #     kern.PCK.allocate()
            #     kern.PCK.address_mangler = addr_mangler
        
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
        log(4, "Kernel setup completed")

    def engine_prepare(self):
        super().engine_prepare()

        for name, s in self.ob.S.items():
            s.gpu, s.data = mppa(s.data)
        for name, s in self.pr.S.items():
            s.gpu, s.data = mppa(s.data)

        for label, d in self.di.storages.items():
            prep = self.diff_info[d.ID]
            prep.addr_gpu = gpuarray.to_gpu(prep.addr)
            if self.do_position_refinement:
                prep.mangled_addr_gpu = prep.addr_gpu.copy()

        for label, d in self.ptycho.new_data:
            dID = d.ID
            prep = self.diff_info[dID]
            pID, oID, eID = prep.poe_IDs

            prep.ma_sum_gpu = gpuarray.to_gpu(prep.ma_sum)
            prep.err_fourier_gpu = gpuarray.to_gpu(prep.err_fourier)
            prep.err_phot_gpu = gpuarray.to_gpu(prep.err_phot)
            prep.err_exit_gpu = gpuarray.to_gpu(prep.err_exit)
            if self.do_position_refinement:
                prep.error_state_gpu = gpuarray.empty_like(prep.err_fourier_gpu)
            prep.obn = gpuarray.to_gpu(prep.obn)
            prep.prn = gpuarray.to_gpu(prep.prn)
            # prepare page-locked mems:
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
                    ex_from, ex_to = prep.addr_ex[i]
                    ex = ex_full[ex_from:ex_to]
                    mag = mag_full[i,None]
                    ma = ma_full[i,None]
                    ma_sum = prep.ma_sum[i,None]
                    obn = prep.obn
                    prn = prep.prn
                    err_phot = prep.err_phot_gpu[i,None]
                    err_fourier = prep.err_fourier_gpu[i,None]
                    err_exit = prep.err_exit_gpu[i,None]

                    ## build auxilliary wave
                    AWK.build_aux2(aux, addr, ob, pr, ex, alpha=self.alpha)

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
                    AWK.build_exit_alpha_tau(aux, addr, ob, pr, ex, alpha=self.alpha, tau=self.tau)
                    if self.p.compute_exit_error:
                        FUK.exit_error(aux,addr)
                        FUK.error_reduce(addr, err_exit)

                    ## build auxilliary wave (ob * pr product)
                    AWK.build_aux2_no_ex(aux, addr, ob, pr)

                    # object update
                    POK.pr_norm_local(addr, pr, prn)
                    POK.ob_update_local(addr, ob, pr, ex, aux, prn, A=self.obA, B=self.obB)

                    # probe update
                    POK.ob_norm_local(addr, ob, obn)
                    POK.pr_update_local(addr, pr, ob, ex, aux, obn, A=self.prA, B=self.prB)

                    ## compute log-likelihood
                    if self.p.compute_log_likelihood:
                        PROP.fw(aux, aux)
                        FUK.log_likelihood2(aux, addr, mag, ma, err_phot)

                    # position update
                    self.position_update()

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
            err_fourier = prep.err_fourier_gpu.get()
            err_phot = prep.err_phot_gpu.get()
            err_exit = prep.err_exit_gpu.get()
            errs = np.ascontiguousarray(np.vstack([err_fourier, err_phot, err_exit]).T)
            error.update(zip(prep.view_IDs, errs))

        # wait for the async transfers
        self.qu_dtoh.synchronize()

        self.error = error
        return error

    def position_update(self):
        """
        Position refinement
        """
        if not self.do_position_refinement or (not self.curiter):
            return
        do_update_pos = (self.p.position_refinement.stop > self.curiter >= self.p.position_refinement.start)
        do_update_pos &= (self.curiter % self.p.position_refinement.interval) == 0

        # Update positions
        if do_update_pos:
            """
            Iterates through all positions and refines them by a given algorithm.
            """
            log(4, "----------- START POS REF -------------")


    def engine_finalize(self):
        """
        clear GPU data and destroy context.
        """
        self.ex_data = None
        self.ma_data = None
        self.mag_data = None

        for name, s in self.ob.S.items():
            del s.gpu
        for name, s in self.pr.S.items():
            del s.gpu
        for dID, prep in self.diff_info.items():
            prep.addr = prep.addr_gpu.get()

        # copy data to cpu 
        # this kills the pagelock memory (otherwise we get segfaults in h5py)
        for name, s in self.pr.S.items():
            s.data = np.copy(s.data)
        for name, s in self.ob.S.items():
            s.data = np.copy(s.data)

        self.context.detach()
        super().engine_finalize()