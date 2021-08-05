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

MPI = False

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
        Difference map reconstruction engine.
        """
        super().__init__(ptycho_parent, pars)

    
    def engine_initialize(self):
        """
        Prepare for reconstruction.
        """
        self.context, self.queue = get_context(new_context=True, new_queue=True)
        super().engine_initialize()

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

            # if self.do_position_refinement:
            #     log(4, "Setting up position correction")
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

            log(4, "Kernel setup completed")


    def engine_prepare(self):

        super().engine_prepare()

        for name, s in self.ob.S.items():
            s.gpu = gpuarray.to_gpu(s.data)
        for name, s in self.pr.S.items():
            s.gpu, s.data = mppa(s.data)

        # TODO : like the serialization this one is needed due to object reformatting
        for label, d in self.di.storages.items():
            prep = self.diff_info[d.ID]
            prep.addr_gpu = gpuarray.to_gpu(prep.addr)

        for label, d in self.ptycho.new_data:
            prep = self.diff_info[d.ID]
            prep.ex = gpuarray.to_gpu(prep.ex)
            prep.mag = gpuarray.to_gpu(prep.mag)
            prep.ma = gpuarray.to_gpu(prep.ma)
            prep.ma_sum = gpuarray.to_gpu(prep.ma_sum)
            prep.err_fourier_gpu = gpuarray.to_gpu(prep.err_fourier)
            prep.err_phot_gpu = gpuarray.to_gpu(prep.err_phot)
            prep.err_exit_gpu = gpuarray.to_gpu(prep.err_exit)
            # if self.do_position_refinement:
            #     prep.error_state_gpu = gpuarray.empty_like(prep.err_fourier_gpu)
            prep.obn = gpuarray.to_gpu(prep.obn)
            prep.prn = gpuarray.to_gpu(prep.prn)


    def engine_iterate(self, num=1):
        """
        Compute one iteration.
        """
        queue = self.queue
        error = {}
        for it in range(num):
            
            for dID in self.di.S.keys():

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

                # Iterate through views
                for i in vieworder:

                    # Get local adress and arrays
                    addr = prep.addr_gpu[i,None]
                    ex_from, ex_to = prep.addr_ex[i]
                    ex = prep.ex[ex_from:ex_to]
                    mag = prep.mag[i,None]
                    ma = prep.ma[i,None]
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
                    if self.p.compute_fourier_error:
                        FUK.fourier_error(aux, addr, mag, ma, ma_sum)
                        FUK.error_reduce(addr, err_fourier)
                    else:
                        FUK.fourier_deviation(aux, addr, mag)
                    FUK.fmag_update_nopbound(aux, addr, mag, ma)

                    ## backward FFT
                    PROP.bw(aux, aux)

                    ## build exit wave
                    AWK.build_exit_alpha_tau(aux, addr, ob, pr, ex, alpha=self.alpha, tau=self.tau)
                    if self.p.compute_exit_error:
                        FUK.exit_error(aux,addr)
                        FUK.error_reduce(addr, err_exit)

                    ## probe/object rescale
                    #if self.p.rescale_probe:
                    #    pr *= np.sqrt(self.mean_power / (np.abs(pr)**2).mean())

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

            self.curiter += 1

        queue.synchronize()
        for name, s in self.ob.S.items():
            s.gpu.get(s.data)
        for name, s in self.pr.S.items():
            s.gpu.get(s.data)

        for dID, prep in self.diff_info.items():
            err_fourier = prep.err_fourier_gpu.get()
            err_phot = prep.err_phot_gpu.get()
            err_exit = prep.err_exit_gpu.get()
            errs = np.ascontiguousarray(np.vstack([err_fourier, err_phot, err_exit]).T)
            error.update(zip(prep.view_IDs, errs))

        self.error = error
        return error

    def engine_finalize(self):
        """
        clear GPU data and destroy context.
        """
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

        self.context.detach()
        super().engine_finalize()