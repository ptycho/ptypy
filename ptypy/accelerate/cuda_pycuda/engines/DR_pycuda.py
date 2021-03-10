# -*- coding: utf-8 -*-
"""
Local Douglas-Rachford reconstruction engine.

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
from ptypy.accelerate.base.engines import DR_serial
from ptypy.accelerate.base import address_manglers
from .. import get_context
from ..kernels import FourierUpdateKernel, AuxiliaryWaveKernel, PoUpdateKernel, PositionCorrectionKernel, PropagationKernel
from ..array_utils import ArrayUtilsKernel, GaussianSmoothingKernel, TransposeKernel
from ..mem_utils import make_pagelocked_paired_arrays as mppa

MPI = False

__all__ = ['DR_pycuda']

@register()
class DR_pycuda(DR_serial.DR_serial):

    """
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
        super(DR_pycuda, self).__init__(ptycho_parent, pars)

    
    def engine_initialize(self):
        """
        Prepare for reconstruction.
        """
        self.context, self.queue = get_context(new_context=True, new_queue=True)

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

            # Currently modes not implemented for DR algorithm 
            assert scan.p.coherence.num_probe_modes == 1
            assert scan.p.coherence.num_object_modes == 1
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

            #logger.info("Setting up TransposeKernel")
            #kern.TK = TransposeKernel(queue=self.queue)

            logger.info("Setting up PropagationKernel")
            kern.PROP = PropagationKernel(aux, geo.propagator, self.queue, self.p.fft_lib)
            kern.PROP.allocate()
            kern.resolution = geo.resolution[0]

            # if self.do_position_refinement:
            #     logger.info("Setting up position correction")
            #     addr_mangler = address_manglers.RandomIntMangle(int(self.p.position_refinement.amplitude // geo.resolution[0]),
            #                                                     self.p.position_refinement.start,
            #                                                     self.p.position_refinement.stop,
            #                                                     max_bound=int(self.p.position_refinement.max_shift // geo.resolution[0]),
            #                                                     randomseed=0)
            #     logger.warning("amplitude is %s " % (self.p.position_refinement.amplitude // geo.resolution[0]))
            #     logger.warning("max bound is %s " % (self.p.position_refinement.max_shift // geo.resolution[0]))

            #     kern.PCK = PositionCorrectionKernel(aux, nmodes, queue_thread=self.queue)
            #     kern.PCK.allocate()
            #     kern.PCK.address_mangler = addr_mangler

            logger.info("Kernel setup completed")


    def engine_prepare(self):

        super(DM_pycuda, self).engine_prepare()

        for name, s in self.ob.S.items():
            s.gpu = gpuarray.to_gpu(s.data)
        for name, s in self.pr.S.items():
            s.gpu, s.data = mppa(s.data)

        # TODO : like the serialization this one is needed due to object reformatting
        for label, d in self.di.storages.items():
            prep = self.diff_info[d.ID]
            prep.addr_gpu = [gpuarray.to_gpu(prep.addr[i,None]) for i in range(len(prep.addr))]

        for label, d in self.ptycho.new_data:
            prep = self.diff_info[d.ID]
            pID, oID, eID = prep.poe_IDs
            s = self.ex.S[eID]
            s.gpu = gpuarray.to_gpu(s.data)
            s = self.ma.S[d.ID]
            s.gpu = gpuarray.to_gpu(s.data.astype(np.float32))

            prep.mag = [gpuarray.to_gpu(prep.mag[i,None]) for i in range(len(prep.mag))]
            prep.ma_sum = [gpuarray.to_gpu(prep.ma_sum[i,None]) for i in range(len(prep.ma_sum))]
            prep.err_fourier_gpu = [gpuarray.to_gpu(prep.err_fourier[i,None]) for i in range(len(prep.err_fourier))]
            prep.err_phot_gpu = [gpuarray.to_gpu(prep.err_phot[i,None]) for i in range(len(prep.err_phot))]
            prep.err_exit_gpu = [gpuarray.to_gpu(prep.err_exit[i,None]) for i in range(len(prep.err_exit))]
            # if self.do_position_refinement:
            #     prep.error_state_gpu = gpuarray.empty_like(prep.err_fourier_gpu)


    def engine_iterate(self, num=1):
        """
        Compute one iteration.
        """
        queue = self.queue

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
                POK = kern.POK
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
                vieworder = prep.vieworder

                # local references
                ma = self.ma.S[dID].gpu
                ob = self.ob.S[oID].gpu
                pr = self.pr.S[pID].gpu
                ex = self.ex.S[eID].gpu

                # randomly shuffle view order
                np.random.shuffle(vieworder)

                # Iterate through views
                for i in vieworder:

                    # Get local adress and arrays
                    addr = prep.addr[i]
                    mag = prep.mag[i]
                    ma = prep.ma[i]
                    ma_sum = prep.ma_sum[i]

                    err_phot = prep.err_phot[i]
                    err_fourier = prep.err_fourier[i]
                    err_exit = prep.err_exit[i]                   

                    ## compute log-likelihood
                    t1 = time.time()
                    AWK.build_aux_no_ex(aux, addr, ob, pr)
                    aux[:] = FW(aux)
                    FUK.log_likelihood(aux, addr, mag, ma, err_phot)
                    self.benchmark.F_LLerror += time.time() - t1

                    ## build auxilliary wave
                    t1 = time.time()
                    AWK.build_aux(aux, addr, ob, pr, ex, alpha=self.p.alpha)
                    self.benchmark.A_Build_aux += time.time() - t1

                    ## forward FFT
                    t1 = time.time()
                    aux[:] = FW(aux)
                    self.benchmark.B_Prop += time.time() - t1

                    ## Deviation from measured data
                    t1 = time.time()
                    FUK.fourier_error(aux, addr, mag, ma, ma_sum)
                    FUK.error_reduce(addr, err_fourier)
                    FUK.fmag_all_update(aux, addr, mag, ma, err_fourier, pbound)
                    self.benchmark.C_Fourier_update += time.time() - t1

                    ## backward FFT
                    t1 = time.time()
                    aux[:] = BW(aux)
                    self.benchmark.D_iProp += time.time() - t1

                    ## build exit wave
                    t1 = time.time()
                    AWK.build_exit_alpha_tau(aux, addr, ob, pr, ex, alpha=self.p.alpha, tau=self.p.tau)
                    FUK.exit_error(aux,addr)
                    FUK.error_reduce(addr, err_exit)
                    self.benchmark.E_Build_exit += time.time() - t1
                    self.benchmark.calls_fourier += 1

                    ## probe/object rescale
                    #if self.p.rescale_probe:
                    #    pr *= np.sqrt(self.mean_power / (np.abs(pr)**2).mean())

                    ## build auxilliary wave (ob * pr product)
                    t1 = time.time()
                    AWK.build_aux(aux, addr, ob, pr, ex, alpha=0)
                    self.benchmark.A_Build_aux += time.time() - t1

                    # object update
                    t1 = time.time()
                    POK.ob_update_local(addr, ob, pr, ex, aux)
                    self.benchmark.object_update += time.time() - t1
                    self.benchmark.calls_object += 1

                    # probe update
                    t1 = time.time()
                    POK.pr_update_local(addr, pr, ob, ex, aux)
                    self.benchmark.probe_update += time.time() - t1
                    self.benchmark.calls_probe += 1

            self.curiter += 1
            queue.synchronize()

        for name, s in self.ob.S.items():
            s.data[:] = s.gpu.get()
        for name, s in self.pr.S.items():
            s.data[:] = s.gpu.get()

        for dID, prep in self.diff_info.items():
            err_fourier = np.array([prep.err_fourier_gpu[i].get() for i in range(len(prep.err_fourier_gpu))])
            err_phot = np.array([prep.err_phot_gpu[i].get() for i in range(len(prep.err_phot_gpu))])
            err_exit = np.array([prep.err_exit_gpu[i].get() for i in range(len(prep.err_exit_gpu))])
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
            prep.addr = np.array([prep.addr_gpu[i].get() for i in range(len(prep.addr_gpu))])

        # copy data to cpu 
        # this kills the pagelock memory (otherwise we get segfaults in h5py)
        for name, s in self.pr.S.items():
            s.data = np.copy(s.data)

        self.context.detach()
        super(DR_pycuda, self).engine_finalize()