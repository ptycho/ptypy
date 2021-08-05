# -*- coding: utf-8 -*-
"""
Serialized stochastic reconstruction engine.

This file is part of the PTYPY package.

    :copyright: Copyright 2014 by the PTYPY team, see AUTHORS.
    :license: GPLv2, see LICENSE for details.
"""
import numpy as np
import time

from ptypy import utils as u
from ptypy.utils.verbose import logger, log
from ptypy.utils import parallel
from ptypy import defaults_tree
from ptypy.engines import register, stochastic
from ptypy.core.manager import Full, Vanilla, Bragg3dModel, BlockVanilla, BlockFull
from ptypy.accelerate.base.engines import DM_serial
from ptypy.accelerate.base.kernels import FourierUpdateKernel, AuxiliaryWaveKernel, PoUpdateKernel, PositionCorrectionKernel
from ptypy.accelerate.base import address_manglers
from ptypy.accelerate.base import array_utils as au


class StochasticBaseEngineSerial(stochastic.StochasticBaseEngine):
    """
    A serialized base implementation of a stochastic algorithm for ptychography

    Defaults:

    [clip_object]
    default = None
    type = tuple
    help = Clip object amplitude into this interval

    [compute_log_likelihood]
    default = True
    type = bool
    help = A switch for computing the log-likelihood error (this can impact the performance of the engine)

    [compute_exit_error]
    default = False
    type = bool
    help = A switch for computing the exitwave error (this can impact the performance of the engine)

    [compute_fourier_error]
    default = False
    type = bool
    help = A switch for computing the fourier error (this can impact the performance of the engine)

    """

    def __init__(self, ptycho_parent, pars=None):
        """
        Stochastic reconstruction engine.
        """
        super().__init__(ptycho_parent, pars)

        # keep track of timings
        self.benchmark = u.Param()

        # Stores all information needed with respect to the diffraction storages.
        self.diff_info = {}
        self.ob_cfact = {}
        self.pr_cfact = {}
        self.kernels = {}

    def engine_initialize(self):
        """
        Prepare for reconstruction.
        """
        super().engine_initialize()

        self.error = []
        self._reset_benchmarks()
        self._setup_kernels()

    def _reset_benchmarks(self):
        self.benchmark.A_Build_aux = 0.
        self.benchmark.B_Prop = 0.
        self.benchmark.C_Fourier_update = 0.
        self.benchmark.D_iProp = 0.
        self.benchmark.E_Build_exit = 0.
        self.benchmark.F_LLerror = 0.
        self.benchmark.probe_update = 0.
        self.benchmark.object_update = 0.
        self.benchmark.calls_fourier = 0
        self.benchmark.calls_object = 0
        self.benchmark.calls_probe = 0

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
            ash = (1 * nmodes,) + tuple(geo.shape)
            aux = np.zeros(ash, dtype=np.complex64)
            kern.aux = aux

            # setup kernels, one for each SCAN.
            kern.FUK = FourierUpdateKernel(aux, nmodes)
            kern.FUK.allocate()

            kern.POK = PoUpdateKernel()
            kern.POK.allocate()

            kern.AWK = AuxiliaryWaveKernel()
            kern.AWK.allocate()

            kern.FW = geo.propagator.fw
            kern.BW = geo.propagator.bw
            kern.resolution = geo.resolution[0]

            if self.do_position_refinement:
                addr_mangler = address_manglers.RandomIntMangle(int(self.p.position_refinement.amplitude // geo.resolution[0]),
                                                                self.p.position_refinement.start,
                                                                self.p.position_refinement.stop,
                                                                max_bound=int(self.p.position_refinement.max_shift // geo.resolution[0]),
                                                                randomseed=0)
                logger.warning("amplitude is %s " % (self.p.position_refinement.amplitude // geo.resolution[0]))
                logger.warning("max bound is %s " % (self.p.position_refinement.max_shift // geo.resolution[0]))

                kern.PCK = PositionCorrectionKernel(aux, nmodes)
                kern.PCK.allocate()
                kern.PCK.address_mangler = addr_mangler

    def engine_prepare(self):
        """
        Last minute initialization.

        Everything that needs to be recalculated when new data arrives.
        """
        if self.ptycho.new_data:

            # recalculate everything
            mean_power = 0.
            for s in self.di.storages.values():
                mean_power += s.mean_power
            self.mean_power = mean_power / len(self.di.storages)

        ## Serialize new data ##
        for label, d in self.ptycho.new_data:
            prep = u.Param()
            prep.label = label
            self.diff_info[d.ID] = prep
            prep.mag = np.sqrt(np.abs(d.data))
            prep.ma = self.ma.S[d.ID].data.astype(np.float32)
            prep.ma_sum = prep.ma.sum(-1).sum(-1)
            prep.err_phot = np.zeros_like(prep.ma_sum)
            prep.err_fourier = np.zeros_like(prep.ma_sum)
            prep.err_exit = np.zeros_like(prep.ma_sum)
            
        # Unfortunately this needs to be done for all pods, since
        # the shape of the probe / object was modified.
        # TODO: possible scaling issue, remove the need for padding
        for label, d in self.di.storages.items():
            prep = self.diff_info[d.ID]
            prep.view_IDs, prep.poe_IDs, prep.addr = DM_serial.serialize_array_access(d)
            if self.do_position_refinement:
                prep.original_addr = np.zeros_like(prep.addr)
                prep.original_addr[:] = prep.addr
            pID, oID, eID = prep.poe_IDs

            # Keep a list of view indices
            prep.rng = np.random.default_rng()
            prep.vieworder = np.arange(prep.addr.shape[0])

            # Modify addresses, copy pa into ea and remove da/ma
            prep.addr_ex = np.vstack([prep.addr[:,0,2,0], prep.addr[:,-1,2,0]+1]).T
            prep.addr[:,:,2] = prep.addr[:,:,0]
            prep.addr[:,:,3:,0] = 0

            # Reference to ex
            prep.ex = self.ex.S[eID].data

            # Object / probe norm
            prep.obn = np.zeros_like(prep.mag[0,None], dtype=np.float32)
            prep.prn = np.zeros_like(prep.mag[0,None], dtype=np.float32)

    def engine_iterate(self, num=1):
        """
        Compute one iteration.
        """
        for it in range(num):   

            error_dct = {}

            for dID in self.di.S.keys():

                # find probe, object and exit ID in dependence of dID
                prep = self.diff_info[dID]
                pID, oID, eID = prep.poe_IDs

                # references for kernels
                kern = self.kernels[prep.label]
                FUK = kern.FUK
                AWK = kern.AWK
                POK = kern.POK
                FW = kern.FW
                BW = kern.BW

                # global aux buffer
                aux = kern.aux

                # references for ob, pr
                ob = self.ob.S[oID].data
                pr = self.pr.S[pID].data

                # shuffle view order
                vieworder = prep.vieworder
                prep.rng.shuffle(vieworder)

                # Iterate through views
                for i in vieworder:

                    # Get local adress and arrays
                    addr = prep.addr[i,None]
                    ex_from, ex_to = prep.addr_ex[i]
                    ex = prep.ex[ex_from:ex_to]
                    mag = prep.mag[i,None]
                    ma = prep.ma[i,None]
                    ma_sum = prep.ma_sum[i,None]
                    obn = prep.obn
                    prn = prep.prn
                    err_phot = prep.err_phot[i,None]
                    err_fourier = prep.err_fourier[i,None]
                    err_exit = prep.err_exit[i,None]

                    ## build auxilliary wave
                    t1 = time.time()
                    AWK.build_aux(aux, addr, ob, pr, ex, alpha=self.alpha)
                    self.benchmark.A_Build_aux += time.time() - t1

                    ## forward FFT
                    t1 = time.time()
                    aux[:] = FW(aux)
                    self.benchmark.B_Prop += time.time() - t1

                    ## Deviation from measured data
                    t1 = time.time()
                    if self.p.compute_fourier_error:
                        FUK.fourier_error(aux, addr, mag, ma, ma_sum)
                        FUK.error_reduce(addr, err_fourier)
                    else:
                        FUK.fourier_deviation(aux, addr, mag)
                    FUK.fmag_update_nopbound(aux, addr, mag, ma)
                    self.benchmark.C_Fourier_update += time.time() - t1

                    ## backward FFT
                    t1 = time.time()
                    aux[:] = BW(aux)
                    self.benchmark.D_iProp += time.time() - t1

                    ## build exit wave
                    t1 = time.time()
                    AWK.build_exit_alpha_tau(aux, addr, ob, pr, ex, alpha=self.alpha, tau=self.tau)
                    if self.p.compute_exit_error:
                        FUK.exit_error(aux,addr)
                        FUK.error_reduce(addr, err_exit)
                    self.benchmark.E_Build_exit += time.time() - t1
                    self.benchmark.calls_fourier += 1

                    ## build auxilliary wave (ob * pr product)
                    t1 = time.time()
                    AWK.build_aux_no_ex(aux, addr, ob, pr)
                    self.benchmark.A_Build_aux += time.time() - t1

                    # object update
                    t1 = time.time()
                    POK.pr_norm_local(addr, pr, prn)
                    POK.ob_update_local(addr, ob, pr, ex, aux, prn, A=self.obA, B=self.obB)
                    self.benchmark.object_update += time.time() - t1
                    self.benchmark.calls_object += 1

                    # probe update
                    t1 = time.time()
                    POK.ob_norm_local(addr, ob, obn)
                    POK.pr_update_local(addr, pr, ob, ex, aux, obn, A=self.prA, B=self.prB)
                    self.benchmark.probe_update += time.time() - t1
                    self.benchmark.calls_probe += 1

                    ## compute log-likelihood
                    if self.p.compute_log_likelihood:
                        t1 = time.time()
                        aux[:] = FW(aux)
                        FUK.log_likelihood(aux, addr, mag, ma, err_phot)
                        self.benchmark.F_LLerror += time.time() - t1

                # update errors
                errs = np.ascontiguousarray(np.vstack([np.hstack(prep.err_fourier), 
                                                       np.hstack(prep.err_phot), 
                                                       np.hstack(prep.err_exit)]).T)
                error_dct.update(zip(prep.view_IDs, errs))

            self.curiter += 1

        #error = parallel.gather_dict(error_dct)
        return error_dct


    def engine_finalize(self):
        """
        try deleting ever helper contianer
        """
        if parallel.master and self.benchmark.calls_fourier:
            print("----- BENCHMARKS ----")
            acc = 0.
            for name in sorted(self.benchmark.keys()):
                t = self.benchmark[name]
                if name[0] in 'ABCDEFGHI':
                    print('%20s : %1.3f ms per iteration' % (name, t / self.benchmark.calls_fourier * 1000))
                    acc += t
                elif str(name) == 'probe_update':
                    print('%20s : %1.3f ms per call. %d calls' % (
                        name, t / self.benchmark.calls_probe * 1000, self.benchmark.calls_probe))
                elif str(name) == 'object_update':
                    print('%20s : %1.3f ms per call. %d calls' % (
                        name, t / self.benchmark.calls_object * 1000, self.benchmark.calls_object))

            print('%20s : %1.3f ms per iteration. %d calls' % (
                'Fourier_total', acc / self.benchmark.calls_fourier * 1000, self.benchmark.calls_fourier))

        self._reset_benchmarks()

        if self.do_position_refinement:
            for label, d in self.di.storages.items():
                prep = self.diff_info[d.ID]
                res = self.kernels[prep.label].resolution
                for i,view in enumerate(d.views):
                    for j,(pname, pod) in enumerate(view.pods.items()):
                        delta = (prep.original_addr[i][j][1][1:] - prep.addr[i][j][1][1:]) * res
                        pod.ob_view.coord += delta 
                        pod.ob_view.storage.update_views(pod.ob_view)
