# -*- coding: utf-8 -*-
"""
Serialized stochastic reconstruction engine.

This file is part of the PTYPY package.

    :copyright: Copyright 2014 by the PTYPY team, see AUTHORS.
    :license: see LICENSE for details.
"""
import numpy as np
import time

from ptypy import utils as u
from ptypy.utils.verbose import logger, log
from ptypy.utils import parallel
from ptypy import defaults_tree
from ptypy.engines import register
from ptypy.engines.stochastic import _StochasticEngine, EPIEMixin, SDRMixin
#from ptypy.core.manager import Full, Vanilla, Bragg3dModel, BlockVanilla, BlockFull
from ptypy.accelerate.base.engines import projectional_serial
from ptypy.accelerate.base.kernels import FourierUpdateKernel, AuxiliaryWaveKernel, PoUpdateKernel, PositionCorrectionKernel
from ptypy.accelerate.base import address_manglers
from ptypy.accelerate.base import array_utils as au

__all__ = ["EPIE_serial", "SDR_serial"]

class _StochasticEngineSerial(_StochasticEngine):
    """
    A serialized base implementation of a stochastic algorithm for ptychography

    Defaults:

    [compute_exit_error]
    default = False
    type = bool
    help = A switch for computing the exitwave error (this can impact the performance of the engine)

    [compute_fourier_error]
    default = False
    type = bool
    help = A switch for computing the fourier error (this can impact the performance of the engine)

    """

    #SUPPORTED_MODELS = [Full, Vanilla, Bragg3dModel, BlockVanilla, BlockFull]

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
            kern.scanmodel = type(scan).__name__
            self.kernels[label] = kern

            # TODO: needs to be adapted for broad bandwidth
            geo = scan.geometries[0]

            # TODO : make this more foolproof
            try:
                nmodes = scan.p.coherence.num_probe_modes * \
                         scan.p.coherence.num_object_modes
            except:
                nmodes = 1

            # create buffer arrays
            ash = (nmodes,) + tuple(geo.shape)
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
                kern.PCK = PositionCorrectionKernel(aux, nmodes, self.p.position_refinement, geo.resolution)
                kern.PCK.allocate()

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
            prep.view_IDs, prep.poe_IDs, prep.addr = projectional_serial.serialize_array_access(d)
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

                    # position update
                    self.position_update_local(prep,i)

                    ## build auxilliary wave
                    t1 = time.time()
                    AWK.make_aux(aux, addr, ob, pr, ex, c_po=self._c, c_e=1-self._c)
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
                    AWK.make_exit(aux, addr, ob, pr, ex, c_a=self._b, c_po=self._a, c_e=-(self._a+self._b))
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
                    POK.ob_update_local(addr, ob, pr, ex, aux, prn, a=self._ob_a, b=self._ob_b)
                    self.benchmark.object_update += time.time() - t1
                    self.benchmark.calls_object += 1

                    # probe update
                    t1 = time.time()
                    if self._object_norm_is_global and self._pr_a == 0:
                        obn_max = au.max_abs2(ob)
                        obn[:] = 0
                    else:
                        POK.ob_norm_local(addr, ob, obn)
                        obn_max = obn.max()
                    if self.p.probe_update_start <= self.curiter:
                        POK.pr_update_local(addr, pr, ob, ex, aux, obn, obn_max, a=self._pr_a, b=self._pr_b)
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

            # Re-center the probe
            self.center_probe()

            self.curiter += 1

        #error = parallel.gather_dict(error_dct)
        return error_dct

    def position_update_local(self, prep, i):
        """
        Position refinement update for current view.
        """
        if not self.do_position_refinement:
            return
        do_update_pos = (self.p.position_refinement.stop > self.curiter >= self.p.position_refinement.start)
        do_update_pos &= (self.curiter % self.p.position_refinement.interval) == 0

        # Update positions
        if do_update_pos:
            """
            Iterates through all positions and refines them by a given algorithm.
            """
            #log(4, "----------- START POS REF -------------")
            pID, oID, eID = prep.poe_IDs
            mag = prep.mag[i,None]
            ma = prep.ma[i,None]
            ma_sum = prep.ma_sum[i,None]
            ob = self.ob.S[oID].data
            pr = self.pr.S[pID].data
            kern = self.kernels[prep.label]
            aux = kern.aux
            addr = prep.addr[i,None]
            original_addr = prep.original_addr[i,None]
            mangled_addr = addr.copy()
            err_fourier = prep.err_fourier[i,None]

            PCK = kern.PCK
            FW = kern.FW

            # Keep track of object boundaries
            max_oby = ob.shape[-2] - aux.shape[-2] - 1
            max_obx = ob.shape[-1] - aux.shape[-1] - 1

            # We first need to calculate the current error
            PCK.build_aux(aux, addr, ob, pr)
            aux[:] = FW(aux)
            if self.p.position_refinement.metric == "fourier":
                PCK.fourier_error(aux, addr, mag, ma, ma_sum)
                PCK.error_reduce(addr, err_fourier)
            if self.p.position_refinement.metric == "photon":
                PCK.log_likelihood(aux, addr, mag, ma, err_fourier)
            error_state = np.zeros_like(err_fourier)
            error_state[:] = err_fourier
            PCK.mangler.setup_shifts(self.curiter, nframes=addr.shape[0])

            #log(4, 'Position refinement trial: iteration %s' % (self.curiter))
            for i in range(PCK.mangler.nshifts):
                PCK.mangler.get_address(i, addr, mangled_addr, max_oby, max_obx)
                PCK.build_aux(aux, mangled_addr, ob, pr)
                aux[:] = FW(aux)
                if self.p.position_refinement.metric == "fourier":
                    PCK.fourier_error(aux, mangled_addr, mag, ma, ma_sum)
                    PCK.error_reduce(mangled_addr, err_fourier)
                if self.p.position_refinement.metric == "photon":
                    PCK.log_likelihood(aux, mangled_addr, mag, ma, err_fourier)
                PCK.update_addr_and_error_state(addr, error_state, mangled_addr, err_fourier)

            prep.err_fourier[i,None] = error_state
            prep.addr[i,None] = addr

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


@register()
class EPIE_serial(_StochasticEngineSerial, EPIEMixin):
    """
    A serialized implementation of the EPIE algorithm.

    Defaults:

    [name]
    default = EPIE_serial
    type = str
    help =
    doc =

    """

    def __init__(self, ptycho_parent, pars=None):
        _StochasticEngineSerial.__init__(self, ptycho_parent, pars)
        EPIEMixin.__init__(self, self.p.alpha, self.p.beta)
        ptycho_parent.citations.add_article(**self.article)

@register()
class SDR_serial(_StochasticEngineSerial, SDRMixin):
    """
    A serialized implemnentation of the semi-implicit relaxed Douglas-Rachford (SDR) algorithm.

    Defaults:

    [name]
    default = SDR_serial
    type = str
    help =
    doc =

    """

    def __init__(self, ptycho_parent, pars=None):
        _StochasticEngineSerial.__init__(self, ptycho_parent, pars)
        SDRMixin.__init__(self, self.p.sigma, self.p.tau, self.p.beta_probe, self.p.beta_object)
        ptycho_parent.citations.add_article(**self.article)
