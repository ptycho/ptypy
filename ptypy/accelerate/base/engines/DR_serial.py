# -*- coding: utf-8 -*-
"""
Local Difference Map/Alternate Projections reconstruction engine.

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
from ptypy.engines import register
from ptypy.engines.base import PositionCorrectionEngine
from ptypy.core.manager import Full, Vanilla, Bragg3dModel, BlockVanilla, BlockFull
from ptypy.accelerate.base.engines import DM_serial
from ptypy.accelerate.base.kernels import FourierUpdateKernel, AuxiliaryWaveKernel, PoUpdateKernel, PositionCorrectionKernel
from ptypy.accelerate.base import address_manglers
from ptypy.accelerate.base import array_utils as au

# for debugging
import h5py

__all__ = ['DR_serial']

@register()
class DR_serial(PositionCorrectionEngine):
    """
    An implementation of the Douglas-Rachford algorithm
    that can be operated like the ePIE algorithm.

    Defaults:

    [name]
    default = DR_serial
    type = str
    help =
    doc =

    [alpha]
    default = 1
    type = float
    lowlim = 0.0
    help = Tuning parameter, a value of 0 makes it equal to ePIE.

    [tau]
    default = 1
    type = float
    lowlim = 0.0
    help = fourier update parameter, a value of 0 means no fourier update.

    [probe_inertia]
    default = 1e-9
    type = float
    lowlim = 0.0
    help = Weight of the current probe estimate in the update

    [object_inertia]
    default = 1e-4
    type = float
    lowlim = 0.0
    help = Weight of the current object in the update

    [clip_object]
    default = None
    type = tuple
    help = Clip object amplitude into this interval

    [rescale_probe]
    default = True
    type = bool
    lowlim = 0
    help = Normalise probe power according to data

    [fourier_power_bound]
    default = None
    type = float
    help = If rms error of model vs diffraction data is smaller than this value, Fourier constraint is met
    doc = For Poisson-sampled data, the theoretical value for this parameter is 1/4. Set this value higher for noisy data.

    [compute_log_likelihood]
    default = True
    type = bool
    help = A switch for computing the log-likelihood error (this can impact the performance of the engine)

    [debug]
    default = None
    type = str
    help = For debugging purposes, dump arrays into given directory

    [debug_iter]
    default = 0
    type = int
    help = For debugging purposes, dump arrays at this iteration

    """

    SUPPORTED_MODELS = [Full, Vanilla, Bragg3dModel, BlockVanilla, BlockFull]

    def __init__(self, ptycho_parent, pars=None):
        """
        Local difference map reconstruction engine.
        """
        super(DR_serial, self).__init__(ptycho_parent, pars)

        # Instance attributes
        self.error = None
        self.pbound = None
        self.mean_power = None

        # keep track of timings
        self.benchmark = u.Param()

        # Stores all information needed with respect to the diffraction storages.
        self.diff_info = {}
        self.ob_cfact = {}
        self.pr_cfact = {}
        self.kernels = {}

        self.ptycho.citations.add_article(
            title='Semi-implicit relaxed Douglas-Rachford algorithm (sDR) for ptychography',
            author='Pham et al.',
            journal='Opt. Express',
            volume=27,
            year=2019,
            page=31246,
            doi='10.1364/OE.27.031246',
            comment='The local douglas-rachford reconstruction algorithm',
        )

    def engine_initialize(self):
        """
        Prepare for reconstruction.
        """
        super(DR_serial, self).engine_initialize()

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
            self.pbound_scan = {}
            for s in self.di.storages.values():            
                self.pbound_scan[s.label] = self.p.fourier_power_bound
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

            ob = self.ob.S[oID]
            misfit = np.asarray(ob.shape[-2:]) % 32
            if (misfit != 0).any():
                pad = 32 - np.asarray(ob.shape[-2:]) % 32
                ob.data = u.crop_pad(ob.data, [[0, pad[0]], [0, pad[1]]], axes=[-2, -1], filltype='project')
                ob.shape = ob.data.shape

            # Keep a list of view indices
            prep.vieworder = np.arange(prep.addr.shape[0])

            # Transform addr array into a list
            prep.addr = [prep.addr[i,None] for i in range(prep.addr.shape[0])]

            # Transform mag, ma, ma_sum into lists
            prep.mag = [prep.mag[i,None] for i in range(prep.mag.shape[0])]
            prep.ma  = [prep.ma[i,None] for i in range(prep.ma.shape[0])]
            prep.ma_sum = [prep.ma_sum[i,None] for i in range(prep.ma_sum.shape[0])]

            # Transform errors into lists
            prep.err_phot = [prep.err_phot[i,None] for i in range(prep.err_phot.shape[0])]
            prep.err_fourier = [prep.err_fourier[i,None] for i in range(prep.err_fourier.shape[0])]
            prep.err_exit = [prep.err_exit[i,None] for i in range(prep.err_exit.shape[0])]

            # calculate c_facts
            #cfact = self.p.object_inertia * self.mean_power
            #self.ob_cfact[oID] = cfact / u.parallel.size

            #pr = self.pr.S[pID]
            #cfact = self.p.probe_inertia * len(pr.views) / pr.data.shape[0]
            #self.pr_cfact[pID] = cfact / u.parallel.size


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

                # global buffers
                pbound = self.pbound_scan[prep.label]
                aux = kern.aux
                vieworder = prep.vieworder

                # references for ob, pr, ex
                ob = self.ob.S[oID].data
                pr = self.pr.S[pID].data
                ex = self.ex.S[eID].data

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

                    # debugging
                    if self.p.debug and parallel.master and (self.curiter == self.p.debug_iter):
                        with h5py.File(self.p.debug + "/before_%04d.h5" %self.curiter, "w") as f:
                            f["aux"] = aux
                            f["addr"] = addr
                            f["ob"] = ob
                            f["pr"] = pr
                            f["mag"] = mag
                            f["ma"] = ma
                            f["ma_sum"] = ma_sum

                    ## compute log-likelihood
                    if self.p.compute_log_likelihood:
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

                # update errors
                errs = np.ascontiguousarray(np.vstack([np.hstack(prep.err_fourier), 
                                                       np.hstack(prep.err_phot), 
                                                       np.hstack(prep.err_exit)]).T)
                error_dct.update(zip(prep.view_IDs, errs))

            self.curiter += 1

        error = parallel.gather_dict(error_dct)
        return error


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
