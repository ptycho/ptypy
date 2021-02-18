# -*- coding: utf-8 -*-
"""
Local Difference Map/Alternate Projections reconstruction engine.

This file is part of the PTYPY package.

    :copyright: Copyright 2014 by the PTYPY team, see AUTHORS.
    :license: GPLv2, see LICENSE for details.
"""
import numpy as np
import time

from .. import utils as u
from ..utils.verbose import logger, log
from ..utils import parallel
from .. import defaults_tree
from . import register, DM_serial
from .base import PositionCorrectionEngine
from ..core.manager import Full, Vanilla, Bragg3dModel, BlockVanilla, BlockFull
from ..accelerate.array_based.kernels import FourierUpdateKernel, AuxiliaryWaveKernel, PoUpdateKernel, PositionCorrectionKernel
from ..accelerate.array_based import address_manglers
from ..accelerate.array_based import array_utils as au

__all__ = ['DM_local']

@register()
class DM_local(PositionCorrectionEngine):
    """
    A local version of the Difference Map engine
    that can be operated like the ePIE algorithm.


    Defaults:

    [name]
    default = DM_local
    type = str
    help =
    doc =

    [alpha]
    default = 1
    type = float
    lowlim = 0.0
    help = Difference map tuning parameter, a value of 0 makes it equal to ePIE.

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

    [compute_log_likelihood]
    default = True
    type = bool
    help = A switch for computing the log-likelihood error (this can impact the performance of the engine)

    """

    SUPPORTED_MODELS = [Full, Vanilla, Bragg3dModel, BlockVanilla, BlockFull]

    def __init__(self, ptycho_parent, pars=None):
        """
        Local difference map reconstruction engine.
        """
        super(DM_local, self).__init__(ptycho_parent, pars)

        # Instance attributes
        self.error = None
        self.pbound = None

        # Required to get proper normalization of object inertia
        # The actual value is computed in engine_prepare
        # Another possibility would be to use the maximum value of all probe storages.
        self.mean_power = None

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
        super(DM_local, self).engine_initialize()

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
            ash = (fpc * nmodes,) + tuple(geo.shape)
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
                if not self.pbound_scan.get(s.label):
                    self.pbound_scan[s.label] = 0.25
                else:
                    self.pbound_scan[s.label] = max(pb, self.pbound_scan[s.label])
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

            # calculate c_facts
            cfact = self.p.object_inertia * self.mean_power
            self.ob_cfact[oID] = cfact / u.parallel.size

            pr = self.pr.S[pID]
            cfact = self.p.probe_inertia * len(pr.views) / pr.data.shape[0]
            self.pr_cfact[pID] = cfact / u.parallel.size


    def engine_iterate(self, num=1):
        """
        Compute one iteration.
        """
        for it in range(num):   

            error_dct = {}
            for name, di_view in self.di.views.items():
                if not di_view.active:
                    continue
                error_dct[name] = np.array([0,0,0])

            time.sleep(0.1)

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