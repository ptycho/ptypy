# -*- coding: utf-8 -*-
"""
Difference Map reconstruction engine.

This file is part of the PTYPY package.

    :copyright: Copyright 2014 by the PTYPY team, see AUTHORS.
    :license: GPLv2, see LICENSE for details.
"""
import numpy as np
import time
from .. import utils as u
from ..utils.verbose import logger, log
from ..utils import parallel
from .utils import basic_fourier_update
from . import register
from DM import DM
from ptypy.gpu.config import init_gpus
from ptypy.gpu import object_probe_interaction as opi
from ptypy.array_based import data_utils as du
from ..core.manager import Full, Vanilla

__all__ = ['DMHybrid']

@register()
class DMHybrid(DM):
    """
    A full-fledged Difference Map engine.


    Defaults:

    [name]
    default = DMHybrid
    type = str
    help =
    doc =

    [gpu_device]
    default = 0
    type = int
    help = The device number for the gpu that we will use

    """

    SUPPORTED_MODELS = [Full, Vanilla]

    def __init__(self, ptycho_parent, pars=None):
        """
        Difference map reconstruction engine.
        """
        super(DMHybrid, self).__init__(ptycho_parent, pars)
        if parallel.master:
            init_gpus(self.p.gpu_device)

    def engine_prepare(self):
        """
        Last minute initialization.

        Everything that needs to be recalculated when new data arrives.
        """

        super(DMHybrid, self).engine_prepare()
        # and then something to convert the arrays to numpy

        self.vectorised_scan = {}
        self.propagator = {}
        for dID, _diffs in self.di.S.iteritems():
            self.vectorised_scan[dID] = du.pod_to_arrays(self, dID)

    def engine_iterate(self, num=1):
        """
        Compute `num` iterations.
        """
        to = 0.
        tf = 0.
        for it in range(num):
            t1 = time.time()

            # Fourier update
            error_dct = self.fourier_update()

            t2 = time.time()
            tf += t2 - t1
            if not parallel.master:
                self.ob.fill(0.0)
                self.ob_nrm.fill(0.)
                self.pr.fill(0.0)
                self.pr_nrm.fill(0.0)
            else:
            # Overlap update
                for dID, _diffs in self.di.S.iteritems():
                    cfact_probe = (self.p.probe_inertia * len(self.vectorised_scan[dID]['meta']['addr']) /
                                   self.vectorised_scan[dID]['probe'].shape[0]) * np.ones_like(
                        self.vectorised_scan[dID]['probe'])

                    cfact_object = self.p.object_inertia * self.mean_power * (self.vectorised_scan[dID]['object viewcover'] + 1.)
                    do_update_probe = (self.p.probe_update_start <= self.curiter)


                    opi.difference_map_overlap_update(addr_info=self.vectorised_scan[dID]['meta']['addr'],
                                                      cfact_object=cfact_object,
                                                      cfact_probe=cfact_probe,
                                                      do_update_probe=do_update_probe,
                                                      exit_wave=self.vectorised_scan[dID]['exit wave'],
                                                      ob=self.vectorised_scan[dID]['obj'],
                                                      object_weights=self.vectorised_scan[dID]['object weights'].astype(np.float32),
                                                      probe=self.vectorised_scan[dID]['probe'],
                                                      probe_support=self.probe_support[self.vectorised_scan[dID]['meta']['poe_IDs'][0]].astype(np.complex64),
                                                      probe_weights=self.vectorised_scan[dID]['probe weights'].astype(np.float32),
                                                      max_iterations=self.p.overlap_max_iterations,
                                                      update_object_first=self.p.update_object_first,
                                                      obj_smooth_std=self.p.obj_smooth_std,
                                                      overlap_converge_factor=self.p.overlap_converge_factor,
                                                      probe_center_tol=self.p.probe_center_tol,
                                                      clip_object=self.p.clip_object)

                    # sync probe array
                    for name, s in self.pr.storages.iteritems():
                        # MPI reduction of results
                        nrm = self.pr_nrm.storages[name].data
                        parallel.allreduce(s.data)
                        parallel.allreduce(nrm)

                    # sync the object array
                    for name, s in self.ob.storages.iteritems():
                        # Get the np arrays
                        nrm = self.ob_nrm.storages[name].data
                        parallel.allreduce(s.data)
                        parallel.allreduce(nrm)




            t3 = time.time()
            to += t3 - t2

            # count up
            self.curiter +=1

        logger.info('Time spent in Fourier update: %.2f' % tf)
        logger.info('Time spent in Overlap update: %.2f' % to)
        error = parallel.gather_dict(error_dct)
        return error

