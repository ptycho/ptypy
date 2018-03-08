# -*- coding: utf-8 -*-
"""
Difference Map reconstruction engine that uses numpy arrays instead of iteration.

This file is part of the PTYPY package.

    :copyright: Copyright 2014 by the PTYPY team, see AUTHORS.
    :license: GPLv2, see LICENSE for details.
"""

import time
from ..utils.verbose import logger, log
from ..utils import parallel
from DM import DM
from ..utils.descriptor import defaults_tree
from ..core.manager import Full, Vanilla
from ..array_based import data_utils as du
from ..array_based import constraints as con
from ..array_based import object_probe_interaction as opi
import numpy as np
__all__ = ['DMNpy']


@defaults_tree.parse_doc('engine.DMNpy')
class DMNpy(DM):
    """
    A full-fledged Difference Map engine that uses numpy arrays instead of iteration.


    Defaults:

    [name]
    default = DMNpy
    type = str
    help =
    doc =

    [alpha]
    default = 1
    type = float
    lowlim = 0.0
    help = Difference map parameter

    [probe_update_start]
    default = 2
    type = int
    lowlim = 0
    help = Number of iterations before probe update starts

    [subpix_start]
    default = 0
    type = int
    lowlim = 0
    help = Number of iterations before starting subpixel interpolation

    [subpix]
    default = 'linear'
    type = str
    help = Subpixel interpolation; 'fourier','linear' or None for no interpolation

    [update_object_first]
    default = True
    type = bool
    help = If True update object before probe

    [overlap_converge_factor]
    default = 0.05
    type = float
    lowlim = 0.0
    help = Threshold for interruption of the inner overlap loop
    doc = The inner overlap loop refines the probe and the object simultaneously. This loop is escaped as soon as the overall change in probe, relative to the first iteration, is less than this value.

    [overlap_max_iterations]
    default = 10
    type = int
    lowlim = 1
    help = Maximum of iterations for the overlap constraint inner loop

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

    [fourier_relax_factor]
    default = 0.05
    type = float
    lowlim = 0.0
    help = If rms error of model vs diffraction data is smaller than this fraction, Fourier constraint is met
    doc = Set this value higher for noisy data.

    [obj_smooth_std]
    default = None
    type = int
    lowlim = 0
    help = Gaussian smoothing (pixel) of the current object prior to update
    doc = If None, smoothing is deactivated. This smoothing can be used to reduce the amplitude of spurious pixels in the outer, least constrained areas of the object.

    [clip_object]
    default = None
    type = tuple
    help = Clip object amplitude into this interval

    [probe_center_tol]
    default = None
    type = float
    lowlim = 0.0
    help = Pixel radius around optical axes that the probe mass center must reside in

    """

    SUPPORTED_MODELS = [Vanilla, Full]

    def __init__(self, ptycho_parent, pars=None):
        """
        Difference map reconstruction engine.
        """
        super(DMNpy, self).__init__(ptycho_parent, pars)

    def engine_prepare(self):
        """
        Last minute initialization.

        Everything that needs to be recalculated when new data arrives.
        """
        super(DMNpy, self).engine_prepare()
        # and then something to convert the arrays to numpy
        self.vectorised_scan = {}
        self.propagator = {}
        for dID, _diffs in self.di.S.iteritems():
            self.vectorised_scan[dID] = du.pod_to_arrays(self, dID)
            first_view_id = self.vectorised_scan[dID]['meta']['view_IDs'][0]
            self.propagator[dID] = self.di.V[first_view_id].pod.geometry.propagator

    def engine_iterate(self, num=1):
        """
        Compute `num` iterations.
        """
        to = 0.
        tf = 0.
        # run the data for `num` iterations on the cards, then pull the relevant off to sync
        # error_dct = {}
        # k=0
        # for idx, name in self.di.views.iteritems():
        #     error_dct[idx] = np.zeros((3,))
        #     k+=1



        for dID, _diffs in self.di.S.iteritems():
            for it in range(num):
                t1 = time.time()

                # Fourier update
                # error_dct = self.fourier_update()
                # if self.thingamejog < 1:
                error_dct = self.numpy_fourier_update(self.vectorised_scan[dID]['mask'],
                                                      self.vectorised_scan[dID]['diffraction'],
                                                      self.vectorised_scan[dID]['obj'],
                                                      self.vectorised_scan[dID]['probe'],
                                                      self.vectorised_scan[dID]['exit wave'],
                                                      self.vectorised_scan[dID]['meta']['addr'],
                                                      self.propagator[dID],
                                                      pbound=self.pbound[dID])

                t2 = time.time()
                tf += t2 - t1

                self.numpy_overlap_update(self.vectorised_scan[dID]['obj'],
                                          self.vectorised_scan[dID]['object weights'],
                                          self.vectorised_scan[dID]['object viewcover'],
                                          self.vectorised_scan[dID]['probe'],
                                          self.vectorised_scan[dID]['probe weights'],
                                          self.vectorised_scan[dID]['probe support'],
                                          self.vectorised_scan[dID]['exit wave'],
                                          self.mean_power,
                                          self.vectorised_scan[dID]['meta']['addr'][:, 0])

                t3 = time.time()
                to += t3 - t2

                # count up
                self.curiter += 1

        # self.mpi_numpy_overlap_update()
        logger.info('Time spent in Fourier update: %.2f' % tf)
        logger.info('Time spent in Overlap update: %.2f' % to)
        error = parallel.gather_dict(error_dct)
        return error

    def numpy_fourier_update(self, mask, Idata, obj, probe, exit_wave, addr, propagator, pbound):
        error_dct = {}

        errors = con.difference_map_fourier_constraint(mask,
                                                       Idata,
                                                       obj,
                                                       probe,
                                                       exit_wave,
                                                       addr,
                                                       prefilter=propagator.pre_fft,
                                                       postfilter=propagator.post_fft,
                                                       pbound=pbound,
                                                       alpha=self.p.alpha,
                                                       LL_error=False)

        k = 0
        for idx, name in self.di.views.iteritems():
            error_dct[idx] = errors[:, k]
            k += 1

        return error_dct

    def numpy_overlap_update(self, ob, object_weights, ob_viewcover ,probe, probe_weights, probe_support, exit_wave, mean_power, addr_info):
        """
        DM overlap constraint update.
        """
        # Condition to update probe
        do_update_probe = (self.p.probe_update_start <= self.curiter)

        for inner in range(self.p.overlap_max_iterations):
            pre_str = 'Iteration (Overlap) #%02d:  ' % inner

            # Update object first
            if self.p.update_object_first or (inner > 0):
                # Update object
                log(4, pre_str + '----- object update -----')
                opi.difference_map_update_object(ob,
                                                 object_weights,
                                                 probe,
                                                 exit_wave,
                                                 ob_viewcover,
                                                 addr_info,
                                                 mean_power,
                                                 ob_inertia=self.p.object_inertia,
                                                 ob_smooth_std=self.p.obj_smooth_std,
                                                 clip_object=self.p.clip_object)

            # Exit if probe should not be updated yet
            if not do_update_probe:
                break

            # Update probe
            log(4, pre_str + '----- probe update -----')
            # change = self.probe_update()
            change = opi.difference_map_update_probe(ob,
                                                     probe_weights,
                                                     probe,
                                                     exit_wave,
                                                     addr_info,
                                                     self.p.probe_inertia,
                                                     probe_support)

            log(4, pre_str + 'change in probe is %.3f' % change)

            # Recenter the probe
            self.center_probe()

            # Stop iteration if probe change is small
            if change < self.p.overlap_converge_factor:
                break

