# -*- coding: utf-8 -*-
"""
Difference Map reconstruction engine that uses numpy arrays instead of iteration.

This file is part of the PTYPY package.

    :copyright: Copyright 2014 by the PTYPY team, see AUTHORS.
    :license: GPLv2, see LICENSE for details.
"""

import time

from ..utils import parallel
from .DM_npy import DMNpy
from ptypy import defaults_tree
from ..core.manager import Full, Vanilla
from archive.cuda_extension.accelerate.cuda import difference_map_iterator
from . import register
import numpy as np

__all__ = ['DMGpu']


@register()
class DMGpu(DMNpy):
    """
    A full-fledged Difference Map engine that uses numpy arrays instead of iteration.


    Defaults:

    [name]
    default = DMGpu
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
    choices = ['fourier','linear',None]

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
        super(DMGpu, self).__init__(ptycho_parent, pars)

    def engine_iterate(self, num=1):
        """
        Compute `num` iterations.
        """

        for dID, _diffs in self.di.S.items():

            cfact_probe = (self.p.probe_inertia * len(self.vectorised_scan[dID]['meta']['addr']) /
                           self.vectorised_scan[dID]['probe'].shape[0]) * np.ones_like(
                self.vectorised_scan[dID]['probe'])


            cfact_object = self.p.object_inertia * self.mean_power * (self.vectorised_scan[dID]['object viewcover'] + 1.)


            pre_fft = self.propagator[dID].pre_fft
            post_fft = self.propagator[dID].post_fft
            psupp = self.probe_support[self.vectorised_scan[dID]['meta']['poe_IDs'][0]].astype(np.complex64)
            #print("{}, {}".format(psupp.shape, psupp.dtype))

            errors =difference_map_iterator(diffraction=self.vectorised_scan[dID]['diffraction'],
                                        obj=self.vectorised_scan[dID]['obj'],
                                        object_weights=self.vectorised_scan[dID]['object weights'].astype(np.float32),
                                        cfact_object=cfact_object,
                                        mask=self.vectorised_scan[dID]['mask'],
                                        probe=self.vectorised_scan[dID]['probe'],
                                        cfact_probe=cfact_probe,
                                        probe_support=psupp, #self.probe_support[self.vectorised_scan[dID]['meta']['poe_IDs'][0]],
                                        probe_weights=self.vectorised_scan[dID]['probe weights'].astype(np.float32),
                                        exit_wave=self.vectorised_scan[dID]['exit wave'],
                                        addr=self.vectorised_scan[dID]['meta']['addr'],
                                        pre_fft=pre_fft,
                                        post_fft=post_fft,
                                        pbound=self.pbound[dID],
                                        overlap_max_iterations=self.p.overlap_max_iterations,
                                        update_object_first=self.p.update_object_first,
                                        obj_smooth_std=self.p.obj_smooth_std,
                                        overlap_converge_factor=self.p.overlap_converge_factor,
                                        probe_center_tol=self.p.probe_center_tol,
                                        probe_update_start=0,
                                        alpha=self.p.alpha,
                                        clip_object=self.p.clip_object,
                                        LL_error=True,
                                        num_iterations=num)


            #yuk yuk yuk
            error_dct = {}
            print(errors.shape)
            jx =0
            for jx in range(num):
                k = 0
                for idx, name in self.di.views.items():
                    error_dct[idx] = errors[jx, :, k]
                    k += 1
                jx +=1
                error = parallel.gather_dict(error_dct)

            # count up
            self.curiter += num

        return error

    