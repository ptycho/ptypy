# -*- coding: utf-8 -*-
"""
Stochastic Douglas-Rachfrod (SDR) reconstruction engine.

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
from .base import PositionCorrectionEngine
from ..core.manager import Full, Vanilla, Bragg3dModel, BlockVanilla, BlockFull

__all__ = ['SDR']

@register()
class SDR(PositionCorrectionEngine):
    """
    An implementation of the stochastic Douglas-Rachford algorithm
    that is equivalent to the ePIE algorithm for alpha=0 and tau=1.

    Defaults:

    [name]
    default = SDR
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

    [compute_log_likelihood]
    default = True
    type = bool
    help = A switch for computing the log-likelihood error (this can impact the performance of the engine)

    """

    SUPPORTED_MODELS = [Full, Vanilla, Bragg3dModel, BlockVanilla, BlockFull]

    def __init__(self, ptycho_parent, pars=None):
        """
        Stochastic Douglas-Rachford reconstruction engine.
        """
        super(SDR, self).__init__(ptycho_parent, pars)

        # Instance attributes
        self.error = None
        self.mean_power = None

        self.ptycho.citations.add_article(
            title='Semi-implicit relaxed Douglas-Rachford algorithm (sDR) for ptychography',
            author='Pham et al.',
            journal='Opt. Express',
            volume=27,
            year=2019,
            page=31246,
            doi='10.1364/OE.27.031246',
            comment='The stochastic douglas-rachford reconstruction algorithm',
        )
        self.ptycho.citations.add_article(
            title='An improved ptychographical phase retrieval algorithm for diffractive imaging',
            author='Maiden A. and Rodenburg J.',
            journal='Ultramicroscopy',
            volume=10,
            year=2009,
            page=1256,
            doi='10.1016/j.ultramic.2009.05.012',
            comment='The ePIE reconstruction algorithm',
        )


    def engine_initialize(self):
        """
        Prepare for reconstruction.
        """
        super(SDR, self).engine_initialize()

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

    def engine_iterate(self, num=1):
        """
        Compute one iteration.
        """
        vieworder = list(self.di.views.keys())
        vieworder.sort()
        rng = np.random.default_rng()

        for it in range(num):   

            error_dct = {}
            rng.shuffle(vieworder)

            for name in vieworder:
                view = self.di.views[name]
                if not view.active:
                    continue

                # Fourier update
                error_dct[name] = basic_fourier_update(view,  
                                                       alpha=self.p.alpha, tau=self.p.tau, 
                                                       LL_error=self.p.compute_log_likelihood)
                
                # A copy of the old exit wave
                exit_ = {}
                for name, pod in view.pods.items():
                    exit_[name] = pod.object * pod.probe

                # Object update
                probe_power = 0
                for name, pod in view.pods.items():
                    probe_power += u.abs2(pod.probe)
                probe_norm = np.max(probe_power)
                for name, pod in view.pods.items():
                    pod.object += np.conj(pod.probe) * (pod.exit - exit_[name]) / probe_norm

                # Probe update
                object_power = 0
                for name, pod in view.pods.items():
                    object_power += u.abs2(pod.object)
                object_norm = np.max(object_power)
                for name, pod in view.pods.items():
                    pod.probe += np.conj(pod.object) * (pod.exit - exit_[name]) / object_norm

            self.curiter += 1

        return error_dct