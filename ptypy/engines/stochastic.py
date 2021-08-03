# -*- coding: utf-8 -*-
"""
Stochastic reconstruction engine.

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

class StochasticBaseEngine(PositionCorrectionEngine):
    """
    The base implementation of a stochastic algorithm for ptychography

    Defaults:

    [clip_object]
    default = None
    type = tuple
    help = Clip object amplitude into this interval

    [compute_log_likelihood]
    default = True
    type = bool
    help = A switch for computing the log-likelihood error (this can impact the performance of the engine)

    """

    def __init__(self, ptycho_parent, pars=None):
        """
        Stochastic Douglas-Rachford reconstruction engine.
        """
        super(StochasticBaseEngine, self).__init__(ptycho_parent, pars)

        # Instance attributes
        self.error = None
        self.mean_power = None

    def engine_initialize(self):
        """
        Prepare for reconstruction.
        """
        super(StochasticBaseEngine, self).engine_initialize()

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
                error_dct[name] = self.fourier_update(view)
                
                # A copy of the old exit wave
                exit_wave = {}
                for name, pod in view.pods.items():
                    exit_wave[name] = pod.object * pod.probe

                # Object update
                self.object_update(view, exit_wave)

                # Probe update
                self.probe_update(view, exit_wave)


            self.curiter += 1

        return error_dct

    def generic_object_update(self, view, exit_wave, alpha=0., beta=1.):
        """
        A generic object update for stochastic algorithms.
        alpha = 0, beta = b is the ePIE update with step parameter b.
        alpha = a, beta = 0 is the SDR update with step parameter a.

        .. math::
            O^{j+1} += (\\alpha + \\beta) * \\bar{P^{j}} * (\\Psi^{\prime} - \\Psi^{j}) / P_{norm}
            P_{norm} = (1 - \\alpha) * ||P^{j}||^2 + \\alpha * |P^{j}|^2

        """
        probe_power = 0
        for name, pod in view.pods.items():
            probe_power += u.abs2(pod.probe)
        probe_norm = (1 - alpha) * np.max(probe_power) + alpha * probe_power
        for name, pod in view.pods.items():
            pod.object += (alpha + beta) * np.conj(pod.probe) * (pod.exit - exit_wave[name]) / probe_norm

    def generic_probe_update(self, view, exit_wave, alpha=0., beta=1.):
        """
        A generic probe update for stochastic algorithms.
        alpha = 0, beta = b is the ePIE update with step parameter b.
        alpha = a, beta = 0 is the SDR update with step parameter a.

        .. math::
            P^{j+1} += (\\alpha + \\beta) * \\bar{O^{j}} * (\\Psi^{\prime} - \\Psi^{j}) / O_{norm}
            O_{norm} = (1 - \\alpha) * ||O^{j}||^2 + \\alpha * |O^{j}|^2

        """
        object_power = 0
        for name, pod in view.pods.items():
            object_power += u.abs2(pod.object)
        object_norm = (1 - alpha) * np.max(object_power) + self.p.probe_update_step * object_power
        for name, pod in view.pods.items():
            pod.probe += (alpha + beta) * np.conj(pod.object) * (pod.exit - exit_wave[name]) / object_norm
