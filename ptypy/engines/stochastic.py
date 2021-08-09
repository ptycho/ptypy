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
        if parallel.MPIenabled:
            raise NotImplementedError("The stochastic engines are not compatible with MPI")

        # Adjustment parameters for fourier update
        self.alpha = 0.0
        self.tau = 1.0

        # Adjustment parameters for probe update
        self.prA = 0.0
        self.prB = 1.0

        # Adjustment parameters for object update
        self.obA = 0.0
        self.obB = 1.0

    def engine_prepare(self):
        """
        Last minute initialization.
        Everything that needs to be recalculated when new data arrives.
        """
        pass

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

                # Position update
                self.position_update_local(view)

            self.curiter += 1

        return error_dct

    def position_update_local(self, view):
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
            refines position of current view by a given algorithm. 
            """
            self.position_refinement.update_constraints(self.curiter) # this stays here

            # Check for new coordinates
            if view.active:
                self.position_refinement.update_view_position(view)

    def fourier_update(self, view):
        """
        Engine-specific implementation of Fourier update

        Parameters
        ----------
        view : View
        View to diffraction data
        """
        return basic_fourier_update(view, alpha=self.alpha, tau=self.tau, 
                                    LL_error=self.p.compute_log_likelihood)

    def object_update(self, view, exit_wave):
        """
        Engine-specific implementation of object update

        Parameters
        ----------
        view : View
        View to diffraction data

        exit_wave: dict
        Collection of exit waves associated with the current view
        """
        self._generic_object_update(view, exit_wave, A=self.obA, B=self.obB)

    def probe_update(self, view, exit_wave):
        """
        Engine-specific implementation of probe update

        Parameters
        ----------
        view : View
        View to diffraction data

        exit_wave: dict
        Collection of exit waves associated with the current view
        """
        self._generic_probe_update(view, exit_wave, A=self.prA, B=self.prB)

    def _generic_object_update(self, view, exit_wave, A=0., B=1.):
        """
        A generic object update for stochastic algorithms.

        Parameters
        ----------
        view : View
        View to diffraction data

        exit_wave: dict
        Collection of exit waves associated with the current view

        A : float
        Generic parameter for adjusting step size of object update

        B : float
        Generic parameter for adjusting step size of object update

        A = 0, B = \\alpha is the ePIE update with parameter \\alpha.
        A = \\beta_O, B = 0 is the SDR update with parameter \\beta_O.

        .. math::
            O^{j+1} += (A + B) * \\bar{P^{j}} * (\\Psi^{\\prime} - \\Psi^{j}) / P_{norm}
            P_{norm} = (1 - A) * ||P^{j}||_{max}^2 + A * |P^{j}|^2

        """
        probe_power = 0
        for name, pod in view.pods.items():
            probe_power += u.abs2(pod.probe)
        probe_norm = (1 - A) * np.max(probe_power) + A * probe_power
        for name, pod in view.pods.items():
            pod.object += (A + B) * np.conj(pod.probe) * (pod.exit - exit_wave[name]) / probe_norm

    def _generic_probe_update(self, view, exit_wave, A=0., B=1.):
        """
        A generic probe update for stochastic algorithms.

        Parameters
        ----------
        view : View
        View to diffraction data

        exit_wave: dict
        Collection of exit waves associated with the current view

        A : float
        Generic parameter for adjusting step size of probe update

        B : float
        Generic parameter for adjusting step size of probe update

        A = 0, B = \\beta is the ePIE update with parameter \\beta.
        A = \\beta_P, B = 0 is the SDR update with parameter \\beta_P.

        .. math::
            P^{j+1} += (A + B) * \\bar{O^{j}} * (\\Psi^{\\prime} - \\Psi^{j}) / O_{norm}
            O_{norm} = (1 - A) * ||O^{j}||_{max}^2 + A * |O^{j}|^2

        """
        object_power = 0
        for name, pod in view.pods.items():
            object_power += u.abs2(pod.object)
        object_norm = (1 - A) * np.max(object_power) + A * object_power
        for name, pod in view.pods.items():
            pod.probe += (A + B) * np.conj(pod.object) * (pod.exit - exit_wave[name]) / object_norm
