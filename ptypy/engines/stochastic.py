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
        super().__init__(ptycho_parent, pars)
        if parallel.MPIenabled:
            raise NotImplementedError("The stochastic engines are not compatible with MPI")

        # Adjustment parameters for fourier update
        self._alpha = 0.0
        self._tau = 1.0

        # Adjustment parameters for probe update
        self._pr_a = 0.0
        self._pr_b = 1.0

        # Adjustment parameters for object update
        self._ob_a = 0.0
        self._ob_b = 1.0

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

                # Position update
                self.position_update_local(view)

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
        return basic_fourier_update(view, alpha=self._alpha, tau=self._tau, 
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
        self._generic_object_update(view, exit_wave, a=self._ob_a, b=self._ob_b)

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
        self._generic_probe_update(view, exit_wave, a=self._pr_a, b=self._pr_b)

    def _generic_object_update(self, view, exit_wave, a=0., b=1.):
        """
        A generic object update for stochastic algorithms.

        Parameters
        ----------
        view : View
        View to diffraction data

        exit_wave: dict
        Collection of exit waves associated with the current view

        a : float
        Generic parameter for adjusting step size of object update

        b : float
        Generic parameter for adjusting step size of object update

        a = 0, b = \\alpha is the ePIE update with parameter \\alpha.
        a = \\beta_O, b = 0 is the SDR update with parameter \\beta_O.

        .. math::
            O^{j+1} += (a + b) * \\bar{P^{j}} * (\\Psi^{\\prime} - \\Psi^{j}) / P_{norm}
            P_{norm} = (1 - a) * ||P^{j}||_{max}^2 + a * |P^{j}|^2

        """
        probe_power = 0
        for name, pod in view.pods.items():
            probe_power += u.abs2(pod.probe)
        probe_norm = (1 - a) * np.max(probe_power) + a * probe_power
        for name, pod in view.pods.items():
            pod.object += (a + b) * np.conj(pod.probe) * (pod.exit - exit_wave[name]) / probe_norm

    def _generic_probe_update(self, view, exit_wave, a=0., b=1.):
        """
        A generic probe update for stochastic algorithms.

        Parameters
        ----------
        view : View
        View to diffraction data

        exit_wave: dict
        Collection of exit waves associated with the current view

        a : float
        Generic parameter for adjusting step size of probe update

        b : float
        Generic parameter for adjusting step size of probe update

        a = 0, b = \\beta is the ePIE update with parameter \\beta.
        a = \\beta_P, b = 0 is the SDR update with parameter \\beta_P.

        .. math::
            P^{j+1} += (a + b) * \\bar{O^{j}} * (\\Psi^{\\prime} - \\Psi^{j}) / O_{norm}
            O_{norm} = (1 - a) * ||O^{j}||_{max}^2 + a * |O^{j}|^2

        """
        object_power = 0
        for name, pod in view.pods.items():
            object_power += u.abs2(pod.object)
        object_norm = (1 - a) * np.max(object_power) + a * object_power
        for name, pod in view.pods.items():
            pod.probe += (a + b) * np.conj(pod.object) * (pod.exit - exit_wave[name]) / object_norm
