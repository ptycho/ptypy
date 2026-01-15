# -*- coding: utf-8 -*-
"""
Stochastic reconstruction engine.

This file is part of the PTYPY package.

    :copyright: Copyright 2014 by the PTYPY team, see AUTHORS.
    :license: see LICENSE for details.
"""
import numpy as np
import time
from .. import utils as u
from ..utils.verbose import logger, log
from ..utils import parallel
from .utils import projection_update_generalized, log_likelihood
from .base import PositionCorrectionEngine
from . import register
from ..core.manager import Full, Vanilla, Bragg3dModel, BlockVanilla, BlockFull, GradFull, BlockGradFull

__all__ = ['EPIE', 'SDR']

class _StochasticEngine(PositionCorrectionEngine):
    """
    The base implementation of a stochastic algorithm for ptychography

    Defaults:

    [probe_update_start]
    default = 0
    type = int
    lowlim = 0
    help = Number of iterations before probe update starts

    [probe_center_tol]
    default = None
    type = float
    lowlim = 0.0
    help = Pixel radius around optical axes that the probe mass center must reside in

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
        self._a = 0
        self._b = 1
        self._c = 1

        # Adjustment parameters for probe update
        self._pr_a = 0.0
        self._pr_b = 1.0

        # Adjustment parameters for object update
        self._ob_a = 0.0
        self._ob_b = 1.0

        # By default the object norm is based on the local object
        self._object_norm_is_global = False

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

            # Recenter the probe
            self.center_probe()

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
        General implementation of Fourier update

        Parameters
        ----------
        view : View
        View to diffraction data
        """
        #return basic_fourier_update(view, alpha=self._alpha, tau=self._tau,
        #                            LL_error=self.p.compute_log_likelihood)

        err_fmag, err_exit = projection_update_generalized(view, self._a, self._b, self._c)
        if self.p.compute_log_likelihood:
            err_phot = log_likelihood(view)
        else:
            err_phot = 0.
        return np.array([err_fmag, err_phot, err_exit])

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
        if self.p.probe_update_start > self.curiter:
            return
        self._generic_probe_update(view, exit_wave, a=self._pr_a, b=self._pr_b)

    def center_probe(self):
        if self.p.probe_center_tol is not None:
            for name, pr_s in self.pr.storages.items():
                c1 = u.mass_center(u.abs2(pr_s.data).sum(0))
                c2 = np.asarray(pr_s.shape[-2:]) // 2
                # fft convention should however use geometry instead
                if u.norm(c1 - c2) < self.p.probe_center_tol:
                    break
                # SC: possible BUG here, wrong input parameter
                pr_s.data[:] = u.shift_zoom(pr_s.data, (1.,)*3,
                        (0, c1[0], c1[1]), (0, c2[0], c2[1]))

                # shift the object
                ob_s = pr_s.views[0].pod.ob_view.storage
                ob_s.data[:] = u.shift_zoom(ob_s.data, (1.,)*3,
                        (0, c1[0], c1[1]), (0, c2[0], c2[1]))

                # shift the exit waves, loop through different exit wave views
                for pv in pr_s.views:
                    pv.pod.exit = u.shift_zoom(pv.pod.exit, (1.,)*2,
                            (c1[0], c1[1]), (c2[0], c2[1]))

                log(4,'Probe recentered from %s to %s'
                            % (str(tuple(c1)), str(tuple(c2))))

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
        # Calculate the object norm based on the global object
        # This only works if a = 0.
        if self._object_norm_is_global and a == 0:
            object_norm = np.max(u.abs2(view.pod.ob_view.storage.data).sum(axis=0))
        # Calculate the object norm based on the local object
        else:
            object_power = 0
            for name, pod in view.pods.items():
                object_power += u.abs2(pod.object)
            object_norm = (1 - a) * np.max(object_power) + a * object_power
        for name, pod in view.pods.items():
            pod.probe += (a + b) * np.conj(pod.object) * (pod.exit - exit_wave[name]) / object_norm

class EPIEMixin:
    """
    Defaults:

    [alpha]
    default = 1.0
    type = float
    lowlim = 0.0
    help = Parameter for adjusting the step size of the object update

    [beta]
    default = 1.0
    type = float
    lowlim = 0.0
    help = Parameter for adjusting the step size of the probe update

    [object_norm_is_global]
    default = False
    type = bool
    help = Calculate the object norm based on the global object instead of the local object

    """
    SUPPORTED_MODELS = [Full, Vanilla, Bragg3dModel, BlockVanilla, BlockFull, GradFull, BlockGradFull]

    def __init__(self, alpha, beta):
        # EPIE adjustment parameters
        self._a = 0
        self._b = 1
        self._c = 1
        self._pr_a = 0.0
        self._ob_a = 0.0
        self._pr_b = alpha
        self._ob_b = beta
        self._object_norm_is_global = self.p.object_norm_is_global
        self.article = dict(
            title='An improved ptychographical phase retrieval algorithm for diffractive imaging',
            author='Maiden A. and Rodenburg J.',
            journal='Ultramicroscopy',
            volume=10,
            year=2009,
            page=1256,
            doi='10.1016/j.ultramic.2009.05.012',
            comment='The ePIE reconstruction algorithm',
        )

    @property
    def alpha(self):
        return self._pr_a

    @alpha.setter
    def alpha(self, alpha):
        self._pr_b = alpha

    @property
    def beta(self):
        return self._ob_b

    @beta.setter
    def beta(self, beta):
        self._ob_b = beta


class SDRMixin:
    """
    Defaults:

    [sigma]
    default = 1
    type = float
    lowlim = 0.0
    help = Relaxed Fourier reflection parameter.

    [tau]
    default = 1
    type = float
    lowlim = 0.0
    help = Relaxed modulus constraint parameter.

    [beta_probe]
    default = 0.1
    type = float
    lowlim = 0.0
    help = Parameter for adjusting the step size of the probe update

    [beta_object]
    default = 0.9
    type = float
    lowlim = 0.0
    help = Parameter for adjusting the step size of the object update

    """
    SUPPORTED_MODELS = [Full, Vanilla, Bragg3dModel, BlockVanilla, BlockFull]

    def __init__(self, sigma, tau, beta_probe, beta_object):
        # SDR Adjustment parameters
        self._sigma = sigma
        self._tau = tau
        self._update_abc()
        self._pr_a = beta_probe
        self._ob_a = beta_object
        self._pr_b = 0.0
        self._ob_b = 0.0

        self.article = dict(
            title='Semi-implicit relaxed Douglas-Rachford algorithm (sDR) for ptychography',
            author='Pham et al.',
            journal='Opt. Express',
            volume=27,
            year=2019,
            page=31246,
            doi='10.1364/OE.27.031246',
            comment='The semi-implicit relaxed Douglas-Rachford reconstruction algorithm',
        )

    def _update_abc(self):
        self._a = 1 - self._tau * (1 + self._sigma)
        self._b = self._tau
        self._c = 1 + self._sigma

    @property
    def sigma(self):
        return self._sigma

    @sigma.setter
    def sigma(self, sigma):
        self._sigma = sigma
        self._update_abc()

    @property
    def tau(self):
        return self._tau

    @tau.setter
    def tau(self, tau):
        self._tau = tau
        self._update_abc()

    @property
    def beta_probe(self):
        return self._pr_a

    @beta_probe.setter
    def beta_probe(self, beta):
        self._pr_a = beta

    @property
    def beta_object(self):
        return self._ob_a

    @beta_object.setter
    def beta_object(self, beta):
        self._ob_a = beta

@register()
class EPIE(_StochasticEngine, EPIEMixin):
    """
    The ePIE algorithm.

    Defaults:

    [name]
    default = EPIE
    type = str
    help =
    doc =

    """
    def __init__(self, ptycho_parent, pars=None):
        _StochasticEngine.__init__(self, ptycho_parent, pars)
        EPIEMixin.__init__(self, self.p.alpha, self.p.beta)
        ptycho_parent.citations.add_article(**self.article)


@register()
class SDR(_StochasticEngine, SDRMixin):
    """
    The stochastic Douglas-Rachford algorithm.

    Defaults:

    [name]
    default = SDR
    type = str
    help =
    doc =

    """
    def __init__(self, ptycho_parent, pars=None):
        _StochasticEngine.__init__(self, ptycho_parent, pars)
        SDRMixin.__init__(self, self.p.sigma, self.p.tau, self.p.beta_probe, self.p.beta_object)
        ptycho_parent.citations.add_article(**self.article)
