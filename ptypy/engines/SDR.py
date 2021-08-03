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
from .stochastic import StochasticBaseEngine
from ..core.manager import Full, Vanilla, Bragg3dModel, BlockVanilla, BlockFull

__all__ = ['SDR']

@register()
class SDR(StochasticBaseEngine):
    """
    The stochastic Douglas-Rachford algorithm.

    Defaults:

    [name]
    default = SDR
    type = str
    help =
    doc =

    [sigma]
    default = 1
    type = float
    lowlim = 0.0
    help = Relaxed Fourier reflaction parameter.

    [tau]
    default = 1
    type = float
    lowlim = 0.0
    help = Relaxed modulus constraint parameter.

    [probe_update_step]
    default = 0.1
    type = float
    lowlim = 0.0
    help = Step size in the probe update

    [object_update_step]
    default = 0.9
    type = float
    lowlim = 0.0
    help = Step size in the object update

    """

    SUPPORTED_MODELS = [Full, Vanilla, Bragg3dModel, BlockVanilla, BlockFull]

    def __init__(self, ptycho_parent, pars=None):
        """
        Stochastic Douglas-Rachford reconstruction engine.
        """
        super(SDR, self).__init__(ptycho_parent, pars)

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

    def engine_initialize(self):
        """
        Prepare for reconstruction.
        """
        super(SDR, self).engine_initialize()

    def fourier_update(self, view):
        """
        Fourier update for Stochastic Douglas-Rachford (SDR).


        """
        return basic_fourier_update(view, alpha=self.p.sigma, tau=self.p.tau, 
                                    LL_error=self.p.compute_log_likelihood)

    def object_update(self, *args, **kwargs):
        """
        Object update for Stochastic Douglas-Rachford (SDR).

        .. math::
            O^{j+1} += \\alpha * \\bar{P^{j}} * (\\Psi^{\prime} - \\Psi^{j}) / P_{norm}
            P_{norm} = (1 - \\alpha) * ||P^{j}||^2 + \\alpha * |P^{j}|^2

        """
        self.generic_object_update(*args, **kwargs, alpha=self.p.object_update_step, beta=0.0)


    def probe_update(self, *args, **kwargs):
        """
        Probe update for Stochastic Douglas-Rachford (SDR).

        .. math::
            P^{j+1} += \\alpha * \\bar{O^{j}} * (\\Psi^{\prime} - \\Psi^{j}) / O_{norm}
            O_{norm} = (1 - \\alpha) * ||O^{j}||^2 + \\alpha * |O^{j}|^2

        """
        self.generic_probe_update(*args, **kwargs, alpha=self.p.probe_update_step, beta=0.0)