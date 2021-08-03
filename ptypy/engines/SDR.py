# -*- coding: utf-8 -*-
"""
Semi-implicit relaxed Douglas-Rachfrod (SDR) reconstruction engine.

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

    def __init__(self, ptycho_parent, pars=None):
        """
        Semi-implicit relaxed Douglas-Rachford (SDR) reconstruction engine.
        """
        super().__init__(ptycho_parent, pars)

        self.ptycho.citations.add_article(
            title='Semi-implicit relaxed Douglas-Rachford algorithm (sDR) for ptychography',
            author='Pham et al.',
            journal='Opt. Express',
            volume=27,
            year=2019,
            page=31246,
            doi='10.1364/OE.27.031246',
            comment='The semi-implicit relaxed Douglas-Rachford reconstruction algorithm',
        )

    def fourier_update(self, view):
        """
        Fourier update for Stochastic Douglas-Rachford (SDR).


        """
        return basic_fourier_update(view, alpha=self.p.sigma, tau=self.p.tau, 
                                    LL_error=self.p.compute_log_likelihood)

    def object_update(self, *args, **kwargs):
        """
        Object update for SDR.

        .. math::
            O^{j+1} += \\alpha * \\bar{P^{j}} * (\\Psi^{\prime} - \\Psi^{j}) / P_{norm}
            P_{norm} = (1 - \\alpha) * ||P^{j}||_{max}^2 + \\alpha * |P^{j}|^2

        """
        self.generic_object_update(*args, **kwargs, A=self.p.beta_object, B=0.0)


    def probe_update(self, *args, **kwargs):
        """
        Probe update for SDR. 

        .. math::
            P^{j+1} += \\alpha * \\bar{O^{j}} * (\\Psi^{\prime} - \\Psi^{j}) / O_{norm}
            O_{norm} = (1 - \\alpha) * ||O^{j}||_{max}^2 + \\alpha * |O^{j}|^2

        """
        self.generic_probe_update(*args, **kwargs, A=self.p.beta_probe, B=0.0)