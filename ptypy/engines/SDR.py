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

        # SDR Adjustment parameters
        self.alpha = self.p.sigma # TODO: replace with generic fourier update params
        self.tau = self.p.tau # TODO: replace with generic fourier update params
        self.prA = self.p.beta_probe
        self.prB = 0.0
        self.obA = self.p.beta_object
        self.obB = 0.0

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