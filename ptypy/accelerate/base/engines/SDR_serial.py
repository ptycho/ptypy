# -*- coding: utf-8 -*-
"""
Serialized semi-implicit relaxed Douglas-Rachford (SDR) reconstruction engine.

This file is part of the PTYPY package.

    :copyright: Copyright 2014 by the PTYPY team, see AUTHORS.
    :license: GPLv2, see LICENSE for details.
"""

from ptypy import utils as u
from ptypy.utils.verbose import logger, log
from ptypy import defaults_tree
from ptypy.engines import register
from .stochastic_serial import StochasticBaseEngineSerial
from ptypy.core.manager import Full, Vanilla, Bragg3dModel, BlockVanilla, BlockFull

@register()
class SDR_serial(StochasticBaseEngineSerial):
    """
    A serialized implementation of the stochastic Douglas-Rachford algorithm
    that is equivalent to the ePIE algorithm for alpha=0 and tau=1.

    Defaults:

    [name]
    default = SDR_serial
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
        Semi-implicit relaxed Douglas-Rachford (SDR) serialized reconstruction engine.
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