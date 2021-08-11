# -*- coding: utf-8 -*-
"""
Accelerated SDR reconstruction engine.

This file is part of the PTYPY package.

    :copyright: Copyright 2014 by the PTYPY team, see AUTHORS.
    :license: GPLv2, see LICENSE for details.
"""

from ptypy import utils as u
from ptypy.utils.verbose import logger, log
from ptypy import defaults_tree
from ptypy.engines import register
from .stochastic import StochasticBaseEnginePycuda
from ptypy.core.manager import Full, Vanilla, Bragg3dModel, BlockVanilla, BlockFull

@register()
class SDR_pycuda(StochasticBaseEnginePycuda):
    """
    An accelerated implementation of the semi-implicit relaxed Douglas-Rachford (SDR) algorithm.

    Defaults:

    [name]
    default = SDR_pycuda
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
        ePIE serialized reconstruction engine.
        """
        super().__init__(ptycho_parent, pars)

        # SDR Adjustment parameters
        self._alpha = self.p.sigma 
        self._tau = self.p.tau
        self.beta_probe = self.p.beta_probe
        self.beta_object = self.p.beta_object
        self._pr_b = 0.0
        self._ob_b = 0.0

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