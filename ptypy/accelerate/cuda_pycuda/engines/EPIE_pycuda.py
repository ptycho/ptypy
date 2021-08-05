# -*- coding: utf-8 -*-
"""
Accelerated EPIE reconstruction engine.

This file is part of the PTYPY package.

    :copyright: Copyright 2014 by the PTYPY team, see AUTHORS.
    :license: GPLv2, see LICENSE for details.
"""

from ptypy import utils as u
from ptypy.utils.verbose import logger, log
from ptypy import defaults_tree
from ptypy.engines import register
from .stochastic_pycuda import StochasticBaseEnginePycuda
from ptypy.core.manager import Full, Vanilla, Bragg3dModel, BlockVanilla, BlockFull

@register()
class EPIE_pycuda(StochasticBaseEnginePycuda):
    """
    An accelerated implementation of the EPIE algorithm.

    Defaults:

    [name]
    default = EPIE_pycuda
    type = str
    help =
    doc =

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

    """

    SUPPORTED_MODELS = [Full, Vanilla, Bragg3dModel, BlockVanilla, BlockFull]

    def __init__(self, ptycho_parent, pars=None):
        """
        ePIE serialized reconstruction engine.
        """
        super().__init__(ptycho_parent, pars)

        # EPIE Adjustment parameters
        self.alpha = 0.0 # TODO: replace with generic fourier update params
        self.tau = 1.0 # TODO: replace with generic fourier update params
        self.prA = 0.0
        self.prB = self.p.alpha
        self.obA = 0.0
        self.obB = self.p.beta

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