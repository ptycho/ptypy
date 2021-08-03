# -*- coding: utf-8 -*-
"""
ePIE reconstruction engine.

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

__all__ = ['EPIE']

@register()
class EPIE(StochasticBaseEngine):
    """
    The ePIE algorithm.

    Defaults:

    [name]
    default = EPIE
    type = str
    help =
    doc =

    [probe_update_step]
    default = 1.0
    type = float
    lowlim = 0.0
    help = Step size in the probe update

    [object_update_step]
    default = 1.0
    type = float
    lowlim = 0.0
    help = Step size in the object update

    """

    SUPPORTED_MODELS = [Full, Vanilla, Bragg3dModel, BlockVanilla, BlockFull]

    def __init__(self, ptycho_parent, pars=None):
        """
        Stochastic Douglas-Rachford reconstruction engine.
        """
        super(EPIE, self).__init__(ptycho_parent, pars)

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

    def engine_initialize(self):
        """
        Prepare for reconstruction.
        """
        super(EPIE, self).engine_initialize()

    def fourier_update(self, view):
        """
        Fourier update for ePIE.

        """
        return basic_fourier_update(view, alpha=0.0, tau=1.0, 
                                    LL_error=self.p.compute_log_likelihood)

    def object_update(self, *args, **kwargs):
        """
        Object update for ePIE.

        .. math::
            O^{j+1} += \\beta * \\bar{P^{j}} * (\\Psi^{\prime} - \\Psi^{j}) / P_{norm}
            P_{norm} = ||P^{j}||^2 

        """
        self.generic_object_update(*args, **kwargs, alpha=0.0, beta=self.p.object_update_step)


    def probe_update(self, *args, **kwargs):
        """
        Probe update for ePIE.

        .. math::
            P^{j+1} += \\beta * \\bar{O^{j}} * (\\Psi^{\prime} - \\Psi^{j}) / O_{norm}
            O_{norm} = ||O^{j}||^2 

        """
        self.generic_probe_update(*args, **kwargs, alpha=0.0, beta=self.p.probe_update_step)