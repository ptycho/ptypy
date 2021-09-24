# -*- coding: utf-8 -*-
"""
ADMM reconstruction engine.

This file is part of the PTYPY package.

    :copyright: Copyright 2014 by the PTYPY team, see AUTHORS.
    :license: GPLv2, see LICENSE for details.
"""
from ptypy.engines import register, projectional

__all__ = ['ADMM']

class ADMMMixin:
    """
    Defaults:

    [beta]
    default = 0.75
    type = float
    lowlim = 0.0
    help = Beta parameter for ADMM algorithm
    """

    def __init__(self, beta):
        self._beta = 1.
        self._a = -1
        self._b = 1
        self._c = 1 + beta
        self.beta = beta

    @property
    def beta(self):
        return self._beta

    @beta.setter
    def beta(self, beta):
        self._beta = beta
        self._a = -1
        self._b = 1
        self._c = 1 + beta


@register()
class ADMM(projectional._ProjectionEngine, ADMMMixin):
    """
    A ADMM engine.

    Defaults:

    [name]
    default = ADMM
    type = str
    help =
    doc =

    """

    def __init__(self, ptycho_parent, pars=None):

        projectional._ProjectionEngine.__init__(self, ptycho_parent, pars)
        ADMMMixin.__init__(self, self.p.beta)
