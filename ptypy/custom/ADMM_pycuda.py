# -*- coding: utf-8 -*-
"""
ADMM reconstruction engine.

This file is part of the PTYPY package.

    :copyright: Copyright 2014 by the PTYPY team, see AUTHORS.
    :license: GPLv2, see LICENSE for details.
"""
from ptypy.engines import register
from ptypy.accelerate.cuda_pycuda.engines import projectional_pycuda
from . import ADMM

@register()
class ADMM_pycuda(projectional_pycuda._ProjectionEngine_pycuda, ADMM.ADMMMixin):
    """
    An ADMM engine accelerated with pycuda.

    Defaults:

    [name]
    default = ADMM_pycuda
    type = str
    help =
    doc =

    """

    def __init__(self, ptycho_parent, pars=None):

        projectional_pycuda._ProjectionEngine_pycuda.__init__(self, ptycho_parent, pars)
        ADMM.ADMMMixin.__init__(self, self.p.beta)

