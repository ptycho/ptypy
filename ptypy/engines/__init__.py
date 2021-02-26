# -*- coding: utf-8 -*-
"""
Engines module.

Implements the difference map (DM) and maximum likelihood (ML) reconstruction
algorithms for ptychography.

This file is part of the PTYPY package.

    :copyright: Copyright 2014 by the PTYPY team, see AUTHORS.
    :license: GPLv2, see LICENSE for details.
"""

from .utils import *



ENGINES = dict()


def register(name=None):
    """Engine registration decorator"""
    return lambda cls: _register_engine_class(cls, name)


def _register_engine_class(cls, name=None):
    from ptypy import defaults_tree
    name = cls.__name__ if name is None else name
    ENGINES[name] = cls
    cls = defaults_tree.parse_doc('engine.' + name, True)(cls)
    return cls


def by_name(name):
    if name not in ENGINES.keys():
        raise RuntimeError('Unknown engine: %s' % name)
    return ENGINES[name]

from .base import BaseEngine, DEFAULT_iter_info

# These imports should be executable separately
from . import DM
from . import DM_simple
from . import DMOPR
from . import ML
from . import MLOPR
from . import dummy
from . import ePIE
from . import Bragg3d_engines

# TODO: make this better / explicit
try:
    from ptypy.accelerate.cuda_pycuda.engines import DM_pycuda
    from ptypy.accelerate.cuda_pycuda.engines import DM_pycuda_streams
    from ptypy.accelerate.cuda_pycuda.engines import ML_pycuda
    from ptypy.accelerate.cuda_pycuda.engines import DM_pycuda_stream
except:
    pass
try:
    from ptypy.accelerate.ocl_pyopencl.engines import DM_ocl
except:
    pass


# dynamic load, maybe discarded in future
dynamic_load('./', ['BaseEngine', 'PositionCorrectionEngine'] + list(ENGINES.keys()), True)
dynamic_load('~/.ptypy/', ['BaseEngine', 'PositionCorrectionEngine'] + list(ENGINES.keys()), True)
