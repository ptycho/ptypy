# -*- coding: utf-8 -*-
"""
Engines module.

Implements the difference map (DM) and maximum likelihood (ML) reconstruction
algorithms for ptychography.

This file is part of the PTYPY package.

    :copyright: Copyright 2014 by the PTYPY team, see AUTHORS.
    :license: GPLv2, see LICENSE for details.
"""

from .. import utils as u
from .. import defaults_tree
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

from base import BaseEngine

# These imports should be executable separately
from . import DM
from . import DM_gpu
from . import DM_npy
from . import DM_simple
from . import ML
from . import dummy
from . import ePIE

# dynamic load, maybe discarded in future
dynamic_load('./', ['BaseEngine'] + ENGINES.keys(), True)
dynamic_load('~/.ptypy/', ['BaseEngine'] + ENGINES.keys(), True)

    


