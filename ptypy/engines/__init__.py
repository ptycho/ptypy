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
from base import BaseEngine, DEFAULT_iter_info
from base import DEFAULT as COMMON
import DM
import DM_simple
import ML
import dummy
#from ePIE import ePIE

__all__ = ['DM', 'ML', 'BaseEngine']

engine_names = ['Dummy', 'DM_simple', 'DM', 'ML', 'ML_new']

DEFAULTS = u.Param(
    common = COMMON,
    Dummy = dummy.DEFAULT,
    DM_simple = DM_simple.DEFAULT,
    DM = DM.DEFAULT,
    ML = ML.DEFAULT,
    DM_minimal = DM_minimal.DEFAULT,
)

ENGINES = u.Param(
    Dummy = dummy.Dummy,
    DM_simple = DM_simple.DM_simple,
    DM = DM.DM,
    ML = ML.ML,
    DM_minimal = DM_minimal.DM_minimal,
)
def by_name(name):
    if name not in ENGINES.keys():
        raise RuntimeError('Unknown engine: %s' % name)
    return ENGINES[name]
