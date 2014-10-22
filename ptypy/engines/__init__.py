# -*- coding: utf-8 -*-
"""
Engines module.

Implements the difference map (DM) and maximum likelihood (ML) reconstruction
algorithms for ptychography.

This file is part of the PTYPY package.

    :copyright: Copyright 2014 by the PTYPY team, see AUTHORS.
    :license: GPLv2, see LICENSE for details.
"""
import os

from base import BaseEngine
from DM import DM
from DM_simple import DM_simple
from ML import ML
from dummy import Dummy
#from ePIE import ePIE

engine_names = ['Dummy','DM_simple','DM', 'ML']

def by_name(name):
    if name not in engine_names: raise RuntimeError('Unknown engine: %s' % name)
    return globals()[name]
