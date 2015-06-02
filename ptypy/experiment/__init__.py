# -*- coding: utf-8 -*-
"""
Beamline-specific data preparation modules.

Currently available:
 * I13DLS

This file is part of the PTYPY package.

    :copyright: Copyright 2014 by the PTYPY team, see AUTHORS.
    :license: GPLv2, see LICENSE for details.
"""
#from data_structure import *

# Import nstrument-specific modules 
#import cSAXS
from I13 import I13Scan
from optiklabor import FliSpecScanMultexp
from plugin import makeScanPlugin

PtyScanTypes = dict(
    fli_spec_multexp=FliSpecScanMultexp,
    i13dls=I13Scan,
    plugin=makeScanPlugin
)
