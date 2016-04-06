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
from AMO_LCLS import AMOScan
from I13 import I13Scan
from optiklabor import FliSpecScanMultexp
from plugin import makeScanPlugin
from I08 import I08Scan
from UCL import UCLLaserScan

PtyScanTypes = dict(
    amo_lcls=AMOScan,
    fli_spec_multexp=FliSpecScanMultexp,
    i13dls=I13Scan,
    plugin=makeScanPlugin,
    i08dls=I08Scan,
    laser_ucl=UCLLaserScan
)
