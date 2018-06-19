# -*- coding: utf-8 -*-
"""
Beamline-specific data preparation modules.

Currently available:
 * cSAXS
 * I13DLS, FFP and NFP
 * I08DLS, FFP and NFP
 * ID16A ESRF, NFP
 * AMO LCLS
 * DiProI FERMI
 * NanoMAX

This file is part of the PTYPY package.

    :copyright: Copyright 2014 by the PTYPY team, see AUTHORS.
    :license: GPLv2, see LICENSE for details.
"""
from .. import defaults_tree
from ..core.data import MoonFlowerScan, PtydScan, PtyScan, QuickScan
from ..simulations import SimScan

__all__ = ['MoonFlowerScan', 'PtydScan', 'PtyScan', 'QuickScan', 'SimScan']
PTYSCANS = {'MoonFlowerScan': MoonFlowerScan,
            'PtydScan': PtydScan,
            'PtyScan': PtyScan,
            'QuickScan': QuickScan,
            'SimScan': SimScan}


def register(name=None):
    """PtyScan subclass registration decorator"""
    return lambda cls: _register_PtyScan_class(cls, name)


def _register_PtyScan_class(cls, name=None):
    # Get class name
    name = cls.__name__ if name is None else name
    PTYSCANS[name] = cls

    # Apply descriptor decorator
    cls = defaults_tree.parse_doc('scandata.' + name, True)(cls)

    # Add class to namespace
    globals()[name] = cls
    __all__.append(name)
    return cls


# Import instrument-specific modules
try:
    from cSAXS import cSAXSScan
    from savu import Savu
    from plugin import makeScanPlugin
    from ID16Anfp import ID16AScan
    from AMO_LCLS import AMOScan
    from DiProI_FERMI import DiProIFERMIScan
    from optiklabor import FliSpecScanMultexp
    from UCL import UCLLaserScan
    from nanomax import NanomaxStepscanMay2017, NanomaxStepscanNov2016, NanomaxFlyscanJune2017
    from ALS_5321 import ALS5321Scan
except:
    pass
#from I13_ffp import I13ScanFFP
#from I13_nfp import I13ScanNFP
#from DLS import DlsScan
#from I08 import I08Scan

