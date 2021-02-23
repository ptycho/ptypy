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
from importlib import import_module
from .. import defaults_tree
from ..core.data import MoonFlowerScan, PtydScan, PtyScan, QuickScan, IAScan
from ..simulations import SimScan
from ..utils.verbose import log

__all__ = ['MoonFlowerScan', 'PtydScan', 'PtyScan', 'QuickScan', 'SimScan']
PTYSCANS = {'MoonFlowerScan': MoonFlowerScan,
            'PtydScan': PtydScan,
            'PtyScan': PtyScan,
            'IAScan': IAScan,
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


ptyscan_modules = [('.hdf5_loader', 'Hdf5Loader'),
                   ('.cSAXS', 'cSAXSScan'),
                   ('.savu', 'Savu'),
                   ('.plugin', 'makeScanPlugin'),
                   ('.ID16Anfp', 'ID16AScan'),
                   ('.AMO_LCLS', 'AMOScan'),
                   ('.DiProI_FERMI', 'DiProIFERMIScan'),
                   ('.optiklabor', 'FliSpecScanMultexp'),
                   ('.UCL', 'UCLLaserScan'),
                   ('.nanomax', 'NanomaxStepscanNov2018'),
                   ('.nanomax', 'NanomaxFlyscanMay2019'),
                   ('.nanomax', 'NanomaxStepscanSep2019'),
                   ('.nanomax', 'NanomaxContrast'),
                   ('.nanomax_streaming', 'NanomaxZmqScan'),
                   ('.ALS_5321', 'ALS5321Scan'),
                   ('.Bragg3dSim', 'Bragg3dSimScan')]

for module, obj in ptyscan_modules:
    try:
        lib = import_module(module, 'ptypy.experiment')
    except ImportError as exception:
        log(2, 'Could not import experiment %s from %s, Reason: %s' % (obj, module, exception))
        pass
    else:
        globals()[obj] = lib.__dict__[obj]


