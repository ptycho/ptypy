# -*- coding: utf-8 -*-
"""
Beamline-specific data preparation modules.

Currently available:
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
# Import instrument-specific modules
#import cSAXS

# from I13_ffp import I13ScanFFP
# from I13_nfp import I13ScanNFP
# from DLS import DlsScan
from I08 import I08Scan
from savu import Savu
from plugin import makeScanPlugin
from ID16Anfp import ID16AScan
from AMO_LCLS import AMOScan
from DiProI_FERMI import DiProIFERMIScan
from optiklabor import FliSpecScanMultexp
from UCL import UCLLaserScan
from nanomax import NanomaxStepscanMay2017, NanomaxStepscanNov2016, NanomaxFlyscanJune2017
from ALS_5321 import ALS5321Scan


if __name__ == "__main__":
    from ptypy.utils.verbose import logger
    from ptypy.core.data import PtydScan, MoonFlowerScan, PtyScan
else:
    from ..utils.verbose import logger
    from ..core.data import PtydScan, MoonFlowerScan, PtyScan

def all_subclasses(cls, names=False):
    """
    Helper function for finding all subclasses of a base class.
    """
    subs = cls.__subclasses__() + [g for s in cls.__subclasses__()
                                    for g in all_subclasses(s)]
    if names:
        return [c.__name__ for c in subs]
    else:
        return subs

def makePtyScan(pars, scanmodel=None):
    """
    Factory for PtyScan object. Return an instance of the appropriate PtyScan subclass based on the
    input parameters.

    Parameters
    ----------
    pars: dict or Param
        Input parameters according to :py:data:`.scan.data`.

    scanmodel: ScanModel object
        FIXME: This seems to be needed for simulations but broken for now.
    """

    # Extract information on the type of object to build
    name = pars.name

    if name in all_subclasses(PtyScan, names=True):
        ps_class = eval(name)
        logger.info('Scan will be prepared with the PtyScan subclass "%s"' % name)
        ps_instance = ps_class(pars)
    else:
        raise RuntimeError('Could not manage source "%s"' % str(name))

    return ps_instance
