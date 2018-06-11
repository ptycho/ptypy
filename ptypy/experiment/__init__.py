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
# Import instrument-specific modules
from cSAXS import cSAXS
from legacy.I13_ffp import I13ScanFFP
from legacy.I13_nfp import I13ScanNFP
from legacy.DLS import DlsScan
from legacy.I08 import I08Scan
from hdf5_loader import Hdf5Loader
from savu import Savu
from plugin import makeScanPlugin
from ID16Anfp import ID16AScan
from AMO_LCLS import AMOScan
from DiProI_FERMI import DiProIFERMIScan
from optiklabor import FliSpecScanMultexp
from UCL import UCLLaserScan
from nanomax import NanomaxStepscanMay2017, NanomaxStepscanNov2016, NanomaxFlyscanJune2017
from ALS_5321 import ALS5321Scan
from ptypy.core.data import MoonFlowerScan

