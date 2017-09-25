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
#from data_structure import *

# Import instrument-specific modules
#import cSAXS
from I13_ffp import I13ScanFFP
from I13_nfp import I13ScanNFP
from cSAXS_new import cSAXSScan
from DLS import DlsScan
from I08 import I08Scan
from savu import Savu
from plugin import makeScanPlugin
from ID16Anfp import ID16AScan
from AMO_LCLS import AMOScan
from DiProI_FERMI import DiProIFERMIScan
from optiklabor import FliSpecScanMultexp
from UCL import UCLLaserScan
from nanomax import NanomaxStepscanNov2016, NanomaxStepscanMay2017, NanomaxFlyscanJune2017
from ALS_5321 import ALS5321Scan

PtyScanTypes = dict(
    i13dls_ffp = I13ScanFFP,
    i13dls_nfp = I13ScanNFP,
    dls = DlsScan,
    i08dls = I08Scan,
    savu = Savu,
    plugin = makeScanPlugin,
    id16a_nfp = ID16AScan,
    amo_lcls = AMOScan,
    diproi_fermi = DiProIFERMIScan,
    fli_spec_multexp = FliSpecScanMultexp,
    laser_ucl = UCLLaserScan,
    nanomaxstepscannov2016 = NanomaxStepscanNov2016,
    nanomaxstepscanmay2017 = NanomaxStepscanMay2017,
    nanomaxflyscanjune2017 = NanomaxFlyscanJune2017,
    als5321 = ALS5321Scan,
    csaxs = cSAXSScan
)

