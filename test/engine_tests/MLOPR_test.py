"""
Test for the DM_simple engine.

This file is part of the PTYPY package.
    :copyright: Copyright 2014 by the PTYPY team, see AUTHORS.
    :license: see LICENSE for details.
"""

import unittest
from test import utils as tu
from ptypy import utils as u
from ptypy.core import Ptycho
import tempfile
import shutil

from ptypy.custom import MLOPR

class MLOPRTest(unittest.TestCase):
    def setUp(self):
        self.outpath = tempfile.mkdtemp(suffix="MLOPR_test")

    def tearDown(self):
        shutil.rmtree(self.outpath)

    def test_MLOPR(self):

        p = u.Param()
        p.verbose_level = 3
        p.io = u.Param()
        p.io.interaction = u.Param()
        p.io.interaction.active = False
        p.io.home = self.outpath
        p.io.rfile = "MLOPRTest.ptyr"
        p.io.autosave = u.Param(active=False)
        p.io.autoplot = u.Param(active=False)
        p.ipython_kernel = False
        p.scans = u.Param()
        p.scans.MF = u.Param()
        p.scans.MF.name = 'OPRModel'
        p.scans.MF.propagation = 'farfield'
        p.scans.MF.data = u.Param()
        p.scans.MF.data.name = 'MoonFlowerScan'
        p.scans.MF.data.positions_theory = None
        p.scans.MF.data.auto_center = None
        p.scans.MF.data.min_frames = 1
        p.scans.MF.data.orientation = None
        p.scans.MF.data.num_frames = 100
        p.scans.MF.data.energy = 6.2
        p.scans.MF.data.shape = 64
        p.scans.MF.data.chunk_format = '.chunk%02d'
        p.scans.MF.data.rebin = None
        p.scans.MF.data.experimentID = None
        p.scans.MF.data.label = None
        p.scans.MF.data.version = 0.1
        p.scans.MF.data.dfile = "MLOPRTest.ptyd"
        p.scans.MF.data.psize = 0.000172
        p.scans.MF.data.load_parallel = None
        p.scans.MF.data.distance = 7.0
        p.scans.MF.data.save = None
        p.scans.MF.data.center = 'fftshift'
        p.scans.MF.data.photons = 100000000.0
        p.scans.MF.data.psf = 0.0
        p.scans.MF.data.density = 0.2
        p.engines = u.Param()

        p.engines.engine00 = u.Param()
        p.engines.engine00.name = "MLOPR"
        P = Ptycho(p, level=5)


if __name__ == "__main__":
    unittest.main()
