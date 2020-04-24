"""
Tests that the assembly of frames into 3d pods gives the original
3d diffraction patterns.
"""

import unittest
from ptypy.core import Ptycho
from ptypy import utils as u
import numpy as np

import os
import tempfile
import shutil

class Bragg3dModelTest(unittest.TestCase):
    def setUp(self):
        self.outpath = tempfile.mkdtemp(suffix="Bragg3dModelTest")

    def tearDown(self):
        shutil.rmtree(self.outpath)

    def test_frame_assembly(self):
        from ptypy.experiment.Bragg3dSim import Bragg3dSimScan
        # parameter tree
        p = u.Param()
        p.scans = u.Param()
        p.scans.scan01 = u.Param()
        p.scans.scan01.name = 'Bragg3dModel'
        p.scans.scan01.data = u.Param()
        p.scans.scan01.data.name = 'Bragg3dSimScan'
        p.scans.scan01.data.dump = os.path.join(self.outpath, 'tmp.npz')

        p.scans.scan01.data.shuffle = True

        # simulate and then load data
        P = Ptycho(p, level=2)

        # load raw simulation data
        diff = np.load(p.scans.scan01.data.dump)

        # check that the pods reflect the raw data
        assert len(diff.keys()) == len(P.pods)
        checked_pods = []
        for i in range(len(diff.keys())):
            diff_raw = diff['diff%02d'%i]
            ok = False
            for j in range(len(P.pods)):
                if j in checked_pods:
                    continue
                diff_pod = P.pods['P%04d'%j].diff
                if np.allclose(diff_raw, diff_pod):
                    checked_pods.append(j)
                    ok = True
                    break
            assert ok

if __name__ == '__main__':
    unittest.main()
