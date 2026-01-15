"""
Test for the ML engine.

This file is part of the PTYPY package.
    :copyright: Copyright 2014 by the PTYPY team, see AUTHORS.
    :license: see LICENSE for details.
"""

import unittest
from test import utils as tu
from ptypy import utils as u
import tempfile
import shutil

class MLTest(unittest.TestCase):

    def setUp(self):
        self.outpath = tempfile.mkdtemp(suffix="DMOPR_test")

    def tearDown(self):
        shutil.rmtree(self.outpath)

    def test_ML_farfield_floating_intensities(self):
        engine_params = u.Param()
        engine_params.name = 'ML'
        engine_params.numiter = 5
        engine_params.probe_update_start = 2
        engine_params.floating_intensities = True
        engine_params.intensity_renormalization = 1.0
        engine_params.reg_del2 =True
        engine_params.reg_del2_amplitude = 0.01
        engine_params.smooth_gradient = 0.0
        engine_params.scale_precond =False
        engine_params.probe_update_start = 0
        tu.EngineTestRunner(engine_params, output_path=self.outpath)

    def test_ML_farfield(self):
        engine_params = u.Param()
        engine_params.name = 'ML'
        engine_params.numiter = 5
        engine_params.probe_update_start = 2
        engine_params.floating_intensities = False
        engine_params.intensity_renormalization = 1.0
        engine_params.reg_del2 =True
        engine_params.reg_del2_amplitude = 0.01
        engine_params.smooth_gradient = 0.0
        engine_params.scale_precond =False
        engine_params.probe_update_start = 0
        tu.EngineTestRunner(engine_params, output_path=self.outpath)


    def test_ML_nearfield(self):
        engine_params = u.Param()
        engine_params.name = 'ML'
        engine_params.numiter = 5
        engine_params.probe_update_start = 2
        engine_params.floating_intensities = False
        engine_params.intensity_renormalization = 1.0
        engine_params.reg_del2 =True
        engine_params.reg_del2_amplitude = 0.01
        engine_params.smooth_gradient = 0.0
        engine_params.scale_precond =False
        engine_params.probe_update_start = 0

        tu.EngineTestRunner(engine_params, propagator='nearfield', output_path=self.outpath)

if __name__ == "__main__":
    unittest.main()
