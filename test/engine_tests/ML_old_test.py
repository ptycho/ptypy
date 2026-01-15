"""
Test for the ML_new engine.

This file is part of the PTYPY package.
    :copyright: Copyright 2014 by the PTYPY team, see AUTHORS.
    :license: see LICENSE for details.
"""

import unittest
from test import utils as tu
from ptypy import utils as u

class MLNewTest(unittest.TestCase):
    @unittest.skip('skip this because it is not supported')
    def test_ML_old_farfield(self):
        engine_params = u.Param()
        engine_params.name = 'ML_new'
        engine_params.ML_type = 'gaussian'
        engine_params.floating_intensities = False
        engine_params.intensity_renormalization = 1.
        engine_params.reg_del2 = False
        engine_params.reg_del2_amplitude = .01
        engine_params.smooth_gradient = 0
        engine_params.scale_precond = False
        engine_params.scale_probe_object = 1.
        tu.EngineTestRunner(engine_params)

    @unittest.skip('skip this because it is not supported')
    def test_ML_nearfield(self):
        engine_params = u.Param()
        engine_params.name = 'ML_new'
        engine_params.ML_type = 'gaussian'
        engine_params.floating_intensities = False
        engine_params.intensity_renormalization = 1.
        engine_params.reg_del2 = False
        engine_params.reg_del2_amplitude = .01
        engine_params.smooth_gradient = 0
        engine_params.scale_precond = False
        engine_params.scale_probe_object = 1.

        tu.EngineTestRunner(engine_params,propagator='nearfield')

if __name__ == "__main__":
    unittest.main()
