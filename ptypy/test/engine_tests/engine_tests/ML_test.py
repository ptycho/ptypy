"""
Test for the ML engine.

This file is part of the PTYPY package.
    :copyright: Copyright 2014 by the PTYPY team, see AUTHORS.
    :license: GPLv2, see LICENSE for details.
"""

import unittest
from ptypy.test import test_utils as tu
from ptypy import utils as u

class MLTest(unittest.TestCase):
    def test_ML(self):
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
        tu.EngineTestRunner(engine_params)


if __name__ == "__main__":
    unittest.main()