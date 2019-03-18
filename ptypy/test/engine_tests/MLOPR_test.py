"""
Test for the DM_simple engine.

This file is part of the PTYPY package.
    :copyright: Copyright 2014 by the PTYPY team, see AUTHORS.
    :license: GPLv2, see LICENSE for details.
"""

import unittest
from ptypy.test import utils as tu
from ptypy import utils as u

class MLOPRTest(unittest.TestCase):

    def test_MLOPR(self):
        engine_params = u.Param()
        engine_params.name = 'MLOPR'
        engine_params.numiter = 10
        engine_params.numiter = 2
        engine_params.reg_del2 =True
        engine_params.reg_del2_amplitude = 0.01
        engine_params.smooth_gradient = 0.0
        engine_params.scale_precond = True
        engine_params.subspace_dim = 10
        tu.EngineTestRunner(engine_params)

if __name__ == "__main__":
    unittest.main()