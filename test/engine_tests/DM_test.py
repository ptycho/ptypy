"""
Test for the DM engine.

This file is part of the PTYPY package.
    :copyright: Copyright 2014 by the PTYPY team, see AUTHORS.
    :license: GPLv2, see LICENSE for details.
"""

import unittest
from test import utils as tu
from ptypy import utils as u

class DMTest(unittest.TestCase):

    def test_DM_position_refinement(self):
        engine_params = u.Param()
        engine_params.name = 'DM'
        engine_params.numiter = 5
        engine_params.alpha =1
        engine_params.probe_update_start = 2
        engine_params.overlap_converge_factor = 0.05
        engine_params.overlap_max_iterations = 10
        engine_params.probe_inertia = 1e-3
        engine_params.object_inertia = 0.1
        engine_params.fourier_relax_factor = 0.01
        engine_params.obj_smooth_std = 20
        engine_params.position_refinement = True
        tu.EngineTestRunner(engine_params)

    def test_DM(self):
        engine_params = u.Param()
        engine_params.name = 'DM'
        engine_params.numiter = 5
        engine_params.alpha =1
        engine_params.probe_update_start = 2
        engine_params.overlap_converge_factor = 0.05
        engine_params.overlap_max_iterations = 10
        engine_params.probe_inertia = 1e-3
        engine_params.object_inertia = 0.1
        engine_params.fourier_relax_factor = 0.01
        engine_params.obj_smooth_std = 20
        tu.EngineTestRunner(engine_params)

if __name__ == "__main__":
    unittest.main()
