"""
Test for the DM engine.

This file is part of the PTYPY package.
    :copyright: Copyright 2014 by the PTYPY team, see AUTHORS.
    :license: GPLv2, see LICENSE for details.
"""

import unittest
from ptypy.test import utils as tu
from ptypy import utils as u

class DMOPRTest(unittest.TestCase):
    def test_DMOPR(self):
        engine_params = u.Param()
        engine_params.name = 'DMOPR'
        engine_params.numiter = 5
        engine_params.numiter_contiguous = 5
        engine_params.overlap_converge_factor = 0.05
        engine_params.overlap_max_iterations = 2
        engine_params.fourier_relax_factor = 0.01
        engine_params.IP_metric = 1.
        engine_params.subspace_dim = 10
        tu.EngineTestRunner(engine_params)

if __name__ == "__main__":
    unittest.main()