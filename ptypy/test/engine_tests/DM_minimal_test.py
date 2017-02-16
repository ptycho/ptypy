"""
Test for the DM_simple engine.

This file is part of the PTYPY package.
    :copyright: Copyright 2014 by the PTYPY team, see AUTHORS.
    :license: GPLv2, see LICENSE for details.
"""

import unittest
from ptypy.test import test_utils as tu 
from ptypy import utils as u

class DMMinimalTest(unittest.TestCase):
    @unittest.skip('gives error "ptycho_parent is not defined" - is this supposed to be used differently?')
    def test_dm_minimal(self):
        engine_params = u.Param()
        engine_params.name = 'DM_minimal'
        engine_params.numiter = 100
        engine_params.numiter_contiguous = 1
        engine_params.fourier_relax_factor = 0.01
        engine_params.alpha = 1.0    
        engine_params.probe_inertia = 0.001             
        engine_params.object_inertia = 0.1              
        tu.EngineTestRunner(engine_params)


if __name__ == "__main__":
    unittest.main()