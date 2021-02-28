"""
Test for the DM_simple engine.

This file is part of the PTYPY package.
    :copyright: Copyright 2014 by the PTYPY team, see AUTHORS.
    :license: GPLv2, see LICENSE for details.
"""

import unittest
from test import utils as tu
from ptypy import utils as u



class DMSimpleTest(unittest.TestCase):
    @unittest.skip('Skipping because of a NotImplementedError in engine_prepare')
    def test_DM_simple(self):
        engine_params = u.Param()
        engine_params.name = 'DM_simple'
        engine_params.alpha = 1.0
        tu.EngineTestRunner(engine_params)
if __name__ == "__main__":
    unittest.main()
