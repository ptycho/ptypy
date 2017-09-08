"""
Test for the Dummy engine.

This file is part of the PTYPY package.
    :copyright: Copyright 2014 by the PTYPY team, see AUTHORS.
    :license: GPLv2, see LICENSE for details.
"""

import unittest
from ptypy.test import test_utils as tu 
from ptypy import utils as u

class DummyTest(unittest.TestCase):
    def test_dummy(self):
        engine_params = u.Param()
        engine_params.name = 'Dummy'
        engine_params.itertime = 2.0
        engine_params.numiter = 5
        tu.EngineTestRunner(engine_params)

if __name__ == "__main__":
    unittest.main()