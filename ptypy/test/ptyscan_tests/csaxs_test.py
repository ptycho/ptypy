"""\
Test for the dls ptyscan, Diamond Light Source.

This file is part of the PTYPY package.
    :copyright: Copyright 2014 by the PTYPY team, see AUTHORS.
    :license: GPLv2, see LICENSE for details.
"""

import unittest
from ptypy.test import test_utils as tu
from ptypy.experiment.cSAXS_new import cSAXSScan
from ptypy import utils as u


class CsaxsTest(unittest.TestCase):
    @unittest.skip("this won't work unless we figure out how to treat the data")
    def test_omny(self):
        r = u.Param()
        r.base_path = '/dls/i14/data/2017/cm16755-2/processing/cSAXS_May2017/'
        r.visit = 'e16403'
        r.detector = 'pilatus_1'
        r.scan_number = 2047
        r.motors = ['Target_x','Target_y']
        r.motors_multiplier = (1e-6,1e-6)
        r.mask_path = None
        r.mask_file = 'binary_mask_6.2keV.mat'
        tu.PtyscanTestRunner(cSAXSScan,r,cleanup=False)

#     def test_flomny(self):
#         r = u.Param()
#         r.experimentID = None
#         r.instrument = 'flomny'
#         r.scan_number = 2046
#         r.detector_name = 'merlin_sw_hdf'
#         r.base_path = ''
#         r.mask_file = '%s/processing/mask2.hdf' % r.base_path
#         r.is_swmr = False
#         tu.PtyscanTestRunner(cSAXSScan,r)

if __name__ == "__main__":
    unittest.main()

