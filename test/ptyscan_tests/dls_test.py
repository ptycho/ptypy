"""\
Test for the dls ptyscan, Diamond Light Source.

This file is part of the PTYPY package.
    :copyright: Copyright 2014 by the PTYPY team, see AUTHORS.
    :license: see LICENSE for details.
"""

import unittest
from .. import utils as tu
from ptypy.experiment.legacy.DLS import DlsScan
from ptypy import utils as u


class DlsTest(unittest.TestCase):
    @unittest.skip("this won't work unless we figure out how to treat the data")
    def test_dls(self):
        r = u.Param()
        r.experimentID = None
        r.scan_number = 68862
        r.energy = None
        r.z = None
        r.detector_name = 'merlin_sw_hdf'
        r.base_path = tu.get_test_data_path('dls')
        r.mask_file = '%s/processing/mask2.hdf' % r.base_path
        r.is_swmr = False
        tu.PtyscanTestRunner(DlsScan,r)

if __name__ == "__main__":
    unittest.main()
