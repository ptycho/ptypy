"""\
Test for the I08 beamline ptyscan, Diamond Light Source.

This file is part of the PTYPY package.
    :copyright: Copyright 2014 by the PTYPY team, see AUTHORS.
    :license: see LICENSE for details.
"""

import unittest
from .. import utils as tu
from ptypy.experiment.legacy.I08 import I08Scan
from ptypy import utils as u


class I08Test(unittest.TestCase):
    @unittest.skip("this won't work unless we figure out how to treat the data")
    def test_I08(self):
        r = u.Param()
        r.base_path = tu.get_test_data_path('i08')
        r.scan_number = '2535'
        r.scan_number_stxm = '010'
        r.dark_number = '2536'
        r.dark_number_stxm = '011'
        r.detector_flat_file = "%s/processing/flat2_xflipped.h5" % r.base_path
        r.date = '2016-03-04'
        tu.PtyscanTestRunner(I08Scan,r)

if __name__ == "__main__":
    unittest.main()
