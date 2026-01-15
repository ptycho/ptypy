"""
This script creates a sample *.ptyd data file using the built-in
test Scan `ptypy.core.data.MoonFlowerScan`
"""

from ptypy import utils as u
from ptypy import io
from ptypy.core.data import MoonFlowerScan
from .. import utils as tu
import unittest
global DATA
DATA = u.Param(
    shape = 128,
    num_frames = 50,
    label=None,
    psize=1.0,
    energy=1.0,
    distance = 1.0,
    center='fftshift',
    auto_center = None ,
    rebin = None,
    orientation = None)

class PtyscanTest(unittest.TestCase):
    def test_moonflower(self):
        '''
        just check it runs
        '''
        tu.PtyscanTestRunner(MoonFlowerScan,data_params=DATA)

    def test_moonflower_with_three_calls(self):
        '''
        check it runs with multiple calls to auto
        '''
        tu.PtyscanTestRunner(MoonFlowerScan, data_params=DATA, auto_frames=30, ncalls=3)

    def test_moonflower_with_three_calls_REGRESSION(self):
        '''
        Same as above, but makes sure the output is sensible
        '''
        out = tu.PtyscanTestRunner(MoonFlowerScan, data_params=DATA, auto_frames=30, ncalls=3)

        self.assertEqual(30,len(out['msgs'][0]['iterable']),
            "Scan did not prepare 30 frames as expected")

        self.assertEqual(20,len(out['msgs'][1]['iterable']),
            "Scan did not prepare 20 frames as expected")

        from ptypy.core.data import EOS
        self.assertEqual(out['msgs'][2], EOS,
            "Last auto call not identified as End of Scan (data.EOS)")

    def test_appended_ptyd(self):
        '''
        technically all of these tests do this, but this is explicit
        '''
        tu.PtyscanTestRunner(MoonFlowerScan,data_params=DATA, save_type='append')

    def test_appended_ptyd_REGRESSION(self):
        '''
        check that we can actually read the result!
        '''
        out = tu.PtyscanTestRunner(MoonFlowerScan,data_params=DATA, save_type='append', cleanup=False)
        d = io.h5read(out['output_file'])

    def test_linked_ptyd(self):
        '''
        test the linking mechanism works
        '''
        tu.PtyscanTestRunner(MoonFlowerScan,data_params=DATA, save_type='link', cleanup=False)

    def test_linked_ptyd_REGRESSION(self):
        '''
        again, can we read it?
        '''
        out = tu.PtyscanTestRunner(MoonFlowerScan,data_params=DATA, save_type='link', cleanup=False)
        d = io.h5read(out['output_file'])



if __name__ == '__main__':
    unittest.main()
