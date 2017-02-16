"""
Testing the xy.py
"""


import unittest
from ptypy.core.xy import from_pars, augment_to_coordlist,spiral_scan,spiral_scan_legacy, round_scan, round_scan_legacy, round_scan_roi_legacy, spiral_scan_roi_legacy,\
    raster_scan, raster_scan_legacy


class XyTest(unittest.TestCase):
    def test_from_pars(self):
        a = from_pars()

    @unittest.skip('this is neat. Figure out how to use it.')
    def test_augment_to_coord_list(self):
        pass

    def test_spiral_scan(self):
        a = spiral_scan()

    def test_raster_scan(self):
        a = raster_scan()

    def test_round_scan(self):
        a = round_scan()

    def test_raster_scan_legacy(self):
        a = raster_scan_legacy()

    def test_round_scan_legacy(self):
        a = round_scan_legacy()

    def test_round_roi_scan_legacy(self):
        a = round_scan_roi_legacy()

    def test_spiral_scan_legacy(self):
        a = spiral_scan_legacy()

if __name__ == '__main__':
    unittest.main()