"""
Testing the xy.py
"""


import unittest
import numpy as np
from ptypy.core.xy import from_pars, augment_to_coordlist,spiral_scan, round_scan, raster_scan

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

    def test_raster_scan_regression(self):
        NY = 5
        NX = 6
        DX = 2.0e-6
        DY = 1.0e-6
        expected_result = np.array([[0.0, 0.0],
                                    [0.0, 1.e-6],
                                    [0.0,2.e-6],
                                    [0.0, 3.e-6],
                                    [0.0, 4.e-6],
                                    [2.e-6, 0.0],
                                    [2.e-6, 1.e-6],
                                    [2.e-6, 2.e-6],
                                    [2.e-6, 3.e-6],
                                    [2.e-6, 4.e-6],
                                    [4.e-6, 0.0],
                                    [4.e-6, 1.e-6],
                                    [4.e-6, 2.e-6],
                                    [4.e-6, 3.e-6],
                                    [4.e-6, 4.e-6],
                                    [6.e-6, 0.0],
                                    [6.e-6, 1.e-6],
                                    [6.e-6, 2.e-6],
                                    [6.e-6, 3.e-6],
                                    [6.e-6, 4.e-6],
                                    [8.e-6, 0.0],
                                    [8.e-6, 1.e-6],
                                    [8.e-6, 2.e-6],
                                    [8.e-6, 3.e-6],
                                    [8.e-6, 4.e-6],
                                    [1.e-05, 0.0],
                                    [1.e-05, 1.e-6],
                                    [1.e-05, 2.e-6],
                                    [1.e-05, 3.e-6],
                                    [1.e-05, 4.e-6]], dtype=np.float64)
        a = raster_scan(dy=DY, dx=DX, ny=NY, nx=NX)
        np.testing.assert_equal(a.shape, (NX*NY, 2))
        np.testing.assert_allclose(a, expected_result)


    def test_round_scan(self):
        a = round_scan()


if __name__ == '__main__':
    unittest.main()