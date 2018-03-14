'''
Tests for the array_utils module
'''


import unittest
from ptypy.array_based import array_utils as au
from ptypy.array_based import FLOAT_TYPE, COMPLEX_TYPE
from ptypy.gpu import array_utils as gau
from ptypy.gpu import FLOAT_TYPE as GPU_FLOAT_TYPE
from copy import deepcopy
from ptypy.gpu import COMPLEX_TYPE as GPU_COMPLEX_TYPE
import numpy as np


class ArrayUtilsTest(unittest.TestCase):

    @unittest.skip("This method is not implemented yet")
    def test_abs2_real_input_UNITY(self):
        x = np.ones((3,3))
        np.testing.assert_array_equal(au.abs2(x), gau.abs2(x))

    @unittest.skip("This method is not implemented yet")
    def test_abs2_complex_input_UNITY(self):
        x = np.ones((3,3)) + 1j*np.ones((3,3))
        np.testing.assert_array_equal(au.abs2(x), gau.abs2(x))

    @unittest.skip("This method is not implemented yet")
    def test_sum_to_buffer_UNITY(self):

        in1 = np.array([[1.0, 1.0, 1.0, 1.0],
                        [2.0, 2.0, 2.0, 2.0],
                        [3.0, 3.0, 3.0, 3.0],
                        [4.0, 4.0, 4.0, 4.0]])

        outshape = (2, 4)

        in1_addr = np.array([(0, 0, 0),
                            (1, 0, 0),
                            (2, 0, 0),
                            (3, 0, 0)])

        out1_addr = np.array([(0, 0, 0),
                              (1, 0, 0),
                              (0, 0, 0),
                              (1, 0, 0)])
        np.testing.assert_array_equal(au.sum_to_buffer(in1, outshape, in1_addr, out1_addr, dtype=FLOAT_TYPE),
                                      gau.sum_to_buffer(in1, outshape, in1_addr, out1_addr, dtype=GPU_FLOAT_TYPE))

    @unittest.skip("This method is not implemented yet")
    def test_sum_to_buffer_complex_UNITY(self):

        I = 4
        X = 2
        M = 4
        N = 4

        in1 = np.empty((I, M, N), dtype=COMPLEX_TYPE)

        # fill the input array
        for idx in range(I):
            in1[idx] = np.ones((M, N))* (idx + 1.0) + 1j * np.ones((M, N))* (idx + 1.0)

        outshape = (X, M, N)
        expected_out = np.empty(outshape, dtype=COMPLEX_TYPE)

        expected_out[0] = np.ones((M, N)) * 4.0 + 1j * np.ones((M, N))* 4.0
        expected_out[1] = np.ones((M, N)) * 6.0+ 1j * np.ones((M, N))* 6.0

        in1_addr = np.empty((I, 3))

        in1_addr = np.array([(0, 0, 0),
                            (1, 0, 0),
                            (2, 0, 0),
                            (3, 0, 0)])

        out1_addr = np.empty_like(in1_addr)
        out1_addr = np.array([(0, 0, 0),
                              (1, 0, 0),
                              (0, 0, 0),
                              (1, 0, 0)])

        out = au.sum_to_buffer(in1, outshape, in1_addr, out1_addr, dtype=COMPLEX_TYPE)
        outg = gau.sum_to_buffer(in1, outshape, in1_addr, out1_addr, dtype=COMPLEX_TYPE)
        np.testing.assert_array_equal(out, outg)

    @unittest.skip("This method is not implemented yet")
    def test_norm2_1d_real_UNITY(self):
        a = np.array([1.0, 2.0], dtype=FLOAT_TYPE)
        out = au.norm2(a)
        outg =gau.norm2(a)
        np.testing.assert_array_equal(out, outg)

    @unittest.skip("This method is not implemented yet")
    def test_norm2_1d_complex_UNITY(self):
        a = np.array([1.0+1.0j, 2.0+2.0j], dtype=COMPLEX_TYPE)
        out = au.norm2(a)
        outg = gau.norm2(a)
        np.testing.assert_array_equal(out, outg)

    @unittest.skip("This method is not implemented yet")
    def test_norm2_2d_real_UNITY(self):
        a = np.array([[1.0, 2.0],
                      [3.0, 4.0]], dtype=FLOAT_TYPE)
        out = au.norm2(a)
        outg = gau.norm2(a)
        np.testing.assert_array_equal(out, outg)

    @unittest.skip("This method is not implemented yet")
    def test_norm2_2d_complex_UNITY(self):
        a = np.array([[1.0+1.0j, 2.0+2.0j],
                      [3.0+3.0j, 4.0+4.0j]], dtype=COMPLEX_TYPE)
        out = au.norm2(a)
        outg = gau.norm2(a)
        np.testing.assert_array_equal(out, outg)

    @unittest.skip("This method is not implemented yet")
    def test_norm2_3d_real_UNITY(self):
        a = np.array([[[1.0, 2.0],
                      [3.0, 4.0]],
                      [[5.0, 6.0],
                       [7.0, 8.0]]], dtype=FLOAT_TYPE)
        out = au.norm2(a)
        outg = gau.norm2(a)
        np.testing.assert_array_equal(out, outg)

    @unittest.skip("This method is not implemented yet")
    def test_norm2_3d_complex_UNITY(self):
        a = np.array([[[1.0+1.0j, 2.0+2.0j],
                      [3.0+3.0j, 4.0+4.0j]],
                      [[5.0 + 5.0j, 6.0 + 6.0j],
                       [7.0 + 7.0j, 8.0 + 8.0j]]], dtype=COMPLEX_TYPE)
        out = au.norm2(a)
        outg = gau.norm2(a)
        np.testing.assert_array_equal(out, outg)

    @unittest.skip("This method is not implemented yet")
    def test_complex_gaussian_filter_2d_UNITY(self):
        data = np.zeros((8, 8), dtype=COMPLEX_TYPE)
        data[3:5, 3:5] = 2.0+2.0j
        mfs = 3.0,4.0
        out = au.complex_gaussian_filter(data, mfs)
        outg = gau.complex_gaussian_filter(data, mfs)
        np.testing.assert_array_almost_equal(np.diagonal(out), np.diagonal(outg))

    @unittest.skip("This method is not implemented yet")
    def test_complex_gaussian_filter_3d_UNITY(self):
        data = np.zeros((4, 4, 4), dtype=COMPLEX_TYPE)
        data[2:3, 2:3, 2:3] = 2.0+2.0j
        mfs = 2.0,3.0,4.0
        out = au.complex_gaussian_filter(data, mfs)
        gout = gau.complex_gaussian_filter(data, mfs)
        np.testing.assert_array_almost_equal(np.diagonal(out), np.diagonal(gout))

    @unittest.skip("This method is not implemented yet")
    def test_mass_center_2d_UNITY(self):
        npts = 64
        probe = np.zeros((1, npts, npts), dtype=COMPLEX_TYPE)
        rad = 10.0
        probe_vals = 2 + 3j
        x = np.array(range(npts)) - npts // 2
        X, Y = np.meshgrid(x, x)
        Xoff = 5.0
        Yoff = 2.0
        probe[0, (X-Xoff)**2 + (Y-Yoff)**2 < rad**2] = probe_vals

        com = au.mass_center(np.abs(probe[0]))
        comg = gau.mass_center(np.abs(probe[0]))

        np.testing.assert_array_almost_equal(com, comg)

    @unittest.skip("This method is not implemented yet")
    def test_mass_center_3d_UNITY(self):
        npts = 64
        probe = np.zeros((npts, npts, npts), dtype=COMPLEX_TYPE)
        rad = 10.0
        probe_vals = 2 + 3j
        x = np.array(range(npts)) - npts // 2
        X, Y, Z = np.meshgrid(x, x, x)
        Xoff = 5.0
        Yoff = 2.0
        Zoff = 10.0
        probe[(X-Xoff)**2 + (Y-Yoff)**2 + (Z-Zoff)**2< rad**2] = probe_vals

        com = au.mass_center(np.abs(probe))
        comg = gau.mass_center(np.abs(probe))
        np.testing.assert_array_almost_equal(com, comg)

    @unittest.skip("This method is not implemented yet")
    def test_interpolated_shift(self):
        npts = 32
        probe = np.zeros((1, npts, npts), dtype=COMPLEX_TYPE)
        rad = 10.0
        probe_vals = 2 + 3j
        x = np.array(range(npts)) - npts // 2
        X, Y = np.meshgrid(x, x)
        Xoff = 5.0
        Yoff = 2.0
        offset = np.array([-Yoff, -Xoff])
        probe[0, (X-Xoff)**2 + (Y-Yoff)**2 < rad**2] = probe_vals
        gprobe = deepcopy(probe)

        probe[0] = au.interpolated_shift(probe[0], offset)
        gprobe[0] = gau.interpolated_shift(gprobe[0], offset)
        np.testing.assert_array_almost_equal(probe, gprobe)

    @unittest.skip("This method is not implemented yet")
    def test_clip_magnitudes_to_range(self):
        data = np.ones((5,5), dtype=COMPLEX_TYPE)
        data[2, 4] = 20.0*np.exp(1j*np.pi/2)
        data[3, 1] = 0.2*np.exp(1j*np.pi/3)
        gdata = deepcopy(data)
        clip_min = 0.5
        clip_max = 2.0

        au.clip_complex_magnitudes_to_range(data, clip_min, clip_max)
        gau.clip_complex_magnitudes_to_range(gdata, clip_min, clip_max)
        np.testing.assert_array_almost_equal(data, gdata)


if __name__=='__main__':
    unittest.main()