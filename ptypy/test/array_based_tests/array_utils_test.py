'''
Tests for the array_utils module
'''


import unittest
import numpy as np
from ptypy.array_based import FLOAT_TYPE, COMPLEX_TYPE
from ptypy.array_based import array_utils as au


class ArrayUtilsTest(unittest.TestCase):

    def test_abs2_real_input(self):
        single_dim = 50.0
        npts = single_dim ** 3
        array_to_be_absed = np.arange(npts)
        absed = np.array([ix**2 for ix in array_to_be_absed])
        array_shape = (int(single_dim), int(single_dim), int(single_dim))
        array_to_be_absed.reshape(array_shape)
        absed.reshape(array_shape)
        out = au.abs2(array_to_be_absed)
        np.testing.assert_array_equal(absed, out)
        self.assertEqual(absed.dtype, np.float)


    def test_abs2_complex_input(self):
        single_dim = 50.0
        array_shape = (int(single_dim), int(single_dim), int(single_dim))
        npts = single_dim ** 3
        array_to_be_absed = np.arange(npts) + 1j * np.arange(npts)
        absed = np.array([np.abs(ix**2) for ix in array_to_be_absed])
        absed.reshape(array_shape)
        array_to_be_absed.reshape(array_shape)
        out = au.abs2(array_to_be_absed)
        np.testing.assert_array_equal(absed, out)
        self.assertEqual(absed.dtype, np.float)

    def test_sum_to_buffer(self):

        I = 4
        X = 2
        M = 4
        N = 4

        in1 = np.empty((I, M, N), dtype=FLOAT_TYPE)

        # fill the input array
        for idx in range(I):
            in1[idx] = np.ones((M, N))* (idx + 1.0)

        outshape = (X, M, N)
        expected_out = np.empty(outshape)

        expected_out[0] = np.ones((M, N)) * 4.0
        expected_out[1] = np.ones((M, N)) * 6.0

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

        out = au.sum_to_buffer(in1, outshape, in1_addr, out1_addr, dtype=FLOAT_TYPE)
        np.testing.assert_array_equal(out, expected_out)


    def test_sum_to_buffer_complex(self):

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

        np.testing.assert_array_equal(out, expected_out)

    def test_norm2_1d_real(self):
        a = np.array([1.0, 2.0], dtype=FLOAT_TYPE)
        out = au.norm2(a)
        np.testing.assert_array_equal(out, 5.0)

    def test_norm2_1d_complex(self):
        a = np.array([1.0+1.0j, 2.0+2.0j], dtype=COMPLEX_TYPE)
        out = au.norm2(a)
        np.testing.assert_array_equal(out, 10.0)

    def test_norm2_2d_real(self):
        a = np.array([[1.0, 2.0],
                      [3.0, 4.0]], dtype=FLOAT_TYPE)
        out = au.norm2(a)
        np.testing.assert_array_equal(out, 30.0)

    def test_norm2_2d_complex(self):
        a = np.array([[1.0+1.0j, 2.0+2.0j],
                      [3.0+3.0j, 4.0+4.0j]], dtype=COMPLEX_TYPE)
        out = au.norm2(a)
        np.testing.assert_array_equal(out, 60.0)

    def test_norm2_3d_real(self):
        a = np.array([[[1.0, 2.0],
                      [3.0, 4.0]],
                      [[5.0, 6.0],
                       [7.0, 8.0]]], dtype=FLOAT_TYPE)
        out = au.norm2(a)
        np.testing.assert_array_equal(out, 204.0)

    def test_norm2_3d_complex(self):
        a = np.array([[[1.0+1.0j, 2.0+2.0j],
                      [3.0+3.0j, 4.0+4.0j]],
                      [[5.0 + 5.0j, 6.0 + 6.0j],
                       [7.0 + 7.0j, 8.0 + 8.0j]]], dtype=COMPLEX_TYPE)
        out = au.norm2(a)
        np.testing.assert_array_equal(out, 408.0)

    def test_complex_gaussian_filter_2d(self):
        data = np.zeros((8, 8), dtype=COMPLEX_TYPE)
        data[3:5, 3:5] = 2.0+2.0j
        mfs = 3.0,4.0
        out = au.complex_gaussian_filter(data, mfs)
        expected_out = np.array([0.11033735 + 0.11033735j, 0.11888228 + 0.11888228j, 0.13116673 + 0.13116673j
                                    , 0.13999543 + 0.13999543j, 0.13999543 + 0.13999543j, 0.13116673 + 0.13116673j
                                    , 0.11888228 + 0.11888228j, 0.11033735 + 0.11033735j], dtype=COMPLEX_TYPE)
        np.testing.assert_array_almost_equal(np.diagonal(out), expected_out)


    def test_complex_gaussian_filter_3d(self):
        data = np.zeros((4, 4, 4), dtype=COMPLEX_TYPE)
        data[2:3, 2:3, 2:3] = 2.0+2.0j
        mfs = 2.0,3.0,4.0
        out = au.complex_gaussian_filter(data, mfs)
        expected_out = np.array([[ 0.02338744+0.02338744j, 0.02814008+0.02814008j, 0.03458820+0.0345882j, 0.03891133+0.03891133j],
                                [0.02345678+0.02345678j, 0.02822352+0.02822352j, 0.03469075+0.03469075j, 0.03902670+0.0390267j],
                                [0.02355688+0.02355688j, 0.02834396+0.02834396j, 0.03483879+0.03483879j, 0.03919324+0.03919324j],
                                [0.02362553+0.02362553j, 0.02842657+0.02842657j, 0.03494033+0.03494033j, 0.03930746+0.03930746j]], dtype=COMPLEX_TYPE)
        np.testing.assert_array_almost_equal(np.diagonal(out), expected_out)

    def test_mass_center_2d(self):
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
        expected_out = np.array([Yoff, Xoff]) + npts // 2
        np.testing.assert_array_almost_equal(com, expected_out, decimal=6)


    def test_mass_center_3d(self):
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
        expected_out = np.array([Yoff, Xoff, Zoff]) + npts // 2
        np.testing.assert_array_almost_equal(com, expected_out, decimal=5)

    def test_interpolated_shift(self):
        npts = 32
        probe = np.zeros((1, npts, npts), dtype=COMPLEX_TYPE)
        rad = 10.0
        probe_vals = 2 + 3j
        x = np.array(range(npts)) - npts // 2
        X, Y = np.meshgrid(x, x)
        Xoff = 5.0
        Yoff = 2.0
        probe[0, (X-Xoff)**2 + (Y-Yoff)**2 < rad**2] = probe_vals
        offset = np.array([-Yoff, -Xoff])

        not_shifted_probe = np.zeros((1, npts, npts), dtype=COMPLEX_TYPE)
        not_shifted_probe[0, (X)**2 + (Y)**2 < rad**2] = probe_vals
        probe[0] = au.interpolated_shift(probe[0], offset)
        np.testing.assert_array_almost_equal(probe, not_shifted_probe, decimal=8)

    def test_clip_magnitudes_to_range(self):
        data = np.ones((5,5), dtype=COMPLEX_TYPE)
        data[2, 4] = 20.0*np.exp(1j*np.pi/2)
        data[3, 1] = 0.2*np.exp(1j*np.pi/3)

        clip_min = 0.5
        clip_max = 2.0
        expected_out = np.ones_like(data)
        expected_out[2, 4] = 2.0*np.exp(1j*np.pi/2)
        expected_out[3, 1] = 0.5*np.exp(1j*np.pi/3)
        au.clip_complex_magnitudes_to_range(data, clip_min, clip_max)
        np.testing.assert_array_almost_equal(data, expected_out, decimal=7) # floating point precision I guess...



if __name__=='__main__':
    unittest.main()