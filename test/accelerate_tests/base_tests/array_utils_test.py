'''
Tests for the array_utils module
'''

import unittest
import numpy as np
from ptypy.accelerate.base import FLOAT_TYPE, COMPLEX_TYPE
from ptypy.accelerate.base import array_utils as au


class ArrayUtilsTest(unittest.TestCase):

    def test_dot_resolution(self):
        X, Y, Z = np.indices((3, 3, 1001), dtype=np.float32)
        A = 10 ** Y + 1j * 10 ** X
        out = au.dot(A, A)
        np.testing.assert_array_equal(out, 60666606.0)

    def test_abs2_real_input(self):
        single_dim = 50.0
        npts = single_dim ** 3
        array_to_be_absed = np.arange(npts)
        absed = np.array([ix ** 2 for ix in array_to_be_absed])
        array_shape = (int(single_dim), int(single_dim), int(single_dim))
        array_to_be_absed.reshape(array_shape)
        absed.reshape(array_shape)
        out = au.abs2(array_to_be_absed)
        np.testing.assert_array_equal(absed, out)
        self.assertEqual(absed.dtype, np.float64)

    def test_abs2_complex_input(self):
        single_dim = 50.0
        array_shape = (int(single_dim), int(single_dim), int(single_dim))
        npts = single_dim ** 3
        array_to_be_absed = np.arange(npts) + 1j * np.arange(npts)
        absed = np.array([np.abs(ix ** 2) for ix in array_to_be_absed])
        absed.reshape(array_shape)
        array_to_be_absed.reshape(array_shape)
        out = au.abs2(array_to_be_absed)
        np.testing.assert_array_equal(absed, out)
        self.assertEqual(absed.dtype, np.float64)

    def test_sum_to_buffer(self):

        I = 4
        X = 2
        M = 4
        N = 4

        in1 = np.empty((I, M, N), dtype=FLOAT_TYPE)

        # fill the input array
        for idx in range(I):
            in1[idx] = np.ones((M, N)) * (idx + 1.0)

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
            in1[idx] = np.ones((M, N)) * (idx + 1.0) + 1j * np.ones((M, N)) * (idx + 1.0)

        outshape = (X, M, N)
        expected_out = np.empty(outshape, dtype=COMPLEX_TYPE)

        expected_out[0] = np.ones((M, N)) * 4.0 + 1j * np.ones((M, N)) * 4.0
        expected_out[1] = np.ones((M, N)) * 6.0 + 1j * np.ones((M, N)) * 6.0

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
        a = np.array([1.0 + 1.0j, 2.0 + 2.0j], dtype=COMPLEX_TYPE)
        out = au.norm2(a)
        np.testing.assert_array_equal(out, 10.0)

    def test_norm2_2d_real(self):
        a = np.array([[1.0, 2.0],
                      [3.0, 4.0]], dtype=FLOAT_TYPE)
        out = au.norm2(a)
        np.testing.assert_array_equal(out, 30.0)

    def test_norm2_2d_complex(self):
        a = np.array([[1.0 + 1.0j, 2.0 + 2.0j],
                      [3.0 + 3.0j, 4.0 + 4.0j]], dtype=COMPLEX_TYPE)
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
        a = np.array([[[1.0 + 1.0j, 2.0 + 2.0j],
                       [3.0 + 3.0j, 4.0 + 4.0j]],
                      [[5.0 + 5.0j, 6.0 + 6.0j],
                       [7.0 + 7.0j, 8.0 + 8.0j]]], dtype=COMPLEX_TYPE)
        out = au.norm2(a)
        np.testing.assert_array_equal(out, 408.0)

    def test_complex_gaussian_filter_2d(self):
        data = np.zeros((8, 8), dtype=COMPLEX_TYPE)
        data[3:5, 3:5] = 2.0 + 2.0j
        mfs = 3.0, 4.0
        out = au.complex_gaussian_filter(data, mfs)
        expected_out = np.array([0.11033735 + 0.11033735j, 0.11888228 + 0.11888228j, 0.13116673 + 0.13116673j
                                    , 0.13999543 + 0.13999543j, 0.13999543 + 0.13999543j, 0.13116673 + 0.13116673j
                                    , 0.11888228 + 0.11888228j, 0.11033735 + 0.11033735j], dtype=COMPLEX_TYPE)
        np.testing.assert_array_almost_equal(np.diagonal(out), expected_out)

    def test_complex_gaussian_filter_2d_batched(self):
        batch_number = 2
        A = 5
        B = 5

        data = np.zeros((batch_number, A, B), dtype=COMPLEX_TYPE)
        data[:, 2:3, 2:3] = 2.0 + 2.0j
        mfs = 3.0, 4.0
        out = au.complex_gaussian_filter(data, mfs)

        expected_out = np.array([[[0.07988770 + 0.0798877j, 0.07989411 + 0.07989411j, 0.07989471 + 0.07989471j,
                                   0.07989411 + 0.07989411j, 0.07988770 + 0.0798877j],
                                  [0.08003781 + 0.08003781j, 0.08004424 + 0.08004424j, 0.08004485 + 0.08004485j,
                                   0.08004424 + 0.08004424j, 0.08003781 + 0.08003781j],
                                  [0.08012911 + 0.08012911j, 0.08013555 + 0.08013555j, 0.08013615 + 0.08013615j,
                                   0.08013555 + 0.08013555j, 0.08012911 + 0.08012911j],
                                  [0.08003781 + 0.08003781j, 0.08004424 + 0.08004424j, 0.08004485 + 0.08004485j,
                                   0.08004424 + 0.08004424j, 0.08003781 + 0.08003781j],
                                  [0.07988770 + 0.0798877j, 0.07989411 + 0.07989411j, 0.07989471 + 0.07989471j,
                                   0.07989411 + 0.07989411j, 0.07988770 + 0.0798877j]],

                                 [[0.07988770 + 0.0798877j, 0.07989411 + 0.07989411j, 0.07989471 + 0.07989471j,
                                   0.07989411 + 0.07989411j, 0.07988770 + 0.0798877j],
                                  [0.08003781 + 0.08003781j, 0.08004424 + 0.08004424j, 0.08004485 + 0.08004485j,
                                   0.08004424 + 0.08004424j, 0.08003781 + 0.08003781j],
                                  [0.08012911 + 0.08012911j, 0.08013555 + 0.08013555j, 0.08013615 + 0.08013615j,
                                   0.08013555 + 0.08013555j, 0.08012911 + 0.08012911j],
                                  [0.08003781 + 0.08003781j, 0.08004424 + 0.08004424j, 0.08004485 + 0.08004485j,
                                   0.08004424 + 0.08004424j, 0.08003781 + 0.08003781j],
                                  [0.07988770 + 0.0798877j, 0.07989411 + 0.07989411j, 0.07989471 + 0.07989471j,
                                   0.07989411 + 0.07989411j, 0.07988770 + 0.0798877j]]], dtype=COMPLEX_TYPE)

        np.testing.assert_array_almost_equal(out, expected_out)

    def test_mass_center_2d(self):
        npts = 64
        probe = np.zeros((1, npts, npts), dtype=COMPLEX_TYPE)
        rad = 10.0
        probe_vals = 2 + 3j
        x = np.array(range(npts)) - npts // 2
        X, Y = np.meshgrid(x, x)
        Xoff = 5.0
        Yoff = 2.0
        probe[0, (X - Xoff) ** 2 + (Y - Yoff) ** 2 < rad ** 2] = probe_vals

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
        probe[(X - Xoff) ** 2 + (Y - Yoff) ** 2 + (Z - Zoff) ** 2 < rad ** 2] = probe_vals

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
        probe[0, (X - Xoff) ** 2 + (Y - Yoff) ** 2 < rad ** 2] = probe_vals
        offset = np.array([-Yoff, -Xoff])

        not_shifted_probe = np.zeros((1, npts, npts), dtype=COMPLEX_TYPE)
        not_shifted_probe[0, (X) ** 2 + (Y) ** 2 < rad ** 2] = probe_vals
        probe[0] = au.interpolated_shift(probe[0], offset)
        np.testing.assert_array_almost_equal(probe, not_shifted_probe, decimal=8)

    def test_clip_magnitudes_to_range(self):
        data = np.ones((5, 5), dtype=COMPLEX_TYPE)
        data[2, 4] = 20.0 * np.exp(1j * np.pi / 2)
        data[3, 1] = 0.2 * np.exp(1j * np.pi / 3)

        clip_min = 0.5
        clip_max = 2.0
        expected_out = np.ones_like(data)
        expected_out[2, 4] = 2.0 * np.exp(1j * np.pi / 2)
        expected_out[3, 1] = 0.5 * np.exp(1j * np.pi / 3)
        au.clip_complex_magnitudes_to_range(data, clip_min, clip_max)
        np.testing.assert_array_almost_equal(data, expected_out, decimal=7)  # floating point precision I guess...

    def test_crop_pad_1(self):
        # pad, integer, 2D
        B = np.indices((4, 4), dtype=np.int32)
        A = np.zeros((6, 6), dtype=B.dtype)
        au.crop_pad_2d_simple(A, B.sum(0))
        exp_A = np.array([[0, 0, 0, 0, 0, 0],
                          [0, 0, 1, 2, 3, 0],
                          [0, 1, 2, 3, 4, 0],
                          [0, 2, 3, 4, 5, 0],
                          [0, 3, 4, 5, 6, 0],
                          [0, 0, 0, 0, 0, 0]])
        np.testing.assert_equal(A, exp_A)

    def test_crop_pad_2(self):
        # crop, float, 3D
        B = np.indices((4, 4), dtype=np.float32)
        A = np.zeros((2, 2, 2), dtype=B.dtype)
        au.crop_pad_2d_simple(A, B)
        exp_A = np.array([[[1., 1.],
                           [2., 2.]],
                          [[1., 2.],
                           [1., 2.]]], dtype=np.float32)
        np.testing.assert_array_almost_equal(A, exp_A)

    def test_crop_pad_3(self):
        # crop/pad, complex, 3D
        B = np.indices((4, 3), dtype=np.complex64)
        B = np.indices((4, 3), dtype=np.complex64) + 1j * B[::-1, :, :]
        A = np.zeros((2, 2, 5), dtype=B.dtype)
        au.crop_pad_2d_simple(A, B)
        exp_A = np.array([[[0. + 0.j, 1. + 0.j, 1. + 1.j, 1. + 2.j, 0. + 0.j],
                           [0. + 0.j, 2. + 0.j, 2. + 1.j, 2. + 2.j, 0. + 0.j]],
                          [[0. + 0.j, 0. + 1.j, 1. + 1.j, 2. + 1.j, 0. + 0.j],
                           [0. + 0.j, 0. + 2.j, 1. + 2.j, 2. + 2.j, 0. + 0.j]]],
                         dtype=np.complex64)
        np.testing.assert_array_almost_equal(A, exp_A)

    def test_downsample_factor_2(self):
        # downsample, complex, 3D
        B = np.indices((4,4), dtype=np.complex64)
        B = np.indices((4,4), dtype=np.complex64) + 1j * B[::-1, :, :]
        A = np.zeros((2,2,2), dtype=B.dtype)
        au.resample(A,B)
        exp_A = np.array([[[ 2. +2.j,  2.+10.j],
                           [10. +2.j, 10.+10.j]],
                          [[ 2. +2.j, 10.+ 2.j],
                           [ 2.+10.j, 10.+10.j]]], dtype=np.complex64)
        np.testing.assert_almost_equal(A.sum(), B.sum())            
        np.testing.assert_array_almost_equal(A, exp_A)

    def test_upsample_factor_2(self):
        # upsample, complex, 3D
        B = np.indices((2,2), dtype=np.complex64)
        B = np.indices((2,2), dtype=np.complex64) + 1j * B[::-1, :, :]
        A = np.zeros((2,4,4), dtype=B.dtype)
        au.resample(A,B)
        exp_A = np.array([[[0.  +0.j  , 0.  +0.j  , 0.  +0.25j, 0.  +0.25j],
                           [0.  +0.j  , 0.  +0.j  , 0.  +0.25j, 0.  +0.25j],
                           [0.25+0.j  , 0.25+0.j  , 0.25+0.25j, 0.25+0.25j],
                           [0.25+0.j  , 0.25+0.j  , 0.25+0.25j, 0.25+0.25j]],
                          [[0.  +0.j  , 0.  +0.j  , 0.25+0.j  , 0.25+0.j  ],
                           [0.  +0.j  , 0.  +0.j  , 0.25+0.j  , 0.25+0.j  ],
                           [0.  +0.25j, 0.  +0.25j, 0.25+0.25j, 0.25+0.25j],
                           [0.  +0.25j, 0.  +0.25j, 0.25+0.25j, 0.25+0.25j]]], dtype=np.complex64)
        np.testing.assert_almost_equal(A.sum(), B.sum())
        np.testing.assert_array_almost_equal(A, exp_A)

    def test_downsample_factor_4(self):
        # downsample, complex, 3D
        B = np.indices((8,8), dtype=np.complex64)
        B = np.indices((8,8), dtype=np.complex64) + 1j * B[::-1, :, :]
        A = np.zeros((2,2,2), dtype=B.dtype)
        au.resample(A,B)
        exp_A = np.array([[[24.+24.j, 24.+88.j],
                           [88.+24.j, 88.+88.j]],
                          [[24.+24.j, 88.+24.j],
                           [24.+88.j, 88.+88.j]]], dtype=np.complex64)
        np.testing.assert_almost_equal(A.sum(), B.sum())
        np.testing.assert_array_almost_equal(A, exp_A)

    def test_upsample_factor_4(self):
        # upsample, complex, 3D
        Bshape = (2,4,4)
        B = np.reshape(np.arange(0, np.prod(Bshape)), Bshape).astype(np.complex64) * 16
        B = B + 1j * B
        A = np.zeros((Bshape[0], Bshape[1]*4, Bshape[2]*4), dtype=B.dtype)
        au.resample(A, B)
        exp_A = np.zeros_like(A)
        
        # building the expected value element-wise, to ensure correctness
        for z in range(A.shape[0]):
            for y in range(A.shape[1]):
                for x in range(A.shape[2]):
                    exp_A[z, y, x] = B[z, y//4, x//4] / 16
                    if np.abs(A[z, y, x]-exp_A[z, y, x]) > 1e-6:
                        print("mismatch! i=({},{},{}): act={}, exp={}".format(z, y, x, A[z, y, x], exp_A[z, y, x]))

        np.testing.assert_almost_equal(A.sum(), B.sum())
        np.testing.assert_array_almost_equal(A, exp_A)


if __name__ == '__main__':
    unittest.main()
