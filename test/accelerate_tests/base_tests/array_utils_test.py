'''
Tests for the array_utils module
'''

import unittest, pytest
import numpy as np
from ptypy.accelerate.base import FLOAT_TYPE, COMPLEX_TYPE
from ptypy.accelerate.base import array_utils as au

_numpy_version = tuple(map(int, np.__version__.split(".")))

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

    @pytest.mark.skipif(_numpy_version < (2,0), reason="requires Numpy 2.0 or higher to match expected output")
    def test_fft_filter(self):
        data = np.zeros((256, 512), dtype=COMPLEX_TYPE)
        data[64:-64,128:-128] = 1 + 1.j

        prefactor = np.zeros_like(data)
        prefactor[:,256:] = 1.
        postfactor = np.zeros_like(data)
        postfactor[128:,:] = 1.

        rk = np.zeros_like(data)
        rk[:30, :30] = 1.
        kernel = np.fft.fftn(rk)

        output = au.fft_filter(data, kernel, prefactor, postfactor)

        known_test_output = np.array([-0.0000000e+00+0.00000000e+00j, -0.0000000e+00+0.00000000e+00j,
                                      0.0000000e+00+0.00000000e+00j,  0.0000000e+00+0.00000000e+00j,
                                      0.0000000e+00+0.00000000e+00j,  0.0000000e+00+0.00000000e+00j,
                                      0.0000000e+00-0.00000000e+00j,  0.0000000e+00+0.00000000e+00j,
                                      -0.0000000e+00+0.00000000e+00j,  0.0000000e+00+0.00000000e+00j,
                                      -0.0000000e+00+0.00000000e+00j, -0.0000000e+00+0.00000000e+00j,
                                      0.0000000e+00+0.00000000e+00j,  0.0000000e+00+0.00000000e+00j,
                                      0.0000000e+00+0.00000000e+00j,  0.0000000e+00+0.00000000e+00j,
                                      0.0000000e+00+0.00000000e+00j,  0.0000000e+00-0.00000000e+00j,
                                      0.0000000e+00+0.00000000e+00j,  0.0000000e+00+0.00000000e+00j,
                                      0.0000000e+00+0.00000000e+00j,  0.0000000e+00+0.00000000e+00j,
                                      0.0000000e+00+0.00000000e+00j,  0.0000000e+00+0.00000000e+00j,
                                      0.0000000e+00+0.00000000e+00j,  0.0000000e+00+0.00000000e+00j,
                                      0.0000000e+00+0.00000000e+00j,  0.0000000e+00+0.00000000e+00j,
                                      0.0000000e+00+0.00000000e+00j,  0.0000000e+00+0.00000000e+00j,
                                      0.0000000e+00+0.00000000e+00j,  0.0000000e+00-0.00000000e+00j,
                                      0.0000000e+00+0.00000000e+00j,  6.1097220e-05+2.92563982e-05j,
                                      4.0044695e-05+2.52102855e-05j,  8.9999994e+02+9.00000000e+02j,
                                      8.9999988e+02+8.99999939e+02j,  5.0999994e+02+5.09999939e+02j,
                                      1.9365043e-05+5.84280206e-05j,  3.0681291e-05+2.31116355e-05j,
                                      1.2552022e-05-1.01537153e-05j, -1.4034913e-05-1.17988075e-05j,
                                      -1.9193330e-05+2.72889110e-07j,  1.3895768e-05+1.64778357e-05j,
                                      6.5228807e-05+2.45708943e-05j,  3.8999994e+02+3.89999939e+02j,
                                      8.9999988e+02+8.99999878e+02j,  8.9999982e+02+8.99999878e+02j,
                                      3.0000015e+01+3.00000248e+01j,  3.8863189e-05+3.26705631e-05j,
                                      2.8768281e-06-1.62116921e-05j, -3.2418033e-05-1.97073969e-05j,
                                      -6.6843757e-05+7.19546824e-06j,  6.5036993e-06+3.95851657e-06j,
                                      -2.4053887e-05+9.88548163e-06j,  1.5231475e-05+1.31202614e-06j,
                                      8.7000000e+01+8.70000305e+01j,  6.1035156e-05+0.00000000e+00j,
                                      6.1035156e-05+0.00000000e+00j, -2.4943074e-07+6.62429193e-06j,
                                      1.6712515e-06-2.97475322e-06j,  1.9025241e-05+2.97752194e-07j,
                                      -9.2436176e-07-3.86252796e-05j, -8.8145862e-06-9.89961700e-06j,
                                      -1.5782407e-06+1.01533060e-05j, -4.7593076e-06+2.96332291e-05j])
        
        np.testing.assert_array_almost_equal(output.flat[::2000], known_test_output, decimal=5)

    @pytest.mark.skipif(_numpy_version < (2,0), reason="requires Numpy 2.0 or higher to match expected output")
    def test_fft_filter_batched(self):
        data = np.zeros((2,256, 512), dtype=COMPLEX_TYPE)
        data[:,64:-64,128:-128] = 1 + 1.j

        prefactor = np.zeros_like(data)
        prefactor[:,:,256:] = 1.
        postfactor = np.zeros_like(data)
        postfactor[:,128:,:] = 1.

        rk = np.zeros_like(data)[0]
        rk[:30, :30] = 1.
        kernel = np.fft.fftn(rk)

        output = au.fft_filter(data, kernel, prefactor, postfactor)

        known_test_output = np.array([ 0.00000000e+00-0.0000000e+00j,  0.00000000e+00-0.0000000e+00j,
                                       0.00000000e+00+0.0000000e+00j,  0.00000000e+00+0.0000000e+00j,
                                       0.00000000e+00+0.0000000e+00j,  0.00000000e+00+0.0000000e+00j,
                                       0.00000000e+00+0.0000000e+00j, 0.00000000e+00+0.0000000e+00j,
                                       -0.00000000e+00+0.0000000e+00j,  0.00000000e+00-0.0000000e+00j,
                                       0.00000000e+00-0.0000000e+00j,  0.00000000e+00-0.0000000e+00j,
                                       0.00000000e+00+0.0000000e+00j, 0.00000000e+00+0.0000000e+00j,
                                       0.00000000e+00+0.0000000e+00j,  0.00000000e+00+0.0000000e+00j,
                                       0.00000000e+00+0.0000000e+00j,  0.00000000e+00+0.0000000e+00j,
                                       0.00000000e+00+0.0000000e+00j,  0.00000000e+00-0.0000000e+00j,
                                       0.00000000e+00-0.0000000e+00j,  0.00000000e+00-0.0000000e+00j,
                                       0.00000000e+00+0.0000000e+00j,  0.00000000e+00+0.0000000e+00j,
                                       0.00000000e+00+0.0000000e+00j,  0.00000000e+00+0.0000000e+00j,
                                       0.00000000e+00+0.0000000e+00j,  0.00000000e+00+0.0000000e+00j,
                                       0.00000000e+00+0.0000000e+00j, 0.00000000e+00-0.0000000e+00j,
                                       0.00000000e+00-0.0000000e+00j,  0.00000000e+00-0.0000000e+00j,
                                       0.00000000e+00-0.0000000e+00j,  4.86995195e-05-9.1511911e-06j,
                                       5.89395277e-05+3.6706428e-05j,  8.99999817e+02+9.0000000e+02j,
                                       8.99999817e+02+9.0000000e+02j,  5.09999969e+02+5.0999997e+02j,
                                       6.86399580e-05+5.5245564e-05j, -2.15578075e-06-8.0761157e-07j,
                                       -5.99612467e-05-3.7489859e-05j, -2.08058154e-05-1.7001423e-05j,
                                       -3.15661709e-05-2.0192698e-05j, -1.17410173e-05-2.3929812e-05j,
                                       8.41844594e-05+4.9635066e-05j,  3.90000031e+02+3.9000003e+02j,
                                       8.99999817e+02+8.9999994e+02j,  8.99999817e+02+8.9999994e+02j,
                                       3.00000153e+01+3.0000000e+01j,  4.75842753e-05+1.7961407e-05j,
                                       -1.28229876e-05-3.3492659e-05j, -1.50405585e-05+3.0159079e-05j,
                                       -1.00799960e-04-6.6932058e-05j, -4.90295024e-05-3.6601130e-05j,
                                       -4.48861247e-05-1.4717044e-05j,  2.60417364e-05-8.3221821e-06j,
                                       8.69999847e+01+8.7000046e+01j,  4.31583721e-05+4.3158372e-05j,
                                       4.31583721e-05+4.3158372e-05j,  4.04649109e-06-1.6836095e-05j,
                                       1.37377283e-05+5.2577798e-06j, -2.30404657e-05-3.4596611e-05j,
                                       -1.33214944e-05-3.2517899e-05j,  2.45428764e-05-3.5186855e-07j,
                                       -1.85950885e-05-2.1921931e-05j, -1.65030433e-05-8.0249208e-07j])
                        
        np.testing.assert_array_almost_equal(output[1].flat[::2000], known_test_output, decimal=5)


    def test_complex_gaussian_filter_fft(self):
        data = np.zeros((8, 8), dtype=COMPLEX_TYPE)
        data[3:5, 3:5] = 2.0 + 2.0j
        mfs = 3.0, 4.0

        out = au.complex_gaussian_filter_fft(data, mfs)
        expected_out = np.array([0.11033735 + 0.11033735j, 0.11888228 + 0.11888228j, 0.13116673 + 0.13116673j
                                    , 0.13999543 + 0.13999543j, 0.13999543 + 0.13999543j, 0.13116673 + 0.13116673j
                                    , 0.11888228 + 0.11888228j, 0.11033735 + 0.11033735j], dtype=COMPLEX_TYPE)
        np.testing.assert_array_almost_equal(np.diagonal(out), expected_out, decimal=5)

    def test_complex_gaussian_filter_fft_batched(self):
        batch_number = 2
        A = 5
        B = 5

        data = np.zeros((batch_number, A, B), dtype=COMPLEX_TYPE)
        data[:, 2:3, 2:3] = 2.0 + 2.0j
        mfs = 3.0, 4.0
        out = au.complex_gaussian_filter_fft(data, mfs)

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

        np.testing.assert_array_almost_equal(out, expected_out, decimal=5)


if __name__ == '__main__':
    unittest.main()
