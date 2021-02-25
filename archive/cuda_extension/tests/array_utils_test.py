'''
Tests for the array_utils module
'''


import unittest
from ptypy.accelerate.array_based import array_utils as au
from ptypy.accelerate.array_based import FLOAT_TYPE, COMPLEX_TYPE
from copy import deepcopy
import numpy as np

from . import have_cuda, only_if_cuda_available
if have_cuda():
    from archive.cuda_extension.accelerate.cuda import array_utils as gau
    from archive.cuda_extension.accelerate.cuda.config import init_gpus, reset_function_cache
    init_gpus(0)

@only_if_cuda_available
class ArrayUtilsTest(unittest.TestCase):

    def tearDown(self):
        # reset the cached GPU functions after each test
        reset_function_cache()

    def test_abs2_real_input_2D_float32_UNITY(self):
        x = np.ones((3,3), dtype=np.float32)
        np.testing.assert_array_equal(au.abs2(x), gau.abs2(x))

    def test_abs2_complex_input_2D_float32_UNITY(self):
        x = np.ones((3,3)) + 1j*np.ones((3,3))
        x = x.astype(np.complex64)
        np.testing.assert_array_equal(au.abs2(x), gau.abs2(x))
    
    def test_abs2_real_input_3D_float32_UNITY(self):
        x = np.ones((3,3,3), dtype=np.float32)
        np.testing.assert_array_equal(au.abs2(x), gau.abs2(x))

    def test_abs2_complex_input_3D_float32_UNITY(self):
        x = np.ones((3,3,3)) + 1j*np.ones((3,3,3))
        x = x.astype(np.complex64)
        np.testing.assert_array_equal(au.abs2(x), gau.abs2(x))

    def test_abs2_real_input_2D_float64_UNITY(self):
        x = np.ones((3,3))
        np.testing.assert_array_equal(au.abs2(x), gau.abs2(x))

    def test_abs2_complex_input_2D_float64_UNITY(self):
        x = np.ones((3,3)) + 1j*np.ones((3,3))
        np.testing.assert_array_equal(au.abs2(x), gau.abs2(x))
    
    def test_abs2_real_input_3D_float64_UNITY(self):
        x = np.ones((3,3,3))
        np.testing.assert_array_equal(au.abs2(x), gau.abs2(x))

    def test_abs2_complex_input_3D_float64_UNITY(self):
        x = np.ones((3,3,3)) + 1j*np.ones((3,3,3))
        np.testing.assert_array_equal(au.abs2(x), gau.abs2(x))

    def test_sum_to_buffer_real_UNITY(self):

        in1 = np.array([np.ones((4, 4)),
                        np.ones((4, 4))*2.0,
                        np.ones((4, 4))*3.0,
                        np.ones((4, 4))*4.0], dtype=FLOAT_TYPE)

        outshape = (2, 4, 4)
        
        in1_addr = np.array([(0, 0, 0),
                            (1, 0, 0),
                            (2, 0, 0),
                            (3, 0, 0)])

        out1_addr = np.array([(0, 0, 0),
                              (1, 0, 0),
                              (0, 0, 0),
                              (1, 0, 0)])

        s1 = au.sum_to_buffer(in1, outshape, in1_addr, out1_addr, dtype=FLOAT_TYPE)
        s2 = gau.sum_to_buffer(in1, outshape, in1_addr, out1_addr, dtype=FLOAT_TYPE)

        np.testing.assert_array_equal(s1, s2)
        
    def test_sum_to_buffer_stride_real_UNITY(self):

        in1 = np.array([np.ones((4, 4)),
                        np.ones((4, 4))*2.0,
                        np.ones((4, 4))*3.0,
                        np.ones((4, 4))*4.0], dtype=FLOAT_TYPE)

        outshape = (2, 4, 4)
        
        addr_info = np.zeros((4, 5, 3), dtype=np.int)
        addr_info[:,2,:] = np.array([(0, 0, 0),
                            (1, 0, 0),
                            (2, 0, 0),
                            (3, 0, 0)])

        addr_info[:,3,:] = np.array([(0, 0, 0),
                              (1, 0, 0),
                              (0, 0, 0),
                              (1, 0, 0)])

        s1 = au.sum_to_buffer(in1, outshape, addr_info[:,2,:], addr_info[:,3,:], dtype=FLOAT_TYPE)
        s2 = gau.sum_to_buffer_stride(in1, outshape, addr_info, dtype=FLOAT_TYPE)
        
        np.testing.assert_array_equal(s1, s2)

    def test_sum_to_buffer_complex_UNITY(self):

        in1 = np.array([np.ones((4,4)),
                        np.ones((4, 4))*2.0,
                        np.ones((4, 4))*3.0,
                        np.ones((4, 4))*4.0], dtype=FLOAT_TYPE) \
            + 1j*np.array([np.ones((4,4)),
                        np.ones((4, 4))*2.0,
                        np.ones((4, 4))*3.0,
                        np.ones((4, 4))*4.0], dtype=FLOAT_TYPE)

        outshape = (2, 4, 4)
        
        in1_addr = np.array([(0, 0, 0),
                            (1, 0, 0),
                            (2, 0, 0),
                            (3, 0, 0)])

        out1_addr = np.array([(0, 0, 0),
                              (1, 0, 0),
                              (0, 0, 0),
                              (1, 0, 0)])

        s1 = au.sum_to_buffer(in1, outshape, in1_addr, out1_addr, dtype=in1.dtype)
        s2 = gau.sum_to_buffer(in1, outshape, in1_addr, out1_addr, dtype=in1.dtype)

        np.testing.assert_array_equal(s1, s2)
    
    def test_sum_to_buffer_stride_complex_UNITY(self):

        in1 = np.array([np.ones((4,4)),
                        np.ones((4, 4))*2.0,
                        np.ones((4, 4))*3.0,
                        np.ones((4, 4))*4.0], dtype=FLOAT_TYPE) \
            + 1j*np.array([np.ones((4,4)),
                        np.ones((4, 4))*2.0,
                        np.ones((4, 4))*3.0,
                        np.ones((4, 4))*4.0], dtype=FLOAT_TYPE)

        outshape = (2, 4, 4)
        
        addr_info = np.zeros((4, 5, 3), dtype=np.int)
        addr_info[:,2,:] = np.array([(0, 0, 0),
                            (1, 0, 0),
                            (2, 0, 0),
                            (3, 0, 0)])

        addr_info[:,3,:] = np.array([(0, 0, 0),
                              (1, 0, 0),
                              (0, 0, 0),
                              (1, 0, 0)])

        s1 = au.sum_to_buffer(in1, outshape, addr_info[:,2,:], addr_info[:,3,:], dtype=in1.dtype)
        s2 = gau.sum_to_buffer_stride(in1, outshape, addr_info, dtype=in1.dtype)
        
        np.testing.assert_array_equal(s1, s2)


    def test_norm2_1d_real_UNITY(self):
        a = np.array([1.0, 2.0], dtype=FLOAT_TYPE)
        out = au.norm2(a)
        outg =gau.norm2(a)
        np.testing.assert_array_equal(out, outg)

    def test_norm2_1d_complex_UNITY(self):
        a = np.array([1.0+1.0j, 2.0+2.0j], dtype=COMPLEX_TYPE)
        out = au.norm2(a)
        outg = gau.norm2(a)
        np.testing.assert_array_equal(out, outg)

    def test_norm2_2d_real_UNITY(self):
        a = np.array([[1.0, 2.0],
                      [3.0, 4.0]], dtype=FLOAT_TYPE)
        out = au.norm2(a)
        outg = gau.norm2(a)
        np.testing.assert_array_equal(out, outg)

    def test_norm2_2d_complex_UNITY(self):
        a = np.array([[1.0+1.0j, 2.0+2.0j],
                      [3.0+3.0j, 4.0+4.0j]], dtype=COMPLEX_TYPE)
        out = au.norm2(a)
        outg = gau.norm2(a)
        np.testing.assert_array_equal(out, outg)

    def test_norm2_3d_real_UNITY(self):
        a = np.array([[[1.0, 2.0],
                      [3.0, 4.0]],
                      [[5.0, 6.0],
                       [7.0, 8.0]]], dtype=FLOAT_TYPE)
        out = au.norm2(a)
        outg = gau.norm2(a)
        np.testing.assert_array_equal(out, outg)

    def test_norm2_3d_complex_UNITY(self):
        a = np.array([[[1.0+1.0j, 2.0+2.0j],
                      [3.0+3.0j, 4.0+4.0j]],
                      [[5.0 + 5.0j, 6.0 + 6.0j],
                       [7.0 + 7.0j, 8.0 + 8.0j]]], dtype=COMPLEX_TYPE)
        out = au.norm2(a)
        outg = gau.norm2(a)
        np.testing.assert_array_equal(out, outg)

    def test_norm2_1d_real_large_UNITY(self):
        a = np.ones(2000000, dtype=FLOAT_TYPE)  # > 1M, to test multi-stage
        out = au.norm2(a)
        outg = gau.norm2(a)
        np.testing.assert_array_equal(out, outg)

    def test_complex_gaussian_filter_1d_UNITY(self):
        data = np.zeros((11,), dtype=COMPLEX_TYPE)
        data[5] = 1.0 +1.0j
        mfs = [1.0]
        out = au.complex_gaussian_filter(data, mfs)
        outg = gau.complex_gaussian_filter(data, mfs)
        #for i in xrange(11):
        #    print("{} vs {}".format(out[i], outg[i]))
        np.testing.assert_allclose(out, outg, rtol=1e-6)

    def test_complex_gaussian_filter_2d_simple_UNITY(self):
        data = np.zeros((11, 11), dtype=COMPLEX_TYPE)
        data[5, 5] = 1.0+1.0j
        mfs = 1.0,0.0
        out = au.complex_gaussian_filter(data, mfs)
        outg = gau.complex_gaussian_filter(data, mfs)
        np.testing.assert_allclose(out, outg, rtol=1e-6)

    def test_complex_gaussian_filter_2d_simple2_UNITY(self):
        data = np.zeros((11, 11), dtype=COMPLEX_TYPE)
        data[5, 5] = 1.0+1.0j
        mfs = 0.0,1.0
        out = au.complex_gaussian_filter(data, mfs)
        outg = gau.complex_gaussian_filter(data, mfs)
        np.testing.assert_allclose(out, outg, rtol=1e-6)

    def test_complex_gaussian_filter_2d_UNITY(self):
        data = np.zeros((8, 8), dtype=COMPLEX_TYPE)
        data[3:5, 3:5] = 2.0+2.0j
        mfs = 3.0,4.0
        out = au.complex_gaussian_filter(data, mfs)
        outg = gau.complex_gaussian_filter(data, mfs)
        np.testing.assert_allclose(out, outg, rtol=1e-6)

    def test_complex_gaussian_filter_2d_batched(self):
        batch_number = 2
        A = 5
        B = 5

        data = np.zeros((batch_number, A, B), dtype=COMPLEX_TYPE)
        data[:, 2:3, 2:3] = 2.0+2.0j
        mfs = 3.0,4.0
        out = au.complex_gaussian_filter(data, mfs)
        gout = gau.complex_gaussian_filter(data, mfs)
        
        np.testing.assert_allclose(out, gout, rtol=1e-6)

    def test_mass_center_2d_simple(self):
        data = np.array([[0,0,0,0],
                         [0,1,1,0],
                         [0,1,1,0],
                         [0,1,1,0],
                         [0,1,1,0]], dtype=FLOAT_TYPE)
        comg = gau.mass_center(data)
        np.testing.assert_array_equal([2.5, 1.5], comg)

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
        np.testing.assert_allclose(com, comg, rtol=1e-6)
        self.assertEqual(com.dtype, comg.dtype)

    def test_interpolated_shift_integer(self):
        A = np.array([[0,0,0,0,0],
                      [0,1,1,0,0],
                      [0,1,1,0,0],
                      [0,0,0,0,0],
                      [0,0,0,0,0]], dtype=COMPLEX_TYPE) + \
            1j * np.array([[0,0,0,0,0],
                      [0,0,1,1,1],
                      [1,0,1,0,1],
                      [0,0,0,1,1],
                      [0,0,0,0,0]], dtype=COMPLEX_TYPE)
        A = A.real + 1j * A.real
        
        shifts = [[-1, 0], 
                  [1, 1],
                  [2,2],
                  [-5,0],
                  [0,4],
                  [-3,-3],
                  [0,0]]
        for s in shifts:
            B1 = au.interpolated_shift(A, s)
            gB1 = gau.interpolated_shift(A, s)
            np.testing.assert_allclose(B1, gB1, rtol=1e-7, atol=1e-7)
            
    def test_interpolated_shift_linear_simple(self):
        A = np.array([[0,0,0,0,0],
                      [0,1,1,0,0],
                      [0,1,1,0,0],
                      [0,0,0,0,0],
                      [0,0,0,0,0]], dtype=COMPLEX_TYPE) + \
            1j * np.array([
                      [0,0,0,0,0],
                      [0,0,1,1,1],
                      [1,0,1,0,1],
                      [0,0,0,1,1],
                      [0,0,0,0,0]], dtype=COMPLEX_TYPE)
        
        
        shifts = [
            [0,-0.25],
            [-1.1,1.4],
            [0.5,-0.12],
            [-2.4,2.3],
            [-5.6,1.2],
            [0.2, 6.2],
            [-0.2, -7.6],
            [-102.1, 951.12],
                  ]
        for s in shifts:
            B1 = au.interpolated_shift(A, s, do_linear=True)
            gB1 = gau.interpolated_shift(A, s, do_linear=True)
            np.testing.assert_allclose(B1, gB1, rtol=1e-6, atol=1e-7, 
                                       err_msg="Failure for shift {}:\n CPU={}\nGPU={}".format(s, B1, gB1))

    def test_interpolated_shift_linear(self):
        npts = 32
        probe = np.zeros((1, npts, npts), dtype=COMPLEX_TYPE)
        rad = 10.0
        probe_vals = 2 + 3j
        x = np.array(range(npts)) - npts // 2
        X, Y = np.meshgrid(x, x)
        Xoff = 5.0
        Yoff = 2.0
        probe[0, (X-Xoff)**2 + (Y-Yoff)**2 < rad**2] = probe_vals
        
        shifts = [[-Xoff, -Yoff],
                  [0.5,0],
                  #[-5.2, -1.9]
                  ]
        
        for s in shifts:
            B = au.interpolated_shift(probe[0], s, do_linear=True)
            gB = gau.interpolated_shift(probe[0], s, do_linear=True)
            np.testing.assert_allclose(B, gB, rtol=1e-6, atol=1e-3, err_msg="failed for shifts {}".format(s))
    
    @unittest.skip("not implemented yet")
    def test_interpolated_shift_bicubic(self):
        npts = 32
        probe = np.zeros((1, npts, npts), dtype=COMPLEX_TYPE)
        rad = 10.0
        probe_vals = 2 + 3j
        x = np.array(range(npts)) - npts // 2
        X, Y = np.meshgrid(x, x)
        Xoff = 5.1
        Yoff = 2.2
        probe[0, (X-Xoff)**2 + (Y-Yoff)**2 < rad**2] = probe_vals
        
        shifts = [[-Xoff, -Yoff],
                  [2.3,1.3],
                  [-5.2, -1.9]]

        for s in shifts:
            B = au.interpolated_shift(probe[0], s)
            gB = gau.interpolated_shift(probe[0], s)
            np.testing.assert_allclose(B, gB, rtol=1e-6, atol=1e-3)

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
