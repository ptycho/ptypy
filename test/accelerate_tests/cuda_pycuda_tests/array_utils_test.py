'''


'''

import unittest
import numpy as np
from . import perfrun, PyCudaTest, have_pycuda
from ptypy.accelerate.base import array_utils as au

if have_pycuda():
    from pycuda import gpuarray
    import ptypy.accelerate.cuda_pycuda.array_utils as gau

class ArrayUtilsTest(PyCudaTest):

    def test_dot_float_float(self):
        ## Arrange
        X,Y,Z = np.indices((3,3,1001), dtype=np.float32)
        A = 10 ** Y
        A_dev = gpuarray.to_gpu(A)

        ## Act
        AU = gau.ArrayUtilsKernel(acc_dtype=np.float32)
        out_dev = AU.dot(A_dev, A_dev)
        out = out_dev.get()

        ## Assert
        np.testing.assert_allclose(out, 30333303.0, rtol=1e-7)

    def test_dot_float_double(self):
        ## Arrange
        X,Y,Z = np.indices((3,3,1001), dtype=np.float32)
        A = 10 ** Y
        A_dev = gpuarray.to_gpu(A)

        ## Act
        AU = gau.ArrayUtilsKernel(acc_dtype=np.float64)
        out_dev = AU.dot(A_dev, A_dev)
        out = out_dev.get()

        ## Assert
        np.testing.assert_equal(out, 30333303.0)
    
    def test_dot_complex_float(self):
        ## Arrange
        X,Y,Z = np.indices((3,3,1001), dtype=np.float32)
        A = 10 ** Y + 1j * 10 ** X
        A_dev = gpuarray.to_gpu(A)

        ## Act
        AU = gau.ArrayUtilsKernel(acc_dtype=np.float32)
        out_dev = AU.dot(A_dev, A_dev)
        out = out_dev.get()

        ## Assert
        np.testing.assert_allclose(out, 60666606.0, rtol=1e-7)

    def test_dot_complex_double(self):
        ## Arrange
        X,Y,Z = np.indices((3,3,1001), dtype=np.float32)
        A = 10 ** Y + 1j * 10 ** X
        A_dev = gpuarray.to_gpu(A)

        ## Act
        AU = gau.ArrayUtilsKernel(acc_dtype=np.float64)
        out_dev = AU.dot(A_dev, A_dev)
        out = out_dev.get()

        ## Assert
        np.testing.assert_array_equal(out, 60666606.0)

    @unittest.skipIf(not perfrun, "Performance test")
    def test_dot_performance(self):
        ## Arrange
        X,Y,Z = np.indices((3,3,1021301), dtype=np.float32)
        A = 10 ** Y + 1j * 10 ** X
        A_dev = gpuarray.to_gpu(A)

        ## Act
        AU = gau.ArrayUtilsKernel(acc_dtype=np.float64)
        out_dev = AU.dot(A_dev, A_dev)

    def test_transpose_2D(self):
        ## Arrange
        inp,_ = np.indices((5,3), dtype=np.int32)
        inp_dev = gpuarray.to_gpu(inp)
        out_dev = gpuarray.empty((3,5), dtype=np.int32)
        
        ## Act
        AU = gau.TransposeKernel()
        AU.transpose(inp_dev, out_dev)

        ## Assert
        out_exp = np.transpose(inp, (1, 0))
        out = out_dev.get()
        np.testing.assert_array_equal(out, out_exp)

    def test_transpose_2D_large(self):
        ## Arrange
        inp,_ = np.indices((137,61), dtype=np.int32)
        inp_dev = gpuarray.to_gpu(inp)
        out_dev = gpuarray.empty((61,137), dtype=np.int32)
        
        ## Act
        AU = gau.TransposeKernel()
        AU.transpose(inp_dev, out_dev)

        ## Assert
        out_exp = np.transpose(inp, (1, 0))
        out = out_dev.get()
        np.testing.assert_array_equal(out, out_exp)

    def test_transpose_4D(self):
        ## Arrange
        inp = np.random.randint(0, 10000, (250, 3, 5, 3), dtype=np.int32) # like addr
        inp_dev = gpuarray.to_gpu(inp)
        out_dev = gpuarray.empty((5, 3, 250, 3), dtype=np.int32)

        ## Act
        AU = gau.TransposeKernel()
        AU.transpose(inp_dev.reshape(750, 15), out_dev.reshape(15, 750))

        ## Assert
        out_exp = np.transpose(inp, (2, 3, 0, 1))
        out = out_dev.get()
        np.testing.assert_array_equal(out, out_exp)

    def test_complex_gaussian_filter_1d_no_blurring_UNITY(self):
        # Arrange
        inp = np.zeros((11,), dtype=np.complex64)
        inp[5] = 1.0 +1.0j
        mfs = [0]
        inp_dev = gpuarray.to_gpu(inp)
        out_dev = gpuarray.empty((11,), dtype=np.complex64)

        # Act
        GS = gau.GaussianSmoothingKernel()
        GS.convolution(inp_dev, out_dev, mfs)

        # Assert
        out_exp = au.complex_gaussian_filter(inp, mfs)
        out = out_dev.get()
        self.assertTrue(np.testing.assert_allclose(out_exp, out, rtol=1e-5) is None)

    def test_complex_gaussian_filter_1d_little_blurring_UNITY(self):
        # Arrange
        inp = np.zeros((11,), dtype=np.complex64)
        inp[5] = 1.0 +1.0j
        mfs = [0.2]
        inp_dev = gpuarray.to_gpu(inp)
        out_dev = gpuarray.empty((11,), dtype=np.complex64)

        # Act
        GS = gau.GaussianSmoothingKernel()
        GS.convolution(inp_dev, out_dev, mfs)

        # Assert
        out_exp = au.complex_gaussian_filter(inp, mfs)
        out = out_dev.get()
        np.testing.assert_allclose(out_exp, out, rtol=1e-5)

    
    def test_complex_gaussian_filter_1d_more_blurring_UNITY(self):
        # Arrange
        inp = np.zeros((11,), dtype=np.complex64)
        inp[5] = 1.0 +1.0j
        mfs = [2.0]
        inp_dev = gpuarray.to_gpu(inp)
        out_dev = gpuarray.empty((11,), dtype=np.complex64)

        # Act
        GS = gau.GaussianSmoothingKernel()
        GS.convolution(inp_dev, out_dev, mfs)

        # Assert
        out_exp = au.complex_gaussian_filter(inp, mfs)
        out = out_dev.get()
        np.testing.assert_allclose(out_exp, out, rtol=1e-5)

    def test_complex_gaussian_filter_2d_no_blurring_UNITY(self):
        # Arrange
        inp = np.zeros((11, 11), dtype=np.complex64)
        inp[5, 5] = 1.0+1.0j
        mfs = 0.0,0.0
        inp_dev = gpuarray.to_gpu(inp)
        out_dev = gpuarray.empty((11,11), dtype=np.complex64)

        # Act
        GS = gau.GaussianSmoothingKernel()
        GS.convolution(inp_dev, out_dev, mfs)

        # Assert
        out_exp = au.complex_gaussian_filter(inp, mfs)
        out = out_dev.get()
        np.testing.assert_allclose(out_exp, out, rtol=1e-5)

    def test_complex_gaussian_filter_2d_little_blurring_UNITY(self):
        # Arrange
        inp = np.zeros((11, 11), dtype=np.complex64)
        inp[5, 5] = 1.0+1.0j
        mfs = 0.2,0.2
        inp_dev = gpuarray.to_gpu(inp)
        out_dev = gpuarray.empty((11,11),dtype=np.complex64)

        # Act
        GS = gau.GaussianSmoothingKernel()
        GS.convolution(inp_dev, out_dev, mfs)

        # Assert
        out_exp = au.complex_gaussian_filter(inp, mfs)
        out = out_dev.get()
        np.testing.assert_allclose(out_exp, out, rtol=1e-5)

    def test_complex_gaussian_filter_2d_more_blurring_UNITY(self):
        # Arrange
        inp = np.zeros((8, 8), dtype=np.complex64)
        inp[3:5, 3:5] = 2.0+2.0j
        mfs = 3.0,4.0
        inp_dev = gpuarray.to_gpu(inp)
        out_dev = gpuarray.empty((8,8), dtype=np.complex64)

        # Act
        GS = gau.GaussianSmoothingKernel()
        GS.convolution(inp_dev, out_dev, mfs)

        # Assert
        out_exp = au.complex_gaussian_filter(inp, mfs)
        out = out_dev.get()
        np.testing.assert_allclose(out_exp, out, rtol=1e-4)

    def test_complex_gaussian_filter_2d_nonsquare_UNITY(self):
        # Arrange
        inp = np.zeros((32, 16), dtype=np.complex64)
        inp[3:4, 11:12] = 2.0+2.0j
        inp[3:5, 3:5] = 2.0+2.0j
        inp[20:25,3:5] = 2.0+2.0j
        mfs = 1.0,1.0
        inp_dev = gpuarray.to_gpu(inp)
        out_dev = gpuarray.empty(inp.shape, dtype=np.complex64)

        # Act
        GS = gau.GaussianSmoothingKernel()
        GS.convolution(inp_dev, out_dev, mfs)

        # Assert
        out_exp = au.complex_gaussian_filter(inp, mfs)
        out = out_dev.get()

        np.testing.assert_allclose(out_exp, out, rtol=1e-4)

    def test_complex_gaussian_filter_2d_batched(self):
        # Arrange
        batch_number = 2
        A = 5
        B = 5
        inp = np.zeros((batch_number, A, B), dtype=np.complex64)
        inp[:, 2:3, 2:3] = 2.0+2.0j
        mfs = 3.0,4.0
        inp_dev = gpuarray.to_gpu(inp)
        out_dev = gpuarray.empty((batch_number,A,B), dtype=np.complex64)

        # Act
        GS = gau.GaussianSmoothingKernel()
        GS.convolution(inp_dev, out_dev, mfs)

        # Assert
        out_exp = au.complex_gaussian_filter(inp, mfs)
        out = out_dev.get()        
        np.testing.assert_allclose(out_exp, out, rtol=1e-4)


    def test_crop_pad_simple_1_UNITY(self):
        # pad, integer, 2D
        B = np.indices((4, 4), dtype=np.int).sum(0)
        A = np.zeros((6, 6), dtype=B.dtype)
        B_dev = gpuarray.to_gpu(B)
        A_dev = gpuarray.to_gpu(A)

        # Act
        au.crop_pad_2d_simple(A, B)
        k = gau.CropPadKernel(queue=self.stream)
        k.crop_pad_2d_simple(A_dev, B_dev)

        # Assert
        np.testing.assert_allclose(A, A_dev.get(), rtol=1e-6, atol=1e-6)

    def test_crop_pad_simple_2_UNITY(self):
        # crop, float, 3D
        B = np.indices((4, 4), dtype=np.float32)
        A = np.zeros((2, 2, 2), dtype=B.dtype)
        B_dev = gpuarray.to_gpu(B)
        A_dev = gpuarray.to_gpu(A)

        # Act
        au.crop_pad_2d_simple(A, B)
        k = gau.CropPadKernel(queue=self.stream)
        k.crop_pad_2d_simple(A_dev, B_dev)


        # Assert
        np.testing.assert_allclose(A, A_dev.get(), rtol=1e-6, atol=1e-6)

    def test_crop_pad_simple_3_UNITY(self):
        # crop/pad, complex, 3D
        B = np.indices((4, 3), dtype=np.complex64)
        B = np.indices((4, 3), dtype=np.complex64) + 1j * B[::-1, :, :]
        A = np.zeros((2, 2, 5), dtype=B.dtype)
        B_dev = gpuarray.to_gpu(B)
        A_dev = gpuarray.to_gpu(A)

        # Act
        au.crop_pad_2d_simple(A, B)
        k = gau.CropPadKernel(queue=self.stream)
        k.crop_pad_2d_simple(A_dev, B_dev)

        # Assert
        np.testing.assert_allclose(A, A_dev.get(), rtol=1e-6, atol=1e-6)

    def test_crop_pad_simple_difflike_UNITY(self):
        np.random.seed(1983)
        # crop/pad, 4D
        D = np.random.randint(0, 3000, (100,256,256)).astype(np.float32)
        A = np.zeros((100,260,260), dtype=D.dtype)
        B = np.zeros((100,250,250), dtype=D.dtype)
        B_dev = gpuarray.to_gpu(B)
        A_dev = gpuarray.to_gpu(A)
        D_dev = gpuarray.to_gpu(D)

        # Act
        au.crop_pad_2d_simple(A, D)
        au.crop_pad_2d_simple(B, D)
        k = gau.CropPadKernel(queue=self.stream)
        k.crop_pad_2d_simple(A_dev, D_dev)
        k.crop_pad_2d_simple(B_dev, D_dev)

        # Assert
        np.testing.assert_allclose(A, A_dev.get(), rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(B, B_dev.get(), rtol=1e-6, atol=1e-6)

    def test_crop_pad_simple_oblike_UNITY(self):
        np.random.seed(1983)
        # crop/pad, 4D
        B = np.random.rand(2,1230,1434).astype(np.complex64) \
           +2j * np.pi * np.random.randn(2,1230,1434).astype(np.complex64)
        A = np.ones((2,1000,1500), dtype=B.dtype)
        B_dev = gpuarray.to_gpu(B)
        A_dev = gpuarray.to_gpu(A)

        # Act
        au.crop_pad_2d_simple(A, B)
        k = gau.CropPadKernel(queue=self.stream)
        k.crop_pad_2d_simple(A_dev, B_dev)

        # Assert
        np.testing.assert_allclose(A, A_dev.get(), rtol=1e-6, atol=1e-6)
