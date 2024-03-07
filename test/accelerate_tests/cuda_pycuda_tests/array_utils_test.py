'''


'''

import unittest
import numpy as np
from . import perfrun, PyCudaTest, have_pycuda
from ptypy.accelerate.base import array_utils as au

if have_pycuda():
    from pycuda import gpuarray
    import ptypy.accelerate.cuda_pycuda.array_utils as gau
    from ptypy.accelerate.cuda_pycuda.kernels import FFTFilterKernel

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

    def test_batched_multiply(self):
        # Arrange
        sh = (3,14,24)
        ksh = (14,24)
        data = (np.random.random(sh) + 1j* np.random.random(sh)).astype(np.complex64)
        kernel = (np.random.random(ksh) + 1j* np.random.random(ksh)).astype(np.complex64)
        data_dev = gpuarray.to_gpu(data)
        kernel_dev = gpuarray.to_gpu(kernel)

        # Act
        BM = gau.BatchedMultiplyKernel(data_dev)
        BM.multiply(data_dev, kernel_dev, scale=2.)

        # Assert
        expected = data * kernel * 2.
        np.testing.assert_array_almost_equal(data_dev.get(), expected)

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
        data = np.zeros((11,), dtype=np.complex64)
        data[5] = 1.0 +1.0j
        mfs = [0]
        data_dev = gpuarray.to_gpu(data)
        tmp_dev = gpuarray.empty((11,), dtype=np.complex64)

        # Act
        GS = gau.GaussianSmoothingKernel()
        GS.convolution(data_dev, mfs, tmp=tmp_dev)

        # Assert
        out_exp = au.complex_gaussian_filter(data, mfs)
        out = data_dev.get()
        self.assertTrue(np.testing.assert_allclose(out_exp, out, rtol=1e-5) is None)

    def test_complex_gaussian_filter_1d_little_blurring_UNITY(self):
        # Arrange
        data = np.zeros((11,), dtype=np.complex64)
        data[5] = 1.0 +1.0j
        mfs = [0.2]
        data_dev = gpuarray.to_gpu(data)
        tmp_dev = gpuarray.empty((11,), dtype=np.complex64)

        # Act
        GS = gau.GaussianSmoothingKernel()
        GS.convolution(data_dev, mfs, tmp=tmp_dev)

        # Assert
        out_exp = au.complex_gaussian_filter(data, mfs)
        out = data_dev.get()
        np.testing.assert_allclose(out_exp, out, rtol=1e-5)


    def test_complex_gaussian_filter_1d_more_blurring_UNITY(self):
        # Arrange
        data = np.zeros((11,), dtype=np.complex64)
        data[5] = 1.0 +1.0j
        mfs = [2.0]
        data_dev = gpuarray.to_gpu(data)
        tmp_dev = gpuarray.empty((11,), dtype=np.complex64)

        # Act
        GS = gau.GaussianSmoothingKernel()
        GS.convolution(data_dev, mfs, tmp=tmp_dev)

        # Assert
        out_exp = au.complex_gaussian_filter(data, mfs)
        out = data_dev.get()
        np.testing.assert_allclose(out_exp, out, rtol=1e-5)

    def test_complex_gaussian_filter_2d_no_blurring_UNITY(self):
        # Arrange
        data = np.zeros((11, 11), dtype=np.complex64)
        data[5, 5] = 1.0+1.0j
        mfs = 0.0,0.0
        data_dev = gpuarray.to_gpu(data)
        tmp_dev = gpuarray.empty((11,11), dtype=np.complex64)

        # Act
        GS = gau.GaussianSmoothingKernel()
        GS.convolution(data_dev, mfs, tmp=tmp_dev)

        # Assert
        out_exp = au.complex_gaussian_filter(data, mfs)
        out = data_dev.get()
        np.testing.assert_allclose(out_exp, out, rtol=1e-5)

    def test_complex_gaussian_filter_2d_little_blurring_UNITY(self):
        # Arrange
        data = np.zeros((11, 11), dtype=np.complex64)
        data[5, 5] = 1.0+1.0j
        mfs = 0.2,0.2
        data_dev = gpuarray.to_gpu(data)
        tmp_dev = gpuarray.empty((11,11),dtype=np.complex64)

        # Act
        GS = gau.GaussianSmoothingKernel()
        GS.convolution(data_dev, mfs, tmp=tmp_dev)

        # Assert
        out_exp = au.complex_gaussian_filter(data, mfs)
        out = data_dev.get()
        np.testing.assert_allclose(out_exp, out, rtol=1e-5)

    def test_complex_gaussian_filter_2d_more_blurring_UNITY(self):
        # Arrange
        data = np.zeros((8, 8), dtype=np.complex64)
        data[3:5, 3:5] = 2.0+2.0j
        mfs = 3.0,4.0
        data_dev = gpuarray.to_gpu(data)
        #tmp_dev = gpuarray.empty((8,8), dtype=np.complex64)

        # Act
        GS = gau.GaussianSmoothingKernel()
        GS.convolution(data_dev, mfs)

        # Assert
        out_exp = au.complex_gaussian_filter(data, mfs)
        out = data_dev.get()
        np.testing.assert_allclose(out_exp, out, rtol=1e-4)

    def test_complex_gaussian_filter_2d_nonsquare_UNITY(self):
        # Arrange
        data = np.zeros((32, 16), dtype=np.complex64)
        data[3:4, 11:12] = 2.0+2.0j
        data[3:5, 3:5] = 2.0+2.0j
        data[20:25,3:5] = 2.0+2.0j
        mfs = 1.0,1.0
        data_dev = gpuarray.to_gpu(data)
        tmp_dev = gpuarray.empty(data_dev.shape, dtype=np.complex64)

        # Act
        GS = gau.GaussianSmoothingKernel()
        GS.convolution(data_dev, mfs, tmp=tmp_dev)

        # Assert
        out_exp = au.complex_gaussian_filter(data, mfs)
        out = data_dev.get()

        np.testing.assert_allclose(out_exp, out, rtol=1e-4)

    def test_complex_gaussian_filter_2d_batched(self):
        # Arrange
        batch_number = 2
        A = 5
        B = 5
        data = np.zeros((batch_number, A, B), dtype=np.complex64)
        data[:, 2:3, 2:3] = 2.0+2.0j
        mfs = 3.0,4.0
        data_dev = gpuarray.to_gpu(data)
        tmp_dev = gpuarray.empty((batch_number,A,B), dtype=np.complex64)

        # Act
        GS = gau.GaussianSmoothingKernel()
        GS.convolution(data_dev, mfs, tmp=tmp_dev)

        # Assert
        out_exp = au.complex_gaussian_filter(data, mfs)
        out = data_dev.get()
        np.testing.assert_allclose(out_exp, out, rtol=1e-4)


    def test_crop_pad_simple_1_UNITY(self):
        # pad, integer, 2D
        B = np.indices((4, 4), dtype=np.int32).sum(0)
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

    def test_max_abs2_complex_UNITY(self):
        np.random.seed(1983)
        X = (np.random.randint(-1000, 1000, (3,100,200)).astype(np.float32) + \
            1j * np.random.randint(-1000, 1000, (3,100,200)).astype(np.float32)).astype(np.complex64)
        out = np.zeros((1,), dtype=np.float32)
        X_dev = gpuarray.to_gpu(X)
        out_dev = gpuarray.to_gpu(out)

        out = au.max_abs2(X)

        MAK = gau.MaxAbs2Kernel(queue=self.stream)
        MAK.max_abs2(X_dev, out_dev)

        np.testing.assert_allclose(out_dev.get(), out, rtol=1e-6, atol=1e-6,
            err_msg="The object norm array has not been updated as expected")

    def test_max_abs2_float_UNITY(self):
        np.random.seed(1983)
        X = np.random.randint(-1000, 1000, (3,100,200)).astype(np.float32)

        out = np.zeros((1,), dtype=np.float32)
        X_dev = gpuarray.to_gpu(X)
        out_dev = gpuarray.to_gpu(out)

        out = au.max_abs2(X)

        MAK = gau.MaxAbs2Kernel(queue=self.stream)
        MAK.max_abs2(X_dev, out_dev)

        np.testing.assert_allclose(out_dev.get(), out, rtol=1e-6, atol=1e-6,
            err_msg="The object norm array has not been updated as expected")


    def test_clip_magnitudes_to_range_UNITY(self):
        np.random.seed(1987)
        A = np.random.random((2,10,10))
        B = A[0] + 1j* A[1]
        B = B.astype(np.complex64)
        B_gpu = gpuarray.to_gpu(B)

        au.clip_complex_magnitudes_to_range(B, 0.2,0.8)
        CMK = gau.ClipMagnitudesKernel()
        CMK.clip_magnitudes_to_range(B_gpu, 0.2, 0.8)

        np.testing.assert_allclose(B_gpu.get(), B, rtol=1e-6, atol=1e-6,
            err_msg="The magnitudes of the array have not been clipped as expected")


    def test_mass_center_2d_UNITY(self):
        np.random.seed(1987)
        A = np.random.random((128, 128)).astype(np.float32)
        A_gpu = gpuarray.to_gpu(A)

        out = au.mass_center(A)

        MCK = gau.MassCenterKernel()
        mc_d = MCK.mass_center(A_gpu)
        mc = mc_d.get()

        np.testing.assert_allclose(out, mc, rtol=1e-6, atol=1e-6,
            err_msg="The centre of mass of the array has not been calculated as expected")


    def test_mass_center_3d_UNITY(self):
        np.random.seed(1987)
        A = np.random.random((128, 128, 128)).astype(np.float32)
        A_gpu = gpuarray.to_gpu(A)

        out = au.mass_center(A)

        MCK = gau.MassCenterKernel()
        mc_d = MCK.mass_center(A_gpu)
        mc = mc_d.get()

        np.testing.assert_allclose(out, mc, rtol=1e-6, atol=1e-6,
            err_msg="The centre of mass of the array has not been calculated as expected")

    def test_abs2sum_complex_float_UNITY(self):
        np.random.seed(1987)
        A = np.random.random((3, 321, 123)).astype(np.float32)
        B = A + A**2 * 1j
        B_gpu = gpuarray.to_gpu(B)

        out = au.abs2(B).sum(0)

        A2SK = gau.Abs2SumKernel(dtype=B_gpu.dtype)
        a2s_d = A2SK.abs2sum(B_gpu)
        a2s = a2s_d.get()

        np.testing.assert_allclose(out, a2s, rtol=1e-6, atol=1e-6,
            err_msg="The sum of absolute values along the first dimension has not been calculated as expected")

    def test_abs2sum_complex_double_UNITY(self):
        np.random.seed(1987)
        A = np.random.random((3, 321, 123)).astype(np.float64)
        B = A + A**2 * 1j
        B_gpu = gpuarray.to_gpu(B)

        out = au.abs2(B).sum(0)

        A2SK = gau.Abs2SumKernel(dtype=B_gpu.dtype)
        a2s_d = A2SK.abs2sum(B_gpu)
        a2s = a2s_d.get()

        np.testing.assert_allclose(out, a2s, rtol=1e-6, atol=1e-6,
            err_msg="The sum of absolute values along the first dimension has not been calculated as expected")

    def test_interpolate_shift_2D_UNITY(self):
        np.random.seed(1987)
        A = np.random.random((259, 252)).astype(np.float32)
        A = A + A**2 * 1j
        A_gpu = gpuarray.to_gpu(A)

        cen_old = np.array([100.123, 5.678]).astype(np.float32)
        cen_new = np.array([128.5, 127.5]).astype(np.float32)
        shift = cen_new - cen_old

        out = au.interpolated_shift(A, shift, do_linear=True)

        ISK = gau.InterpolatedShiftKernel()
        isk_d = ISK.interpolate_shift(A_gpu, shift)
        isk = isk_d.get()

        np.testing.assert_allclose(out, isk, rtol=1e-6, atol=1e-6,
            err_msg="The shifting of array has not been calculated as expected")

    def test_interpolate_shift_3D_UNITY(self):
        np.random.seed(1987)
        A = np.random.random((3, 200, 300)).astype(np.float32)
        A = A + A**2 * 1j
        A_gpu = gpuarray.to_gpu(A)

        cen_old = np.array([0., 180.123, 5.678]).astype(np.float32)
        cen_new = np.array([0., 128.5, 127.5]).astype(np.float32)
        shift = cen_new - cen_old

        out = au.interpolated_shift(A, shift, do_linear=True)

        ISK = gau.InterpolatedShiftKernel()
        isk_d = ISK.interpolate_shift(A_gpu, shift[1:])
        isk = isk_d.get()

        np.testing.assert_allclose(out, isk, rtol=1e-6, atol=1e-6,
            err_msg="The shifting of array has not been calculated as expected")

    def test_interpolate_shift_integer_UNITY(self):
        np.random.seed(1987)
        A = np.random.random((3, 200, 300)).astype(np.float32)
        A = A + A**2 * 1j
        A_gpu = gpuarray.to_gpu(A)

        cen_old = np.array([0, 180, 5]).astype(np.float32)
        cen_new = np.array([0, 128, 127]).astype(np.float32)
        shift = cen_new - cen_old

        out = au.interpolated_shift(A, shift, do_linear=True)

        ISK = gau.InterpolatedShiftKernel()
        isk_d = ISK.interpolate_shift(A_gpu, shift[1:])
        isk = isk_d.get()

        np.testing.assert_allclose(out, isk, rtol=1e-6, atol=1e-6,
            err_msg="The shifting of array has not been calculated as expected")

    def test_interpolate_shift_no_shift_UNITY(self):
        np.random.seed(1987)
        A = np.random.random((3, 200, 300)).astype(np.float32)
        A = A + A**2 * 1j
        A_gpu = gpuarray.to_gpu(A)

        cen_old = np.array([0, 0, 0]).astype(np.float32)
        cen_new = np.array([0, 0, 0]).astype(np.float32)
        shift = cen_new - cen_old

        out = au.interpolated_shift(A, shift, do_linear=True)

        ISK = gau.InterpolatedShiftKernel()
        isk_d = ISK.interpolate_shift(A_gpu, shift[1:])
        isk = isk_d.get()

        np.testing.assert_allclose(out, isk, rtol=1e-6, atol=1e-6,
            err_msg="The shifting of array has not been calculated as expected")

    def test_fft_filter_UNITY(self):
        sh = (16, 35)
        data = np.zeros(sh, dtype=np.complex64)
        data.flat[:] = np.arange(np.prod(sh))
        kernel = np.zeros_like(data)
        kernel[0, 0] = 1.
        kernel[0, 1] = 0.5

        prefactor = np.zeros_like(data)
        prefactor[:,2:] = 1.
        postfactor = np.zeros_like(data)
        postfactor[2:,:] = 1.

        data_dev = gpuarray.to_gpu(data)
        kernel_dev = gpuarray.to_gpu(kernel)
        pre_dev = gpuarray.to_gpu(prefactor)
        post_dev = gpuarray.to_gpu(postfactor)

        FF = FFTFilterKernel(queue_thread=self.stream)
        FF.allocate(kernel=kernel_dev, prefactor=pre_dev, postfactor=post_dev)
        FF.apply_filter(data_dev)

        output = au.fft_filter(data, kernel, prefactor, postfactor)

        np.testing.assert_allclose(output, data_dev.get(), rtol=1e-5, atol=1e-6)

    def test_fft_filter_batched_UNITY(self):
        sh = (2,16, 35)
        data = np.zeros(sh, dtype=np.complex64)
        data.flat[:] = np.arange(np.prod(sh))
        kernel = np.zeros_like(data)
        kernel[:,0, 0] = 1.
        kernel[:,0, 1] = 0.5

        prefactor = np.zeros_like(data)
        prefactor[:,:,2:] = 1.
        postfactor = np.zeros_like(data)
        postfactor[:,2:,:] = 1.

        data_dev = gpuarray.to_gpu(data)
        kernel_dev = gpuarray.to_gpu(kernel)
        pre_dev = gpuarray.to_gpu(prefactor)
        post_dev = gpuarray.to_gpu(postfactor)

        FF = FFTFilterKernel(queue_thread=self.stream)
        FF.allocate(kernel=kernel_dev, prefactor=pre_dev, postfactor=post_dev)
        FF.apply_filter(data_dev)

        output = au.fft_filter(data, kernel, prefactor, postfactor)
        print(data_dev.get())

        np.testing.assert_allclose(output, data_dev.get(), rtol=1e-5, atol=1e-6)

    def test_complex_gaussian_filter_fft_little_blurring_UNITY(self):
        # Arrange
        data = np.zeros((21, 21), dtype=np.complex64)
        data[10:12, 10:12] = 2.0+2.0j
        mfs = 0.2,0.2
        data_dev = gpuarray.to_gpu(data)

        # Act
        FGSK = gau.FFTGaussianSmoothingKernel(queue=self.stream)
        FGSK.filter(data_dev, mfs)

        # Assert
        out_exp = au.complex_gaussian_filter_fft(data, mfs)
        out = data_dev.get()
        
        np.testing.assert_allclose(out_exp, out, atol=1e-6)

    def test_complex_gaussian_filter_fft_more_blurring_UNITY(self):
        # Arrange
        data = np.zeros((8, 8), dtype=np.complex64)
        data[3:5, 3:5] = 2.0+2.0j
        mfs = 3.0,4.0
        data_dev = gpuarray.to_gpu(data)

        # Act
        FGSK = gau.FFTGaussianSmoothingKernel(queue=self.stream)
        FGSK.filter(data_dev, mfs)

        # Assert
        out_exp = au.complex_gaussian_filter_fft(data, mfs)
        out = data_dev.get()

        np.testing.assert_allclose(out_exp, out, atol=1e-6)

    def test_complex_gaussian_filter_fft_nonsquare_UNITY(self):
        # Arrange
        data = np.zeros((32, 16), dtype=np.complex64)
        data[3:4, 11:12] = 2.0+2.0j
        data[3:5, 3:5] = 2.0+2.0j
        data[20:25,3:5] = 2.0+2.0j
        mfs = 1.0,1.0
        data_dev = gpuarray.to_gpu(data)

        # Act
        FGSK = gau.FFTGaussianSmoothingKernel(queue=self.stream)
        FGSK.filter(data_dev, mfs)

        # Assert
        out_exp = au.complex_gaussian_filter_fft(data, mfs)
        out = data_dev.get()

        np.testing.assert_allclose(out_exp, out, atol=1e-6)

    def test_complex_gaussian_filter_fft_batched(self):
        # Arrange
        batch_number = 2
        A = 5
        B = 5
        data = np.zeros((batch_number, A, B), dtype=np.complex64)
        data[:, 2:3, 2:3] = 2.0+2.0j
        mfs = 3.0,4.0
        data_dev = gpuarray.to_gpu(data)

        # Act
        FGSK = gau.FFTGaussianSmoothingKernel(queue=self.stream)
        FGSK.filter(data_dev, mfs)

        # Assert
        out_exp = au.complex_gaussian_filter_fft(data, mfs)
        out = data_dev.get()

        np.testing.assert_allclose(out_exp, out, atol=1e-6)