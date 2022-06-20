'''
'''

import unittest
import numpy as np
import scipy.fft as fft
from test.accelerate_tests.cuda_pycuda_tests import PyCudaTest, have_pycuda


if have_pycuda():
    from pycuda import gpuarray
    from ptypy.accelerate.cuda_pycuda.fft import FFT as ReiknaFFT
    from ptypy.accelerate.cuda_pycuda.cufft import FFT_cuda as cuFFT

class FftAccurracyTest(PyCudaTest):

    def gen_input(self):
        rows = cols = 32
        batches = 1
        f = np.random.randn(batches, rows, cols) + 1j * np.random.randn(batches,rows, cols)
        f = np.ascontiguousarray(f.astype(np.complex64))
        return f

    def test_random_cufft_fwd(self):
        f = self.gen_input()
        cuft = cuFFT(f, self.stream, inplace=True, pre_fft=None, post_fft=None, symmetric=None, forward=True).ft
        reikft = ReiknaFFT(f, self.stream, inplace=True, pre_fft=None, post_fft=None, symmetric=False).ft
        for i in range(10):
            f = self.gen_input()
            y = fft.fft2(f)

            x_d = gpuarray.to_gpu(f)
            cuft(x_d, x_d)
            y_cufft = x_d.get().reshape(y.shape)

            x_d = gpuarray.to_gpu(f)
            reikft(x_d, x_d)
            y_reikna = x_d.get().reshape(y.shape)

            # cufft_diff = np.max(np.abs(y_cufft - y))
            # reikna_diff = np.max(np.abs(y_reikna-y))
            # cufft_rdiff = np.max(np.abs(y_cufft - y) / np.abs(y))
            # reikna_rdiff = np.max(np.abs(y_reikna - y) / np.abs(y))
            # print('{}: {}\t{}\t{}\t{}'.format(i, cufft_diff, reikna_diff, cufft_rdiff, reikna_rdiff))
        
            # Note: check if this tolerance and test case is ok
            np.testing.assert_allclose(y, y_cufft, atol=1e-6, rtol=5e-5, err_msg='cuFFT error at index {}'.format(i))
            np.testing.assert_allclose(y, y_reikna, atol=1e-6, rtol=5e-5, err_msg='reikna FFT error at index {}'.format(i))
