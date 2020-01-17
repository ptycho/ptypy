'''
'''

import unittest
import numpy as np
import scipy.fft as fft

def have_pycuda():
    try:
        import pycuda.driver
        return True
    except:
        return False

if have_pycuda():
    import pycuda.driver as cuda
    from pycuda import gpuarray
    from pycuda.tools import make_default_context
    from ptypy.accelerate.py_cuda.fft import FFT as ReiknaFFT
    from ptypy.accelerate.py_cuda.cufft import FFT as cuFFT
   

COMPLEX_TYPE = np.complex64
FLOAT_TYPE = np.float32
INT_TYPE = np.int32

@unittest.skipIf(not have_pycuda(), "no PyCUDA or GPU drivers available")
class FftAccurracyTest(unittest.TestCase):

    def setUp(self):
        import sys
        np.set_printoptions(threshold=sys.maxsize, linewidth=np.inf)
        self.ctx = make_default_context()
        self.stream = cuda.Stream()

    def tearDown(self):
        np.set_printoptions()
        self.ctx.pop()
        self.ctx.detach()

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

            if False:
                cufft_diff = np.max(np.abs(y_cufft - y))
                reikna_diff = np.max(np.abs(y_reikna-y))
                cufft_rdiff = np.max(np.abs(y_cufft - y) / np.abs(y))
                reikna_rdiff = np.max(np.abs(y_reikna - y) / np.abs(y))
                print('{}: {}\t{}\t{}\t{}'.format(i, cufft_diff, reikna_diff, cufft_rdiff, reikna_rdiff))
            
            # Note: check if this tolerance and test case is ok
            np.testing.assert_allclose(y, y_cufft, rtol=5e-5, err_msg='cuFFT error at index {}'.format(i))
            np.testing.assert_allclose(y, y_reikna, rtol=5e-5, err_msg='reikna FFT error at index {}'.format(i))
