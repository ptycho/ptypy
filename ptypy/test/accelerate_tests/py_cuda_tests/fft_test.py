'''


'''

import unittest
import numpy as np

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
    from ptypy.accelerate.py_cuda.fft import FFT
    

COMPLEX_TYPE = np.complex64
FLOAT_TYPE = np.float32
INT_TYPE = np.int32


@unittest.skipIf(not have_pycuda(), "no PyCUDA or GPU drivers available")
class FftTest(unittest.TestCase):

    def setUp(self):
        print("Called setup")
        import sys
        np.set_printoptions(threshold=sys.maxsize, linewidth=np.inf)
        self.ctx = make_default_context()
        self.stream = cuda.Stream()

    def tearDown(self):
        print("Called teardown")
        np.set_printoptions()
        self.ctx.pop()
        self.ctx.detach()

    def test_fft_works_1(self):
        '''
        setup
        '''
        print("This one")
        B = 64 # frame size y
        C = 64  # frame size x

        D = 2  # number of probe modes
        G = 2  # number og object modes

        E = B  # probe size y
        F = C  # probe size x

        scan_pts = 2  # one dimensional scan point number

        N = scan_pts ** 2
        total_number_modes = G * D
        A =2226# N * total_number_modes  # this is a 16 point scan pattern (4x4 grid) over all the modes

        f = np.empty(shape=(A, B, C), dtype=COMPLEX_TYPE)
        for idx in range(A):
            f[idx] = np.ones((B, C)) * (idx + 1) + 1j * np.ones((B, C)) * (idx + 1)

        prefilter = (np.arange(B*C).reshape((B, C)) + 1j* np.arange(B*C).reshape((B, C))).astype(COMPLEX_TYPE)
        postfilter = (np.arange(13, 13 + B*C).reshape((B, C)) + 1j*np.arange(13, 13 + B*C).reshape((B, C))).astype(COMPLEX_TYPE)



        f_d = gpuarray.to_gpu(f)

        propagator_forward = FFT(f, self.stream, pre_fft=prefilter, post_fft=postfilter, inplace=True, symmetric=True)
        propagator_forward.ft(f_d, f_d)

        propagator_backward = FFT(f, self.stream, pre_fft=prefilter, post_fft=postfilter, inplace=True, symmetric=True)
        propagator_backward.ift(f_d, f_d)

        a = f_d.get()
        print("here")
        print(type(a))
        # np.testing.assert_array_equal(a, f)
        print("Freeing the mem")
        f_d.gpudata.free()
        print("done Freeing the mem")

    def test_fft_works_2(self):
        '''
        setup
        '''
        print("Now this one")
        B = 64 # frame size y
        C = 64  # frame size x

        D = 2  # number of probe modes
        G = 2  # number og object modes

        E = B  # probe size y
        F = C  # probe size x

        scan_pts = 2  # one dimensional scan point number

        N = scan_pts ** 2
        total_number_modes = G * D
        A =2226# N * total_number_modes  # this is a 16 point scan pattern (4x4 grid) over all the modes

        f = np.empty(shape=(A, B, C), dtype=COMPLEX_TYPE)
        for idx in range(A):
            f[idx] = np.ones((B, C)) * (idx + 1) + 1j * np.ones((B, C)) * (idx + 1)


        f_d = gpuarray.to_gpu(f)
        f_out = gpuarray.to_gpu(np.zeros_like(f))
        propagator_forward = FFT(f, self.stream, pre_fft=None, post_fft=None, inplace=True, symmetric=True)
        propagator_forward.ft(f_d, f_d)

        propagator_backward = FFT(f, self.stream, pre_fft=None, post_fft=None, inplace=True, symmetric=True)
        propagator_backward.ift(f_d, f_d)

        print("done with the ffts")

        a = f_d.get()
        print("here")
        print(type(a))
        # np.testing.assert_array_equal(a, f)
        print("Freeing the mem")
        f_d.gpudata.free()
        print("done Freeing the mem")

if __name__ == '__main__':
    unittest.main()
