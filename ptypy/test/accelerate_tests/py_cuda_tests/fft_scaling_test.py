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
    from ptypy.accelerate.py_cuda.fft import FFT as ReiknaFFT
    from ptypy.accelerate.py_cuda.cufft import FFT as cuFFT
   

COMPLEX_TYPE = np.complex64
FLOAT_TYPE = np.float32
INT_TYPE = np.int32


def get_forward_cuFFT(f, stream,
                      pre_fft, post_fft, inplace, 
                      symmetric):
    return cuFFT(f, stream,
               pre_fft=pre_fft, post_fft=post_fft, inplace=inplace, symmetric=symmetric, forward=True).ft

def get_reverse_cuFFT(f, stream,
                      pre_fft, post_fft, inplace, 
                      symmetric):
    return cuFFT(f, stream,
               pre_fft=pre_fft, post_fft=post_fft, inplace=inplace, symmetric=symmetric, forward=False).ift

def get_forward_Reikna(f, stream,
                      pre_fft, post_fft, inplace, 
                      symmetric):
    return ReiknaFFT(f, stream,
               pre_fft=pre_fft, post_fft=post_fft, inplace=inplace, symmetric=symmetric).ft

def get_reverse_Reikna(f, stream,
                      pre_fft, post_fft, inplace, 
                      symmetric):
    return ReiknaFFT(f, stream,
               pre_fft=pre_fft, post_fft=post_fft, inplace=inplace, symmetric=symmetric).ift



@unittest.skipIf(not have_pycuda(), "no PyCUDA or GPU drivers available")
class FftScalingTest(unittest.TestCase):

    def setUp(self):
        import sys
        np.set_printoptions(threshold=sys.maxsize, linewidth=np.inf)
        cuda.init()
        self.ctx = make_default_context()
        self.stream = cuda.Stream()

    def tearDown(self):
        np.set_printoptions()
        self.ctx.pop()
        self.ctx.detach()

    def get_input(self):
        rows = cols = 32
        batches = 1
        f = np.ones(shape=(batches, rows, cols), dtype=COMPLEX_TYPE)
        return f

    #### Trivial foward transform tests ####

    def fwd_test(self, symmetric, factory, preffact=None, postfact=None):
        f = self.get_input()
        f_d = gpuarray.to_gpu(f)
        if preffact is not None:
            pref = preffact * np.ones(shape=f.shape[-2:], dtype=np.complex64)
            pref_d = gpuarray.to_gpu(pref)
        else:
            preffact=1.0
            pref_d = None
        if postfact is not None:
            post = postfact * np.ones(shape=f.shape[-2:], dtype=np.complex64)
            post_d = gpuarray.to_gpu(post)
        else:
            postfact=1.0
            post_d = None
        ft = factory(f, self.stream,
                  pre_fft=pref_d, post_fft=post_d, inplace=True, 
                symmetric=symmetric)
        ft(f_d, f_d)
        f_back = f_d.get()
        elements = f.shape[-2] * f.shape[-1]
        scale = 1.0 if not symmetric else 1.0 / np.sqrt(elements)
        expected = elements * scale * preffact * postfact
        self.assertAlmostEqual(f_back[0,0,0], expected)
        np.testing.assert_array_almost_equal(f_back.flat[1:], 0)

    def test_fwd_noscale_reikna(self):
        self.fwd_test(False, get_forward_Reikna)
    
    def test_fwd_noscale_cufft(self):
        self.fwd_test(False, get_forward_cuFFT)

    def test_fwd_scale_reikna(self):
        self.fwd_test(True, get_forward_Reikna)
    
    def test_fwd_scale_cufft(self):
        self.fwd_test(True, get_forward_cuFFT)

    def test_prefilt_fwd_noscale_reikna(self):
        self.fwd_test(False, get_forward_Reikna, preffact=2.0)
    
    def test_prefilt_fwd_noscale_cufft(self):
        self.fwd_test(False, get_forward_cuFFT, preffact=2.0)

    def test_prefilt_fwd_scale_reikna(self):
        self.fwd_test(True, get_forward_Reikna, preffact=2.0)
    
    def test_prefilt_fwd_scale_cufft(self):
        self.fwd_test(True, get_forward_cuFFT, preffact=2.0)

    def test_postfilt_fwd_noscale_reikna(self):
        self.fwd_test(False, get_forward_Reikna, postfact=2.0)
    
    def test_postfilt_fwd_noscale_cufft(self):
        self.fwd_test(False, get_forward_cuFFT, postfact=2.0)

    def test_postfilt_fwd_scale_reikna(self):
        self.fwd_test(True, get_forward_Reikna, postfact=2.0)
    
    def test_postfilt_fwd_scale_cufft(self):
        self.fwd_test(True, get_forward_cuFFT, postfact=2.0)

    def test_prepostfilt_fwd_noscale_reikna(self):
        self.fwd_test(False, get_forward_Reikna, postfact=2.0, preffact=1.5)
    
    def test_prepostfilt_fwd_noscale_cufft(self):
        self.fwd_test(False, get_forward_cuFFT, postfact=2.0, preffact=1.5)

    def test_prepostfilt_fwd_scale_reikna(self):
        self.fwd_test(True, get_forward_Reikna, postfact=2.0, preffact=1.5)
    
    def test_prepostfilt_fwd_scale_cufft(self):
        self.fwd_test(True, get_forward_cuFFT, postfact=2.0, preffact=1.5)


    ############# Trivial inverse transform tests #########

    def rev_test(self, symmetric, factory, preffact=None, postfact=None):
        f = self.get_input()
        f_d = gpuarray.to_gpu(f)
        if preffact is not None:
            pref = preffact * np.ones(shape=f.shape[-2:], dtype=np.complex64)
            pref_d = gpuarray.to_gpu(pref)
        else:
            preffact=1.0
            pref_d = None
        if postfact is not None:
            post = postfact * np.ones(shape=f.shape[-2:], dtype=np.complex64)
            post_d = gpuarray.to_gpu(post)
        else:
            postfact=1.0
            post_d = None
        ift = factory(f, self.stream,
                pre_fft=pref_d, post_fft=post_d, inplace=True, symmetric=symmetric)
        ift(f_d, f_d)
        f_back = f_d.get()
        elements = f.shape[-2] * f.shape[-1]
        scale = 1.0 if not symmetric else np.sqrt(elements)
        expected = scale * preffact * postfact
        self.assertAlmostEqual(f_back[0,0,0], expected)
        np.testing.assert_array_almost_equal(f_back.flat[1:], 0)


    def test_rev_noscale_reikna(self):
        self.rev_test(False, get_reverse_Reikna)

    def test_rev_noscale_cufft(self):
        self.rev_test(False, get_reverse_cuFFT)

    def test_rev_scale_reikna(self):
        self.rev_test(True, get_reverse_Reikna)

    def test_rev_scale_cufft(self):
        self.rev_test(True, get_reverse_cuFFT)

    def test_prefilt_rev_noscale_reikna(self):
        self.rev_test(False, get_reverse_Reikna, preffact=1.5)

    def test_prefilt_rev_noscale_cufft(self):
        self.rev_test(False, get_reverse_cuFFT, preffact=1.5)

    def test_prefilt_rev_scale_reikna(self):
        self.rev_test(True, get_reverse_Reikna, preffact=1.5)

    def test_prefilt_rev_scale_cufft(self):
        self.rev_test(True, get_reverse_cuFFT, preffact=1.5)
    
    def test_postfilt_rev_noscale_reikna(self):
        self.rev_test(False, get_reverse_Reikna, postfact=1.5)

    def test_postfilt_rev_noscale_cufft(self):
        self.rev_test(False, get_reverse_cuFFT, postfact=1.5)

    def test_postfilt_rev_scale_reikna(self):
        self.rev_test(True, get_reverse_Reikna, postfact=1.5)

    def test_postfilt_rev_scale_cufft(self):
        self.rev_test(True, get_reverse_cuFFT, postfact=1.5)

    def test_prepostfilt_rev_noscale_reikna(self):
        self.rev_test(False, get_reverse_Reikna, postfact=1.5, preffact=2.0)

    def test_prepostfilt_rev_noscale_cufft(self):
        self.rev_test(False, get_reverse_cuFFT, postfact=1.5, preffact=2.0)

    def test_prepostfilt_rev_scale_reikna(self):
        self.rev_test(True, get_reverse_Reikna, postfact=1.5, preffact=2.0)

    def test_prepostfilt_rev_scale_cufft(self):
        self.rev_test(True, get_reverse_cuFFT, postfact=1.5, preffact=2.0)



"""     def test_fft_works_1(self):
        '''
        setup
        '''
        
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

        propagator_forward = ReiknaFFT(f, self.stream, pre_fft=prefilter, post_fft=postfilter, inplace=True, symmetric=True)
        propagator_forward.ft(f_d, f_d)

        propagator_backward = ReikanFFT(f, self.stream, pre_fft=prefilter, post_fft=postfilter, inplace=True, symmetric=True)
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
 """
if __name__ == '__main__':
    unittest.main()
