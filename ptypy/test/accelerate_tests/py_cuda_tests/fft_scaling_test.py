'''


'''

import unittest
import numpy as np
from . import PyCudaTest, have_pycuda

if have_pycuda():
    from pycuda import gpuarray
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



class FftScalingTest(PyCudaTest):

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



if __name__ == '__main__':
    unittest.main()
