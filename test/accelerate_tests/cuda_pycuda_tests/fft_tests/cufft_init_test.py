
import unittest
from test.accelerate_tests.cuda_pycuda_tests import PyCudaTest, have_pycuda

if have_pycuda():
    from filtered_cufft import FilteredFFT

class CuFFTInitTest(PyCudaTest):

    def test_import_fft(self):
        ft = FilteredFFT(2, 32, 32, False, True, 0, 0, 0)
    
    
    def test_import_fft_different_shape(self):
        ft = FilteredFFT(2, 128, 128, False, True, 0, 0, 0)
    

    @unittest.expectedFailure
    def test_import_fft_not_square(self):
        ft = FilteredFFT(2, 32, 64, False, True, 0, 0, 0)
    
    @unittest.expectedFailure
    def test_import_fft_not_pow2(self):
        ft = FilteredFFT(2, 40, 40, False, True, 0, 0, 0)


if __name__=="__main__":
    unittest.main()
