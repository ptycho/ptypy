
import unittest
from test.accelerate_tests.cuda_pycuda_tests import PyCudaTest, have_pycuda

if have_pycuda():
    from ptypy.accelerate.cuda_pycuda import import_fft

class ImportFFTTest(PyCudaTest):

    def test_import_fft(self):
        mod = import_fft.ImportFFT().get_mod()
        ft = mod.FilteredFFT(2, 32, 32, False, True, 0, 0, 0)
    
    
    def test_import_fft_different_shape(self):
        mod = import_fft.ImportFFT(quiet=False).get_mod()
        ft = mod.FilteredFFT(2, 128, 128, False, True, 0, 0, 0)
    
    def test_import_fft_same_module_again(self):
        mod = import_fft.ImportFFT().get_mod()
        ft = mod.FilteredFFT(2, 32, 32, False, True, 0, 0, 0)

    @unittest.expectedFailure
    def test_import_fft_not_square(self):
        mod = import_fft.ImportFFT().get_mod()
        ft = mod.FilteredFFT(2, 32, 64, False, True, 0, 0, 0)
    
    @unittest.expectedFailure
    def test_import_fft_not_pow2(self):
        mod = import_fft.ImportFFT().get_mod()
        ft = mod.FilteredFFT(2, 40, 40, False, True, 0, 0, 0)


if __name__=="__main__":
    unittest.main()
