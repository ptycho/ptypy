
import unittest, pytest
from test.accelerate_tests.py_cuda_tests import PyCudaTest, have_pycuda
import os, shutil
from distutils import sysconfig

if have_pycuda():
    import pycuda.driver as cuda
    from pycuda import gpuarray
    from ptypy.accelerate.cuda_pycuda import import_fft
    from pycuda.tools import make_default_context

class ImportFFTTest(PyCudaTest):

    def test_import_fft(self):
        import_fft.ImportFFT(32, 32)
    
    
    def test_import_fft_different_shape(self):
        import_fft.ImportFFT(128, 128)
    
    def test_import_fft_same_module_again(self):
        import_fft.ImportFFT(32, 32)


if __name__=="__main__":
    unittest.main()
