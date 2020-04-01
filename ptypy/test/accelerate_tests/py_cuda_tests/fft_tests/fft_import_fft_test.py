
import unittest
from ptypy.test.accelerate_tests.py_cuda_tests import PyCudaTest, have_pycuda
from ptypy.accelerate.py_cuda import import_fft
import os, shutil
from pycuda.tools import make_default_context
from distutils import sysconfig

if have_pycuda():
    import pycuda.driver as cuda
    from pycuda import gpuarray

class ImportFFTTest(PyCudaTest):

    def test_import_fft(self):
        import_fft.import_fft(32, 32)
    
    
    def test_import_fft_twice(self):
        import_fft.import_fft(128, 128)
    
    def test_import_fft_twice_again(self):
        import_fft.import_fft(32, 32)

    # def test_32_32(self):
    #     rows = columns = 32
    #     module_name = 'module_' + str(rows) + '_' + str(columns)
    #     dirname = os.path.join("/home/clb02321/PycharmProjects/ptypy_accelerate_fresh/build/lib/ptypy/accelerate/py_cuda", 'cuda', 'filtered_fft')
    #     dst  = os.path.join(dirname, module_name)
    #     src = dirname # os.path.join(dirname, 'module.cpp')
    #     # dst = os.path.join(dirname, module_name + '.cpp')
    #     shutil.copytree(src, dst)
    #     # print('copies {} to {}'.format(src, dst))
    #
    #     # monkey-patch the customize_compiler function
    #     old = sysconfig.customize_compiler
    #     sysconfig.customize_compiler = get_customize_compiler(rows, columns, old)
    #
    #     import cppimport
#     # cppimport.force_rebuild()
    #     # cppimport.set_quiet(True)
    #     # cppimport
    #     import_module_name = "ptypy.accelerate.py_cuda.cuda.filtered_fft.%s.module" % module_name
    #     print("Import module name is %s" % import_module_name)
    #     filtered_fft = cppimport.imp(import_module_name)
    #
    #     # revert the monkey-patch
    #     sysconfig.customize_compiler = old
    #
    #
    # def test_128_128(self):
    #     rows = columns = 128
    #     module_name = 'module_' + str(rows) + '_' + str(columns)
    #     dirname = os.path.join("/home/clb02321/PycharmProjects/ptypy_accelerate_fresh/build/lib/ptypy/accelerate/py_cuda", 'cuda', 'filtered_fft')
    #     dst  = os.path.join(dirname, module_name)
    #     src = dirname # os.path.join(dirname, 'module.cpp')
    #     # dst = os.path.join(dirname, module_name + '.cpp')
    #     shutil.copytree(src, dst)
    #     # print('copies {} to {}'.format(src, dst))
    #
    #     # monkey-patch the customize_compiler function
    #     old = sysconfig.customize_compiler
    #     sysconfig.customize_compiler = get_customize_compiler(rows, columns, old)
    #
    #     import cppimport
    #     # cppimport.force_rebuild()
    #     # cppimport.set_quiet(True)
    #     # cppimport
    #     import_module_name = "ptypy.accelerate.py_cuda.cuda.filtered_fft.%s.module" % module_name
    #     print("Import module name is %s" % import_module_name)
    #     filtered_fft = cppimport.imp(import_module_name)
    #
    #     # revert the monkey-patch
    #     sysconfig.customize_compiler = old

if __name__=="__main__":
    unittest.main()
