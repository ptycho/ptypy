import unittest

def have_cuda():
    try:
        from ptypy.accelerate.cuda import gpu_extension
        return True
    except:
        return False

only_if_cuda_available = unittest.skipIf(not have_cuda(), "no (cythonized) CUDA extension available")
