import unittest
import numpy as np

# shall we run the performance tests?
perfrun = False

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
    from ptypy.accelerate import cuda_pycuda

    # make sure this is called once
    cuda.init()

@unittest.skipIf(not have_pycuda(), "no PyCUDA or GPU drivers available")
class PyCudaTest(unittest.TestCase):

    def setUp(self):
        import sys
        np.set_printoptions(threshold=sys.maxsize, linewidth=np.inf)

        def _retain_primary_context(dev):
            ctx = dev.retain_primary_context()
            ctx.push()
            return ctx
        self.ctx = make_default_context(_retain_primary_context)

        self.stream = cuda.Stream()
        # enable assertions in CUDA kernels for testing
        if not 'perf' in self._testMethodName:
            self.opts_old = cuda_pycuda.debug_options.copy()
            if '-DNDEBUG' in cuda_pycuda.debug_options:
               cuda_pycuda.debug_options.remove('-DNDEBUG')

    def tearDown(self):
        np.set_printoptions()
        self.ctx.pop()
        self.ctx = None
        if not 'perf' in self._testMethodName:
           cuda_pycuda.debug_options = self.opts_old

