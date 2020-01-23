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
    from ptypy.accelerate.py_cuda.kernels import DerivativesKernel

COMPLEX_TYPE = np.complex64
FLOAT_TYPE = np.float32
INT_TYPE = np.int32


@unittest.skipIf(not have_pycuda(), "no PyCUDA or GPU drivers available")
class DerivativesKernelTest(unittest.TestCase):

    def setUp(self):
        import sys
        np.set_printoptions(threshold=sys.maxsize, linewidth=np.inf)
        self.ctx = make_default_context()

    def tearDown(self):
        np.set_printoptions()
        self.ctx.pop()
        self.ctx.detach()

    @unittest.skip("not implemented")
    def test_delxf_1dim(self):
        inp = np.array([0, 1, 2, 4, 8, 0, 6], dtype=np.float32)
        inp_dev = gpuarray.to_gpu(inp)
        outp = np.zeros_like(inp)
        outp_dev = gpuarray.to_gpu(outp)

        DK = DerivativesKernel(inp)
        DK.allocate()
        DK.delxf(inp_dev, out=outp_dev)

        outp[:] = outp_dev.get()

        exp = np.array([1, 1, 2, 4, -8, 6, 0], dtype=np.float32)
        np.testing.assert_array_equal(outp, exp)

    @unittest.skip("not implemented")
    def test_delxf_1dim_inplace(self):
        inp = np.array([0, 1, 2, 4, 8, 0, 6], dtype=np.float32)
        inp_dev = gpuarray.to_gpu(inp)

        DK = DerivativesKernel(inp)
        DK.allocate()
        DK.delxf(inp_dev, out=inp_dev)

        outp[:] = inp_dev.get()

        exp = np.array([1, 1, 2, 4, -8, 6, 0], dtype=np.float32)
        np.testing.assert_array_equal(outp, exp)

    @unittest.skip("not implemented")
    def test_delxf_2dim1(self):
        inp  = np.array([
            [0, 2, 6],
            [1, -4, 5]
        ], dtype=np.float32)

        inp_dev = gpuarray.to_gpu(inp)
        outp = np.zeros_like(inp)
        outp_dev = gpuarray.to_gpu(outp)

        DK = DerivativesKernel(inp)
        DK.allocate()
        DK.delxf(inp_dev, out=outp_dev, axis=0)

        outp[:] = outp_dev.get()


        exp = np.array([
            [1, -6, -1],
            [0, 0, 0]
        ], dtype=np.float32)
        np.testing.assert_array_equal(outp, exp)

    @unittest.skip("not implemented")
    def test_delxf_2dim2(self):
        inp  = np.array([
            [0, 2, 6],
            [1, -4, 5]
        ], dtype=np.float32)
        inp_dev = gpuarray.to_gpu(inp)
        outp = np.zeros_like(inp)
        outp_dev = gpuarray.to_gpu(outp)

        DK = DerivativesKernel(inp)
        DK.allocate()
        DK.delxf(inp_dev, out=outp_dev, axis=1)

        outp[:] = outp_dev.get()

        exp = np.array([
            [2, 4, 0],
            [-5, 9, 0]
        ], dtype=np.float32)
        np.testing.assert_array_equal(outp, exp)

    @unittest.skip("not implemented")
    def test_delxb_1dim(self):
        inp = np.array([0, 1, 2, 4, 8, 0, 6], dtype=np.float32)
        inp_dev = gpuarray.to_gpu(inp)
        outp = np.zeros_like(inp)
        outp_dev = gpuarray.to_gpu(outp)

        DK = DerivativesKernel(inp)
        DK.allocate()
        DK.delxb(inp_dev, out=outp_dev)

        outp[:] = outp_dev.get()

        exp = np.array([0, 1, 1, 2, 4, -8, 6], dtype=np.float32)
        np.testing.assert_array_equal(outp, exp)

    @unittest.skip("not implemented")
    def test_delxb_2dim1(self):
        inp  = np.array([
            [0, 2, 6],
            [1, -4, 5]
        ], dtype=np.float32)
        inp_dev = gpuarray.to_gpu(inp)
        outp = np.zeros_like(inp)
        outp_dev = gpuarray.to_gpu(outp)

        DK = DerivativesKernel(inp)
        DK.allocate()
        DK.delxb(inp_dev, out=outp_dev, axis=0)

        outp[:] = outp_dev.get()


        exp = np.array([
            [0, 0, 0],
            [1, -6, -1],
        ], dtype=np.float32)
        np.testing.assert_array_equal(outp, exp)

    @unittest.skip("not implemented")
    def test_delxb_2dim2(self):
        inp  = np.array([
            [0, 2, 6],
            [1, -4, 5]
        ], dtype=np.float32)
        inp_dev = gpuarray.to_gpu(inp)
        outp = np.zeros_like(inp)
        outp_dev = gpuarray.to_gpu(outp)

        DK = DerivativesKernel(inp)
        DK.allocate()
        DK.delxf(inp_dev, out=outp_dev, axis=1)

        outp[:] = outp_dev.get()


        exp = np.array([
            [0, 2, 4],
            [0, -5, 9]
        ], dtype=np.float32)
        np.testing.assert_array_equal(outp, exp)
        