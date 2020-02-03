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
from ptypy.utils.math_utils import delxf, delxb 

COMPLEX_TYPE = np.complex64
FLOAT_TYPE = np.float32
INT_TYPE = np.int32


@unittest.skipIf(not have_pycuda(), "no PyCUDA or GPU drivers available")
class DerivativesKernelTest(unittest.TestCase):

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

    def test_delxf_1dim(self):
        inp = np.array([0, 1, 2, 4, 8, 0, 6], dtype=np.float32)
        inp_dev = gpuarray.to_gpu(inp)
        outp = np.zeros_like(inp)
        outp_dev = gpuarray.to_gpu(outp)

        DK = DerivativesKernel(inp.dtype, stream=self.stream)
        DK.delxf(inp_dev, out=outp_dev)

        outp[:] = outp_dev.get()

        exp = np.array([1, 1, 2, 4, -8, 6, 0], dtype=np.float32)
        np.testing.assert_array_equal(outp, exp)

    def test_delxf_1dim_inplace(self):
        inp = np.array([0, 1, 2, 4, 8, 0, 6], dtype=np.float32)
        inp_dev = gpuarray.to_gpu(inp)

        DK = DerivativesKernel(inp.dtype, stream=self.stream)
        DK.delxf(inp_dev, out=inp_dev)

        outp = inp_dev.get()

        exp = np.array([1, 1, 2, 4, -8, 6, 0], dtype=np.float32)
        np.testing.assert_array_equal(outp, exp)

    def test_delxf_2dim1(self):
        inp  = np.array([
            [0, 2, 6],
            [1, -4, 5]
        ], dtype=np.float32)

        inp_dev = gpuarray.to_gpu(inp)
        outp = np.zeros_like(inp)
        outp_dev = gpuarray.to_gpu(outp)

        DK = DerivativesKernel(inp.dtype, stream=self.stream)
        DK.delxf(inp_dev, out=outp_dev, axis=0)

        outp[:] = outp_dev.get()


        exp = np.array([
            [1, -6, -1],
            [0, 0, 0]
        ], dtype=np.float32)
        np.testing.assert_array_equal(outp, exp)

    def test_delxf_2dim2(self):
        inp  = np.array([
            [0, 2, 6],
            [1, -4, 5]
        ], dtype=np.float32)
        inp_dev = gpuarray.to_gpu(inp)
        outp = np.zeros_like(inp)
        outp_dev = gpuarray.to_gpu(outp)

        DK = DerivativesKernel(inp.dtype, stream=self.stream)
        DK.delxf(inp_dev, out=outp_dev, axis=1)

        outp[:] = outp_dev.get()

        exp = np.array([
            [2, 4, 0],
            [-5, 9, 0]
        ], dtype=np.float32)
        np.testing.assert_array_equal(outp, exp)

    def test_delxb_1dim(self):
        inp = np.array([0, 1, 2, 4, 8, 0, 6], dtype=np.float32)
        inp_dev = gpuarray.to_gpu(inp)
        outp = np.zeros_like(inp)
        outp_dev = gpuarray.to_gpu(outp)

        DK = DerivativesKernel(inp.dtype, stream=self.stream)
        DK.delxb(inp_dev, out=outp_dev)

        outp[:] = outp_dev.get()

        exp = np.array([0, 1, 1, 2, 4, -8, 6], dtype=np.float32)
        np.testing.assert_array_equal(outp, exp)

    def test_delxb_2dim1(self):
        inp  = np.array([
            [0, 2, 6],
            [1, -4, 5]
        ], dtype=np.float32)
        inp_dev = gpuarray.to_gpu(inp)
        outp = np.zeros_like(inp)
        outp_dev = gpuarray.to_gpu(outp)

        DK = DerivativesKernel(inp.dtype, stream=self.stream)
        DK.delxb(inp_dev, out=outp_dev, axis=0)

        outp[:] = outp_dev.get()


        exp = np.array([
            [0, 0, 0],
            [1, -6, -1],
        ], dtype=np.float32)
        np.testing.assert_array_equal(outp, exp)

    def test_delxb_2dim2(self):
        inp  = np.array([
            [0, 2, 6],
            [1, -4, 5]
        ], dtype=np.float32)
        inp_dev = gpuarray.to_gpu(inp)
        outp = np.zeros_like(inp)
        outp_dev = gpuarray.to_gpu(outp)

        DK = DerivativesKernel(inp.dtype, stream=self.stream)
        DK.delxb(inp_dev, out=outp_dev, axis=1)

        outp[:] = outp_dev.get()


        exp = np.array([
            [0, 2, 4],
            [0, -5, 9]
        ], dtype=np.float32)
        np.testing.assert_array_equal(outp, exp)

    def test_delxf_2dim2complex(self):
        inp = np.array([
            [0, 2, 6],
            [1, -4, 5]
        ],dtype=np.float32) + 1j * np.array([
            [0, 4, 12],
            [2, -8, 10]
        ],dtype=np.float32)
        inp_dev = gpuarray.to_gpu(inp)
        outp = np.zeros_like(inp)
        outp_dev = gpuarray.to_gpu(outp)

        DK = DerivativesKernel(inp.dtype, stream=self.stream)
        DK.delxf(inp_dev, out=outp_dev, axis=1)

        outp[:] = outp_dev.get()

        exp = np.array([
            [2, 4, 0],
            [-5, 9, 0]
        ], dtype=np.float32) + 1j * np.array([
            [4, 8, 0],
            [-10, 18, 0]
        ], dtype=np.float32)
        np.testing.assert_array_equal(outp, exp)
        
    def test_delxf_3dim2(self):
        inp = np.array([
            [
                [1, 2, 4,],
                [7, 11, 16,],
            ],
            [
                [22, 29, 37,],
                [46, 56, 67]
            ]
        ], dtype=np.float32)
        
        inp_dev = gpuarray.to_gpu(inp)
        outp = np.zeros_like(inp)
        outp_dev = gpuarray.to_gpu(outp)

        DK = DerivativesKernel(inp.dtype, stream=self.stream)
        DK.delxf(inp_dev, out=outp_dev, axis=1)

        outp[:] = outp_dev.get()

        exp = np.array([
            [
                [6, 9, 12,],
                [0, 0, 0,],
            ],
            [
                [24, 27, 30,],
                [0, 0, 0],
            ]
        ], dtype=np.float32)

        np.testing.assert_array_equal(outp, exp)

    def test_delxf_3dim1_unity(self):
        inp = np.ascontiguousarray(np.random.randn(33, 283, 142), dtype=np.float32)
        
        inp_dev = gpuarray.to_gpu(inp)
        outp = np.zeros_like(inp)
        outp_dev = gpuarray.to_gpu(outp)

        DK = DerivativesKernel(inp.dtype, stream=self.stream)
        DK.delxf(inp_dev, out=outp_dev, axis=0)
        outp[:] = outp_dev.get()

        exp = delxf(inp, axis=0)
        np.testing.assert_array_almost_equal(outp, exp)

    def test_delxf_3dim2_unity1(self):
        inp = np.array([
            [ [1], [2], [4]],
            [ [8], [16], [32]]
        ], dtype=np.float32)
       
        inp_dev = gpuarray.to_gpu(inp)
        outp = np.zeros_like(inp)
        outp_dev = gpuarray.to_gpu(outp)

        DK = DerivativesKernel(inp.dtype, stream=self.stream)
        DK.delxf(inp_dev, out=outp_dev, axis=1)
        outp[:] = outp_dev.get()

        exp = delxf(inp, axis=1)

        np.testing.assert_array_almost_equal(np.squeeze(outp), np.squeeze(exp))

    def test_delxf_3dim2_unity2(self):
        inp = np.array([
            [ [1, 2], [4, 7], [11,16] ],
            [ [22,29], [37,46], [56,67]]
        ], dtype=np.float32)
       
        inp_dev = gpuarray.to_gpu(inp)
        outp = np.zeros_like(inp)
        outp_dev = gpuarray.to_gpu(outp)

        DK = DerivativesKernel(inp.dtype, stream=self.stream)
        DK.delxf(inp_dev, out=outp_dev, axis=1)
        outp[:] = outp_dev.get()

        exp = delxf(inp, axis=1)

        np.testing.assert_array_almost_equal(np.squeeze(outp), np.squeeze(exp))

    def test_delxf_3dim2_unity(self):
        inp = np.ascontiguousarray(np.random.randn(33, 283, 142), dtype=np.float32)
        
        inp_dev = gpuarray.to_gpu(inp)
        outp = np.zeros_like(inp)
        outp_dev = gpuarray.to_gpu(outp)

        DK = DerivativesKernel(inp.dtype, stream=self.stream)
        DK.delxf(inp_dev, out=outp_dev, axis=1)
        outp[:] = outp_dev.get()

        exp = delxf(inp, axis=1)
        np.testing.assert_array_almost_equal(outp, exp)

    def test_delxf_3dim3_unity(self):
        inp = np.ascontiguousarray(np.random.randn(33, 283, 142), dtype=np.float32)
        
        inp_dev = gpuarray.to_gpu(inp)
        outp = np.zeros_like(inp)
        outp_dev = gpuarray.to_gpu(outp)

        DK = DerivativesKernel(inp.dtype, stream=self.stream)
        DK.delxf(inp_dev, out=outp_dev, axis=2)
        outp[:] = outp_dev.get()

        exp = delxf(inp, axis=2)
        np.testing.assert_array_almost_equal(outp, exp)

    @unittest.skip("performance test")
    def test_perf_3d_0(self):
        shape = [500, 1024, 1024]
        inp = np.ones(shape, dtype=np.complex64)
        inp_dev = gpuarray.to_gpu(inp)
        outp = np.ones_like(inp)
        outp_dev = gpuarray.to_gpu(outp)

        DK = DerivativesKernel(inp.dtype, stream=self.stream)
        DK.delxf(inp_dev, out=outp_dev, axis=0)
        outp[:] = outp_dev.get()
        np.testing.assert_array_equal(outp, 0)

    @unittest.skip("performance test")
    def test_perf_3d_1(self):
        shape = [500, 1024, 1024]
        inp = np.ones(shape, dtype=np.complex64)
        inp_dev = gpuarray.to_gpu(inp)
        outp = np.ones_like(inp)
        outp_dev = gpuarray.to_gpu(outp)

        DK = DerivativesKernel(inp.dtype, stream=self.stream)
        DK.delxf(inp_dev, out=outp_dev, axis=1)
        outp[:] = outp_dev.get()
        np.testing.assert_array_equal(outp, 0)

    @unittest.skip("performance test")
    def test_perf_3d_2(self):
        shape = [500, 1024, 1024]
        inp = np.ones(shape, dtype=np.complex64)
        inp_dev = gpuarray.to_gpu(inp)
        outp = np.ones_like(inp)
        outp_dev = gpuarray.to_gpu(outp)

        DK = DerivativesKernel(inp.dtype, stream=self.stream)
        DK.delxf(inp_dev, out=outp_dev, axis=2)
        outp[:] = outp_dev.get()
        np.testing.assert_array_equal(outp, 0)