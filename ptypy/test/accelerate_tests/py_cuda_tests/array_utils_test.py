'''


'''

import unittest
import numpy as np
from . import perfrun, PyCudaTest, have_pycuda

if have_pycuda():
    from pycuda import gpuarray
    import ptypy.accelerate.py_cuda.array_utils as au

class ArrayUtilsTest(PyCudaTest):

    def test_dot_float_float(self):
        ## Arrange
        X,Y,Z = np.indices((3,3,1001), dtype=np.float32)
        A = 10 ** Y
        A_dev = gpuarray.to_gpu(A)

        ## Act
        AU = au.ArrayUtilsKernel(acc_dtype=np.float32)
        out_dev = AU.dot(A_dev, A_dev)
        out = out_dev.get()

        ## Assert
        np.testing.assert_allclose(out, 30333303.0, rtol=1e-7)

    def test_dot_float_double(self):
        ## Arrange
        X,Y,Z = np.indices((3,3,1001), dtype=np.float32)
        A = 10 ** Y
        A_dev = gpuarray.to_gpu(A)

        ## Act
        AU = au.ArrayUtilsKernel(acc_dtype=np.float64)
        out_dev = AU.dot(A_dev, A_dev)
        out = out_dev.get()

        ## Assert
        np.testing.assert_equal(out, 30333303.0)
    
    def test_dot_complex_float(self):
        ## Arrange
        X,Y,Z = np.indices((3,3,1001), dtype=np.float32)
        A = 10 ** Y + 1j * 10 ** X
        A_dev = gpuarray.to_gpu(A)

        ## Act
        AU = au.ArrayUtilsKernel(acc_dtype=np.float32)
        out_dev = AU.dot(A_dev, A_dev)
        out = out_dev.get()

        ## Assert
        np.testing.assert_allclose(out, 60666606.0, rtol=1e-7)

    def test_dot_complex_double(self):
        ## Arrange
        X,Y,Z = np.indices((3,3,1001), dtype=np.float32)
        A = 10 ** Y + 1j * 10 ** X
        A_dev = gpuarray.to_gpu(A)

        ## Act
        AU = au.ArrayUtilsKernel(acc_dtype=np.float64)
        out_dev = AU.dot(A_dev, A_dev)
        out = out_dev.get()

        ## Assert
        np.testing.assert_array_equal(out, 60666606.0)

    @unittest.skipIf(not perfrun, "Performance test")
    def test_dot_performance(self):
        ## Arrange
        X,Y,Z = np.indices((3,3,1021301), dtype=np.float32)
        A = 10 ** Y + 1j * 10 ** X
        A_dev = gpuarray.to_gpu(A)

        ## Act
        AU = au.ArrayUtilsKernel(acc_dtype=np.float64)
        out_dev = AU.dot(A_dev, A_dev)
