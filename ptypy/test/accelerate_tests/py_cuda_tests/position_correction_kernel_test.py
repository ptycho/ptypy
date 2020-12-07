'''


'''

import unittest
import numpy as np
from . import PyCudaTest, have_pycuda

if have_pycuda():
    from pycuda import gpuarray
    from ptypy.accelerate.py_cuda.kernels import PositionCorrectionKernel
    from ptypy.accelerate.array_based.kernels import PositionCorrectionKernel as abPositionCorrectionKernel

COMPLEX_TYPE = np.complex64
FLOAT_TYPE = np.float32
INT_TYPE = np.int32


class PositionCorrectionKernelTest(PyCudaTest):

    def update_addr_and_error_state_UNITY_helper(self, size, modes):
        ## Arrange
        addr = np.ones((size, modes, 5, 3), dtype=np.int32)
        mangled_addr = 2 * addr
        err_state = np.zeros((size,), dtype=np.float32)
        err_state[5:] = 2.
        err_sum = np.ones((size, ), dtype=np.float32)
        addr_gpu = gpuarray.to_gpu(addr)
        mangled_addr_gpu = gpuarray.to_gpu(mangled_addr)
        err_state_gpu = gpuarray.to_gpu(err_state)
        err_sum_gpu = gpuarray.to_gpu(err_sum)
        aux = np.ones((1,1,1), dtype=np.complex64)

        ## Act
        PCK = PositionCorrectionKernel(aux, modes, queue_thread=self.stream)
        PCK.update_addr_and_error_state(addr_gpu, err_state_gpu, mangled_addr_gpu, err_sum_gpu)
        abPCK = abPositionCorrectionKernel(aux, modes)
        abPCK.update_addr_and_error_state(addr, err_state, mangled_addr, err_sum)

        ## Assert
        np.testing.assert_array_equal(addr_gpu.get(), addr)
        np.testing.assert_array_equal(err_state_gpu.get(), err_state)

    def test_update_addr_and_error_state_UNITY_small_onemode(self):
        self.update_addr_and_error_state_UNITY_helper(4, 1)

    def test_update_addr_and_error_state_UNITY_large_onemode(self):
        self.update_addr_and_error_state_UNITY_helper(323, 1)
    
    def test_update_addr_and_error_state_UNITY_small_multimode(self):
        self.update_addr_and_error_state_UNITY_helper(4, 3)

    def test_update_addr_and_error_state_UNITY_large_multimode(self):
        self.update_addr_and_error_state_UNITY_helper(323, 3)
        