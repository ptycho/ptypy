'''


'''

import unittest
import numpy as np
from . import perfrun, PyCudaTest, have_pycuda

if have_pycuda():
    from pycuda import gpuarray
    from ptypy.accelerate.cuda_pycuda.engines.ML_pycuda import Regul_del2_pycuda
    from ptypy.engines.ML import Regul_del2
    from pycuda.tools import make_default_context
    from pycuda.driver import mem_alloc


class EngineUtilsTest(PyCudaTest):

    def test_regul_del2_grad_unity(self):
        ## Arrange
        A = (np.random.randn(40,40)
        +1j*np.random.randn(40,40)).astype(np.complex64)
        A_dev = gpuarray.to_gpu(A)

        ## Act
        Reg = Regul_del2(0.1)
        Reg_dev = Regul_del2_pycuda(0.1, allocator=mem_alloc)
        grad_dev = Reg_dev.grad(A_dev).get()
        grad = Reg.grad(A)
        #grad_dev = grad
        ## Assert
        np.testing.assert_allclose(grad_dev, grad, rtol=1e-7)
        np.testing.assert_allclose(Reg_dev.LL, Reg.LL, rtol=1e-7)


    def test_regul_del2_coeff_unity(self):
        ## Arrange
        A = (np.random.randn(40,40)
        +1j*np.random.randn(40,40)).astype(np.complex64)
        B = (np.random.randn(40,40)
        +1j*np.random.randn(40,40)).astype(np.complex64)
        A_dev = gpuarray.to_gpu(A)
        B_dev = gpuarray.to_gpu(B)

        ## Act
        Reg = Regul_del2(0.1)
        Reg_dev = Regul_del2_pycuda(0.1, allocator=mem_alloc)
        d = Reg_dev.poly_line_coeffs(A_dev, B_dev)
        c = Reg.poly_line_coeffs(A, B)
        #grad_dev = grad
        #d = c
        ## Assert
        np.testing.assert_allclose(c, d, rtol=1e-6)
