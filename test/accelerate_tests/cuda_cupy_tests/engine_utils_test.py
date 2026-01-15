'''


'''

import unittest
import numpy as np
from . import perfrun, CupyCudaTest, have_cupy

if have_cupy():
    import cupy as cp
    from ptypy.accelerate.cuda_cupy.engines.ML_cupy import Regul_del2_cupy
    from ptypy.engines.ML import Regul_del2


class EngineUtilsTest(CupyCudaTest):

    def test_regul_del2_grad_unity(self):
        ## Arrange
        A = (np.random.randn(40,40)
        +1j*np.random.randn(40,40)).astype(np.complex64)
        A_dev = cp.asarray(A)

        ## Act
        Reg = Regul_del2(0.1)
        Reg_dev = Regul_del2_cupy(0.1)
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
        A_dev = cp.asarray(A)
        B_dev = cp.asarray(B)

        ## Act
        Reg = Regul_del2(0.1)
        Reg_dev = Regul_del2_cupy(0.1)
        d = Reg_dev.poly_line_coeffs(A_dev, B_dev)
        c = Reg.poly_line_coeffs(A, B)
        #grad_dev = grad
        #d = c
        ## Assert
        np.testing.assert_allclose(c, d, rtol=1e-6)
