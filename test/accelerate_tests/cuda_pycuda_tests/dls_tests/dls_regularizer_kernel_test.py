'''
Testing on real data
'''

import h5py
import unittest
import numpy as np
from parameterized import parameterized
from .. import PyCudaTest, have_pycuda

if have_pycuda():
    from pycuda import gpuarray
    from ptypy.accelerate.cuda_pycuda.engines.ML_pycuda import Regul_del2_pycuda
    import pycuda.driver as cuda
from ptypy.engines.ML import Regul_del2

COMPLEX_TYPE = np.complex64
FLOAT_TYPE = np.float32
INT_TYPE = np.int32

class DlsRegularizerTest(PyCudaTest):

    datadir = "/dls/science/users/iat69393/gpu-hackathon/test-data-%s/"
    rtol = 1e-6
    atol = 1e-6

    @parameterized.expand([
        ["regul", 50]
    ])
    def test_regularizer_grad_UNITY(self, name, iter):

        # Load data
        with h5py.File(self.datadir %name + "regul_grad_%04d.h5" %iter, "r") as f:
            ob = f["ob"][:]

        # Copy data to device
        ob_dev = gpuarray.to_gpu(ob)

        # CPU Kernel
        regul = Regul_del2(0.1)
        obr = regul.grad(ob)

        # GPU Kernel
        regul_pycuda = Regul_del2_pycuda(0.1, queue=self.stream, allocator=cuda.mem_alloc)
        obr_dev = regul_pycuda.grad(ob_dev)

        ## Assert
        np.testing.assert_allclose(obr, obr_dev.get(),  atol=self.atol, rtol=self.rtol, 
            err_msg="The object array has not been updated as expected")
        np.testing.assert_allclose(regul.LL, regul_pycuda.LL,  atol=self.atol, rtol=self.rtol, 
            err_msg="The LL array has not been updated as expected")

    @parameterized.expand([
        ["regul", 50],
    ])
    def test_regularizer_poly_line_ceoffs_UNITY(self, name, iter):

        # Load data
        with h5py.File(self.datadir % name + "regul_poly_line_coeffs_%04d.h5" %iter, "r") as f:
            ob  = f["ob"][:]
            obh = f["obh"][:]

        # Copy data to device
        ob_dev = gpuarray.to_gpu(ob)
        obh_dev = gpuarray.to_gpu(obh)

        # CPU Kernel
        regul = Regul_del2(0.1)
        res = regul.poly_line_coeffs(obh, ob)

        # GPU Kernel
        regul_pycuda = Regul_del2_pycuda(0.1, queue=self.stream, allocator=cuda.mem_alloc)
        res_pycuda = regul_pycuda.poly_line_coeffs(obh_dev, ob_dev)

        ## Assert
        np.testing.assert_allclose(res, res_pycuda,  atol=self.atol, rtol=self.rtol, 
            err_msg="The B array has not been updated as expected")
