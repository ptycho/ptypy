'''
Testing based on real data
'''
import h5py
import unittest
import numpy as np
from parameterized import parameterized
from .. import perfrun, PyCudaTest, have_pycuda

if have_pycuda():
    from pycuda import gpuarray
    from ptypy.accelerate.cuda_pycuda.kernels import AuxiliaryWaveKernel
from ptypy.accelerate.base.kernels import AuxiliaryWaveKernel as BaseAuxiliaryWaveKernel

COMPLEX_TYPE = np.complex64
FLOAT_TYPE = np.float32
INT_TYPE = np.int32

class DlsAuxiliaryWaveKernelTest(PyCudaTest):

    datadir = "/dls/science/users/iat69393/gpu-hackathon/test-data-%s/"
    rtol = 1e-6
    atol = 1e-6

    @parameterized.expand([
        ["base", 10],
        ["regul", 50],
        ["floating", 0],
    ])
    def test_build_aux_no_ex_noadd_UNITY(self, name, iter):

        # Load data
        with h5py.File(self.datadir % name + "build_aux_no_ex_%04d.h5" %iter, "r") as f:
            aux = f["aux"][:]
            addr = f["addr"][:]
            ob = f["ob"][:]
            pr = f["pr"][:]

        # Copy data to device
        aux_dev = gpuarray.to_gpu(aux)
        addr_dev = gpuarray.to_gpu(addr)
        ob_dev = gpuarray.to_gpu(ob)
        pr_dev = gpuarray.to_gpu(pr)

        # CPU kernel
        BAWK = BaseAuxiliaryWaveKernel()
        BAWK.allocate()
        BAWK.build_aux_no_ex(aux, addr, ob, pr, add=False)

        ## GPU kernel
        AWK = AuxiliaryWaveKernel(self.stream)
        AWK.allocate()
        AWK.build_aux_no_ex(aux_dev, addr_dev, ob_dev, pr_dev, add=False)

        ## Assert
        np.testing.assert_allclose(aux_dev.get(), aux, rtol=self.rtol, atol=self.atol, 
            err_msg="The auxiliary_wave does not match the base kernel output")