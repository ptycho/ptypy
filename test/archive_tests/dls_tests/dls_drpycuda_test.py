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
    from ptypy.accelerate.cuda_pycuda.kernels import PoUpdateKernel
from ptypy.accelerate.base.kernels import PoUpdateKernel as BasePoUpdateKernel

COMPLEX_TYPE = np.complex64
FLOAT_TYPE = np.float32
INT_TYPE = np.int32

class DlsDRpycudaTest(PyCudaTest):

    datadir = "/dls/science/users/iat69393/gpu-hackathon/test-data-dr/"
    iter = 0
    rtol = 1e-6
    atol = 1e-6

    def test_ob_update_local_UNITY(self):

        # Load data
        with h5py.File(self.datadir + "ob_update_local_%04d.h5" %self.iter, "r") as f:
            aux = f["aux"][:]
            addr = f["addr"][:]
            ob = f["ob"][:]
            pr = f["pr"][:]
            ex = f["ex"][:]

        # Copy data to device
        aux_dev = gpuarray.to_gpu(aux)
        ob_dev = gpuarray.to_gpu(ob)
        pr_dev = gpuarray.to_gpu(pr)
        ex_dev = gpuarray.to_gpu(ex)
        addr_dev = gpuarray.to_gpu(addr)

        # CPU Kernel
        BPOK = BasePoUpdateKernel()
        BPOK.ob_update_local(addr, ob, pr, ex, aux)

        # GPU Kernel
        POK = PoUpdateKernel()
        POK.ob_update_local(addr_dev, ob_dev, pr_dev, ex_dev, aux_dev)

        ## Assert
        np.testing.assert_allclose(ob_dev.get(), ob, atol=self.atol, rtol=self.rtol, verbose=False,
            err_msg="The object array has not been updated as expected")

    def test_pr_update_local_UNITY(self):

        # Load data
        with h5py.File(self.datadir + "pr_update_local_%04d.h5" %self.iter, "r") as f:
            aux = f["aux"][:]
            addr = f["addr"][:]
            ob = f["ob"][:]
            pr = f["pr"][:]
            ex = f["ex"][:]

        # Copy data to device
        aux_dev = gpuarray.to_gpu(aux)
        ob_dev = gpuarray.to_gpu(ob)
        pr_dev = gpuarray.to_gpu(pr)
        ex_dev = gpuarray.to_gpu(ex)
        addr_dev = gpuarray.to_gpu(addr)

        # CPU Kernel
        BPOK = BasePoUpdateKernel()
        BPOK.pr_update_local(addr, pr, ob, ex, aux)

        # GPU Kernel
        POK = PoUpdateKernel()
        POK.pr_update_local(addr_dev, pr_dev, ob_dev, ex_dev, aux_dev)

        ## Assert
        np.testing.assert_allclose(pr_dev.get(), pr, atol=self.atol, rtol=self.rtol, verbose=False,
            err_msg="The object array has not been updated as expected")
