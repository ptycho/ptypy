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

class DlsPoUpdateKernelTest(PyCudaTest):

    datadir = "/dls/science/users/iat69393/gpu-hackathon/test-data-%s/"
    rtol = 1e-6
    atol = 1e-6

    @parameterized.expand([
        ["base", 10, False],
        ["regul", 50, False],
        ["floating", 0, False],
        ["base", 10, True],
        ["regul", 50, True],
        ["floating", 0, True],
    ])
    def test_op_update_ml_UNITY(self, name, iter, atomics):

        # Load data
        with h5py.File(self.datadir %name + "op_update_ml_%04d.h5" %iter, "r") as f:
            aux = f["aux"][:]
            addr = f["addr"][:]
            obg = f["obg"][:]
            pr = f["pr"][:]

        # Copy data to device
        aux_dev = gpuarray.to_gpu(aux)
        obg_dev = gpuarray.to_gpu(obg)
        pr_dev = gpuarray.to_gpu(pr)

        # If not using atomics we need to change the addresses
        if not atomics:
            addr2 = np.ascontiguousarray(np.transpose(addr, (2, 3, 0, 1)))
            addr_dev = gpuarray.to_gpu(addr2)
        else:
            addr_dev = gpuarray.to_gpu(addr)

        # CPU Kernel
        BPOK = BasePoUpdateKernel()
        BPOK.ob_update_ML(addr, obg, pr, aux)

        # GPU Kernel
        POK = PoUpdateKernel()
        POK.ob_update_ML(addr_dev, obg_dev, pr_dev, aux_dev, atomics=atomics)

        ## Assert
        np.testing.assert_allclose(obg_dev.get(), obg, atol=self.atol, rtol=self.rtol, verbose=False,
            err_msg="The object array has not been updated as expected")

    @parameterized.expand([
        ["base", 10, False],
        ["regul", 50, False],
        ["floating", 0, False],
        ["base", 10, True],
        ["regul", 50, True],
        ["floating", 0, True],
    ])
    def test_pr_update_ml_UNITY(self, name, iter, atomics):

        # Load data
        with h5py.File(self.datadir %name + "pr_update_ml_%04d.h5" %iter, "r") as f:
            aux = f["aux"][:]
            addr = f["addr"][:]
            ob = f["ob"][:]
            prg = f["prg"][:]

        # Copy data to device
        aux_dev = gpuarray.to_gpu(aux) 
        ob_dev = gpuarray.to_gpu(ob)
        prg_dev = gpuarray.to_gpu(prg)

        # If not using atomics we need to change the addresses
        if not atomics:
            addr2 = np.ascontiguousarray(np.transpose(addr, (2, 3, 0, 1)))
            addr_dev = gpuarray.to_gpu(addr2)
        else:
            addr_dev = gpuarray.to_gpu(addr)

        # CPU Kernel
        BPOK = BasePoUpdateKernel()
        BPOK.pr_update_ML(addr, prg, ob, aux)

        # GPU Kernel
        POK = PoUpdateKernel()
        POK.pr_update_ML(addr_dev, prg_dev, ob_dev, aux_dev, atomics=atomics)
        
        ## Assert
        np.testing.assert_allclose(prg, prg_dev.get(),  atol=self.atol, rtol=self.rtol, verbose=False, 
            err_msg="The probe array has not been updated as expected")