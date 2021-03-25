'''
'''

import unittest
from mpi4py.MPI import Get_version
import numpy as np
from . import PyCudaTest, have_pycuda

if have_pycuda():
    from pycuda import gpuarray
    import pycuda.driver as cuda
    from ptypy.accelerate.cuda_pycuda import multi_gpu as mgpu
    from ptypy.utils import parallel

from pkg_resources import parse_version

class GpuDataTest(PyCudaTest):
    """
    This is a test class for MPI - to really check if it all works, it needs
    to be run as:

    mpirun -np 2 pytest multi_gpu_test.py

    For CUDA-aware MPI testing, currently the environment variable

    OMPI_MCA_opal_cuda_support=true

    needs to be set, mpi4py version 3.1.0+ used, a pycuda build from master,
    and a cuda-aware MPI version.
    """

    def setUp(self):
        if parallel.rank_local < cuda.Device.count():
            self.ctx = cuda.Device(parallel.rank_local).make_context()
            self.ctx.push()
            self.stream = cuda.Stream()
        else:
            self.ctx = None

    def tearDown(self):
        if self.ctx is not None:
            self.ctx.pop()
            self.ctx.detach()

    @unittest.skipIf(parallel.rank != 0, "Only in MPI rank 0")
    def test_version(self):
        v1 = parse_version("3.1.0")
        v2 = parse_version(parse_version("3.1.0a").base_version)

        self.assertGreaterEqual(v2, v1)

    def multigpu_tester(self, com):
        if self.ctx is None:
            return

        data = np.ones((2, 1), dtype=np.float32)
        data_dev = gpuarray.to_gpu(data)
        sz = parallel.size
        com.allReduceSum(data_dev)

        out = data_dev.get()
        np.testing.assert_allclose(out, sz * data, rtol=1e-6)

    def test_multigpu_auto(self):
        self.multigpu_tester(mgpu.MultiGpuCommunicator())
        

    def test_multigpu_mpi(self):
        self.multigpu_tester(mgpu.MultiGpuCommunicatorMpi())

    @unittest.skipIf(not mgpu.have_cuda_mpi, "Cuda-aware MPI not available")
    def test_multigpu_cudampi(self):
        self.multigpu_tester(mgpu.MultiGpuCommunicatorCudaMpi())

    @unittest.skipIf(not mgpu.have_nccl, "NCCL not available")
    def test_multigpu_nccl(self):
        self.multigpu_tester(mgpu.MultiGpuCommunicatorNccl())