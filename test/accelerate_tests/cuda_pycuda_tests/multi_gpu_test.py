'''
'''

import unittest
from mpi4py.MPI import Get_version
import numpy as np
from . import PyCudaTest, have_pycuda

if have_pycuda():
    from pycuda import gpuarray
    import pycuda.driver as cuda
    from pycuda.tools import make_default_context
    from ptypy.accelerate.cuda_pycuda import multi_gpu as mgpu
    from ptypy.utils import parallel

from pkg_resources import parse_version

class GpuDataTest(unittest.TestCase):
    """
    This is a test class for MPI - to really check if it all works, it needs
    to be run as:

    mpirun -np 2 pytest multi_gpu_test.py

    For CUDA-aware MPI testing, currently the environment variable

    OMPI_MCA_opal_cuda_support=true

    needs to be set, mpi4py version 3.1.0+ used, a pycuda build from master,
    and a cuda-aware MPI version.

    To check if it is a cuda-aware MPI version:
        ompi_info --parsable --all | grep mpi_built_with_cuda_support:value
    """

    def setUp(self):
        if parallel.rank_local < cuda.Device.count():
            def _retain_primary_context(dev):
                ctx = dev.retain_primary_context()
                ctx.push()
                return ctx
            self.ctx = make_default_context(_retain_primary_context)
            self.device = self.ctx.get_device()
        else:
            self.ctx = None

    def tearDown(self):
        if self.ctx is not None:
            self.ctx.pop()
            self.ctx = None

    @unittest.skipIf(parallel.rank != 0, "Only in MPI rank 0")
    def test_version(self):
        v1 = parse_version("3.1.0")
        v2 = parse_version(parse_version("3.1.0a").base_version)

        self.assertGreaterEqual(v2, v1)

    def test_compute_mode(self):
        attr = cuda.Context.get_device().get_attributes()
        self.assertIn(cuda.device_attribute.COMPUTE_MODE, attr)
        mode = attr[cuda.device_attribute.COMPUTE_MODE]
        self.assertIn(mode,
            [cuda.compute_mode.DEFAULT, cuda.compute_mode.PROHIBITED, cuda.compute_mode.EXCLUSIVE_PROCESS]
        )

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
        self.multigpu_tester(mgpu.get_multi_gpu_communicator())

    def test_multigpu_mpi(self):
        self.multigpu_tester(mgpu.MultiGpuCommunicatorMpi())

    @unittest.skipIf(not mgpu.have_cuda_mpi, "Cuda-aware MPI not available")
    def test_multigpu_cudampi(self):
        self.multigpu_tester(mgpu.MultiGpuCommunicatorCudaMpi())
