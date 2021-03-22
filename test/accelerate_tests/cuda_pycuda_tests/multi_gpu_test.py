'''
'''

import numpy as np
from . import PyCudaTest, have_pycuda

if have_pycuda():
    from pycuda import gpuarray
    import pycuda.driver as cuda
    from ptypy.accelerate.cuda_pycuda.multi_gpu import MultiGpuCommunicator, MultiGpuCommunicatorMpi
    from ptypy.utils import parallel


class GpuDataTest(PyCudaTest):

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

    def test_multigpu_auto(self):
        if self.ctx is None:
            return

        data = np.ones((2,1), dtype=np.float32)
        data_dev = gpuarray.to_gpu(data)
        com = MultiGpuCommunicator()
        sz = parallel.size
        com.allReduceSum(data_dev, data_dev)

        out = data_dev.get()
        np.testing.assert_allclose(out, sz * data, rtol=1e-6)
        

    def test_multigpu_mpi(self):
        if self.ctx is None:
            return

        data = np.ones((2,1), dtype=np.float32)
        data_dev = gpuarray.to_gpu(data)
        com = MultiGpuCommunicatorMpi()
        sz = parallel.size
        com.allReduceSum(data_dev, data_dev)

        out = data_dev.get()
        np.testing.assert_allclose(out, sz * data, rtol=1e-6)