import numpy as np
from pycuda import gpuarray
import pycuda.driver as cuda
from pycuda.tools import make_default_context

from ptypy.accelerate.cuda_pycuda.multi_gpu import MultiGpuCommunicatorNccl
from ptypy.utils import parallel
from ptypy.accelerate.cuda_pycuda import get_context
from ptypy.accelerate.cuda_pycuda.array_utils import TransposeKernel

context, queue = get_context(new_context=True, new_queue=True)
multigpu = MultiGpuCommunicatorNccl()

# create a few gpu arrays
a1_dev = gpuarray.empty((10,10), dtype=np.float32)
a2 = np.ones((20,20), dtype=np.float32)
a2_dev = gpuarray.to_gpu(a2)
str = cuda.Stream()
ev = cuda.Event()
inta = gpuarray.zeros((30,40), dtype=np.int32)
intb = gpuarray.empty((40,30), dtype=np.int32)
TSP = TransposeKernel(queue=str)
TSP.transpose(inta, intb)
ev.record(str)
a2_dev = gpuarray.to_gpu(np.array((2,2), dtype=np.complex64))
ev.synchronize()

print("rank {}: done synchronising...".format(parallel.rank))

context.pop()
context.detach()


