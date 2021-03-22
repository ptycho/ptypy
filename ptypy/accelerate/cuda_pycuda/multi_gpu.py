"""
Multi-GPU AllReduce Wrapper, that uses NCCL via cupy if it's available,
and otherwise falls back to MPI + host/device copies
"""

import numpy as np
from pycuda import gpuarray
from ptypy.utils import parallel

try:
    from cupy.cuda import nccl
    import cupy as cp
except ImportError:
    import sys
    print("NCCL could not be loaded - falling back to MPI", file=sys.stderr)
    nccl = None
    del sys


class MultiGpuCommunicatorMpi:
    """Communicator for AllReduce that uses MPI on the CPU, i.e. D2H, allreduce, H2D"""

    def __init__(self):
        self.rank = parallel.rank
        self.ndev = parallel.size

    def allReduceSum(self, send_arr, recv_arr):
        """Call MPI.all_reduce from send to recv array, with both arrays on GPU"""

        if not isinstance(send_arr, gpuarray.GPUArray) \
            or not isinstance(recv_arr, gpuarray.GPUArray):
            raise NotImplementedError("AllReduce only implemented for gpuarrays")

        if self.ndev > 1:
            # note: this creates a temporary CPU array
            data = send_arr.get()
            parallel.allreduce(data)
            recv_arr.set(data)
    
class MultiGpuCommunicatorNccl(MultiGpuCommunicatorMpi):
    def __init__(self):
        super().__init__()
        # get a unique identifier for the NCCL communicator and 
        # broadcast it to all MPI processes (assuming one device per process)
        if self.rank == 0:
            self.id = nccl.get_unique_id()
        else:
            self.id = None

        self.id = parallel.bcast(self.id)
        self.com = nccl.NcclCommunicator(self.ndev, self.id, self.rank)

    def allReduceSum(self, send_arr, recv_arr):
        """Call MPI.all_reduce from send to recv array, with both arrays on GPU"""
        if not isinstance(send_arr, gpuarray.GPUArray) \
            or not isinstance(recv_arr, gpuarray.GPUArray):
            raise NotImplementedError("AllReduce only implemented for gpuarrays")
        sendbuf = int(send_arr.gpudata)
        recvbuf = int(recv_arr.gpudata)
        count, datatype = self.__get_NCCL_count_dtype(send_arr)
        
        # no stream support here for now - it fails in NCCL when 
        # pycuda.Stream.handle is used for some unexplained reason
        stream = cp.cuda.Stream.null.ptr
        
        self.com.allReduce(sendbuf, recvbuf, count, datatype, nccl.NCCL_SUM, stream)

    def __get_NCCL_count_dtype(self, arr):
            if arr.dtype == np.complex64:
                return arr.size*2, nccl.NCCL_FLOAT32
            elif arr.dtype == np.complex128:
                return arr.size*2, nccl.NCCL_FLOAT64
            elif arr.dtype == np.float32:
                return arr.size, nccl.NCCL_FLOAT32
            elif arr.dtype == np.float64:
                return arr.size, nccl.NCCL_FLOAT64
            else:
                raise ValueError("This dtype is not supported by NCCL.")

# pick the appropriate communicator depending on installed packages
if nccl is None:
    MultiGpuCommunicator = MultiGpuCommunicatorMpi
else:
    MultiGpuCommunicator = MultiGpuCommunicatorNccl
            