"""
Multi-GPU AllReduce Wrapper, that uses NCCL via cupy if it's available,
and otherwise falls back to MPI + host/device copies
"""

import mpi4py
from pkg_resources import parse_version
import numpy as np
from pycuda import gpuarray
from ptypy.utils import parallel
import os

try:
    from cupy.cuda import nccl
    import cupy as cp
except ImportError:
    nccl = None

# properties to check which versions are available
have_nccl = (nccl is not None)

# at the moment, we require the OpenMPI env var to be set,
# mpi4py >= 3.1.0
# pycuda with __cuda_array_interface__
#
# -> we ideally want to allow enabling support from a parameter in ptypy
have_cuda_mpi = "OMPI_MCA_opal_cuda_support" in os.environ and \
    os.environ["OMPI_MCA_opal_cuda_support"] == "true" and \
    parse_version(parse_version(mpi4py.__version__).base_version) >= parse_version("3.1.0") and \
    hasattr(gpuarray.GPUArray, '__cuda_array_interface__')


class MultiGpuCommunicatorBase:
    """Base class for multi-GPU communicator options, to aggregate common bits"""

    def __init__(self):
        self.rank = parallel.rank
        self.ndev = parallel.size

    def allReduceSum(self, arr):
        """Call MPI.all_reduce in-place, with array on GPU"""
        # base class only checks properties of arrays
        assert isinstance(arr, gpuarray.GPUArray), "Input must be a GPUArray"


class MultiGpuCommunicatorMpi(MultiGpuCommunicatorBase):
    """Communicator for AllReduce that uses MPI on the CPU, i.e. D2H, allreduce, H2D"""

    def allReduceSum(self, arr):
        """Call MPI.all_reduce in-place, with array on GPU"""
        super().allReduceSum(arr)

        if parallel.MPIenabled:
            # note: this creates a temporary CPU array
            data = arr.get()
            parallel.allreduce(data)
            arr.set(data)

class MultiGpuCommunicatorCudaMpi(MultiGpuCommunicatorBase):

    def allReduceSum(self, arr):
        """Call MPI.all_reduce in-place, with array on GPU"""

        assert hasattr(arr, '__cuda_array_interface__'), "input array should have a cuda array interface"

        if parallel.MPIenabled:
            comm = parallel.comm
            comm.Allreduce(parallel.MPI.IN_PLACE, arr)
            
    
class MultiGpuCommunicatorNccl(MultiGpuCommunicatorBase):
    
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

    def allReduceSum(self, arr):
        """Call MPI.all_reduce in-place, with array on GPU"""

        buf = int(arr.gpudata)
        count, datatype = self.__get_NCCL_count_dtype(arr)
        
        # no stream support here for now - it fails in NCCL when 
        # pycuda.Stream.handle is used for some unexplained reason
        stream = cp.cuda.Stream.null.ptr
       
        self.com.allReduce(buf, buf, count, datatype, nccl.NCCL_SUM, stream)

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
if have_nccl:
    MultiGpuCommunicator = MultiGpuCommunicatorNccl
elif have_cuda_mpi:
    MultiGpuCommunicator = MultiGpuCommunicatorCudaMpi
else:
    MultiGpuCommunicator = MultiGpuCommunicatorMpi
    