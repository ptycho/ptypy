"""
Multi-GPU AllReduce Wrapper, that uses NCCL via cupy if it's available,
and otherwise falls back to CUDA-aware MPI,
and if that doesn't work, uses host/device copies with regular MPI.

Findings:

1) NCCL works with unit tests, but not in the engines. It seems to 
add something to the existing pycuda Context or create a new one,
as a later event recording on an exit wave transfer fails with
'ivalid resource handle' Cuda Error. This error typically happens if for example
a CUDA event is created in a different context than what it is used in,
or on a different device. PyCuda uses the driver API, NCCL uses the runtime.
Even though those are interoperable, there seems to be an issue.
Note that this is before any allreduce call - straight after initialising.

2) NCCL requires cupy - the Python wrapper is in there

3) OpenMPI with CUDA support needs to be available, and:
  - mpi4py needs to be compiled from master (3.1.0a - latest stable release 3.0.x doesn't have it)
  - pycuda needs to be compile from master (for __cuda_array_interface__ - 2020.1 version doesn't have it)
  - OpenMPI in a conda install needs to have the environment variable
  --> if cuda support isn't enabled, the application simply crashes with a seg fault

4) For NCCL peer-to-peer transfers, the EXCLUSIVE compute mode cannot be used. 
   It should be in DEFAULT mode.

5) NCCL support has been dropped from PyCUDA module, but can be used with CuPy module instead

"""

from pkg_resources import parse_version
import numpy as np
from pycuda import gpuarray
import pycuda.driver as cuda
from ptypy.utils import parallel
from ptypy.utils.verbose import logger, log
import os

try:
    import mpi4py
except ImportError:
    mpi4py = None

# properties to check which versions are available

# At the moment, we require:
# the OpenMPI env var OMPI_MCA_opal_cuda_support to be set to true,
# mpi4py >= 3.1.0
# pycuda with __cuda_array_interface__
# and not setting the PTYPY_USE_MPI environment variable
#
# -> we ideally want to allow enabling support from a parameter in ptypy
have_cuda_mpi = (mpi4py is not None) and \
    "OMPI_MCA_opal_cuda_support" in os.environ and \
    os.environ["OMPI_MCA_opal_cuda_support"] == "true" and \
    parse_version(parse_version(mpi4py.__version__).base_version) >= parse_version("3.1.0") and \
    hasattr(gpuarray.GPUArray, '__cuda_array_interface__') and \
    not ('PTYPY_USE_MPI' in os.environ)


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

        # Check if cuda array interface is available
        if not hasattr(arr, '__cuda_array_interface__'):
            raise RuntimeError("input array should have a cuda array interface")

        if parallel.MPIenabled:
            comm = parallel.comm
            comm.Allreduce(parallel.MPI.IN_PLACE, arr)
            

# pick the appropriate communicator depending on installed packages
def get_multi_gpu_communicator(use_cuda_mpi=True):
    if have_cuda_mpi and use_cuda_mpi:
        try:
            comm = MultiGpuCommunicatorCudaMpi()
            log(4, "Using CUDA-aware MPI communicator")
            return comm
        except RuntimeError:
            pass
    comm = MultiGpuCommunicatorMpi()
    log(4, "Using MPI communicator")
    return comm
