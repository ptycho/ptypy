"""
Multi-GPU AllReduce Wrapper, that uses NCCL via cupy if it's available,
and otherwise falls back to CUDA-aware MPI,
and if that doesn't work, uses host/device copies with regular MPI.

Findings:

1) OpenMPI with CUDA support needs to be available, and:
  - mpi4py needs to be compiled from master (3.1.0a - latest stable release 3.0.x doesn't have it)
  - OpenMPI in a conda install needs to have the environment variable
  --> if cuda support isn't enabled, the application simply crashes with a seg fault

2) For NCCL peer-to-peer transfers, the EXCLUSIVE compute mode cannot be used. 
   It should be in DEFAULT mode.

"""

from pkg_resources import parse_version
import numpy as np
import cupy as cp
from ptypy.utils import parallel
from ptypy.utils.verbose import logger, log
import os
from cupy.cuda import nccl

try:
    import mpi4py
except ImportError:
    mpi4py = None

# properties to check which versions are available

# use NCCL if it is available, and the user didn't override the
# default selection with environment variables
have_nccl = (not 'PTYPY_USE_CUDAMPI' in os.environ) and \
    (not 'PTYPY_USE_MPI' in os.environ)

# At the moment, we require:
# the OpenMPI env var OMPI_MCA_opal_cuda_support to be set to true,
# mpi4py >= 3.1.0
# and not setting the PTYPY_USE_MPI environment variable
#
# -> we ideally want to allow enabling support from a parameter in ptypy
have_cuda_mpi = (mpi4py is not None) and \
    "OMPI_MCA_opal_cuda_support" in os.environ and \
    os.environ["OMPI_MCA_opal_cuda_support"] == "true" and \
    parse_version(parse_version(mpi4py.__version__).base_version) >= parse_version("3.1.0") and \
    not ('PTYPY_USE_MPI' in os.environ)


class MultiGpuCommunicatorBase:
    """Base class for multi-GPU communicator options, to aggregate common bits"""

    def __init__(self):
        self.rank = parallel.rank
        self.ndev = parallel.size

    def allReduceSum(self, arr):
        """Call MPI.all_reduce in-place, with array on GPU"""
        # base class only checks properties of arrays
        assert isinstance(arr, cp.ndarray), "Input must be a GPU Array"


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

        if parallel.MPIenabled:
            comm = parallel.comm
            comm.Allreduce(parallel.MPI.IN_PLACE, arr)


class MultiGpuCommunicatorNccl(MultiGpuCommunicatorBase):

    def __init__(self):
        super().__init__()

        # Check if GPUs are in default mode
        if cp.cuda.Device().attributes["ComputeMode"] != 0:   ## ComputeModeDefault
            raise RuntimeError(
                "Compute mode must be default in order to use NCCL")

        # get a unique identifier for the NCCL communicator and
        # broadcast it to all MPI processes (assuming one device per process)
        if self.rank == 0:
            try:
                self.id = nccl.get_unique_id()
            except:
                self.id = None
        else:
            self.id = None

        self.id = parallel.bcast(self.id)

        self.com = nccl.NcclCommunicator(self.ndev, self.id, self.rank)

    def allReduceSum(self, arr):
        """Call MPI.all_reduce in-place, with array on GPU"""

        count, datatype = self.__get_NCCL_count_dtype(arr)

        self.com.allReduce(arr.data.ptr, arr.data.ptr, count, datatype, nccl.NCCL_SUM, 
            cp.cuda.get_current_stream().ptr)

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
def get_multi_gpu_communicator(use_nccl=True, use_cuda_mpi=True):
    if have_nccl and use_nccl:
        try:
            comm = MultiGpuCommunicatorNccl()
            log(4, "Using NCCL communicator")
            return comm
        except RuntimeError:
            pass
        except AttributeError:
            # see issue #323
            pass
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
