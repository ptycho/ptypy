from ptypy.accelerate.cuda_pycuda import load_kernel
import numpy as np
from ptypy.accelerate.base import address_manglers as npam
from pycuda import gpuarray

class BaseMangler(npam.BaseMangler):

    def __init__(self, max_step_per_shift,  start, stop, nshifts, max_bound=None, randomseed=None, queue_thread=None):
        super().__init__(
            max_step_per_shift=max_step_per_shift, 
            start=start, 
            stop=stop,
            nshifts=nshifts,
            max_bound=max_bound,
            randomseed=randomseed)
        self.queue = queue_thread
        self.get_address_cuda = load_kernel("get_address")
        self.delta = None

    def get_address(self, index, addr_current, mangled_addr, max_oby, max_obx):
        assert addr_current.dtype == np.int32, "addresses must be int32"
        assert mangled_addr.dtype == np.int32, "addresses must be int32"
        assert len(addr_current.shape) == 4, "addresses must be 4 dimensions"
        assert addr_current.shape == mangled_addr.shape, "output addresses must be pre-allocated"
        assert self.delta is not None, "Deltas are not set yet - call setup_shifts first"
        assert index < self.delta.shape[0], "Index out of range for deltas"

        # only using a single thread block here as it's not enough work
        # otherwise
        self.get_address_cuda(
            addr_current,
            mangled_addr,
            np.int32(addr_current.shape[0] * addr_current.shape[1]),
            self.delta[index,None],
            np.int32(max_oby),
            np.int32(max_obx),
            block=(64,1,1),
            grid=(1, 1, 1),
            stream=self.queue)

# with multiple inheritance, we have to be explicit which super class 
# we are calling in the methods

class RandomIntMangler(BaseMangler, npam.RandomIntMangler):

    def __init__(self, *args, **kwargs):
        BaseMangler.__init__(self, *args, **kwargs)

    def setup_shifts(self, current_iteration, nframes):
        npam.RandomIntMangler.setup_shifts(self, current_iteration, nframes=nframes)
        self.delta = gpuarray.to_gpu(self.delta.astype(np.int32))


class GridSearchMangler(BaseMangler, npam.GridSearchMangler):

    def __init__(self, *args, **kwargs):
        BaseMangler.__init__(self, *args, **kwargs)

    def setup_shifts(self, current_iteration, nframes):
        npam.GridSearchMangler.setup_shifts(self, current_iteration, nframes=nframes)
        self.delta = gpuarray.to_gpu(self.delta.astype(np.int32))

    def get_address(self, index, addr_current, mangled_addr, max_oby, max_obx):
        BaseMangler.get_address(self, index, addr_current, mangled_addr, max_oby, max_obx)