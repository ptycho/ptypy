from . import load_kernel
import numpy as np
from ptypy.accelerate.base import address_manglers as npam
import cupy as cp


class BaseMangler(npam.BaseMangler):

    def __init__(self, *args, queue_thread=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.queue = queue_thread
        self.get_address_cuda = load_kernel("get_address")
        self.delta = None
        self.delta_gpu = None

    def _setup_delta_gpu(self):
        if self.queue is not None:
            self.queue.use()
        assert self.delta is not None, "Setup delta using the setup_shifts method first"
        self.delta = np.ascontiguousarray(self.delta, dtype=np.int32)
        
        if self.delta_gpu is None or self.delta_gpu.shape[0] < self.delta.shape[0]:
            self.delta_gpu = cp.empty(self.delta.shape, dtype=np.int32)
        # in case self.delta is smaller than delta_gpu, this will only copy the
        # relevant part
        cp.cuda.runtime.memcpy(dst=self.delta_gpu.data.ptr,
                               src=self.delta.ctypes.data,
                               size=self.delta.size * self.delta.itemsize,
                               kind=1) # host to device
        

    def get_address(self, index, addr_current, mangled_addr, max_oby, max_obx):
        assert addr_current.dtype == np.int32, "addresses must be int32"
        assert mangled_addr.dtype == np.int32, "addresses must be int32"
        assert len(addr_current.shape) == 4, "addresses must be 4 dimensions"
        assert addr_current.shape == mangled_addr.shape, "output addresses must be pre-allocated"
        assert self.delta_gpu is not None, "Deltas are not set yet - call setup_shifts first"
        assert index < self.delta_gpu.shape[0], "Index out of range for deltas"
        assert isinstance(self.delta_gpu, cp.ndarray), "Only GPU arrays are supported for delta"
        
        if self.queue is not None:
            self.queue.use()

        # only using a single thread block here as it's not enough work
        # otherwise
        self.get_address_cuda(
            (1, 1, 1),
            (64, 1, 1),
            (addr_current,
            mangled_addr,
            np.int32(addr_current.shape[0] * addr_current.shape[1]),
            self.delta_gpu[index,None],
            np.int32(max_oby),
            np.int32(max_obx)))

# with multiple inheritance, we have to be explicit which super class 
# we are calling in the methods
class RandomIntMangler(BaseMangler, npam.RandomIntMangler):

    def __init__(self, *args, **kwargs):
        BaseMangler.__init__(self, *args, **kwargs)

    def setup_shifts(self, *args, **kwargs):
        npam.RandomIntMangler.setup_shifts(self, *args, **kwargs)
        self._setup_delta_gpu()

    def get_address(self, *args, **kwargs):
        BaseMangler.get_address(self, *args, **kwargs)


class GridSearchMangler(BaseMangler, npam.GridSearchMangler):

    def __init__(self, *args, **kwargs):
        BaseMangler.__init__(self, *args, **kwargs)

    def setup_shifts(self, *args, **kwargs):
        npam.GridSearchMangler.setup_shifts(self, *args, **kwargs)
        self._setup_delta_gpu()

    def get_address(self, *args, **kwargs):
        BaseMangler.get_address(self, *args, **kwargs)