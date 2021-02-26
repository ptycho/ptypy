import skcuda.fft as cu_fft
from skcuda.fft import cufft as cufftlib
from pycuda.compiler import SourceModule
from pycuda import gpuarray
from . import load_kernel
import numpy as np

class FFT(object):

    def __init__(self, array, queue=None,
                 inplace=False,
                 pre_fft=None,
                 post_fft=None,
                 symmetric=True,
                 forward=True,
                 use_external=True):
        self._queue = queue
        dims = array.ndim
        if dims < 2:
            raise AssertionError('Input array must be at least 2-dimensional')
        self.arr_shape = (array.shape[-2], array.shape[-1])
        self.batches = int(np.product(array.shape[0:dims-2]) if dims > 2 else 1)
        self.use_external = use_external
        self.forward = forward

        if use_external:
            self._load_filtered_fft(array, pre_fft, post_fft, symmetric, forward)
        else:
            self._load_separate_knls(array, pre_fft, post_fft, symmetric, forward)

    def _load_filtered_fft(self, array, pre_fft, post_fft, symmetric, forward):
        if pre_fft is not None:
            self.pre_fft = gpuarray.to_gpu(pre_fft)
            self.pre_fft_ptr = self.pre_fft.gpudata
        else:
            self.pre_fft_ptr = 0
        if post_fft is not None:
            self.post_fft = gpuarray.to_gpu(post_fft)
            self.post_fft_ptr = self.post_fft.gpudata
        else:
            self.post_fft_ptr = 0

        from . import import_fft
        mod = import_fft.ImportFFT(self.arr_shape[0], self.arr_shape[1]).get_mod()
        self.fftobj = mod.FilteredFFT(
                self.batches, 
                symmetric, 
                forward,
                self.pre_fft_ptr,
                self.post_fft_ptr, 
                self._queue.handle)

        self.ft = self._ft_ext
        self.ift = self._ift_ext

    @property
    def queue(self):
        return self._queue

    @queue.setter
    def queue(self, queue):
        self._queue = queue
        if not self.use_external:
            cufftlib.cufftSetStream(self.plan.handle, queue.handle)
        else:
            self.fftobj.queue = queue.handle
    
    def _ft_ext(self, input, output):
        self.fftobj.fft(input.gpudata, output.gpudata)
    
    def _ift_ext(self, input, output):
        self.fftobj.ifft(input.gpudata, output.gpudata)
        
    def _load_separate_knls(self, array, pre_fft, post_fft, symmetric, forward):
        self.pre_fft_knl = load_kernel("batched_multiply", {
            'MPY_DO_SCALE': 'false',
            'MPY_DO_FILT': 'true'
        }) if pre_fft is not None else None

        self.post_fft_knl = load_kernel("batched_multiply", {
            'MPY_DO_SCALE': 'true' if (not forward and not symmetric) or symmetric else 'false',
            'MPY_DO_FILT': 'true' if post_fft is not None else 'false'
        }) if (not (forward and not symmetric) or post_fft is not None) else None

        self.block = (32, 32, 1)
        self.grid = (
            int((self.arr_shape[0] + 31) // 32),
            int((self.arr_shape[1] + 31) // 32),
            int(self.batches)
        )
        self.plan = cu_fft.Plan(
            self.arr_shape,
            array.dtype,
            array.dtype,
            self.batches,
            self.queue
        )
        # with cuFFT, we need to scale ifft
        if not symmetric and not forward:
            self.scale = 1 / np.product(self.arr_shape)
        elif forward and not symmetric:
            self.scale = 1.0
        else:
            self.scale = 1 / np.sqrt(np.product(self.arr_shape))
        
        if pre_fft is not None:
            self.pre_fft = gpuarray.to_gpu(pre_fft)
        else:
            self.pre_fft = np.intp(0)  # NULL
        if post_fft is not None:
            self.post_fft = gpuarray.to_gpu(post_fft)
        else:
            self.post_fft = np.intp(0)
        
        self.ft = self._ft_separate
        self.ift = self._ift_separate


    def _prefilt(self, x, y):
        if self.pre_fft_knl:
            self.pre_fft_knl(x, y, self.pre_fft, 
                             np.float32(self.scale),
                             np.int32(self.batches),
                             np.int32(self.arr_shape[0]),
                             np.int32(self.arr_shape[1]),
                             block=self.block,
                             grid=self.grid,
                             stream=self._queue)
            return y
        else:
            return x

    def _postfilt(self, y):
        if self.post_fft_knl:
            assert self.post_fft is not None
            assert self.scale is not None
            self.post_fft_knl(y, y, self.post_fft, np.float32(self.scale),
                              np.int32(self.batches),
                              np.int32(self.arr_shape[0]), 
                              np.int32(self.arr_shape[1]),
                              block=self.block, grid=self.grid,
                              stream=self._queue)

    def _ft_separate(self, x, y):
        d = self._prefilt(x, y)            
        cu_fft.fft(d, y, self.plan)
        self._postfilt(y)
    
    def _ift_separate(self, x, y):
        d = self._prefilt(x, y)
        cu_fft.ifft(d, y, self.plan)
        self._postfilt(y)