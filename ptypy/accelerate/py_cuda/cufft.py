import skcuda.fft as cu_fft
from pycuda.compiler import SourceModule
from pycuda import gpuarray
from . import load_kernel
import numpy as np

class FFT(object):

    def __init__(self, array, queue=None,
                 inplace=False,
                 pre_fft=None,
                 post_fft=None,
                 symmetric=True):
        self.queue = queue
        
        self.pre_fft_knl = load_kernel("batched_multiply", {
            'MPY_DO_SCALE': 'false',
            'MPY_DO_FILT': 'true'
        }) if pre_fft is not None else None

        self.post_fft_knl = load_kernel("batched_multiply", {
            'MPY_DO_SCALE': 'true' if symmetric else 'false',
            'MPY_DO_FILT': 'true' if post_fft is not None else 'false'
        }) if (symmetric or post_fft is not None) else None

        dims = array.ndim
        if dims < 2:
            raise AssertionError('Input array must be at least 2-dimensional')
        self.arr_shape = (array.shape[-2], array.shape[-1])
        self.batches = int(np.product(array.shape[0:dims-2]) if dims > 2 else 1)
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
        self.scale = 1 / np.sqrt(np.product(self.arr_shape))
        
        if pre_fft is not None:
            self.pre_fft = gpuarray.to_gpu(pre_fft)
        else:
            self.pre_fft = gpuarray.empty((1,), dtype=np.complex64)
        if post_fft is not None:
            self.post_fft = gpuarray.to_gpu(post_fft)
        else:
            self.post_fft = gpuarray.empty((1,), dtype=np.complex64)

    def _prefilt(self, x, y):
        if self.pre_fft_knl:
            self.pre_fft_knl(x, y, self.pre_fft, 
                             np.float32(self.scale),
                             np.int32(self.batches),
                             np.int32(self.arr_shape[0]),
                             np.int32(self.arr_shape[1]),
                             block=self.block,
                             grid=self.grid,
                             stream=self.queue)
            return y
        else:
            return x

    def _postfilt(self, y):
        if self.post_fft_knl:
            self.post_fft_knl(y, y, self.post_fft, np.float32(self.scale),
                              np.int32(self.batches),
                              np.int32(self.arr_shape[0]), 
                              np.int32(self.arr_shape[1]),
                              block=self.block, grid=self.grid,
                              stream=self.queue)

    def ft(self, x, y):
        d = self._prefilt(x, y)            
        cu_fft.fft(d, y, self.plan)
        self._postfilt(y)
    
    def ift(self, x, y):
        d = self._prefilt(x, y)
        cu_fft.ifft(d, y, self.plan)
        self._postfilt(y)