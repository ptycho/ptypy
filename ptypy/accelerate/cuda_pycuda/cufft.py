
from pycuda import gpuarray
from . import load_kernel
import numpy as np

class FFT_base(object):

    def __init__(self, array, queue=None,
                 inplace=False,
                 pre_fft=None,
                 post_fft=None,
                 symmetric=True,
                 forward=True):
        self._queue = queue
        dims = array.ndim
        if dims < 2:
            raise AssertionError('Input array must be at least 2-dimensional')
        self.arr_shape = (array.shape[-2], array.shape[-1])
        self.batches = int(np.prod(array.shape[0:dims-2]) if dims > 2 else 1)
        self.forward = forward

        self._load(array, pre_fft, post_fft, symmetric, forward)

class FFT_cuda(FFT_base):

    def __init__(self, array, queue=None,
                 inplace=False,
                 pre_fft=None,
                 post_fft=None,
                 symmetric=True,
                 forward=True):
        rows, columns = (array.shape[-2], array.shape[-1])
        if rows != columns or rows not in [16, 32, 64, 128, 256, 512, 1024, 2048]:
            raise ValueError("CUDA FFT only supports powers of 2 for rows/columns, from 16 to 2048")
        super(FFT_cuda, self).__init__(array, queue=queue, 
                                       inplace=inplace,
                                       pre_fft=pre_fft,
                                       post_fft=post_fft,
                                       symmetric=symmetric,
                                       forward=forward)
        
    def _load(self, array, pre_fft, post_fft, symmetric, forward):
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

        import filtered_cufft
        self.fftobj = filtered_cufft.FilteredFFT(
                self.batches, 
                self.arr_shape[0], 
                self.arr_shape[1],
                symmetric, 
                forward,
                self.pre_fft_ptr,
                self.post_fft_ptr, 
                self._queue.handle)

        self.ft = self._ft
        self.ift = self._ift

    @property
    def queue(self):
        return self._queue

    @queue.setter
    def queue(self, queue):
        self._queue = queue
        self.fftobj.queue = queue.handle
    
    def _ft(self, input, output):
        self.fftobj.fft(input.gpudata, output.gpudata)
    
    def _ift(self, input, output):
        self.fftobj.ifft(input.gpudata, output.gpudata)
        

class FFT_skcuda(FFT_base):

    def __init__(self, array, queue=None,
                 inplace=False,
                 pre_fft=None,
                 post_fft=None,
                 symmetric=True,
                 forward=True):
        import skcuda.fft as cu_fft
        self._fft = cu_fft.fft
        self._ifft = cu_fft.ifft
        super(FFT_cuda, self).__init__(array, queue=queue, 
                                inplace=inplace,
                                pre_fft=pre_fft,
                                post_fft=post_fft,
                                symmetric=symmetric,
                                forward=forward)

    @property
    def queue(self):
        return self._queue

    @queue.setter
    def queue(self, queue):
        self._queue = queue
        from skcuda.fft import cufft as cufftlib
        cufftlib.cufftSetStream(self.plan.handle, queue.handle)

    def _load(self, array, pre_fft, post_fft, symmetric, forward):
        assert(array.dtype in [np.complex64, np.complex128])
        assert(pre_fft.dtype in [np.complex64, np.complex128] if pre_fft is not None else True)
        assert(post_fft.dtype in [np.complex64, np.complex128] if post_fft is not None else True)

        math_type = 'float' if array.dtype == np.complex64 else 'double'
        if pre_fft is not None:
            math_type = 'float' if pre_fft.dtype == np.complex64 else 'double'
        self.pre_fft_knl = load_kernel("batched_multiply", {
            'MPY_DO_SCALE': 'false',
            'MPY_DO_FILT': 'true',
            'IN_TYPE': 'float' if array.dtype == np.complex64 else 'double',
            'OUT_TYPE': 'float' if array.dtype == np.complex64 else 'double',
            'MATH_TYPE': math_type
        }) if pre_fft is not None else None

        math_type = 'float' if array.dtype == np.complex64 else 'double'
        if post_fft is not None:
            math_type = 'float' if post_fft.dtype == np.complex64 else 'double'
        self.post_fft_knl = load_kernel("batched_multiply", {
            'MPY_DO_SCALE': 'true' if (not forward and not symmetric) or symmetric else 'false',
            'MPY_DO_FILT': 'true' if post_fft is not None else 'false',
            'IN_TYPE': 'float' if array.dtype == np.complex64 else 'double',
            'OUT_TYPE': 'float' if array.dtype == np.complex64 else 'double',
            'MATH_TYPE': math_type
        }) if (not (forward and not symmetric) or post_fft is not None) else None

        self.block = (32, 32, 1)
        self.grid = (
            int((self.arr_shape[0] + 31) // 32),
            int((self.arr_shape[1] + 31) // 32),
            int(self.batches)
        )
        import skcuda.fft as cu_fft
        self.plan = cu_fft.Plan(
            self.arr_shape,
            array.dtype,
            array.dtype,
            self.batches,
            self._queue
        )
        # with cuFFT, we need to scale ifft
        if not symmetric and not forward:
            self.scale = 1 / np.prod(self.arr_shape)
        elif forward and not symmetric:
            self.scale = 1.0
        else:
            self.scale = 1 / np.sqrt(np.prod(self.arr_shape))

        if pre_fft is not None:
            self.pre_fft = gpuarray.to_gpu(pre_fft)
        else:
            self.pre_fft = np.intp(0)  # NULL
        if post_fft is not None:
            self.post_fft = gpuarray.to_gpu(post_fft)
        else:
            self.post_fft = np.intp(0)

        self.ft = self._ft
        self.ift = self._ift

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

    def _ft(self, x, y):
        d = self._prefilt(x, y)
        self._fft(d, y, self.plan)
        self._postfilt(y)

    def _ift(self, x, y):
        d = self._prefilt(x, y)
        self._ifft(d, y, self.plan)
        self._postfilt(y)

