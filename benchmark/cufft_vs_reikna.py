"""
Tests cuFFT vs Reikna FFT performance (not accuracy). 

Together with a C++ implementation of cuFFT with callbacks,
we get the following numbers on a P100 GPU:

For 100 calls of 256x256 with batch size 2000:
- Reikna with or without filters: 1,470ms
- cuFFT without filters         :   792ms
- cuFFT with separate filters   : 1,564ms
- cuFFT with callbacks          :   916ms

For 128x128 with batch size 2000:
- Reikna with or without filters:   389ms
- cuFFT without filters         :   194ms
- cuFFT with separate filters   :   388ms
- cuFFT with callbacks          :   223ms
"""


import numpy as np
import pycuda.driver as cuda
from pycuda import gpuarray
from pycuda.tools import make_default_context
from ptypy.accelerate.cuda_pycuda.fft import FFT 
from ptypy.accelerate.cuda_pycuda.cufft import FFT_cuda as cuFFT
import time

ctx = make_default_context()
stream = cuda.Stream()

A = 2000
B = 256
C = 256

COMPLEX_TYPE = np.complex64


f = np.empty(shape=(A, B, C), dtype=np.complex64)
for idx in range(A):
    f[idx] = np.ones((B, C)) * (idx + 1) + 1j * np.ones((B, C)) * (idx + 1)

prefilter = (np.arange(B*C).reshape((B, C)) + 1j* np.arange(B*C).reshape((B, C))).astype(COMPLEX_TYPE)
postfilter = (np.arange(13, 13 + B*C).reshape((B, C)) + 1j*np.arange(13, 13 + B*C).reshape((B, C))).astype(COMPLEX_TYPE)

f_d = gpuarray.to_gpu(f)

prop_fwd = FFT(f, stream, pre_fft=prefilter, post_fft=postfilter, inplace=True, symmetric=True)

start_reikna = cuda.Event()
stop_reikna = cuda.Event()
start_cufft = cuda.Event()
stop_cufft = cuda.Event()

start_reikna.record(stream)
for p in range(100):
    prop_fwd.ft(f_d, f_d)
stop_reikna.record(stream)
start_reikna.synchronize()
stop_reikna.synchronize()
time_reikna = stop_reikna.time_since(start_reikna) 

print('Reikna for {}: {}ms'.format((A,B,C), time_reikna))

# with pre- and post-filter
cuprop_fw = cuFFT(f, stream, pre_fft=prefilter, post_fft=postfilter, inplace=True, symmetric=True)
# without filters, and symmetric=False avoids scaling the result
#cuprop_fw = cuFFT(f, stream, pre_fft=None, post_fft=None, inplace=True, symmetric=False)

start_cufft.record(stream)
for p in range(100):
    cuprop_fw.ft(f_d, f_d)
stop_cufft.record(stream)
start_cufft.synchronize()
stop_cufft.synchronize()
time_cufft = stop_cufft.time_since(start_cufft)

print('CUFFT for {}: {}ms'.format((A,B,C), time_cufft))
print('CUFFT Speedup: {}x'.format(time_reikna/time_cufft))

ctx.pop()
ctx.detach()