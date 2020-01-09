import numpy as np
import pycuda.driver as cuda
from pycuda import gpuarray
from pycuda.tools import make_default_context
from ptypy.accelerate.py_cuda.fft import FFT
import time
import skcuda.fft as cu_fft 

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

prop_fwd = FFT(f, stream, pre_fft=None, post_fft=None, inplace=True, symmetric=True)

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

plan_fwd = cu_fft.Plan((B, C), np.complex64, np.complex64, A, stream)

start_cufft.record(stream)
for p in range(100):
    cu_fft.fft(f_d, f_d, plan_fwd)
stop_cufft.record(stream)
start_cufft.synchronize()
stop_cufft.synchronize()
time_cufft = stop_cufft.time_since(start_cufft)

print('CUFFT for {}: {}ms'.format((A,B,C), time_cufft))
print('CUFFT Speedup: {}x'.format(time_reikna/time_cufft))

ctx.pop()
ctx.detach()