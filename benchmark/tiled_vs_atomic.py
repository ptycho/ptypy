
import numpy as np
import pycuda.driver as cuda
from pycuda import gpuarray
from pycuda.tools import make_default_context
from ptypy.accelerate.cuda_pycuda.kernels import PoUpdateKernel as POK
import gc

cuda.init()


COMPLEX_TYPE = np.complex64
FLOAT_TYPE = np.float32
INT_TYPE = np.int32

def prepare_arrays(
    overlap=0.2,
    scan_pts=20,
    frame_size =128,
    atomics=True,
    num_pr_modes=2,
    num_ob_modes=1):

    fsh = (frame_size,frame_size)
    shift=int(frame_size*overlap)
    X, Y = np.indices((scan_pts,scan_pts)) * shift
    X = X.flatten()
    Y = Y.flatten()
    num_pts=len(X)
    X+=5
    Y+=5
    osh=(X.max()+5+fsh[0],Y.max()+5+fsh[1]) # ob shape
    #print(fsh, osh, X.min(), X.max()+fsh[0], Y.min(),Y.max()+fsh[1])
    num_modes = num_ob_modes * num_pr_modes
    A = num_pts * num_modes  # this is a 16 point scan pattern (4x4 grid) over all the modes

    probe = np.empty(shape=(num_pr_modes,fsh[0],fsh[1]), dtype=COMPLEX_TYPE)
    for idx in range(num_pr_modes):
        probe[idx] = np.ones(fsh) * (idx + 1) + 1j * np.ones(fsh) * (idx + 1)

    object_array = np.empty(shape=(num_ob_modes,osh[0],osh[1]), dtype=COMPLEX_TYPE)
    for idx in range(num_ob_modes):
        object_array[idx] = np.ones(osh) * (3 * idx + 1) + 1j * np.ones(osh) * (3 * idx + 1)

    exit_wave = np.empty(shape=(A,fsh[0],fsh[1]), dtype=COMPLEX_TYPE)
    for idx in range(A):
        exit_wave[idx] = np.ones(fsh) * (idx + 1) + 1j * np.ones(fsh) * (idx + 1)

    addr = np.zeros((num_pts, num_modes, 5, 3), dtype=INT_TYPE)
    exit_idx = 0
    position_idx = 0
    for xpos, ypos in zip(X, Y):  #
        mode_idx = 0
        for pr_mode in range(num_pr_modes):
            for ob_mode in range(num_ob_modes):
                addr[position_idx, mode_idx] = np.array([[pr_mode, 0, 0],
                                                         [ob_mode, ypos, xpos],
                                                         [exit_idx, 0, 0],
                                                         [0, 0, 0],
                                                         [0, 0, 0]], dtype=INT_TYPE)
                mode_idx += 1
                exit_idx += 1
        position_idx += 1

    if not atomics:
        addr = np.ascontiguousarray(np.transpose(addr, (2, 3, 0, 1)))

    #print(addr)
    object_array_denominator = np.empty_like(object_array, dtype=FLOAT_TYPE)
    for idx in range(num_ob_modes):
        object_array_denominator[idx] = np.ones(osh) * (5 * idx + 2)  # + 1j * np.ones(osh) * (5 * idx + 2)

    probe_denominator = np.empty_like(probe, dtype=FLOAT_TYPE)
    for idx in range(num_pr_modes):
        probe_denominator[idx] = np.ones(fsh) * (5 * idx + 2)  # + 1j * np.ones(fsh) * (5 * idx + 2)

    return (gpuarray.to_gpu(addr),
            gpuarray.to_gpu(object_array),
            gpuarray.to_gpu(object_array_denominator),
            gpuarray.to_gpu(probe),
            gpuarray.to_gpu(exit_wave),
            gpuarray.to_gpu(probe_denominator))

#for overlap in [0.2]:
#    for
ctx = make_default_context()
stream = cuda.Stream()
pok = POK(stream)

overlap=0.2
scan_pts=20
for frame_size in [64,128,256,512]:
    for atomics in [True, False]:
        for overlap in [0.01,0.01,0.02,0.04,0.08]:

            addr, ob, obn, pr, ex, prn = prepare_arrays(overlap, scan_pts, frame_size, atomics)
            stream.synchronize()
            start = cuda.Event()
            stop = cuda.Event()
            start.record(stream)
            for p in range(1):
                pok.ob_update(addr, ob, obn, pr, ex, atomics)
            stop.record(stream)
            stop.synchronize()
            dt = stop.time_since(start)
            stream.synchronize()
            print('10x for {}: {}ms'.format((overlap,frame_size,scan_pts, atomics), dt))

del stream
ctx.pop()
ctx.detach()
del addr, ob, obn, pr, ex, prn
gc.collect()