"""
Temporary file to test the quality-map unwrapping function.

Compile the c file with:
    gcc -O3 -fPIC -shared -o libqmunwrap.so _qmunwrap.c -lm
"""
import numpy as np
import ctypes

# Load the shared library
lib = ctypes.CDLL('./libqmunwrap.so')

# Define the argument and return types for the unwrap function
lib.unwrap.argtypes = [
    ctypes.POINTER(ctypes.c_double),  # phase
    ctypes.c_int,                     # N0
    ctypes.c_int,                     # N1
    ctypes.c_int,                     # num_levels
    ctypes.c_int,                     # start0
    ctypes.c_int,                     # start1
    ctypes.POINTER(ctypes.c_double)   # aout
]
lib.unwrap.restype = None

def unwrap(phase, num_levels=8, start=(0, 0)):
    phase = np.ascontiguousarray(phase, dtype=np.float64)
    N0, N1 = phase.shape
    aout = np.empty_like(phase)
    # Call the C function
    lib.unwrap(
        phase.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        N0,
        N1,
        num_levels,
        start[0],
        start[1],
        aout.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    )
    return aout

def test():
    from scipy.ndimage import gaussian_filter
    sh = (100,100)
    T = np.exp(1j*20*(gaussian_filter(np.random.normal(size=sh),10) + 3*gaussian_filter(np.random.normal(size=sh), 5)))
    phase = np.angle(T)

    # Unwrap the phase
    unwrapped_phase = unwrap(phase, num_levels=8)

    # Plot the results
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title('Original Phase')
    plt.imshow(phase)
    plt.colorbar()
    plt.subplot(1, 2, 2)
    plt.title('Unwrapped Phase')
    plt.imshow(unwrapped_phase)
    plt.colorbar()
    plt.show()