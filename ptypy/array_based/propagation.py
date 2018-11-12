'''
All propagation based kernels 
'''
import numpy as np
from scipy import fftpack

# this is now the same behaviour as the geo class.
try:
    pe = 'FFTW_MEASURE'
    from pyfftw.interfaces import numpy_fft
    fft2 = lambda x: numpy_fft.fft2(x, planner_effort=pe)
    ifft2 = lambda x: numpy_fft.ifft2(x, planner_effort=pe)
except ImportError:
    fft2 = lambda x: fftpack.fft2(x).astype(x.dtype)
    ifft2 = lambda x: fftpack.ifft2(x).astype(x.dtype)

from . import COMPLEX_TYPE

def farfield_propagator(data_to_be_transformed, prefilter=None, postfilter=None, direction='forward'):
    '''
    performs a fourier transform on the nd exit wave stack. FFT shift and normalisation performed by
    multiplication with prefilter and postfilter
    :param data_to_be_transformed. The nd stack of the current iterant.
    :param prefilter. The filter to multiply before fourier transforming. Default: None.
    :param postfilter. The filter to multiply after fourier transforming. Default: None.
    :param direction. The direction of the transform forward or backward. Default: Forward.
    :return: The transformed stack.
    '''

    dtype = data_to_be_transformed.dtype
    if direction is 'forward':
        def fft(x):
            output = np.zeros(x.shape, dtype=dtype)
            for idx in range(output.shape[0]):
                output[idx] = fft2(x[idx])
            return output

        sc = 1.0 / np.sqrt(np.prod(data_to_be_transformed.shape[-2:]))

    elif direction is 'backward':
        def fft(x):
            output = np.zeros(x.shape, dtype=dtype)
            for idx in range(output.shape[0]):
                output[idx] = ifft2(x[idx])
            return output

        sc = np.sqrt(np.prod(data_to_be_transformed.shape[-2:]))

    if (prefilter is None) and (postfilter is None):
        return fft(data_to_be_transformed) * sc
    elif (prefilter is None) and (postfilter is not None):
        postfilter = postfilter.astype(dtype)
        return np.multiply(postfilter, fft(data_to_be_transformed)) * sc
    elif (prefilter is not None) and (postfilter is None):
        prefilter = prefilter.astype(dtype)
        return fft(np.multiply(data_to_be_transformed, prefilter))* sc
    elif (prefilter is not None) and (postfilter is not None):
        prefilter = prefilter.astype(dtype)
        postfilter = postfilter.astype(dtype)
        return np.multiply(postfilter, fft(np.multiply(data_to_be_transformed, prefilter))) * sc

def sqrt_abs(diffraction):
    return np.sqrt(np.abs(diffraction))

