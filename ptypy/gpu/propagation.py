'''
All propagation based kernels 
'''
import numpy as np


def farfield_propagator(data_to_be_transformed, prefilter=None, postfilter=None, direction='forward'):
    '''
    performs a fourier transform on the exit wave. Can be either farfield or nearfield depending on argument
    useful for technique development
    also, will be the fastest.
    '''
    dt = data_to_be_transformed.dtype
    if direction is 'forward':
        fft = np.fft.fft2
        sc = 1.0 / np.sqrt(np.prod(data_to_be_transformed.shape[-2:]))

    elif direction is 'backward':
        fft  = np.fft.ifft2
        sc = np.sqrt(np.prod(data_to_be_transformed.shape[-2:]))
    
    if (prefilter is None) and (postfilter is None):
        
        return fft(data_to_be_transformed, axes=(-2,-1)).astype(dt) * sc
    elif (prefilter is None) and (postfilter is not None):
        return np.multiply(postfilter.astype(dt), fft(data_to_be_transformed, axes=(-2,-1)).astype(dt)) * sc
    elif (prefilter is not None) and (postfilter is None):
        return fft(np.multiply(data_to_be_transformed, prefilter.astype(dt)), axes=(-2,-1)).astype(dt) * sc
    elif (prefilter is not None) and (postfilter is not None):
        return np.multiply(postfilter.astype(dt), fft(np.multiply(data_to_be_transformed, prefilter.astype(dt)), axes=(-2,-1)).astype(dt)) * sc

def fourier_constraint():