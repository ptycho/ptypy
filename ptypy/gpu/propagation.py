'''
All propagation based kernels 
'''
import numpy as np
from .array_utils import grids

def forward_transform(serialized_scan, mode='farfield', prefilter=None, postfilter=None):
    '''
    performs a fourier transform on the exit wave. Can be either farfield or nearfield depending on argument
    useful for technique development
    also, will be the fastest.
    '''
    exit_wave = serialized_scan['exit wave']
    if (prefilter and postfilter) is None:
        return np.fft.fft2(exit_wave, axes=(-2,-1))
    elif (prefilter is None) and postfilter:
        return np.multiply(postfilter, np.fft.fft2(exit_wave, axes=(-2,-1)))
    elif prefilter and (postfilter is None):
        return np.fft.fft2(np.multiply(exit_wave, prefilter), axes=(-2,-1))
    elif prefilter and postfilter:
        return np.multiply(postfilter, np.fft.fft2(np.multiply(exit_wave, prefilter), axes=(-2,-1)))


def inverse_transform(corrected_farfield_stack, mode='farfield', prefilter=None, postfilter=None):
    '''
    performs a fourier transform on the exit wave. Can be either farfield or nearfield depending on argument
    useful for technique development
    useful for technique development
    also, will be the fastest.
    '''
    if (prefilter and postfilter) is None:
        return np.fft.ifft2(corrected_farfield_stack, axes=(-2,-1))
    elif (prefilter is None) and postfilter:
        return np.multiply(postfilter, np.fft.ifft2(corrected_farfield_stack, axes=(-2,-1)))
    elif prefilter and (postfilter is None):
        return np.fft.ifft2(np.multiply(corrected_farfield_stack, prefilter), axes=(-2,-1))
    elif prefilter and postfilter:
        return np.multiply(postfilter, np.fft.ifft2(np.multiply(corrected_farfield_stack, prefilter), axes=(-2,-1)))

def generate_far_field_fft_filters(lz, shape, psam, pdet):
    '''
    generates the filters required for the fourier transforms in case we decide to do these on the gpu
    Params:
        lz = Scaling parameter distance x wavelength
        shape = array shape
        psam = sample space pixel size
        pdet = detector space pixel size
    Returns:
        prefft =  filter for before the fft
        postfft = filter for after the fft
        preifft = filter for before the ifft
        postifft = filter for after the ifft
    '''
    
    [X, Y] = grids(shape, psam, center='geometric')
    [V, W] = grids(shape, pdet, center='geometric')
    
    fftshiftA = np.exp(1j * np.pi * (X ** 2 + Y ** 2) / lz)
    filtA = np.exp(-2.0 * np.pi * 1j * ((X-X[0, 0]) * V[0, 0] + (Y-Y[0, 0]) * W[0, 0]) / lz) # not sure what this is for
    prefft = np.multiply(filtA, fftshiftA)
    fftshiftB = np.exp(1j * np.pi * (V**2 + W**2) / lz)
    filtB = np.exp(-2.0 * np.pi * 1j * (X[0, 0]*V + Y[0, 0]*W) / lz)
    postfft = np.multiply(fftshiftB, filtB) * 1.0 / np.sqrt(np.prod(filtB.shape))
    return prefft, postfft, prefft.conj(), postfft.conj()
