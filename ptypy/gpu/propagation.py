'''
All propagation based kernels 
'''
import numpy as np


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

    if direction is 'forward':
        fft = np.fft.fft2
        # fft = sci.fftpack.fft2
        sc = 1.0 / np.sqrt(np.prod(data_to_be_transformed.shape[-2:]))

    elif direction is 'backward':
        fft  = np.fft.ifft2
        # fft = sci.fftpack.ifft2
        sc = np.sqrt(np.prod(data_to_be_transformed.shape[-2:]))
    
    if (prefilter is None) and (postfilter is None):
        return fft(data_to_be_transformed, axes=(-2,-1)) * sc
    elif (prefilter is None) and (postfilter is not None):
        return np.multiply(postfilter, fft(data_to_be_transformed, axes=(-2,-1))) * sc
    elif (prefilter is not None) and (postfilter is None):
        return fft(np.multiply(data_to_be_transformed, prefilter), axes=(-2,-1))* sc
    elif (prefilter is not None) and (postfilter is not None):
        return np.multiply(postfilter, fft(np.multiply(data_to_be_transformed, prefilter), axes=(-2,-1))) * sc

def sqrt_abs(diffraction):
    return np.sqrt(np.abs(diffraction))






