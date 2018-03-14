'''
useful utilities from ptypy that should be ported to gpu. These don't ahve external dependencies
'''
import numpy as np
from . import COMPLEX_TYPE


def abs2(input):
    raise NotImplementedError("This method is not implemented yet.")


def sum_to_buffer(in1, outshape, in1_addr, out1_addr, dtype):
    raise NotImplementedError("This method is not implemented yet.")


def norm2(input):
    '''
    Input here could be a variety of 1D, 2D, 3D complex or real. all will be single precision at least.
    return should be real
    '''
    raise NotImplementedError("This method is not implemented yet.")

def complex_gaussian_filter(input, mfs):
    '''
    takes 2D and 3D arrays. Complex input, complex output. mfs has len==input.ndim
    '''
    raise NotImplementedError("This method is not implemented yet.")



def mass_center(A):
    '''
    Input will always be real, and 2d or 3d, single precision here
    '''
    raise NotImplementedError("This method is not implemented yet.")



def interpolated_shift(c, shift):
    '''
    complex bicubic interpolated shift.
    complex output. This shift should be applied to 2D arrays. shift should have len=c.ndims 
    '''
    raise NotImplementedError("This method is not implemented yet.")


def clip_complex_magnitudes_to_range(complex_input, clip_min, clip_max):
    '''
    This takes a single precision 2D complex input, clips the absolute magnitudes to be within a range, but leaves the phase untouched.
    '''
    raise NotImplementedError("This method is not implemented yet.")
