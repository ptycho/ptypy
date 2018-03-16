'''
useful utilities from ptypy that should be ported to gpu. These don't ahve external dependencies
'''
import numpy as np
from scipy import ndimage as ndi


def abs2(input):
    '''
    
    :param input. An array that we want to take the absolute value of and square. Can be inplace. Can be complex or real. 
    :return: The real valued abs**2 array
    '''
    return np.multiply(input, input.conj()).real

def sum_to_buffer(in1, outshape, in1_addr, out1_addr, dtype):
    '''
    :param in1. An array . Can be inplace. Can be complex or real.
    :param outshape. An array.
    :param in1_addr. An array . Can be inplace. Can be complex or real.
    :param out1_addr. An array . Can be inplace. Can be complex or real.
    :return: The real valued abs**2 array
    '''
    out1 = np.zeros(outshape, dtype=dtype)
    inshape = in1.shape
    for i1, o1 in zip(in1_addr, out1_addr):
        out1[o1[0], o1[1]:(o1[1] + inshape[1]), o1[2]:(o1[2] + inshape[2])] += in1[i1[0]]
    return out1

def norm2(input):
    '''
    Input here could be a variety of 1D, 2D, 3D complex or real. all will be single precision at least.
    return should be real
    '''
    return np.sum(abs2(input))

def complex_gaussian_filter(input, mfs):
    '''
    takes 2D and 3D arrays. Complex input, complex output. mfs has len==input.ndim
    '''
    return (ndi.gaussian_filter(np.real(input), mfs) +1j *ndi.gaussian_filter(np.imag(input), mfs)).astype(input.dtype)

def mass_center(A):
    '''
    Input will always be real, and 2d or 3d, single precision here
    '''
    return np.array(ndi.measurements.center_of_mass(A))

def interpolated_shift(c, shift):
    '''
    complex bicubic interpolated shift.
    complex output. This shift should be applied to 2D arrays. shift should have len=c.ndims 
    
    '''
    return ndi.interpolation.shift(np.real(c), shift, order=5) + 1j*ndi.interpolation.shift(np.imag(c), shift, order=5)


def clip_complex_magnitudes_to_range(complex_input, clip_min, clip_max):
    '''
    This takes a single precision 2D complex input, clips the absolute magnitudes to be within a range, but leaves the phase untouched.
    '''
    ampl = np.abs(complex_input)
    phase = np.exp(1j * np.angle(complex_input))
    ampl = np.clip(ampl, clip_min, clip_max)
    complex_input[:] = ampl * phase


def delxf(h, axis=-1, out=None):
    """\
    Forward first order derivative for finite difference calculation.
    f(x+h)-f(x)
    .. note::
        The last element along the derivative direction is set to 0.\n
        Pixel units are used (:math:`\\Delta x = \\Delta h = 1`).

    Parameters
    ----------
    h : ndarray
        Input array.

    axis : int, Default=-1, optional
        Which direction used for the derivative.

    out : ndarray, Default=None, optional
        Array in wich the resault is written (same size as ``a``).

    Returns
    -------
    out : ndarray
        Derived array.
    """
    nd = h.ndim
    axis = range(nd)[axis]

    slice1 = [slice(1, None) if i == axis else slice(None) for i in range(nd)]
    slice2 = [slice(None, -1) if i == axis else slice(None) for i in range(nd)]

    if out is None:
        out = np.zeros_like(h)

    out[slice2] = h[slice1] - h[slice2]

    if out is h:
        # required for in-place operation
        slice3 = [slice(-2, None) if i == axis else slice(None)
                  for i in range(nd)]
        out[slice3] = 0.0

    return out


def delxb(h, axis=-1):
    """\
    Backward first order derivative for finite difference calculation.
    f(x)-f(x-h)
    .. note::
        The first element along the derivative direction is set to 0.\n
        Pixel units are used (:math:`\\Delta x = \\Delta h = 1`).

    Parameters
    ----------
    h : ndarray
        Input array.

    axis : int, Default=-1, optional
        Which direction used for the derivative.

    Returns
    -------
    out : ndarray
        Derived array.
    """

    nd = h.ndim
    axis = range(nd)[axis]
    slice1 = [slice(1, None) if i == axis else slice(None) for i in range(nd)]
    slice2 = [slice(None, -1) if i == axis else slice(None) for i in range(nd)]
    b = np.zeros_like(h)
    b[slice1] = h[slice1] - h[slice2]
    return b


def delxc(h, axis=-1):
    """\
    Central first order derivative for finite difference calculation.
    f(x+h/2) - f(x-h/2)
    .. note::
        Forward and backward derivatives are used for first and last
        elements along the derivative direction.\n
        Pixel units are used (:math:`\\Delta x = \\Delta h = 1`).

    Parameters
    ----------
    h : nd-numpy-array
        Input array.

    axis : int, Default=-1, optional
        Which direction used for the derivative.

    Returns
    -------
    out : nd-numpy-array
        Derived array.
    """
    nd = h.ndim
    axis = range(nd)[axis]
    slice_middle = [slice(1,-1) if i==axis else slice(None) for i in range(nd)]
    b = delxf(h, axis) + delxb(h, axis)
    b[slice_middle] *= 0.5
    return b


def dot(a, b):
    return np.dot(a, b)


def vdot(a, b):
    return np.vdot(a, b)


def regul_del2_grad(x, amplitude, delxy, g, LL, axes=None):
    '''
    In place update of delxy, g, and LL for the del2 regulariser
    '''
    if axes is None:
        ax0, ax1 = (-2, -1)
    else:
        ax0, ax1 = axes

    del_xf = delxf(x, axis=ax0)
    del_yf = delxf(x, axis=ax1)
    del_xb = delxb(x, axis=ax0)
    del_yb = delxb(x, axis=ax1)

    delxy[:] = [del_xf, del_yf, del_xb, del_yb]
    g[:] = 2. * amplitude * (del_xb + del_yb - del_xf - del_yf)

    LL[:] = amplitude * (norm2(del_xf)
                         + norm2(del_yf)
                         + norm2(del_xb)
                         + norm2(del_yb))


def regul_del2_poly_line_coeffs(h, amplitude, delxy, x=None, axes=None):
    if axes is None:
        ax0, ax1 = (-2, -1)
    else:
        ax0, ax1 = axes

    if x is None:
        del_xf, del_yf, del_xb, del_yb = delxy
    else:
        del_xf = delxf(x, axis=ax0)
        del_yf = delxf(x, axis=ax1)
        del_xb = delxb(x, axis=ax0)
        del_yb = delxb(x, axis=ax1)

    hdel_xf = delxf(h, axis=ax0)
    hdel_yf = delxf(h, axis=ax1)
    hdel_xb = delxb(h, axis=ax0)
    hdel_yb = delxb(h, axis=ax1)

    c0 = amplitude * (norm2(del_xf)
                      + norm2(del_yf)
                      + norm2(del_xb)
                      + norm2(del_yb))

    c1 = 2 * amplitude * np.real(vdot(del_xf, hdel_xf)
                                 + vdot(del_yf, hdel_yf)
                                 + vdot(del_xb, hdel_xb)
                                 + vdot(del_yb, hdel_yb))

    c2 = amplitude * (norm2(hdel_xf)
                      + norm2(hdel_yf)
                      + norm2(hdel_xb)
                      + norm2(hdel_yb))

    return np.array([c0, c1, c2])
