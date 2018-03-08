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
    :param outshape. An array. Can be inplace. Can be complex or real.
    :param in1_addr. An array . Can be inplace. Can be complex or real.
    :param out1_addr. An array . Can be inplace. Can be complex or real.
    :return: The real valued abs**2 array
    '''
    out1 = np.zeros(outshape, dtype=dtype)
    for i1, o1 in zip(in1_addr, out1_addr):
        out1[o1[0]] += in1[i1[0]]
    return out1

def norm2(input):
    return np.sum(abs2(input))

def complex_gaussian_filter(input, mfs):
    return ndi.gaussian_filter(np.real(input), mfs).astype(input.dtype) +1j *ndi.gaussian_filter(np.imag(input))

def mass_center(A, axes=None):
    """
    Calculates mass center of n-dimensional array `A`
    along tuple of axis `axes`.

    Parameters
    ----------
    A : ndarray
        input array

    axes : list, tuple
        Sequence of axes that contribute to distributed mass. If
        ``axes==None``, all axes are considered.

    Returns
    -------
    mass : 1d array
        Center of mass in pixel for each `axis` selected.
    """
    A = np.asarray(A)

    if axes is None:
        axes = tuple(range(1, A.ndim + 1))
    else:
        axes = tuple(np.array(axes) + 1)

    return np.sum(A * np.indices(A.shape), axis=axes, dtype=np.float) / np.sum(A, dtype=np.float)


def shift_zoom(c, zoom, cen_old, cen_new):
    """\
    Move array from center `cen_old` to `cen_new` and perform a zoom `zoom`.

    This function wraps `scipy.ndimage.affine_transform <https://docs.scipy.org/
    doc/scipy/reference/generated/scipy.ndimage.affine_transform.html>`_ and 
    uses the same keyword arguments.

    Addiionally, it allows for complex input and out by complex overloading, see
    :any:`complex_overload`\ . 

    Parameters
    ----------
    c : numpy.ndarray
        Array to shiftzoom. Can be float or complex

    zoom : float
        Zoom factor

    cen_old : array_like
        Center in input array `c`

    cen_new : array_like
        Desired new center position in shiftzoomed array

    Returns
    -------
    numpy.ndarray
        Shifted and zoomed array
    """

    from scipy.ndimage import affine_transform
    zoom = np.diag(zoom)
    offset = np.asarray(cen_old) - np.asarray(cen_new).dot(zoom)

    return affine_transform(np.real(c), zoom, offset) + 1j*affine_transform(np.imag(c), zoom, offset)
