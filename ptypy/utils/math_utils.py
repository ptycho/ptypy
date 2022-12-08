# -*- coding: utf-8 -*-
"""
Numerical util functions.

This file is part of the PTYPY package.

    :copyright: Copyright 2014 by the PTYPY team, see AUTHORS.
    :license: see LICENSE for details.
"""
import numpy as np
from scipy.special import erf
from scipy.linalg import eig
from scipy import ndimage as ndi

from .misc import *

__all__ = ['smooth_step', 'abs2', 'norm2', 'norm', 'delxb', 'delxc', 'delxf',
           'ortho', 'gauss_fwhm', 'gaussian', 'gf', 'cabs2', 'gf_2d', 'c_gf',
           'gaussian2D', 'rl_deconvolution']


def cabs2(A):
    """
    Squared absolute value for an array `A`.
    If `A` is complex, the returned value is complex as well, with the
    imaginary part of zero.
    """
    return A * A.conj()

def abs2(A):
    """
    Squared absolute value for an array `A`.
    """
    return cabs2(A).real

def norm2(A):
    """
    Squared norm
    """
    return np.sum(abs2(A))

def norm(A):
    """
    Norm.
    """
    return np.sqrt(norm2(A))

def smooth_step(x, mfs):
    """
    Smoothed step function with fwhm `mfs`
    Evaluates the error function `scipy.special.erf`.
    """
    return 0.5 * erf(x * 2.35 / mfs) + 0.5

def gaussian(x, std=1.0, off=0.0):
    """
    Evaluates gaussian standard normal

    .. math::
        g(x)=\\frac{1}{\\mathrm{std}\\sqrt{2\\pi}}\\,\\exp
        \\left(-\\frac{(x-\\mathrm{off})^2}{2 \\mathrm{std}^2 }\\right)

    Parameters
    ----------
    x : ndarray
        input array

    std : float,optional
        Standard deviation

    off : float, optional
        Offset / shift

    See also
    --------
    gauss_fwhm
    smooth_step
    """
    return np.exp(-(x - off)**2 / (2 * std**2)) / (std * np.sqrt(2 * np.pi))

def gauss_fwhm(x, fwhm=1.0, off=0.0):
    """
    Evaluates gaussian with full width half maximum

    Parameters
    ----------
    x : ndarray
        input array

    fwhm : float,optional
        Full width at half maximum

    off : float, optional
        Offset / shift

    See also
    --------
    gaussian

    """
    return gaussian(x, fwhm / 2 / np.sqrt(2 * np.log(2)), off)

def gaussian2D(size, std_x=1.0, std_y=1.0, off_x=0.0, off_y=0.0):
    """
    Evaluates normalized 2D gaussian on array of dimension size.
    Origin of coordinate system is in the center of the array.

    Parameters
    ----------
    size : int
           length of requested array

    std_x : float,optional
            Standard deviation in x direction

    std_y : float,optional
            Standard deviation in y direction

    off_x : float, optional
            Offset / shift in x direction

    off_y : float, optional
            Offset / shift in y direction

    """
    if not isinstance(size, int):
        raise RuntimeError('Input size has to be integer.')

    y, x = np.mgrid[0:size, 0:size]
    x = x - size // 2
    y = y - size // 2
    xpart = (x - off_x)**2 / (2 * std_x**2)
    ypart = (y - off_y)**2 / (2 * std_y**2)
    return np.exp(-(xpart + ypart)) / (2 * np.pi * std_x * std_y)

def delxf(a, axis=-1, out=None):
    """\
    Forward first order derivative for finite difference calculation.

    .. note::
        The last element along the derivative direction is set to 0.\n
        Pixel units are used (:math:`\\Delta x = \\Delta h = 1`).

    Parameters
    ----------
    a : ndarray
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
    nd = a.ndim
    axis = list(range(nd))[axis]

    slice1 = [slice(1, None) if i == axis else slice(None) for i in range(nd)]
    slice2 = [slice(None, -1) if i == axis else slice(None) for i in range(nd)]

    if (out is None):
        out = np.zeros_like(a)

    out[tuple(slice2)] = a[tuple(slice1)] - a[tuple(slice2)]

    if out is a:
        # required for in-place operation
        slice3 = [slice(-2, None) if i == axis else slice(None)
                  for i in range(nd)]
        out[tuple(slice3)] = 0.0

    return out

def delxb(a, axis=-1):
    """\
    Backward first order derivative for finite difference calculation.

    .. note::
        The first element along the derivative direction is set to 0.\n
        Pixel units are used (:math:`\\Delta x = \\Delta h = 1`).

    Parameters
    ----------
    a : ndarray
        Input array.

    axis : int, Default=-1, optional
        Which direction used for the derivative.

    Returns
    -------
    out : ndarray
        Derived array.
    """

    nd = a.ndim
    axis = list(range(nd))[axis]
    slice1 = [slice(1, None) if i == axis else slice(None) for i in range(nd)]
    slice2 = [slice(None, -1) if i == axis else slice(None) for i in range(nd)]
    b = np.zeros_like(a)
    b[tuple(slice1)] = a[tuple(slice1)] - a[tuple(slice2)]
    return b

def delxc(a,axis=-1):
    """\
    Central first order derivative for finite difference calculation.

    .. note::
        Forward and backward derivatives are used for first and last
        elements along the derivative direction.\n
        Pixel units are used (:math:`\\Delta x = \\Delta h = 1`).

    Parameters
    ----------
    a : nd-numpy-array
        Input array.

    axis : int, Default=-1, optional
        Which direction used for the derivative.

    Returns
    -------
    out : nd-numpy-array
        Derived array.
    """
    nd = a.ndim
    axis = list(range(nd))[axis]
    slice_middle = [slice(1,-1) if i==axis else slice(None) for i in range(nd)]
    b = delxf(a, axis) + delxb(a, axis)
    b[slice_middle] *= 0.5
    return b


def ortho(modes):
    """\
    Orthogonalize the given list of modes or ndarray along first axis.
    **specify procedure**

    Parameters
    ----------
    modes : array-like or list
        List equally shaped arrays or array of higher dimension

    Returns
    -------
    amp : vector
        relative power of each mode
    nplist : list
        List of modes, sorted in descending order
    """
    N = len(modes)
    A = np.array([[np.vdot(p2,p1) for p1 in modes] for p2 in modes])
    e, v = eig(A)
    ei = (-e).argsort()
    nplist = [sum(modes[i] * v[i,j] for i in range(N)) for j in ei]
    amp = np.array([norm2(npi) for npi in nplist])
    amp /= amp.sum()
    return amp, nplist


c_gf= complex_overload(ndi.gaussian_filter)
# ndi.gaussian_filter is a little special in the docstring
c_gf.__doc__='    *complex input*\n\n    '+c_gf.__doc__

def gf(c, *arg, **kwargs):
    """
    Wrapper for scipy.ndimage.gaussian_filter, that determines whether
    original or the complex function shall be used.

    See also
    --------
    c_gf
    """
    if np.iscomplexobj(c):
        return c_gf(c, *arg, **kwargs)
    else:
        return ndi.gaussian_filter(c, *arg, **kwargs)

def gf_2d(c, sigma, **kwargs):
    """
    Gaussian filter along the last 2 axes

    See also
    --------
    gf
    c_gf
    """
    if c.ndim > 2:
        n=c.ndim
        return gf(c, (0,) * (n - 2) + tuple(expect2(sigma)), **kwargs)
    else:
        return gf(c, sigma, **kwargs)

def rl_deconvolution(data, mtf, numiter):
    """
    Richardson Lucy deconvolution on a 2D numpy array.

    Parameters
    ----------
    data : 2darray
        Diffraction data (measured intensity).

    mtf : 2darray
        Modulation transfer function (Fourier transform of detector PSF).

    numiter : int
        Number of iterations.

    Returns
    -------
    out : 2darray
        Approximated intensity after numiter iterations.

    Note:
    Assumes that mtf is symmetric and that data is real and positive.
    mtf is non fft shifted that means that the dc component is on mtf[0,0].
    Todo:
    non symmetric mtf: mtf.conj()[-q] somewhere
    optimisation: FFTW? scipy fft? error metric cancel iter?
    Original code provided by M. Stockmar
    """
    convolve = lambda x: np.abs(np.fft.ifft2(np.fft.fft2(x)*mtf)).astype(x.dtype)
    u = data.copy()
    for n in range(numiter):
        u *= convolve(data / (convolve(u) + 1e-6))
    return u
