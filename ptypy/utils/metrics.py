# -*- coding: utf-8 -*-
"""
Metric utility functions.

This file is part of the PTYPY package.

    :copyright: Copyright 2014 by the PTYPY team, see AUTHORS.
    :license: see LICENSE for details.
"""
import numpy as np
from scipy.fft import fftshift, ifftshift
from scipy.ndimage import fourier_shift, gaussian_filter
from skimage.registration import phase_cross_correlation
import warnings

__all__ = ['nyquist', 'ringthickness', 'apodization', 'frc',
            'fsc', 'imgregistration', 'compute_intersection']
            
def frc(input1, input2, apod_width = 1, ringthick=1, threshold = 'onebit'):
    """
    Routine to compute the FRC
    
    Parameters
    ----------
    input1 :  array-like
        array containing the first image, must be two-dimensional
    
    input2 : array-like
        array containing the second image, must be two-dimensional
    
    apod_width : array-like
        width of the apodization margin
    
    ringthick : int
        thickness of the ring for averaging the correlation
    
    threshold : str, optional
        The option `onebit` means 1 bit threshold with ``SNRt = 0.5``, which
        should be used for two independent measurements. The option `halfbit`
        means 1/2 bit threshold with ``SNRt = 0.2071``, which should be
        with caution, because it states the information is split in two.
        The default option is ``onebit``.

    Returns
    -------
    FRC : array-like
        1D array containing the FRC values
    
    T : array-like
        1D array containing the 1-bit threshold
    
    fn : array-like
        1D array containing the normalized frequencies
    
    """
    # Check if the arrays have 2 dimensions
    if input1.ndim==2 and input2.ndim==2:
        nr,nc = input1.shape
    else:
        raise ValueError("The arrays must have 2 dimensions")
    # Check if the arrays have the same size
    if input1.shape != input2.shape:
        raise ValueError("The arrays must have the same size")
        
    # image registration
    # need to align the two image
    input2 = imgregistration(input1,input2,upsamp=100)
    
    # Apodization of the borders
    window = apodization(input1, apod_width)
    img1_apod = input1 * window
    img2_apod = input2 * window
    
    # calculating the Fourier correlation
    FRC, T, fn = _fouriercorrelation(img1_apod, img2_apod, apod_width, ringthick, threshold)

    return FRC, T, fn
    
def fsc(input1, input2, apod_width = 1, ringthick=1, threshold = 'onebit', apod_type='transaxial'):
    """
    Routine to compute the FSC
    
    Parameters
    ----------
    input1 :  array-like
        array containing the first image, must be three-dimensional
    
    input2 : array-like
        array containing the second image, must be three-dimensional
    
    apod_width : array-like
        width of the apodization margin
    
    ringthick : int, optional
        thickness of the ring for averaging the correlation
    
    threshold : str, optional
        The option `onebit` means 1 bit threshold with ``SNRt = 0.5``, which
        should be used for two independent measurements. The option `halfbit`
        means 1/2 bit threshold with ``SNRt = 0.2071``, which should be
        use for split tomogram. The default option is ``half-bit``.
    
    apod_type : str, optional
        Type of apodization for 3D images. There are two types:
        `transverse` and `transaxial`.
    Returns
    -------
    FRC : array-like
        1D array containing the FRC values
    
    T : array-like
        1D array containing the 1-bit threshold
    
    fn : array-like
        1D array containing the normalized frequencies
    
    """
    # Check if the arrays have 2 dimensions
    if input1.ndim==3 and input2.ndim==3:
        ns, nr, nc = input1.shape
    else:
        raise ValueError("The arrays must have 3 dimensions")
    # Check if the arrays have the same size
    if input1.shape != input2.shape:
        raise ValueError("The arrays must have the same size")
        
    # image registration
    # need to align the two image
    warnings.warn("For now, the alignment of 3D images has not been implemented yet!")
    
    # Apodization of the borders    
    if apod_width == 0:
        window = 1
    else:
        print("Apodization in 3D. This takes time and memory...")
        window = apodization(input1, apod_width, apod_type)

    # sagital slices
    img1_apod = (window * input1)
    img2_apod = (window * input2)
    
    # calculating the Fourier correlation
    FSC, T, fn = _fouriercorrelation(img1_apod, img2_apod, apod_width, ringthick, threshold)

    return FSC, T, fn
    
def _fouriercorrelation(input1, input2, apod_width = 1, ringthick=1, threshold = 'onebit'):
    """
    Auxiliary function for the calculation of either FRC or FSC
    """
    # Setting the threshold
    if threshold == 'halfbit' or threshold == 'half-bit':
        print('Computing FRC using half-bit threshold')
        snrt = 0.2071
        raise Warning('FRC should be used with a one-bit threshold. If you proceed with half-bit, you do it on your own responsibility.')
    elif threshold == 'onebit' or threshold =='one-bit':
        print('Computing FRC using 1-bit threshold')
        snrt = 0.5
    else:
        raise ValueError(
            "You must choose between 'halfbit' or 'onebit' threshold"
        )

    # Computation of the FFTs
    F1 = np.fft.fft2(np.fft.ifftshift(input1)) # FFT of input1
    F2 = np.fft.fft2(np.fft.ifftshift(input2)) # FFT of input2
    
    # normalized frequencies
    # Obtain the shape of the images
    sh = np.asarray(input1.shape)
    f,fnyquist = nyquist(sh) # Frequencies and Nyquist frequency
    fn = f/fnyquist
    
    # initializing variables
    C = np.empty_like(f)
    C1 = np.empty_like(f)
    C2 = np.empty_like(f)
    npts = np.zeros_like(f)
    
    print("Calculating the correlation...")
    index = ringthickness(F1) # indexes for the ring thickness
    for ii in range(len(f)):
    #for ii in (range(len(f))):
        if ringthick == 0 or ringthick == 1:
            auxF1 = F1[np.where(index == ii)]
            auxF2 = F2[np.where(index == ii)]
        else:
            auxF1 = F1[
                (
                    np.where(
                        (index >= (ii - ringthick / 2))
                        & (index <= (ii + ringthick / 2))
                    )
                )
            ]
            auxF2 = F2[
                (
                    np.where(
                        (index >= (ii - ringthick / 2))
                        & (index <= (ii + ringthick / 2))
                    )
                )
            ]
        C[ii] = np.abs((auxF1 * np.conj(auxF2)).sum()) # Cross-correlation
        C1[ii] = np.abs((auxF1 * np.conj(auxF1)).sum()) # auto-correlation
        C2[ii] = np.abs((auxF2 * np.conj(auxF2)).sum()) # auto-correlation
        npts[ii] = auxF1.shape[0]

    # The correlation
    FC = C / (np.sqrt(C1 * C2))

    # The computation of the threshold
    Tnum = (
        snrt
        + (2 * np.sqrt(snrt) / np.sqrt(npts))
        + 1 / np.sqrt(npts)
    )
    Tden = (
        snrt
        + (2 * np.sqrt(snrt) / np.sqrt(npts))
        + 1
    )
    # The threshold
    T = Tnum / Tden
    
    print('Done')

    return FC, T, fn


def nyquist(sh):
    """
    Evaluate the Nyquist Frequency
    
    Parameters
    ----------
    sh :  int
        input array length

    Returns
    -------
    f : array-like
        Array containing the frequencies

    fnyquist : array-like
        The Nyquist-frequency
    """
    sh = np.asarray(sh)
    
    nmax = np.max(sh)
    f = np.fft.rfftfreq(nmax)
    fnyquist = np.max(f)
    return f, fnyquist
    
    
def ringthickness(A):
    """
    Defines indexes for ring thickness given the array `A`
    
    Parameters
    ----------
    A :  array-like
        input array, must be two-dimensional or three-dimensional

    Returns
    -------
    index : array-like
        Indexes for the rings
    """
    if A.ndim==2:
        nr, nc = A.shape
    elif A.ndim==3:
        ns, nr, nc = A.shape 

    nmax = np.max((nr,nc)).astype(np.int16)
    x = (
        np.arange(-np.fix(nc / 2.0), np.ceil(nc / 2.0))
        * np.floor(nmax / 2.0)
        / np.floor(nc / 2.0)
    )
    y = (
        np.arange(-np.fix(nr / 2.0), np.ceil(nr / 2.0))
        * np.floor(nmax / 2.0)
        / np.floor(nr / 2.0)
    )
    # bring the central pixel to the corners (important for odd array dimensions)
    x = ifftshift(x)
    y = ifftshift(y)

    if A.ndim==2:
        # meshgriding
        X = np.meshgrid(x, y)
    elif A.ndim==3:
        z = (
            np.arange(-np.fix(ns / 2.0), np.ceil(ns / 2.0))
            * np.floor(nmax / 2.0)
            / np.floor(ns / 2.0)
        )
        # bring the central pixel to the corners  (important for odd array dimensions)
        z = ifftshift(z)
        # meshgriding
        X = np.meshgrid(y, z, x)

    # sum of the squares independent of ndim
    sumsquares = np.zeros_like(X[0])
    for ii in range(len(X)):
        sumsquares += X[ii] ** 2
    index = np.round(np.sqrt(sumsquares)).astype(int)

    return index
    

def _circle(apod_width, nr, nc):
    """
    Create a circle with apodized edges.
    """
    Y, X = np.indices((nr, nc))
    Y -= np.round(nr / 2).astype(int)
    X -= np.round(nc / 2).astype(int)
    R = np.sqrt(X ** 2 + Y ** 2)
    Rmax = np.round(np.max(R.shape) / 2.0)
    maskout = R < Rmax
    t = (
        maskout
        * (1 - np.cos(np.pi * (R - Rmax - 2 * apod_width) / apod_width))
        / 2.0
    )
    t[np.where(R < (Rmax - apod_width))] = 1
    return t
    
def apodization(A, apod_width=1, apod_type='transverse'):
    """
    Compute a tapered Hanning-like window of the size of the data
    for the apodization
    
    Parameters
    ----------
    A :  array-like
        input array, must be two-dimensional or three-dimensional
    
    apod_width : array-like
        width of the apodization margin
    
    apod_type : str, optional
        Type of apodization for 3D images. There are two types:
        `transverse` and `transaxial`. Only considered for 3D
        apodization.

    Returns
    -------
    out : array-like
        2D array containing the apodization mask
    
    """
    if A.ndim==2:
        windowfunc = _window2D
    elif A.ndim==3:
        if apod_type == 'transverse':
            windowfunc = _window3Dtransverse
        elif apod_type == 'transaxial':
            windowfunc = _window3Dtransaxial
        else:
            raise ValueError('Wrong value for apod_type')
    
    return windowfunc(A,apod_width)
    

def imgregistration(ref_img,mov_img,upsamp=1):
    """
    TODO: To be replaced by a custom alignment function
    
    Routine for image registration before the FRC
    
    Parameters
    ----------
    ref_img :  array-like
        array containing the reference image, must be two-dimensional
    
    mov_img : array-like
        array containing the moving image, must be two-dimensional
    
    upsamp : int
        upsampling factor for subpixel registration

    Returns
    -------
    offset_image : array-like
        2D array containing the registered moving image
    
    """
    shift, err, phasediff = phase_cross_correlation(ref_img,mov_img,upsample_factor=upsamp)
    offset_image = np.fft.ifft2(fourier_shift(np.fft.fft2(mov_img),shift))
    return offset_image.real


def _hann_window(sh, width):
    """
    Compute a tapered 1D Hanning-like window of the size of the data.
    
    Parameters
    ----------
    sh : int
        The length of the array
    width : int
        The extent of the window
    
    Returns
    -------
    ndarray
        The Hanning-like window
    """
    sh = np.asarray(sh)
    Np = fftshift(np.arange(sh))
    window = (
        1.0
        + np.cos(
            2
            * np.pi
            * (Np - np.floor((sh - 2 * width - 1) / 2))
            / (1 + 2 * width)
        )
    ) / 2.0
    
    window[width:-width]=1

    return window
    
def _window2D(A, width):
    """
    doc TODO
    """
    sh = A.shape
    window1D1 = _hann_window(sh[0], width)
    window1D2 = _hann_window(sh[1], width)

    return np.outer(window1D1, window1D2)
    
def __window3D(A, width):
    """
    Auxiliary function for the 3D transverse and axial apodization
    """
    sh = A.shape 
    window1D1 = _hann_window(sh[0], width)
    window1D2 = _hann_window(sh[1], width)
    window1D3 = _hann_window(sh[2], width)
    
    result = [np.outer(window1D1, window1D2), np.outer(window1D1, window1D3)]
    return result
    
def _window3Dtransverse(A, width):
    """
    Same transverse apodization in all three directions
    """
    sh = np.asarray(A.shape)
    window1D1 = _hann_window(sh[0], width)
    window1D2 = _hann_window(sh[1], width)
    window1D3 = _hann_window(sh[2], width)
    
    windowaxial = np.outer(window1D2, window1D3)
    windowsag = np.array([window1D1 for ii in range(sh[1])]).swapaxes(0, 1)
    win2d = np.array([np.tile(windowaxial, (1, 1)) for ii in range(sh[0])])
    window = (
        np.array(
            [np.squeeze(win2d[:, :, ii]) * windowsag for ii in range(sh[2])]
        )
        .swapaxes(0, 1)
        .swapaxes(1, 2)
    )
    
    return window
    
def _window3Dtransaxial(A, width):
    """
    Transverse apodization of the sagital and coronal section, and axial apodization
    for the axial slice
    """
    sh = np.asarray(A.shape)
    circular_region = _circle(width, sh[1], sh[2])
    window3D = __window3D(A, width)    
    circle3D = np.asarray([circular_region for ii in range(sh[0])])
    window = (
        np.array(
            [
                np.squeeze(circle3D[:, :, ii]) * window3D[0]
                for ii in range(sh[2])
            ]
        )
        .swapaxes(0, 1)
        .swapaxes(1, 2)
    )
    window = np.array(
        [
            np.squeeze(window[:, ii, :]) * window3D[1]
            for ii in range(sh[1])
        ]
    ).swapaxes(0, 1)
    
    return window

def compute_intersection(x,f1,f2):
    
    """
    Function used to extract the resolution given the FRC/FSC and the threshold curve
    
    Parameters
    ----------
    f1 : ndarray
        First curve (1-D set of values)
    
    f2 : ndarray
        Second curve (1-D set of values)
    
    x : ndarray 
        x-axis (same size and shape as f1 and f2) 
    
    Returns
    -------
    
    x : float
        Associated with the intersection point
    
    """
    
    # --- CHECK INPUT PARAMETERS ---
    
    if not (isinstance(x,np.ndarray) and isinstance(f1,np.ndarray) and isinstance(f2,np.ndarray)):
        
        raise TypeError('Parameters must be np arrays')
        
    if not (x.ndim == 1 and f1.ndim==1 and f2.ndim==1):
        
        raise ValueError('Parameters must be 1-D arrays')
        
    # --------------------
    
    diff = f2 - f1
    
    # WHERE SIGN CHANGES
    
    idx = np.where(np.diff(np.sign(diff)))[0]
    
    crossings = []
    
    for i in idx:
        
        x1, x2 = x[i], x[i+1]
        
        d1, d2 = diff[i], diff[i+1]
        
        # INTERSECTION
        
        x_cross = x1 - d1 * (x2 -x1) / (d2-d1)
        
        crossings.append(x_cross)
        
    # RESOLUTION IS THE FIRST CROSSING
    
    return crossings[0] if crossings else None
