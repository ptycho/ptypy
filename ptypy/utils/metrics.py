# -*- coding: utf-8 -*-
"""
Metric utility functions.

This file is part of the PTYPY package.

    :copyright: Copyright 2014 by the PTYPY team, see AUTHORS.
    :license: see LICENSE for details.
"""
import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fftshift, ifftshift
from scipy.ndimage import fourier_shift, gaussian_filter
from skimage.registration import phase_cross_correlation

__all__ = ['nyquist', 'ringthickness', 'apodization', 'fourierringcorrelation',
            'imgregistration','frc_plot']

def nyquist(arraysize):
    """
    Evaluate the Nyquist Frequency
    
    Parameters
    ----------
    arraysize :  int
        input array length

    Returns
    -------
    f : array-like
        Array containing the frequencies

    fnyquist : array-like
        The Nyquist-frequency
    """
    nmax = np.max(arraysize)
    f = np.fft.rfftfreq(nmax)
    fnyquist = np.max(f)
    return f, fnyquist
    
    
def ringthickness(inputarray):
    """
    Defines indexes for ring thickness
    
    Parameters
    ----------
    inputarray :  array-like
        input array, must be at least two-dimensional

    Returns
    -------
    index : array-like
        Indexes for the rings
    """
    nr,nc = inputarray.shape
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
    # meshgriding
    X = np.meshgrid(x,y)
    # sum of the squares
    sumsquares = X[0]**2 + X[1]**2
    index = np.round(np.sqrt(sumsquares)).astype(np.int16)
    return index
    
def apodization(inputarray, apod_width=1):
    """
    Compute a tapered Hanning-like window of the size of the data
    for the apodization
    
    Parameters
    ----------
    inputarray :  array-like
        input array, must be two-dimensional
    
    apod_width : array-like
        width of the apodization margin

    Returns
    -------
    out : array-like
        2D array containing the apodization mask
    
    """
    print("Calculating the transverse apodization")
    nr, nc = inputarray.shape
    Nr = fftshift(np.arange(nr))
    Nc = fftshift(np.arange(nc))
    window1D1 = (
        1.0
        + np.cos(
            2
            * np.pi
            * (Nr - np.floor((nr - 2 * apod_width - 1) / 2))
            / (1 + 2 * apod_width)
        )
    ) / 2.0
    window1D2 = (
        1.0
        + np.cos(
            2
            * np.pi
            * (Nc - np.floor((nc - 2 * apod_width - 1) / 2))
            / (1 + 2 * apod_width)
        )
    ) / 2.0
    window1D1[apod_width : -apod_width] = 1
    window1D2[apod_width : -apod_width] = 1
    
    return np.outer(window1D1, window1D2)
    
def imgregistration(ref_img,mov_img,upsamp=1):
    """
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
    
def fourierringcorrelation(input1, input2, apod_width = 1, ringthick=1):
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
    
    # Forcing to using 1 bit threshold because it is ring correlation
    # 1/2 bit threshold must only be used for tomography
    print('Computing FRC using 1 bit threshold')
    snrt = 0.5
    
    # Apodization of the borders
    window = apodization(input1, apod_width)
    img1_apod = input1 * window
    img2_apod = input2 * window
    
    # plotting
    fig1 = plt.figure(1)
    ax1 = fig1.add_subplot(121)
    ax2 = fig1.add_subplot(122)
    im1 = ax1.imshow(img1_apod, cmap="bone", interpolation="none")
    ax1.set_title("image1")
    ax1.set_axis_off()
    im2 = ax2.imshow(img2_apod, cmap="bone", interpolation="none")
    ax2.set_title("image2")
    ax2.set_axis_off()

    # Computation of the FFTs
    F1 = np.fft.fft2(np.fft.ifftshift(img1_apod)) # FFT of input1
    F2 = np.fft.fft2(np.fft.ifftshift(img2_apod)) # FFT of input2
    
    # normalized frequencies
    f,fnyquist = nyquist((nr,nc)) # Frequencies and Nyquist frequency
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
    FRC = C / (np.sqrt(C1 * C2))

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

    return FRC, T, fn
    
    
def frc_plot(FRC, T, fn):
    """
    Routine to plot the FRC curves
    
    Parameters
    ----------
    FRC : array-like
        1D array containing the FRC values
    
    T : array-like
        1D array containing the 1-bit threshold
    
    fn : array-like
        1D array containing the normalized frequencies    
    """
    
    plt.figure()
    plt.clf()
    plt.plot(fn, FRC.real, "-b", label="FRC")
    plt.plot(fn, T, "--r", label="1 bit threshold")
    plt.legend()
    plt.xlim(0, 1)
    plt.ylim(0, 1.1)
    plt.xlabel("Spatial frequency/Nyquist [normalized units]")
    plt.ylabel("Magnitude [normalized units]")
    plt.show()
    
    return None
    