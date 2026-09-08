# -*- coding: utf-8 -*-
"""
Metric utility functions:

* radial_power_spectrum: returns the power spectrum of an image averaged azimuthally.
* frc: Fourier ring correlation between two images
* fsc: Foutier shell correlation between two volumes
* frc_thresdhold: Compute the threshold curve for fsc or frc
* compute_intersection: utility to find intersection of two curves.

Typical usage for FSC between vol1 and vol2. Voxel size is vx

```
freq, fsc12, npts = fsc(vol1, vol2)
fsc_curve = frc_threshold(npts)
freq_intersect = compute_intersection(freq, fsc12, fsc_curve)
print(f"FSC resolution: {vx/freq_intersect}")
```

This file is part of the PTYPY package.

    :copyright: Copyright 2014 by the PTYPY team, see AUTHORS.
    :license: see LICENSE for details.
"""
import numpy as np

__all__ = ['radial_power_spectrum', 'frc', 'fsc', 'frc_threshold', 'compute_intersection']


def fourier_ring_sum(f_input, ringthick=1, pixel_size=1.0):
    """
    Ring (or shell) sum of the Fourier transform of an image or volume

    Parameters
    ----------
    f_input :  array-like
        array containing the Fourier transform of the image or volume.
        Note that the (0,0) frequency must be the first element of the array.
        (not fftshifted)

    ringthick : int
        thickness of the ring (or shell) in pixel units for the azimuthal averaging
    
    pixel_size : float
        size of the pixels in the image or volume (default is 1.0)

    Returns
    -------
    freq: array-like
        1D array containing the spatial frequencies
    FRP : array-like
        1D array containing the sum in each ring (or shell)
    npts : array-like
        1D array containing the number of points in each ring (or shell)
    """
    sh = f_input.shape
    ndim = f_input.ndim
    size = f_input.size

    # Spatial frequencies
    ulist = [
        np.fft.fftfreq(sh[i]).reshape([-1 if j==i else 1 for j in range(ndim)]) for i in range(ndim)
    ]
    
    # Radial spatial frequency norm (flatten again)
    unorm = np.sqrt(sum([u**2 for u in ulist])).reshape(size)

    # Ring/shell width in frequency space (largest frequency step is 1/min(sh))
    uw = ringthick/min(sh)

    # Spatial frequencies (center of rings/shells)
    # and bins. A bit complicated to ensure that
    # the first ring is centered at 0
    urange = np.arange(0, unorm.max() + 1.5*uw - 1e-12, uw)
    frequencies = urange[:-1]
    shell_bins = (urange - .5*uw).clip(0)

    # Number of bins
    nbins = len(shell_bins) - 1

    # create the rings/shells
    shells = np.digitize(unorm, shell_bins)

    # Count number of point in each ring/shell
    npts = np.bincount(shells)[1:]

    # Sum in each ring/shell
    # Bincount does not work with complex numbers
    f_flat = f_input.reshape(size)
    if np.iscomplexobj(f_input):
        FRP = np.bincount(shells, weights=f_flat.real)[1:] + 1j*np.bincount(shells, weights=f_flat.imag)[1:]
    else:
        FRP = np.bincount(shells, weights=f_flat)[1:]

    return frequencies, FRP, npts


def radial_power_spectrum(input, ringthick=1, pixel_size=1.0):
    """
    Radial power spectrum of an image or volume

    Parameters
    ----------
    input :  array-like
        array containing the image or volume
    
    ringthick : int
        thickness of the ring (or shell) in pixel units for the azimuthal averaging
    
    pixel_size : float
        size of the pixels in the image or volume (default is 1.0)

    Returns
    -------
    freq: array-like
        1D array containing the spatial frequencies
    RPS : array-like
        1D array containing the radial power spectrum
    """
    # Fourier transform of the input
    f_input = np.fft.fftn(input)
    # Power spectrum
    PS = np.abs(f_input)**2
    # Ring/shell sum
    freq, RPS, npts = fourier_ring_sum(f_input=PS, ringthick=ringthick, pixel_size=pixel_size)
    # Normalization by the number of points in each ring/shell
    RPS /= npts
    
    return freq, RPS


def _fouriercorrelation(input1, input2, ringthick=1):
    """
    Auxiliary function for the calculation of either FRC or FSC
    """
    # Computation of the FFTs
    F1 = np.fft.fftn(np.fft.ifftshift(input1)) # FFT of input1
    F2 = np.fft.fftn(np.fft.ifftshift(input2)) # FFT of input2
    
    freq, C11, npts = fourier_ring_sum(np.abs(F1)**2, ringthick=ringthick)
    _, C22, _ = fourier_ring_sum(np.abs(F2)**2, ringthick=ringthick)
    _, C12, _ = fourier_ring_sum(F1 * np.conj(F2), ringthick=ringthick)

    FRC = np.abs(C12) / np.sqrt(C11 * C22)

    return freq, FRC, npts


def frc(input1, input2, ringthick=1, apod_width=0, align=False):
    """
    Compute the Fourier Ring Correlation (FRC) between input1 and input2.
    
    Parameters
    ----------
    input1 :  array-like
        array containing the first image, must be two-dimensional
    
    input2 : array-like
        array containing the second image, must be two-dimensional
    
    ringthick : int
        thickness of the ring for averaging the correlation

    apod_width: int
        width of image apodization to limit cyclic discontinuities (default: 0 = no apodization)
        NOT IMPLEMENTED

    align: bool
        pre-align the images before comparison (default: False)
        NOT IMPLEMENTED

    Returns
    -------
    freq: array-like
        1D array containing the spatial frequencies

    FRC : array-like
        1D array containing the Fourier Ring Correlation values

    npts : array-like
        1D array containing the number of points in each ring    
    """

    # Check if the arrays have 2 dimensions
    if input1.ndim!=2 or input2.ndim!=2:
        raise ValueError("The arrays must have 2 dimensions")
 
    # Check if the arrays have the same size
    if input1.shape != input2.shape:
        raise ValueError("The arrays must have the same size")
        
    # Image apodization
    if apod_width > 0:
        raise NotImplementedError('Apodization has not been implemented yet.')

    # Image alignment
    if align:
        raise NotImplementedError('Image pre-alignment has not been implemented yet.')

    # Run Fourier correlation
    return _fouriercorrelation(input1, input2, ringthick)


def fsc(input1, input2, ringthick=1, apod_width=0, align=False):
    """
    Compute the Fourier Shell Correlation (FSC) between input1 and input2.

    Parameters
    ----------
    input1 :  array-like
        array containing the first image, must be three-dimensional
    
    input2 : array-like
        array containing the second image, must be three-dimensional
    
    ringthick : int, optional
        thickness of the ring for averaging the correlation

    apod_width: int
        width of volume apodization to limit cyclic discontinuities (default: 0 = no apodization)
        NOT IMPLEMENTED

    align: bool
        pre-align the volumes before comparison (default: False)
        NOT IMPLEMENTED

    Returns
    -------
    freq: array-like
        1D array containing the spatial frequencies
    
    FRC : array-like
        1D array containing the FRC values

    npts : array-like
        1D array containing the number of points in each ring    
    """

    # Check if the arrays have 3 dimensions
    if input1.ndim!=3 or input2.ndim!=3:
        raise ValueError("The arrays must have 3 dimensions")

    # Check if the arrays have the same size
    if input1.shape != input2.shape:
        raise ValueError("The arrays must have the same size")

    # Image apodization
    if apod_width > 0:
        raise NotImplementedError('Apodization has not been implemented yet.')

    # Image alignment
    if align:
        raise NotImplementedError('Volume pre-alignment has not been implemented yet.')
        
    # run Fourier correlation
    return _fouriercorrelation(input1, input2, ringthick)

def frc_threshold(npts, threshold = 'onebit'):
    """
    Compute the FRC threshold curve
    
    Parameters
    ----------
    npts : array-like
        1D array containing the number of points in each ring
    
    threshold : str, optional
        The option `onebit` means 1 bit threshold with ``SNRt = 0.5``, which
        should be used for two independent measurements. The option `halfbit`
        means 1/2 bit threshold with ``SNRt = 0.2071``, which should be
        use for split tomogram. The default option is ``onebit``.

    Returns
    -------
    T : array-like
        1D array containing the FRC threshold values
    
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
    
    Tnum = (snrt + (2*np.sqrt(snrt)/np.sqrt(npts)) + 1/np.sqrt(npts))
    Tden = (snrt + (2*np.sqrt(snrt)/np.sqrt(npts)) + 1)

    return Tnum / Tden


def compute_intersection(x, f1, f2, exclude_origin=True):
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
    
    exclude_origin : bool, optional
        If True, the intersection point at the origin (if any) is ignored.
    
    Returns
    -------
    
    x : float
        Associated with the intersection point
    
    """
    x = np.asarray(x)
    f1 = np.asarray(f1)
    f2 = np.asarray(f2)

    if x.ndim!=1 or f1.ndim!=1 or f2.ndim!=1:        
        raise ValueError('Parameters must be 1-D arrays')
    
    diff = f2 - f1
    
    # Find indices where the sign of the difference changes
    idx = np.where(np.diff(np.sign(diff)))[0]
    
    crossings = []    
    for i in idx:
        x1, x2 = x[i], x[i+1]
        d1, d2 = diff[i], diff[i+1]

        # Linear interpolation to find the crossing point
        x_cross = x1 - d1 * (x2 -x1) / (d2-d1)
        crossings.append(x_cross)

    if exclude_origin and crossings[0]==x[0]:
        crossings = crossings[1:]

    # Return the first crossing point, or None if there are no crossings    
    return crossings[0] if crossings else None
