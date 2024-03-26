import numpy as np
import scipy.interpolate
import os
import scipy.ndimage as ndi

from numpy.random import MT19937
from numpy.random import RandomState, SeedSequence

from scipy.sparse.linalg import eigsh

dirname = os.path.dirname(__file__)


def scan_pos_pix(x,y, dx, dy):
    """
    Calculate scan positions in pixel units.

    Parameters
    ----------
    x : ndarray
        Scan positions in [m]
    y : ndarray
        Scan positions in [m]
    dx : float
        Pixelsize in [m]
    dy : float
        Pixelsize in [m]

    Returns
    -------
    xpix : ndarray
        Scan positions in units of pixels.
    ypix : ndarray
        Scan positions in units of pixels.
    """
    centerx = (x.max() + x.min()) / 2.
    centery = (y.max() + y.min()) / 2.
    x -= centerx
    y -= centery
    xpix = np.floor(x/dx).astype(np.int)
    ypix = np.floor(y/dy).astype(np.int)
    return xpix, ypix


def scan_view_limits(x, y, probe_shape):
    """
    Calculate the limits for each scan view.

    Parameters
    ----------
    x : ndarray
        Scan positions in units of pixels.
    y : ndarray
        Scan positions in units of pixels.
    probe_shape : tuple
        Shape of the probe (illumination)

    Returns
    -------
    xlow : ndarray
        Lower limit of scan view in x.
    xhigh : ndarray
        Upper limit of scan view in x.
    ylow : ndarray
        Lower limit of scan view in y.
    yhigh : ndarray
        Upper limit of scan view in y.
    """
    # Probe shape and its center
    py,px = probe_shape
    cy,cx = py//2, px//2

    # View limits in X
    xlow = x - cx
    xhigh = x + cx
    xmin = xlow.min()
    xlow -= xmin
    xhigh -= xmin

    # View limits in Y
    ylow = y - cy
    yhigh = y + cy
    ymin = ylow.min()
    ylow -= ymin
    yhigh -= ymin

    return xlow, xhigh, ylow, yhigh

def downsample(img, factor=2):
    """
    Downsampling an image by a given factor.

    Parameters
    ----------
    img : ndarray
        Image to be downsampled.
    factor : int
        Downsampling factor
    """
    sh = img.shape
    return img.reshape(sh[0]//factor, factor, sh[1]//factor,factor).sum(axis=(1,3))

def fvec2(sh, psize=None):
    """\
    Squared norm of reciprocal space coordinates, with pixel size ``psize``.

    Parameters
    ----------
    sh : nd-array
        Shape of array.

    psize : int, Defualt=None, optional
        Pixel size in each dimensions.

    Returns
    --------
    ndarray
        Squared norm of reciprocal space coordinates.

    Examples
    --------
    >>> q2 = fvec2(sh, psize):

    .. note::
        Uses function :py:func:`fgrid`
    """
    return np.sum(fgrid(sh,psize)**2, axis=0)

def fgrid(sh,psize=None):
    """\
    Returns Fourier-space coordinates for a N-dimensional array of shape ``sh`` (pixel units).

    Parameters
    ----------
    sh : nd-array
        Shape of array.

    psize : int, Defualt=None, optional
        Pixel size in each dimensions.

    Returns
    --------
    nd-array
        Returns Fourier-space coordinates.

    Examples
    --------
    Returns Fourier-space coordinates for a N-dimensional array of shape sh (pixel units)
        >>> sh = [5,5]
        >>> q0,q1 = fgrid(sh)
        >>> q0
        array([[ 0.,  0.,  0.,  0.,  0.],
               [ 2.,  2.,  2.,  2.,  2.],
               [ 4.,  4.,  4.,  4.,  4.],
               [-4., -4., -4., -4., -4.],
               [-2., -2., -2., -2., -2.]])
        >>> q1
        array([[ 0.,  2.,  4., -4., -2.],
               [ 0.,  2.,  4., -4., -2.],
               [ 0.,  2.,  4., -4., -2.],
               [ 0.,  2.,  4., -4., -2.],
               [ 0.,  2.,  4., -4., -2.]])

    Gives the coordinates according to the given pixel size psize.
        >>> q0,q1 = fgrid([3,3],psize=5)
        >>> q0
        array([[ 0.,  0.,  0.],
               [ 5.,  5.,  5.],
               [-5., -5., -5.]])
        >>> q1
        array([[ 0.,  5., -5.],
               [ 0.,  5., -5.],
               [ 0.,  5., -5.]])
    """
    if psize is None:
        return np.fft.ifftshift(np.indices(sh).astype(float) - np.reshape(np.array(sh)//2,(len(sh),) + len(sh)*(1,)), range(1,len(sh)+1))
    else:
        psize = np.asarray(psize)
        if psize.size == 1:
            psize = psize * np.ones((len(sh),))
        psize = np.asarray(psize).reshape( (len(sh),) + len(sh)*(1,))
        return np.fft.ifftshift(np.indices(sh).astype(float) - np.reshape(np.array(sh)//2,(len(sh),) + len(sh)*(1,)), range(1,len(sh)+1)) * psize


def free_nf(w, l, z, pixsize=1.):
    """\
    Free-space propagation (near field) of the wavefield of a distance z.
    l is the wavelength.
    """
    if w.ndim != 2:
        raise RuntimeError("A 2-dimensional wave front 'w' was expected")

    sh = w.shape

    if sh[0] != sh[1]:
        raise RuntimeError("Only implemented for square arrays...")

    # Convert to pixel units.
    z = z / pixsize
    l = l / pixsize

    q2 = fvec2(sh, psize=(1./sh[0], 1./sh[1]))
    return np.fft.ifftn(np.fft.fftn(w) * np.exp(2j * np.pi * (z / l) * (np.sqrt(1 - q2*l**2) - 1) ) )


def random_points(sh=(10,10), frac=0.1):
    img = np.zeros(np.prod(sh))
    img[:int(frac*np.prod(sh))] = 1.
    np.random.shuffle(img)
    return img.reshape(sh)

def follow_random_path(img, iter=1):
    y,x = np.where(img==1)
    loc = np.vstack([x,y]).T
    new_loc = np.copy(loc)
    choices = np.array([[i,j] for i in range(-1,2) for j in range(-1,2)])
    for i in range(iter):
        for k in range(len(loc)):
            new_loc[k] = loc[k] + choices[np.random.randint(9)]
            xnew = max(min(new_loc[k][0], img.shape[1]-1),0)
            ynew = max(min(new_loc[k][1], img.shape[0]-1),0)
            img[ynew, xnew] = 1
        loc = new_loc
    return img

def randomized_test_image(sh=(128,128), frac=0.01, iter=60, blur=1):
    img = random_points(sh, frac)
    img = follow_random_path(img, iter=iter)
    img = ndi.morphology.binary_fill_holes(img).astype(np.float)
    img = ndi.gaussian_filter(img,blur)
    return img

def load_test_image(filepath, target_shape):
    from PIL import Image
    im = Image.open(filepath)
    ar = np.array(im.resize(target_shape[::-1]), dtype=np.float32)
    if ar.ndim == 3:
        ar = ar.sum(axis=2)
    return ar

def rmask(sh, radius, cent=None):
    if cent is None:
        cent = np.divide(sh,2)
    x = np.arange(0,sh[0]) - cent[0]
    y = np.arange(0,sh[1]) - cent[1]

    yy, xx = np.meshgrid(y, x)
    grid = np.sqrt((xx**2)+(yy**2))
    return grid < radius

def crop_pad(img, k=5):
    if img.ndim == 3:
        out = np.pad(img[:,k:-k,k:-k], ((0,0),(k,k),(k,k)), mode="edge")
    elif img.ndim == 2:
        out = np.pad(img[k:-k,k:-k], ((k,k),(k,k)), mode="edge")
    return out