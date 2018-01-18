'''
useful utilities from ptypy that should be ported to gpu. These don't ahve external dependencies
'''
import numpy as np


def _confine(A):
    """\
    Doc TODO.
    """
    sh=np.asarray(A.shape)[1:]
    A=A.astype(float)
    m=np.reshape(sh,(len(sh),) + len(sh)*(1,))
    return (A+m//2.0) % m - m//2.0

def _translate_to_pix(sh,center):
    """\
    Take arbitrary input and translate it to a pixel position with respect to sh.
    """
    sh=np.array(sh)
    if center=='fftshift':
        cen=sh//2.0
    elif center=='geometric':
        cen=sh/2.0-0.5
    elif center=='fft':
        cen=sh*0.0
    elif center is not None:
        cen = np.asarray(center) % sh

    return cen


def grids(sh,psize=None,center='geometric',FFTlike=True):
    """\
    ``q0,q1,... = grids(sh)``
    returns centered coordinates for a N-dimensional array of shape sh (pixel units)

    ``q0,q1,... = grids(sh,psize)``
    gives the coordinates scaled according to the given pixel size psize.

    ``q0,q1,... = grids(sh,center='fftshift')``
    gives the coordinates shifted according to fftshift convention for the origin

    ``q0,q1,... = grids(sh,psize,center=(c0,c1,c2,...))``
    gives the coordinates according scaled with psize having the origin at (c0,c1,..)


    Parameters
    ----------
    sh : tuple of int
        The shape of the N-dimensional array

    psize : float or tuple of float
        Pixel size in each dimensions

    center : tupel of int
        Tuple of pixel, or use ``center='fftshift'`` for fftshift-like grid
        and ``center='geometric'`` for the matrix center as grid origin

    FFTlike : bool
        If False, grids ar not bound by the interval [-sh//2:sh//2[

    Returns
    -------
    ndarray
        The coordinate grids
    """
    sh=np.asarray(sh)

    cen = _translate_to_pix(sh,center)

    grid=np.indices(sh).astype(float) - np.reshape(cen,(len(sh),) + len(sh)*(1,))

    if FFTlike:
        grid=_confine(grid)

    if psize is None:
        return grid
    else:
        psize = np.asarray(psize)
        if psize.size == 1:
            psize = psize * np.ones((len(sh),))
        psize = np.asarray(psize).reshape( (len(sh),) + len(sh)*(1,))
        return grid * psize
