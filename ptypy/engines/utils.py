# -*- coding: utf-8 -*-
"""\
Engine-specific utilities.
This could be compiled, or GPU accelerated.

This file is part of the PTYPY package.

    :copyright: Copyright 2014 by the PTYPY team, see AUTHORS.
    :license: GPLv2, see LICENSE for details.
"""
import numpy as np

from .. import utils as u
from ..utils.verbose import logger


def basic_fourier_update(diff_view, pbound=None, alpha=1., LL_error=True):
    """\
    Fourier update a single view using its associated pods.
    Updates on all pods' exit waves.

    Parameters
    ----------
    diff_view : View
        View to diffraction data

    alpha : float, optional
        Mixing between old and new exit wave. Valid interval ``[0, 1]``

    pbound : float, optional
        Power bound. Fourier update is bypassed if the quadratic deviation
        between diffraction data and `diff_view` is below this value.
        If ``None``, fourier update always happens.

    LL_error : bool
        If ``True``, calculates log-likelihood and puts it in the last entry
        of the returned error vector, else puts in ``0.0``

    Returns
    -------
    error : ndarray
        1d array, ``error = np.array([err_fmag, err_phot, err_exit])``.

        - `err_fmag`, Fourier magnitude error; quadratic deviation from
          root of experimental data
        - `err_phot`, quadratic deviation from experimental data (photons)
        - `err_exit`, quadratic deviation of exit waves before and after
          Fourier iteration
    """
    # Prepare dict for storing propagated waves
    f = {}

    # Buffer for accumulated photons
    af2 = np.zeros_like(diff_view.data)
    # Get measured data
    I = diff_view.data

    # Get the mask
    fmask = diff_view.pod.mask

    # For log likelihood error
    if LL_error is True:
        LL = np.zeros_like(diff_view.data)
        for name, pod in diff_view.pods.iteritems():
            LL += u.abs2(pod.fw(pod.probe * pod.object))
        err_phot = (np.sum(fmask * (LL - I)**2 / (I + 1.))
                    / np.prod(LL.shape))
    else:
        err_phot = 0.

    # Propagate the exit waves
    for name, pod in diff_view.pods.iteritems():
        if not pod.active:
            continue
        f[name] = pod.fw((1 + alpha) * pod.probe * pod.object
                         - alpha * pod.exit)

        af2 += u.abs2(f[name])

    fmag = np.sqrt(np.abs(I))
    af = np.sqrt(af2)

    # Fourier magnitudes deviations
    fdev = af - fmag
    err_fmag = np.sum(fmask * fdev**2) / fmask.sum()
    err_exit = 0.

    if pbound is None:
        # No power bound
        fm = (1 - fmask) + fmask * fmag / (af + 1e-10)
        for name, pod in diff_view.pods.iteritems():
            if not pod.active:
                continue
            df = pod.bw(fm * f[name]) - pod.probe * pod.object
            pod.exit += df
            err_exit += np.mean(u.abs2(df))
    elif err_fmag > pbound:
        # Power bound is applied
        renorm = np.sqrt(pbound / err_fmag)
        fm = (1 - fmask) + fmask * (fmag + fdev * renorm) / (af + 1e-10)
        for name, pod in diff_view.pods.iteritems():
            if not pod.active:
                continue
            df = pod.bw(fm * f[name]) - pod.probe * pod.object
            pod.exit += df
            err_exit += np.mean(u.abs2(df))
    else:
        # Within power bound so no constraint applied.
        for name, pod in diff_view.pods.iteritems():
            if not pod.active:
                continue
            df = alpha * (pod.probe * pod.object - pod.exit)
            pod.exit += df
            err_exit += np.mean(u.abs2(df))

    if pbound is not None:
        # rescale the fmagnitude error to some meaning !!!
        # PT: I am not sure I agree with this.
        err_fmag /= pbound

    return np.array([err_fmag, err_phot, err_exit])





def Cnorm2(c):
    """\
    Computes a norm2 on whole container `c`.

    :param Container c: Input
    :returns: The norm2 (*scalar*)

    See also
    --------
    ptypy.utils.math_utils.norm2
    """
    r = 0.
    for name, s in c.storages.iteritems():
        r += u.norm2(s.data)
    return r


def Cdot(c1, c2):
    """\
    Compute the dot product on two containers `c1` and `c2`.
    No check is made to ensure they are of the same kind.

    :param Container c1, c2: Input
    :returns: The dot product (*scalar*)
    """
    r = 0.
    for name, s in c1.storages.iteritems():
        r += np.vdot(c1.storages[name].data.flat, c2.storages[name].data.flat)
    return r


class Regul_del2(object):
    """\
    Squared gradient regularizer (Gaussian prior).

    This class applies to any numpy array.
    """

    def __init__(self, amplitude, axes=(-2, -1)):  # TODO: This default argument should not be mutable!
        self.axes = axes
        self.amplitude = amplitude
        self.delxy = None
        self.g = None
        self.LL = None

    def grad(self, x):
        """
        Compute and return the regularizer gradient given the array x.
        """
        ax0, ax1 = self.axes
        del_xf = u.delxf(x, axis=ax0)
        del_yf = u.delxf(x, axis=ax1)
        del_xb = u.delxb(x, axis=ax0)
        del_yb = u.delxb(x, axis=ax1)

        self.delxy = [del_xf, del_yf, del_xb, del_yb]
        self.g = 2. * self.amplitude * (del_xb + del_yb - del_xf - del_yf)

        self.LL = self.amplitude * (u.norm2(del_xf)
                                    + u.norm2(del_yf)
                                    + u.norm2(del_xb)
                                    + u.norm2(del_yb))

        return self.g

    def poly_line_coeffs(self, h, x=None):
        ax0, ax1 = self.axes
        if x is None:
            del_xf, del_yf, del_xb, del_yb = self.delxy
        else:
            del_xf = u.delxf(x, axis=ax0)
            del_yf = u.delxf(x, axis=ax1)
            del_xb = u.delxb(x, axis=ax0)
            del_yb = u.delxb(x, axis=ax1)
        hdel_xf = u.delxf(h, axis=ax0)
        hdel_yf = u.delxf(h, axis=ax1)
        hdel_xb = u.delxb(h, axis=ax0)
        hdel_yb = u.delxb(h, axis=ax1)

        c0 = self.amplitude * (u.norm2(del_xf)
                               + u.norm2(del_yf)
                               + u.norm2(del_xb)
                               + u.norm2(del_yb))

        c1 = 2 * self.amplitude * np.real(np.vdot(del_xf, hdel_xf)
                                          + np.vdot(del_yf, hdel_yf)
                                          + np.vdot(del_xb, hdel_xb)
                                          + np.vdot(del_yb, hdel_yb))

        c2 = self.amplitude * (u.norm2(hdel_xf)
                               + u.norm2(hdel_yf)
                               + u.norm2(hdel_xb)
                               + u.norm2(hdel_yb))

        self.coeff = np.array([c0, c1, c2])
        return self.coeff


def prepare_smoothing_preconditioner(amplitude):
    """
    Factory for smoothing preconditioner.
    """
    if amplitude == 0.:
        return None

    class GaussFilt:
        def __init__(self, sigma):
            self.sigma = sigma

        def __call__(self, x):
            return u.c_gf(x, [0, self.sigma, self.sigma])

    if amplitude > 0.:
        logger.debug(
            'Using a smooth gradient filter (Gaussian blur - only for ML)')
        return GaussFilt(amplitude)

    elif amplitude < 0.:
        raise RuntimeError('Hann filter not implemented (negative smoothing amplitude not supported)')
