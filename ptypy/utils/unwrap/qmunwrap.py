# -*- coding: utf-8 -*-
"""
Quality-map guided phase unwrapping.

This file is part of the PTYPY package.

    :copyright: Copyright 2014 by the PTYPY team, see AUTHORS.
    :license: see LICENSE for details.
"""

import numpy as np

from . import _qmunwrap

__all__ = ['unwrap', 'qualitymap']


def unwrap(phase, num_levels=8, start=(0, 0)):
    """
    Unwraps a two-dimensional phase array.

    Pixels are unwrapped in order of decreasing quality, the quality being
    measured by the local squared wrapped gradient (see :any:`qualitymap`).
    The quality map is quantized into `num_levels` bins, which avoids sorting
    at the price of a (low-risk) non-sequential unwrapping.

    Parameters
    ----------
    phase : array-like
        Two-dimensional wrapped phase.

    num_levels : int
        Number of quality bins. Behaviour is not expected to be much
        different for num_levels > 20 or so.

    start : tuple of int
        Coordinates of the pixel the unwrapping starts from.

    Returns
    -------
    ndarray
        The unwrapped phase, equal to `phase` at the starting pixel.
    """
    phase = np.ascontiguousarray(phase, dtype=np.float64)
    if phase.ndim != 2:
        raise ValueError("phase must be a 2D array, got %d dimension(s)"
                         % phase.ndim)

    num_levels = int(num_levels)
    if num_levels < 1:
        raise ValueError("num_levels must be >= 1, got %d" % num_levels)

    start0, start1 = (int(s) for s in start)
    if not (0 <= start0 < phase.shape[0] and 0 <= start1 < phase.shape[1]):
        raise ValueError("start %r is outside an array of shape %r"
                         % ((start0, start1), phase.shape))

    out = np.empty_like(phase)
    _qmunwrap.unwrap(phase, out, num_levels, start0, start1)
    return out


def qualitymap(phase):
    """
    Computes the quality map of a two-dimensional wrapped phase.

    The quality map is the sum, over the (up to four) edges a pixel takes part
    in, of the squared wrapped phase gradient along that edge. Low values mean
    high quality.

    Parameters
    ----------
    phase : array-like
        Two-dimensional wrapped phase.

    Returns
    -------
    ndarray
        The quality map, with the same shape as `phase`.
    """
    phase = np.ascontiguousarray(phase, dtype=np.float64)
    if phase.ndim != 2:
        raise ValueError("phase must be a 2D array, got %d dimension(s)"
                         % phase.ndim)

    qmap = np.zeros_like(phase)
    _qmunwrap.qualitymap(phase, qmap)
    return qmap
