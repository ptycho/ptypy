# -*- coding: utf-8 -*-
"""
This module generates the scan patterns.

This file is part of the PTYPY package.

    :copyright: Copyright 2014 by the PTYPY team, see AUTHORS.
    :license: see LICENSE for details.
"""
import numpy as np
import warnings

from .. import utils as u
from ..utils.verbose import logger
from ..utils.descriptor import EvalDescriptor

warnings.simplefilter('always', DeprecationWarning)

__all__ = ['xy_desc', 'from_pars', 'round_scan', 'raster_scan', 'spiral_scan']

TEMPLATES = u.Param()


# Local, module-level defaults. These can be appended to the defaults of
# other classes.
xy_desc = EvalDescriptor('xy')
xy_desc.from_string(r"""
    [override]
    default =
    type = array
    help =

    [model]
    default =
    type = str
    help = None, 'round', 'raster', 'spiral' or array-like

    [extent]
    default = 15e-6
    type = float, tuple
    help =

    [spacing]
    default = 1.5e-6
    type = float
    help = Step size (grid spacing)

    [steps]
    default = 10
    type = int
    help =

    [offset]
    default = 0.
    type = float
    help =

    [jitter]
    default =
    type = float
    help =

    [count]
    default =
    type = int
    help =
    """)

DEFAULT = xy_desc.make_default(99)

def from_pars(xypars=None):
    """
    Creates position array from parameter tree `pars`. See :py:data:`DEFAULT`

    :param xypars: TreeDict
        Input parameters
    :return: ndarray pos
        A numpy.ndarray of shape ``(N,2)`` for *N* positions
    """
    p = DEFAULT.copy(depth=3)
    model = None
    if hasattr(xypars, 'items') or hasattr(xypars, 'items'):
        # This is a dict
        p.update(xypars, in_place_depth=3)
    elif xypars is None:
        return None
    elif str(xypars) == xypars:
        if xypars in TEMPLATES.keys():
            return from_pars(TEMPLATES[xypars])
        else:
            raise RuntimeError(
                'Template string `%s` for pattern creation is not understood'
                % xypars)
    elif type(xypars) in [np.ndarray, list]:
        return np.array(xypars)
    else:
        ValueError('Input type `%s` for scan pattern creation is not understood'
                   % str(type(xypars)))

    if p.override is not None:
        return np.asarray(p.override)

    elif p.model is None:
        logger.debug('Scan pattern model `None` is chosen.\n'
                     'Will use positions provided by meta information.')
        return None
    else:
        if type(p.model) in [np.ndarray, list]:
            pos = np.asarray(p.model)
        elif p.model == 'round':
            e, l, s = _complete(p.extent, p.steps, p.spacing)
            pos = round_scan(s[0], l[0]//2)
        elif p.model == 'spiral':
            e, l, s = _complete(p.extent, p.steps, p.spacing)
            pos = spiral_scan(s[0], e[0]/2)
        elif p.model == 'raster':
            e, l, s = _complete(p.extent, p.steps, p.spacing)
            pos = raster_scan(s[0], s[1], l[0], l[1])
        else:
            raise NameError('Unknown pattern type %s' % str(p.model))

        if p.offset is not None:
            pos += u.expect2(p.offset)
        # Filter roi
        if p.extent is not None:
            roi = u.expect2(p.extent) / 2.
            new = []
            for posi in pos:
                if ((posi[0] >= -roi[0])
                        and (posi[0] <= roi[0])
                        and (posi[1] >= -roi[1])
                        and (posi[1] <= roi[1])):
                    new.append(posi)

            pos = np.array(new)

        if p.jitter is not None:
            pos = pos

        if p.count is not None and p.count > 0:
            pos = pos[:int(p.count)]

        logger.info('Prepared %d positions' % len(pos))
        return pos


def _complete(extent, steps, spacing):
    a = np.sum([item is None for item in [extent, steps, spacing]])
    if a >= 2:
        raise ValueError(
            'Only one of <extent>, <layer> or <spacing> may be None')
    elif steps is None:
        e = u.expect2(extent)
        s = u.expect2(spacing)
        l = (e / s).astype(np.int)
    elif spacing is None:
        e = u.expect2(extent)
        l = u.expect2(steps)
        s = e / l
    else:
        l = u.expect2(steps)
        s = u.expect2(spacing)
        e = l * s

    return e, l, s


def augment_to_coordlist(a, Npos):
    # Force into a 2 column matrix
    # Drop element if size is not a modulo of 2
    a = np.asarray(a)
    if a.size == 1:
        a = np.atleast_2d([a, a])

    if a.size % 2 != 0:
        a = a.flatten()[:-1]

    a = a.reshape(a.size//2, 2)
    # Append multiples of a until length is greater equal than Npos
    if a.shape[0] < Npos:
        b = np.concatenate((1 + Npos//a.shape[0]) * [a], axis=0)
    else:
        b = a

    return b[:Npos, :2]


def raster_scan(dy=1.5e-6, dx=1.5e-6, ny=10, nx=10, ang=0.):
    """
    Generates a raster scan.

    Parameters
    ----------
    ny, nx : int
        Number of steps in *y* (vertical) and *x* (horizontal) direction
        *x* is the fast axis

    dy, dx : float
        Step size (grid spacing) in *y* and *x*
        
    ang: float
        Rotation angle of the raster grid (counterclockwise, in degrees)

    Returns
    -------
    pos : ndarray
        A (N,2)-array of positions. It is ``N = (nx+1)*(nx+1)``

    Examples
    --------
    >>> from ptypy.core import xy
    >>> from matplotlib import pyplot as plt
    >>> pos = xy.raster_scan()
    >>> plt.plot(pos[:, 1], pos[:, 0], 'o-'); plt.show()
    """
    iix, iiy = np.indices((nx, ny))
    if ang != 0.:
        ang *= np.pi/180.
        iix, iiy = np.cos(ang)*iix + np.sin(ang)*iiy, np.cos(ang)*iiy - np.sin(ang)*iix
    positions = [(dx*i, dy*j) for i, j in zip(iix.ravel(), iiy.ravel())]
    return np.asarray(positions)


def round_scan(dr=1.5e-6, nr=5, nth=5, bullseye=True):
    """
    Generates a round scan

    Parameters
    ----------
    nr : int
        Number of radial steps from center, ``nr + 1`` shells will be made

    dr : float
        Step size (shell spacing)

    nth : int, optional
        Number of points in first shell

    bullseye : bool
        If set false, point of origin will be ignored

    Returns
    -------
    pos : ndarray
        A (N,2)-array of positions.

    Examples
    --------
    >>> from ptypy.core import xy
    >>> from matplotlib import pyplot as plt
    >>> pos = xy.round_scan()
    >>> plt.plot(pos[:,1], pos[:,0], 'o-'); plt.show()
    """
    if bullseye:
        positions = [(0., 0.)]
    else:
        positions = []

    for ir in range(1, nr+2):
        rr = ir * dr
        dth = 2 * np.pi / (nth * ir)
        positions.extend([(rr * np.sin(ith*dth), rr * np.cos(ith*dth))
                          for ith in range(nth*ir)])
    return np.asarray(positions)


def spiral_scan(dr=1.5e-6, r=7.5e-6, maxpts=None):
    """
    Generates a spiral scan.

    Parameters
    ----------
    r : float
        Number of radial steps from center, ``nr + 1`` shells will be made

    dr : float
        Step size (shell spacing)

    nth : int, optional
        Number of points in first shell

    Returns
    -------
    pos : ndarray
        A (N,2)-array of positions. It is

    Examples
    --------
    >>> from ptypy.core import xy
    >>> from matplotlib import pyplot as plt
    >>> pos = xy.spiral_scan()
    >>> plt.plot(pos[:, 1], pos[:, 0], 'o-'); plt.show()
    """
    alpha = np.sqrt(4 * np.pi)
    beta = dr / (2*np.pi)

    if maxpts is None:
        maxpts = 100000

    positions = []
    for k in range(maxpts):
        theta = alpha * np.sqrt(k)
        rr = beta * theta
        if rr > r:
            break
        positions.append((rr * np.sin(theta), rr * np.cos(theta)))
    return np.asarray(positions)


