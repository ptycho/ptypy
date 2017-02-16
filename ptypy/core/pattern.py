# -*- coding: utf-8 -*-
"""
This module generates the scan patterns. 

This file is part of the PTYPY package.

    :copyright: Copyright 2014 by the PTYPY team, see AUTHORS.
    :license: GPLv2, see LICENSE for details.
"""
from .. import utils as u
#from ..utils import prop 
from ..utils.verbose import logger
import numpy as np

__all__=['DEFAULT','from_pars','round_scan','round_scan_roi'\
         'raster_scan','spiral_scan','spiral_scan_roi']
DEFAULT=u.Param(
    #### Paramaters for popular scan methods 
    scan_type = None, # [None,'round', 'raster', 'round_roi','spiral','spiral_roi','custom']
    dr = 1.5e-6,             # round,round_roi :width of shell 
    nr = 5,                 # round : number of intervals (# of shells - 1) 
    nth = 5,                 # round,round_roi: number of points in the first shell 
    lx = 15e-6,               # round_roi: Width of ROI 
    ly = 15e-6,               # round_roi: Height of ROI 
    nx = 10,                 # raster scan: number of steps in x
    ny = 10,                 # raster scan: number of steps in y
    dx = 1.5e-6,               # raster scan: step size (grid spacing)
    dy = 1.5e-6,               # raster scan: step size (grid spacing)
    #### other 
    positions = None,        # fill this list with your own script if you want other scan patterns, choose 'custom' as san type
)

import warnings
warnings.simplefilter('always', DeprecationWarning)
warnings.warn('This module is deprecated and will be removed from the package on 30/11/16',DeprecationWarning)

DEFAULT = u.validator.make_sub_default('.scan.xy',depth=3)
"""Default pattern parameters. See :py:data:`.scan.xy` and a short listing below"""

def from_pars(pars=None):
    """
    Creates position array from parameter tree `pars`. See :py:data:`DEFAULT`
    
    :param Param pars: Input parameters
    :returns ndarray pos: A numpy.ndarray of shape ``(N,2)`` for *N* positios
    """
    p=u.Param(DEFAULT)
    if pars is not None: # and (isinstance(pars,dict) or isinstance(pars,u.Param)):
        p.update(pars)
    
    if p.type is None:
        logger.debug('Scan_type `None` is chosen . Will use positions provided by meta information')
        return None
        
    elif p.type=='round':
        pos=round_scan(**p.round)
    elif p.type=='round_roi':
        pos=round_scan_roi(**p.round_roi)
    elif p.type=='spiral':
        pos=spiral_scan(**p.spiral)
    elif p.type=='spiral_roi':
        pos=spiral_scan_roi(**p.spiral_roi)
    elif p.type=='raster':
        pos=raster(**p.raster)
    else: 
        pos = p.positions
    pos=np.asarray(pos)
    logger.info('Prepared %d positions' % len(pos))
    return pos
    


def augment_to_coordlist(a,Npos):
 
    # force into a 2 column matrix
    # drop element if size is not a modulo of 2
    a = np.asarray(a)
    if a.size == 1:
        a=np.atleast_2d([a,a])
        
    if a.size % 2 != 0:
        a=a.flatten()[:-1]
    
    a=a.reshape(a.size//2,2)
    # append multiples of a until length is greater equal than Npos
    if a.shape[0] < Npos:
        b=np.concatenate((1+Npos//a.shape[0])*[a],axis=0)
    else:
        b=a
    
    return b[:Npos,:2]
    
def raster_scan(ny=10,nx=10,dy=1.5e-6,dx=1.5e-6):
    """
    Generates as raster scan.
    
    Parameters
    ----------
    ny, nx : int
        Number of steps in *y* (vertical) and *x* (horizontal) direction
        *x* is the fast axis
        
    dy, dx : float
        Step size (grid spacinf) in *y* and *x*  
        
    Returns
    -------
    pos : ndarray
        A (N,2)-array of positions.
        
    Examples
    --------
    >>> from ptypy.core import import xy
    >>> from matplotlib import pyplot as plt
    >>> plt.plot(xy.raster_scan());plt.show()
    """
    iix, iiy = np.indices((nx+1,ny+1))
    positions = [(sx*i, sy*j) for i,j in zip(iix.ravel(), iiy.ravel())]
    return positions

def round_scan(r_in, r_out, nr, nth):
    """\
    Round scan positions, defined as in spec and matlab.
    """
    dr = (r_out - r_in)/ nr
    positions = []
    for ir in range(1,nr+2):
        rr = r_in + ir*dr
        dth = 2*np.pi / (nth*ir)
        positions.extend([(rr*np.sin(ith*dth), rr*np.cos(ith*dth)) for ith in range(nth*ir)])
    return positions

def round_scan_roi(dr, lx, ly, nth):
    """\
    Round scan positions with ROI, defined as in spec and matlab.
    """
    rmax = np.sqrt( (lx/2)**2 + (ly/2)**2 )
    nr = np.floor(rmax/dr) + 1
    positions = []
    for ir in range(1,int(nr+2)):
        rr = ir*dr
        dth = 2*np.pi / (nth*ir)
        th = 2*np.pi*np.arange(nth*ir)/(nth*ir)
        x1 = rr*np.sin(th)
        x2 = rr*np.cos(th)
        positions.extend([(xx1,xx2) for xx1,xx2 in zip(x1,x2) if (np.abs(xx1) <= ly/2) and (np.abs(xx2) <= lx/2)])
    return positions


def spiral_scan(dr,r_out=None,maxpts=None):
    """\
    Spiral scan positions.
    """
    alpha = np.sqrt(4*np.pi)
    beta = dr/(2*np.pi)
    
    if maxpts is None:
        assert r_out is not None
        maxpts = 100000000

    if r_out is None:
        r_out = np.inf

    positions = []
    for k in xrange(maxpts):
        theta = alpha*np.sqrt(k)
        r = beta * theta
        if r > r_out: break
        positions.append( (r*np.sin(theta), r*np.cos(theta)) )
    return positions

def spiral_scan_roi(dr,lx,ly):
    """\
    Spiral scan positions. ROI
    """
    alpha = np.sqrt(4*np.pi)
    beta = dr/(2*np.pi)
    
    rmax = .5*np.sqrt(lx**2 + ly**2)
    positions = []
    for k in xrange(1000000000):
        theta = alpha*np.sqrt(k)
        r = beta * theta
        if r > rmax: break
        x,y = r*np.sin(theta), r*np.cos(theta)
        if abs(x) > lx/2: continue
        if abs(y) > ly/2: continue
        positions.append( (x,y) )
    return positions

