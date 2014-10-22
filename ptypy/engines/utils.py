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
from ..utils import parallel

def basic_fourier_update(diff_view,mask_view,pbound=None,alpha=1.):
    """\
    fourier update one a single view using its associated pods.
    updates on all pods' exit waves.
    
    Parameters:
    -----------
    diff_view : view to diffraction data
    alpha :  [0,1] mixing between old and new exit wave
    pbound : (float, None) power bound. If None pbound is deactivated
    
    Returns:
    --------
    error array consisting of
    err_fmag : Fourier magnitude error; quadratic deviation from root of experimental data
    err_phot : quadratic deviation from experimental data (photons)
    err_exit : quadratic deviation of exit waves before and after Fourier iteration
    """
    
    #pods = self.pty.pods
    # Fourier modulus constraint
    err_fmag = -1
    err_phot = -1
    err_exit = -1
    
    # exit function with Nones if view is not used by this process
    if not diff_view.active: return np.array([err_fmag,err_phot,err_exit])
    
    # prepare dict for storing propagated waves
    f = {}
    
    # buffer for accumulated photons
    af2= np.zeros_like(diff_view.data) 
    
    # get the mask
    fmask = mask_view.data if mask_view is not None else np.ones_like(af2)
    
    # propagate the exit waves
    for name,pod in diff_view.pods.iteritems():
        if not pod.active: continue
        f[name]=( pod.fw( (1+alpha)*pod.probe*pod.object - alpha* pod.exit ))
        af2 += u.abs2(f[name])
        
    # calculate deviations from measured data
    I = np.abs(diff_view.data)
    err_phot = np.sum(fmask*(af2 - I)**2/(I+1.))/fmask.sum()
    fmag = np.sqrt(np.abs(I))
    af=np.sqrt(af2)
    fdev = af - fmag 
    err_fmag = np.sum(fmask*fdev**2)/fmask.sum()
    err_exit = 0.

    # apply changes and backpropagate
    if pbound is None or err_fmag > pbound:
        renorm = np.sqrt(pbound / err_fmag) if pbound is not None else 0.0 # don't know if that is correct
        fm = (1-fmask) + fmask*(fmag + fdev*renorm)/(af + 1e-10)
        for name,pod in diff_view.pods.iteritems():
            if not pod.active: continue
            df = pod.bw(fm*f[name])-pod.probe*pod.object
            pod.exit += df
            err_exit +=np.mean(u.abs2(df))
    else:
        for name,pod in diff_view.pods.iteritems():
            if not pod.active: continue
            df = pod.probe*pod.object-pod.exit
            pod.exit += df
            err_exit +=np.mean(u.abs2(df))
     
    return np.array([err_fmag,err_phot,err_exit])

def FourierProjectionBasic(psi,fmag,out=None):
    pass
    
def FourierProjectionPowerBound(psi,fmag,out=None):
    pass

def FourierProjectionModes(psilist,fmag,out=None):
    pass
    
def Cnorm2(c):
    """\
    Compute a norm2 on a whole container.
    """
    r = 0.
    for name, s in c.S.iteritems():
        r += u.norm2(s.data)
    return r

def Cdot(c1,c2):
    """\
    Compute the dot product on two containers.
    No check is made to ensure they are of the same kind.
    """
    r = 0.
    for name, s in c1.S.iteritems():
        r += np.vdot(c1.S[name].data.flat, c2.S[name].data.flat)
    return r
    
    
    
    
    
    
    
