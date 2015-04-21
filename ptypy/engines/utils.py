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
    
def basic_fourier_update(diff_view,pbound=None,alpha=1.,LL_error=True):
    """\
    Fourier update one a single view using its associated pods.
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
        If ``True``, calculates log-likehood and puts it in the last entry 
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
    
    #pods = self.pty.pods
    ## Fourier modulus constraint
    #err_fmag = -1
    #err_phot = -1
    #err_exit = -1
    
    ## exit function with Nones if view is not used by this process
    #if not diff_view.active: return np.array([err_fmag,err_phot,err_exit])
    
    # prepare dict for storing propagated waves
    f = {}
    
    # buffer for accumulated photons
    af2= np.zeros_like(diff_view.data) 

    # calculate deviations from measured data
    I = diff_view.data

    # get the mask
    #fmask = mask_view.data if mask_view is not None else np.ones_like(af2)
    fmask = diff_view.pod.mask
        
    # for log likelihood error
    if LL_error is True:
        LL= np.zeros_like(diff_view.data) 
        for name,pod in diff_view.pods.iteritems():
            LL +=  u.abs2(pod.fw( pod.probe*pod.object))
        err_phot = np.sum(fmask*np.square(LL - I)/(I+1.)) /np.prod(LL.shape)           
    else:
        err_phot=0.
    
    # propagate the exit waves
    for name,pod in diff_view.pods.iteritems():
        if not pod.active: continue
        f[name]= pod.fw( (1+alpha)*pod.probe*pod.object - alpha* pod.exit )
        af2 += u.cabs2(f[name]).real
    
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
            err_exit +=np.mean(u.cabs2(df).real)
    else:
        for name,pod in diff_view.pods.iteritems():
            if not pod.active: continue
            df = pod.probe*pod.object-pod.exit
            pod.exit += df
            err_exit +=np.mean(u.cabs2(df).real)
    
    return np.array([err_fmag,err_phot,err_exit])
    
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
    for name, s in c.S.iteritems():
        r += u.norm2(s.data)
    return r

def Cdot(c1,c2):
    """\
    Compute the dot product on two containers `c1` and `c2`.
    No check is made to ensure they are of the same kind.
    
    :param Container c1,c2: Input
    :returns: The dot product (*scalar*)
    """
    r = 0.
    for name, s in c1.S.iteritems():
        r += np.vdot(c1.S[name].data.flat, c2.S[name].data.flat)
    return r
    
def Callreduce(c):
    """
    Performs MPI parallel ``allreduce`` with a sum as reduction
    for all :any:`Storage` instances held by this :any:`Container`
    
    :param Container c: Input
    
    See also
    --------
    ptypy.utils.parallel.allreduce
    """
    for s in c.S.itervalues():
        parallel.allreduce(s.data)    
    
    
    
    
    
