# -*- coding: utf-8 -*-
"""
Miscellaneous utility functions.

This file is part of the PTYPY package.

    :copyright: Copyright 2014 by the PTYPY team, see AUTHORS.
    :license: GPLv2, see LICENSE for details.
"""
import os
import scipy.ndimage as ndi
from scipy.special import erf 
import numpy as np

__all__ = ['confine','translate_to_pix','mass_center','center_2d','grids',\
           'rebin','expect2','expect3','c_zoom','c_gf','c_shift_zoom',\
           'shift_zoom','keV2m', 'clean_path','zoom','gf','rebin']

def rebin(a, *args,**kwargs):
    '''\
    Rebin ndarray data into a smaller ndarray of the same rank whose dimensions
    are factors of the original dimensions.
    
    .. note::
        eg. An array with 6 columns and 4 rows
        can be reduced to have 6,3,2 or 1 columns and 4,2 or 1 rows.
    
    Parameters
    ----------
    a : nd-numpy-array
        Input array.
        
    axis : int, Default=-1, optional
        The laplacian is computed along the provided axis or list of axes, or all axes if None
    
    Returns
    -------
    out : nd-numpy-array
        Rebined array.
          
    Examples
    --------
    >>> import ptypy
    >>> import numpy as np
    >>> a=np.random.rand(6,4) 
    >>> b=ptypy.utils.rebin(a,3,2)
    a.reshape(args[0],factor[0],args[1],factor[1],).sum(1).sum(2)*( 1./factor[0]/factor[1])
    >>> a2=np.random.rand(6)
    >>> b2=ptypy.utils.rebin(a2,2)
    a.reshape(args[0],factor[0],).sum(1)*( 1./factor[0])
    '''
    shape = a.shape
    lenShape = a.ndim
    factor = np.asarray(shape)/np.asarray(args)
    evList = ['a.reshape('] + \
             ['args[%d],factor[%d],'%(i,i) for i in range(lenShape)] + \
             [')'] + ['.sum(%d)'%(i+1) for i in range(lenShape)] + \
             ['*( 1.'] + ['/factor[%d]'%i for i in range(lenShape)] + [')']
    if kwargs.get('verbose',False):
        print ''.join(evList)
    return eval(''.join(evList))
           
def confine(A):
    """\
    Doc TODO.
    """
    sh=np.asarray(A.shape)[1:]
    A=A.astype(float)
    m=np.reshape(sh,(len(sh),) + len(sh)*(1,))
    return (A+m//2.0) % m - m//2.0

    
def keV2m(keV):
    """\
    Convert photon energy in keV to wavelength (in vacuum) in meters.
    """
    wl = 1./(keV*1000)*4.1356e-7*2.9998    
    
    return wl

def translate_to_pix(sh,center):
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
        cen=sh*np.asarray(center) % sh - 0.5

    return cen

def mass_center(A, axes=None):
    """\
    Returns center of mass of n-dimensional array 'A' 
    along tuple of axis 'axes'
    """
    A=np.asarray(A)
    
    if axes is None:
        axes=tuple(range(1,A.ndim+1))
    else:
        axes=tuple(np.array(axes)+1)

    return np.sum(A*np.indices(A.shape),axis=axes)/np.sum(A)

def center_2d(sh,center):
    return translate_to_pix(sh[-2:],expect2(center))

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
    sh : the shape of the N-dimensional array
    
    psize : pixel size in each dimensions
    
    center : tupel of pixels, or use center='fftshift' for fftshift-like grid
             and center='geometric' for the matrix center as grid origin
    
    FFTlike : if False, grids ar not bound by the interval [-sh//2:sh//2[
    
    """
    sh=np.asarray(sh)
    
    cen=translate_to_pix(sh,center)

    grid=np.indices(sh).astype(float) - np.reshape(cen,(len(sh),) + len(sh)*(1,))
        
    if FFTlike:
        grid=confine(grid)
        
    if psize is None:
        return grid
    else:
        psize = np.asarray(psize)
        if psize.size == 1:
            psize = psize * np.ones((len(sh),))
        psize = np.asarray(psize).reshape( (len(sh),) + len(sh)*(1,))
        return grid * psize


def rebin(a, *args):
    """\
    Rebin ndarray data into a smaller ndarray of the same rank whose dimensions
    are factors of the original dimensions. eg. An array with 6 columns and 4 rows
    can be reduced to have 6,3,2 or 1 columns and 4,2 or 1 rows.
    example usages:
    >>> a=rand(6,4); b=rebin(a,3,2)
    >>> a=rand(6); b=rebin(a,2)
    """
    shape = a.shape
    lenShape = a.ndim
    factor = np.asarray(shape)/np.asarray(args)
    evList = ['a.reshape('] + \
             ['args[%d],factor[%d],'%(i,i) for i in range(lenShape)] + \
             [')'] + ['.sum(%d)'%(i+1) for i in range(lenShape)] + \
             ['/factor[%d]'%i for i in range(lenShape)]
    #print ''.join(evList)
    return eval(''.join(evList))

def expect2(a):
    """\
    generates 1d numpy array with 2 entries generated from multiple inputs 
    (tuples, arrays, scalars). main puprose of this function is to circumvent
    debugging of input.
    
    expect2( 3.0 ) -> np.array([3.0,3.0])
    expect2( (3.0,4.0) ) -> np.array([3.0,4.0])
    
    even higher order inputs possible, though not tested much
    """
    a=np.atleast_1d(a)
    if len(a)==1:
        b=np.array([a.flat[0],a.flat[0]])
    else: #len(psize)!=2:
        b=np.array([a.flat[0],a.flat[1]])        
    return b
    
def expect3(a):
    """\
    generates 1d numpy array with 3 entries generated from multiple inputs 
    (tuples, arrays, scalars). main puprose of this function is to circumvent
    debugging of input.
    
    expect2( 3.0 ) -> np.array([3.0,3.0,3.0])
    expect2( (3.0,4.0) ) -> np.array([3.0,4.0,4.0])
    
    even higher order inputs possible, though not tested much
    """
    a=np.atleast_1d(a)
    if len(a)==1:
        b=np.array([a.flat[0]]*3)
    elif len(a)==2:
        b=np.array([a.flat[0],a.flat[1],a.flat[1]])
    else:
        b=np.array([a.flat[0],a.flat[1],a.flat[2]])
    return b
 
def complex_overload(func):
    """\
    Doc TODO
    """
    def overloaded(c,*args,**kwargs):
        return func(np.real(c),*args,**kwargs) +1j *func(np.imag(c),*args,**kwargs)
    return overloaded
    
@complex_overload
def c_zoom(c,*arg,**kwargs):
    return ndi.zoom(c,*arg,**kwargs)

@complex_overload
def c_gf(c,*arg,**kwargs):
    return ndi.gaussian_filter(c,*arg,**kwargs)
    
def zoom(c,*arg,**kwargs):
    if np.iscomplexobj(c):
        return c_zoom(c,*arg,**kwargs)
    else:
        return ndi.zoom(c,*arg,**kwargs)
        
def gf(c,*arg,**kwargs):
    if np.iscomplexobj(c):
        return c_gf(c,*arg,**kwargs)
    else:
        return ndi.gaussian_filter(c,*arg,**kwargs)
        
def shift_zoom(c,zoom,cen_old,cen_new,**kwargs):
    """\
    Move array from center cen_old to cen_new and perform a zoom.

    This function uses scipy.ndimage.affine_transfrom.
    """
    zoom = np.diag(zoom)
    offset=np.asarray(cen_old)-np.asarray(cen_new).dot(zoom)
    return ndi.affine_transform(c,zoom,offset,**kwargs)


def c_shift_zoom(c,*args,**kwargs):
    return complex_overload(shift_zoom)(c,*args,**kwargs)


def fill3D(A,B,offset=[0,0,0]):
    """\
    Fill 3-dimensional array A with B.
    """
    if A.ndim != 3 or B.ndim!=3:
        raise ValueError('3D a numpy arrays expected')    
    Alim=np.array(A.shape)
    Blim=np.array(B.shape)
    off=np.array(offset)
    Ao = off.copy()
    Ao[Ao<0]=0
    Bo = -off.copy()
    Bo[Bo<0]=0
    print Ao,Bo
    if (Bo > Blim).any() or (Ao > Alim).any():
        print "misfit"
        pass
    else:
        A[Ao[0]:min(off[0]+Blim[0],Alim[0]),Ao[1]:min(off[1]+Blim[1],Alim[1]),Ao[2]:min(off[2]+Blim[2],Alim[2])] \
        =B[Bo[0]:min(Alim[0]-off[0],Blim[0]),Bo[1]:min(Alim[1]-off[1],Blim[1]),Bo[2]:min(Alim[2]-off[2],Blim[2])] 
        


def clean_path(filename):
    """\
    Makes path absolute and create all sub directories if needed.
    
    Parameters
    ----------
    filename : str
               A filename. 
    """
    filename = os.path.abspath(os.path.expanduser(filename))
    base = os.path.split(filename)[0]
    if not os.path.exists(base):
        os.makedirs(base)
    return filename

