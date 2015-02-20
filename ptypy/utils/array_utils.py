# -*- coding: utf-8 -*-
"""
utility functions to manipulate/reshape numpy arrays.

This file is part of the PTYPY package.

    :copyright: Copyright 2014 by the PTYPY team, see AUTHORS.
    :license: GPLv2, see LICENSE for details.
"""

import scipy.ndimage as ndi
import numpy as np
from misc import *

__all__ = ['grids','switch_orientation',\
           'mirror','crop_pad_axis','crop_pad',\
           'zoom','shift_zoom','rebin','rebin_2d','crop_pad_symmetric_2d']

def switch_orientation(A, orientation, center=None):
    """
    Switches orientation of Array A along the last two axes (-2,-1)
        
    orientation : 3-tuple of booleans (transpose,flipud,fliplr)
    
    returns
    --------
        Flipped array, new center
    """
    # switch orientation
    if orientation[0]:
        axes = list(range(A.ndim - 2)) + [-1, -2]
        A = np.transpose(A, axes)
        center = (center[1], center[0]) if center is not None else None
    if orientation[1]:
        A = A[..., ::-1, :]
        center = (A.shape[-2] - 1 - center[0], center[1]) if center is not None else None
    if orientation[2]:
        A = A[..., ::-1]
        center = (center[0], A.shape[-1] - 1 - center[1]) if center is not None else None

    return A, np.array(center)


def rebin_2d(A, rebin=1):
    """
    Rebins array A symmetrically along last 2 axes with a factor `rebin`
    """
    newdim = np.asarray(A.shape[-2:]) / rebin
    return A.reshape(-1, newdim[0], rebin, newdim[1], rebin).sum(-1).sum(-2)


def crop_pad_symmetric_2d(A, newshape, center=None):
    """
    Crops or pads Array A symmetrically along the last two axes (-2,-1)
    around center `center` to a new shape `newshape`
    
    """
    # crop / pad symmetrically around center
    osh = np.array(A.shape[-2:])
    c = np.round(center) if center is not None else osh // 2
    sh = np.array(newshape[-2:])
    low = -c + sh // 2
    high = -osh + c + (sh + 1) // 2
    hplanes = np.array([[low[0], high[0]], [low[1], high[1]]])

    if (hplanes != 0).any():
        A = crop_pad(A, hplanes)

    return A, c + low

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
        #cen=sh*np.asarray(center) % sh - 0.5
        cen = np.asarray(center) % sh
        
    return cen
"""
def center_2d(sh,center):
    return translate_to_pix(sh[-2:],expect2(center))
"""
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



@complex_overload
def c_zoom(c,*arg,**kwargs):
    return ndi.zoom(c,*arg,**kwargs)


    
def zoom(c,*arg,**kwargs):
    if np.iscomplexobj(c):
        return c_zoom(c,*arg,**kwargs)
    else:
        return ndi.zoom(c,*arg,**kwargs)
        
@complex_overload
def c_affine_transform(c,*args,**kwargs):
    return ndi.affine_transform(c,*args,**kwargs)
    
def shift_zoom(c,zoom,cen_old,cen_new,**kwargs):
    """\
    Move array from center cen_old to cen_new and perform a zoom.

    This function uses scipy.ndimage.affine_transfrom.
    """
    zoom = np.diag(zoom)
    offset=np.asarray(cen_old)-np.asarray(cen_new).dot(zoom)
    if np.iscomplexobj(c):
        return c_affine_transform(c,zoom,offset,**kwargs)
    else:
        return ndi.affine_transform(c,zoom,offset,**kwargs)




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
        

def mirror(A,axis):
    """\
    mirrors array A along one axis 
    """
    return np.flipud(A.swapaxes(axis,0)).swapaxes(0,axis)
    
def pad_lr(A,axis,l,r,fillpar=0.0, filltype='scalar'):
    """\
    Pads ndarray 'A' orthogonal to 'axis' with 'l' layers (pixels,lines,planes,...)
    on low side an 'r' layers on high side. 
    if filltype=
        'scalar' : uniformly pad with fillpar
        'mirror' : mirror A
        'periodic' : well, periodic fill
        'custom' : pad according arrays found in fillpar
         
    """ 
    fsh=np.array(A.shape)
    if l>fsh[axis]: #rare case
        l-=fsh[axis]
        A=pad_lr(A,axis,fsh[axis],0,fillpar, filltype)
        return pad_lr(A,axis,l,r,fillpar, filltype)
    elif r>fsh[axis]: 
        r-=fsh[axis]
        A=pad_lr(A,axis,0,fsh[axis],fillpar, filltype)
        return pad_lr(A,axis,l,r,fillpar, filltype)
    elif filltype=='mirror':        
        left=mirror(np.split(A,[l],axis)[0],axis)
        right=mirror(np.split(A,[A.shape[axis]-r],axis)[1],axis)
    elif filltype=='periodic':
        right=np.split(A,[r],axis)[0]
        left=np.split(A,[A.shape[axis]-l],axis)[1]
    elif filltype=='project':
        fsh[axis]=l
        left=np.ones(fsh,A.dtype)*np.split(A,[1],axis)[0]
        fsh[axis]=r
        right=np.ones(fsh,A.dtype)*np.split(A,[A.shape[axis]-1],axis)[1] 
    if filltype=='scalar' or l==0:
        fsh[axis]=l
        left=np.ones(fsh,A.dtype)*fillpar
    if filltype=='scalar' or r==0:
        fsh[axis]=r
        right=np.ones(fsh,A.dtype)*fillpar 
    if filltype=='custom':
        left=fillpar[0].astype(A.dtype)
        rigth=fillpar[1].astype(A.dtype)   
    return np.concatenate((left,A,right),axis=axis)


def _roll_from_pixcenter(sh,center):
    """\
    returns array of ints as input for np.roll
    use np.roll(A,-roll_from_pixcenter(sh,cen)[ax],ax) to put 'cen' in geometric center of array A
    """
    sh=np.array(sh)
    if center != None:
        if center=='fftshift':
            cen=sh//2.0
        elif center=='geometric':
            cen=sh/2.0-0.5
        elif center=='fft':
            cen=sh*0.0
        elif center is not None:
            cen=sh*np.asarray(center) % sh - 0.5
            
        roll=np.ceil(cen - sh/2.0) % sh
    else:
        roll=np.zeros_like(sh)
    return roll.astype(int)
       
    

def crop_pad_axis(A,hplanes,axis,roll=0,fillpar=0.0, filltype='scalar'):
    """\
    crops or pads a volume array 'A' at beginning and end of axis 'axis' 
    with a number of hyperplanes specified by 'hplanes'

    Paramters:
    -------------
    A : nd-numpy array
    
    hplanes: tuple or scalar int
    axis: int, axis to be used for cropping / padding
    roll: int, roll array backwards by this number prior to padding / cropping. the roll is reversed afterwards
   
    if 'hplanes' is,
    -scalar and negativ : 
        crops symmetrically, low-index end of axis is preferred if hplane is odd,
    -scalar and positiv : 
        pads symmetrically with a fill specified with 'fillpar' and 'filltype'
        look at function pad_lr() for detail.
    -is tupel : function pads /crops asymmetrically according to the tupel.
    
    Usage:
    -------------
    A=np.ones((8,9))
    B=crop_pad_axis(A,2,0)
    -> a total of 2 rows, one at top, one at bottom (same as crop_pad_axis(A,(1,1),0))
    B=crop_pad_axis(A,(-3,2),1)
    -> crop 3 columns on left side and pad 2 columns on right
    V=np.random.rand(3,5,5)
    B=crop_pad_axis(V,-2,0)
    -> crop one plane on low-side and high-side (total of 2) of Volume V
    B=crop_pad_axis(V,(3,-2),1,filltype='mirror')
    -> mirror volume 3 planes on low side of row axis, crop 2 planes on high side
    
    Author: Bjoern Enders
    """
    if np.isscalar(hplanes):
        hplanes=int(hplanes)
        r=np.abs(hplanes) / 2 * np.sign(hplanes)
        l=hplanes - r
    elif len(hplanes)==2:
        l=int(hplanes[0])
        r=int(hplanes[1])
    else:
        raise RuntimeError('unsupoorted input for \'hplanes\'')
        
    if roll!=0:
        A=np.roll(A,-roll,axis=axis)
        
    if l<=0 and r<=0:
        A=np.split(A,[-l,A.shape[axis]+r],axis)[1]
    elif l>0 and r>0:
        A=pad_lr(A,axis,l,r,fillpar,filltype)
    elif l>0 and r<=0:
        A=pad_lr(A,axis,l,0,fillpar,filltype)
        A=np.split(A,[0,A.shape[axis]+r],axis)[1]
    elif l<=0 and r>0:
        A=pad_lr(A,axis,0,r,fillpar,filltype)
        A=np.split(A,[-l,A.shape[axis]],axis)[1]
        
        
    if roll!=0:
        return np.roll(A,roll+r,axis=axis)
    else:
        return A

 
def crop_pad(A,hplane_list,axes=None,cen=None,fillpar=0.0,filltype='scalar'):
    """\
    crops or pads a volume array 'A' with a number of hyperplanes according to parameters in 'hplanes'
    wrapper for crop_pad_axis
    
    Parameters
    ----------------------
    hplane_list : 
     -list of scalars or tupels counting the number of hyperplanes to crop / pad 
     -see crop_pad_axis() for detail
     -if N=len(hplane_list) has less entries than dimensions of A, the last N axes are used 

    axes: list of axes to be used for cropping / padding, has to be same length as hplanes
    
    cen: center of array, padding/cropping occurs at cen + A.shape / 2
    
    Usage:
    ----------------------
    V=np.random.rand(3,5,5)
    B=crop_pad(V,[3,4])
    ->  pads 4 planes of zeros on the last axis (2 on low side and 2 on high side),
        and pads 3 planes of zeros on the second last axis (2 on low side and 1 on high side)
        equivalent: B=crop_pad(V,[(2,1),(2,2)])
                    B=crop_pad(V,[(2,1),(2,2)], axes=[-2,-1],fillpar=0.0,filltype='scalar')
    
    C=pyE17.utils.fgrid_2d((4,5))
    cropped_fgrid=crop_pad(V,[-2,4],cen='fft')
    -> note that cropping/ padding now occurs at the start and end of fourier coordinates
    -> useful for cropping /padding high frequencies in fourier space.
    
    Author: Bjoern Enders
    """
    if axes is None:
        axes=np.arange(len(hplane_list))-len(hplane_list)
    elif not(len(axes)==len(hplane_list)):
        raise RuntimeError('if axes is specified, hplane_list has to be same length as axes')
    
    sh=np.array(A.shape)
    roll = _roll_from_pixcenter(sh,cen)
        
    for ax,cut in zip(axes,hplane_list):
        A=crop_pad_axis(A,cut,ax,roll[ax],fillpar,filltype)
    return A

