# -*- coding: utf-8 -*-
"""
Miscellaneous utility functions with little dependencies from other modules

This file is part of the PTYPY package.

    :copyright: Copyright 2014 by the PTYPY team, see AUTHORS.
    :license: GPLv2, see LICENSE for details.
"""
import os
import numpy as np

__all__ = ['str2int','str2range',\
           'complex_overload','expect2','expect3',\
           'keV2m', 'clean_path']

def str2range(s):
    """
    generates an index list
    range_from_string('1:4:2') == range(1,4,2)
    BUT
    range_from_string('1') == range(1,2)
     
    Author: Bjoern Enders
    """
    start = 0
    stop = 1
    step = 1
    l=s.split(':')
    
    il = [int(ll) for ll in l]
    
    if len(il)==0:
        pass    
    elif len(il)==1:
        start=il[0]; stop=start+1
    elif len(il)==2:
        start, stop= il
    elif len(il)==3:
        start, stop, step = il
        
    return range(start,stop,step)

def str2int(A):
    """
    Transforms numpy array A of strings to uint8 and back
    """
    A=np.asarray(A)
    dt = A.dtype.str
    if '|S' in A.dtype.str:
        depth = int(A.dtype.str.split('S')[-1])
        # make all the same length
        sh = A.shape +(depth,)
        #B = np.empty(sh,dtype=np.uint8)
        return np.array([[ord(l) for l in s.ljust(depth,'\x00')] for s in A.flat],dtype=np.uint8).reshape(sh)
    elif 'u' in dt or 'i' in dt:
        return np.array([s.tostring() for s in np.split(A.astype(np.uint8).ravel(),np.prod(A.shape[:-1]))]).reshape(A.shape[:-1])
    else:
        raise TypeError('Data type `%s` not understood for string - ascii conversion' % dt)
    
def keV2m(keV):
    """\
    Convert photon energy in keV to wavelength (in vacuum) in meters.
    """
    wl = 1./(keV*1000)*4.1356e-7*2.9998    
    
    return wl


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

