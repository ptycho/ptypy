# -*- coding: utf-8 -*-
"""
Miscellaneous utility functions with little dependencies from other modules

This file is part of the PTYPY package.

    :copyright: Copyright 2014 by the PTYPY team, see AUTHORS.
    :license: GPLv2, see LICENSE for details.
"""
import os
import numpy as np
from functools import wraps

__all__ = ['str2int','str2range',\
           'complex_overload','expect2','expect3','expectN',\
           'keV2m','keV2nm','nm2keV', 'clean_path','unique_path']

def str2index(s):
    """
    Converts a str that is supposed to represent a numpy index expression
    into an index expression
    """
    return eval("np.index_exp["+s+"]")

def str2range(s):
    """
    Generates an index list from string input `s`

    Examples
    --------
    >>> # Same as range(1,4,2)
    >>> str2range('1:4:2')
    >>> # Same as range(1,2)
    >>> str2range('1')
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
    Transforms numpy array `A` of strings to ``np.uint8`` and back

    Examples
    --------
    >>> from ptypy.utils import str2int
    >>> A=np.array('hallo')
    >>> A
    array('hallo', dtype='|S5')
    >>> str2int(A)
    array([104,  97, 108, 108, 111], dtype=uint8)
    >>> str2int(str2int(A))
    array('hallo', dtype='|S5')
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

def keV2nm(keV):
    """\
    Convert photon energy in keV to wavelength (in vacuum) in nanometers.
    """
    nm = 1./(keV*10)*4.1356*2.9998

    return nm

def nm2keV(nm):
    """\
    Convert wavelength in nanometers to photon energy in keV.
    """
    keV = keV2nm(1.)/nm

    return keV

def expect2(a):
    """\
    Generates 1d numpy array with 2 entries generated from multiple inputs
    (tuples, arrays, scalars). Main puprose of this function is to circumvent
    debugging of input as very often a 2-vector is requred in scripts

    Examples
    --------
    >>> from ptypy.utils import expect2
    >>> expect2( 3.0 )
    array([ 3.,  3.])
    >>> expect2( (3.0,4.0) )
    array([ 3.,  4.])
    >>> expect2( (1.0, 3.0,4.0) )
    array([ 1.,  3.])
    """
    a=np.atleast_1d(np.asarray(a))
    if len(a)==1:
        b=np.array([a.flat[0],a.flat[0]])
    else: #len(psize)!=2:
        b=np.array([a.flat[0],a.flat[1]])
    return b

def expect3(a):
    """\
    Generates 1d numpy array with 3 entries generated from multiple inputs
    (tuples, arrays, scalars). Main puprose of this function is to circumvent
    debugging of input.

    Examples
    --------
    >>> from ptypy.utils import expect3
    >>> expect3( 3.0 )
    array([ 3.,  3.,  3.])
    >>> expect3( (3.0,4.0) )
    array([ 3.,  4.,  4.])
    >>> expect3( (1.0, 3.0,4.0) )
    array([ 1.,  3.,  4.])

    """
    a=np.atleast_1d(a)
    if len(a)==1:
        b=np.array([a.flat[0]]*3)
    elif len(a)==2:
        b=np.array([a.flat[0],a.flat[1],a.flat[1]])
    else:
        b=np.array([a.flat[0],a.flat[1],a.flat[2]])
    return b

def expectN(a, N):
    if N==2:
        return expect2(a)
    elif N==3:
        return expect3(a)
    else:
        raise ValueError('N must be 2 or 3')
 
def complex_overload(func):
    """\
    Overloads function specified only for floats in the following manner

    .. math::

        \mathrm{complex\_overload}\{f\}(c) = f(\Re(c)) + \mathrm{i} f(\Im(c))
    """
    @wraps(func)
    def overloaded(c,*args,**kwargs):
        return func(np.real(c),*args,**kwargs) +1j *func(np.imag(c),*args,**kwargs)

    return overloaded


def unique_path(filename):
    """\
    Makes path absolute and unique

    Parameters
    ----------
    filename : str
                A filename.
    """
    return os.path.abspath(os.path.expanduser(filename))

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

