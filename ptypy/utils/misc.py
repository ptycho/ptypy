# -*- coding: utf-8 -*-
"""
Miscellaneous utility functions with little dependencies from other modules

This file is part of the PTYPY package.

    :copyright: Copyright 2014 by the PTYPY team, see AUTHORS.
    :license: see LICENSE for details.
"""
import os
import numpy as np
from functools import wraps
from collections import OrderedDict
from collections import namedtuple

__all__ = ['str2int', 'str2range', 'complex_overload', 'expect2',
           'expect3', 'keV2m', 'keV2nm', 'nm2keV', 'm2keV', 'clean_path',
           'unique_path', 'Table', 'all_subclasses', 'expectN', 'isstr',
           'electron_wavelength']


def all_subclasses(cls, names=False):
    """
    Helper function for finding all subclasses of a base class.
    If names is True, returns the names of the classes rather than
    their object handles.
    """
    subs = cls.__subclasses__() + [g for s in cls.__subclasses__()
                                    for g in all_subclasses(s)]
    if names:
        return [c.__name__ for c in subs]
    else:
        return subs

class Table(object):
    """
    Basic table implemented using numpy.recarray
    Ideally subclassed to be used with faster or more
    flexible databases.
    """
    def __init__(self,dct,name='pods'):
        self._table_name = name
        self._record_factory_from_dict(dct)
        
    def _record_factory_from_dict(self,dct,suffix='_record'):
        self._record_factory = namedtuple(self._table_name+suffix,list(dct.keys()))
        self._record_default = self._record_factory._make(list(dct.values()))
        self._record_dtype = [np.array(v).dtype for v in self._record_default]
    
    def new_table(self, records = 0):
        r = self._record_default
        dtype = list(zip(r._fields,self._record_dtype))
        self._table = np.array([tuple(self._record_default)] * records,dtype)
        
    def new_fields(self,**kwargs):
        """ 
        Add fields (columns) to the table. This is probably slow. 
        """
        # save old stuff
        addition = OrderedDict(**kwargs)
        t = self._table
        records = self.pull_records()       
        new =  self._record_factory._asdict(self._record_default)
        new.update(addition)
        self._record_factory_from_dict(new)
        self.new_table()
        a  = tuple(addition.values())
        self.add_records( [r + a for r in records] )
        
        
    def pull_records(self,record_ids=None):
        if record_ids is None:
            return list(map(self._record_factory._make, self._table))
        else:
            return list(map(self._record_factory._make, self._table[record_ids]))
            
    def add_records(self,records):
        """ Add records at the end of the table. """
        start = len(self._table)
        stop = len(records)+start
        record_ids = list(range(start,stop))
        self._table.resize((len(self._table)+len(records),))
        self._table[start:stop]=records
        
        return record_ids
        
    def insert_records(self,records, record_ids):
        """ Insert records and overwrite existing content. """
        self._table[record_ids]=records
        
    def select_func(self,func,fields=None):
        """
        Find all records where search function `func` evaluates True. 
        Arguments to the function are selected by `fields`. 
        The search function will always receive the record_id as first argument. 
        """
        a = list(range(len(self._table)))
        if fields is None:
            res = [n for n in a if func(a)]
        else:
            t = self._table[fields].T # pretty inefficient. Is like a dual transpose
            res =[n for n,rec in zip(a,t) if func(n, *rec)]
            
        return np.array(res)
        
    def select_range(self,field,low,high):
        """
        Find all records whose values are in the range [`low`,`high`] for
        the field entry `field`. Should be a numerical value. 
        """
        t = self._table
        record_ids =  np.argwhere(t[field] >=low and t[field] <=high).squeeze(-1)
        return record_ids
                    
    def select_match(self,field,match):
        """
        Find all records whose values are in the range [`low`,`high`] for
        the field entry `field` 
        """
        t = self._table
        record_ids =  np.argwhere(t[field] == match).squeeze(-1)
        return record_ids
        
    def modify_add(self,record_ids,**kwargs):
        """
        Take selected record ids and overwrite fields with values
        `**kwargs`.
        """
        old_records = self.pull_records(record_ids)
        recs = [r._replace(**kwargs) for r in old_records]
        self.add_records(recs)
        

def isstr(s):
    """
    This function should be used for checking if an object is of string type
    """
    import sys

    if sys.version_info[0] == 3:
        string_types = str,
    else:
        string_types = basestring, # noqa: F821
    
    return isinstance(s, string_types)
    
    
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

    return list(range(start,stop,step))

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

def m2keV(m):
    """\
    Convert wavelength in meters to photon energy in keV.
    """
    keV = keV2m(1.)/m

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
    r"""\
    Overloads function specified only for floats in the following manner

    .. math::

        \mathrm{complex\_overload}\{f\}(c) = f(\Re(c)) + \mathrm{i} f(\Im(c))
    """
    @wraps(func)
    def overloaded(c,*args,**kwargs):
        return func(np.real(c),*args,**kwargs).astype(c.dtype) +1j *func(np.imag(c),*args,**kwargs)

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


def electron_wavelength(electron_energy):
    """
    Calculate electron wavelength based on energy in keV:

    .. math::
        \lambda = hc /  \sqrt{E * (2moc^2 + E)}

    Parameters
    ----------
    electron_energy : float
        in units of keV

    Returns
    -------
    wavelength : float
        in units of meter
    """
    hc = 12.398 # keV-Angstroms
    moc2 = 511 # keV
    wavelength = hc / np.sqrt(electron_energy * (2 * moc2 + electron_energy)) * 1e-10
    return wavelength
