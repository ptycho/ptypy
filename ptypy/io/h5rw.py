# -*- coding: utf-8 -*-
"""
Wrapper to store nearly anything in an hdf5 file.

This file is part of the PTYPY package.

    :copyright: Copyright 2014 by the PTYPY team, see AUTHORS.
    :license: see LICENSE for details.
"""
import h5py
import numpy as np
import time
import os
import glob
from collections import OrderedDict
import pickle
from ..utils import Param
from ..utils.verbose import logger

__all__ = ['h5write', 'h5append', 'h5read', 'h5info', 'h5options']

h5options = dict(
    H5RW_VERSION='0.1',
    H5PY_VERSION=h5py.version.version,
    # UNSUPPORTED = 'ignore',
    UNSUPPORTED='fail',
    SLASH_ESCAPE='_SLASH_')
STR_CONVERT = [type]


def sdebug(f):
    """
    debugging decorator for _store functions
    """

    def newf(*args, **kwds):
        print('{0:20} {1:20}'.format(f.__name__, args[2]))
        return f(*args, **kwds)

    newf.__doc__ = f.__doc__
    return newf


# Helper functions to load slices
class Str_to_Slice(object):
    def __getitem__(self, x):
        return x

    def __call__(self, s):
        return eval('self' + s)


str_to_slice = Str_to_Slice()


def _h5write(filename, mode, *args, **kwargs):
    """\
    _h5write(filename, mode, {'var1'=..., 'var2'=..., ...})
    _h5write(filename, mode, var1=..., var2=..., ...)
    _h5write(filename, mode, dict, var1=..., var2=...)

    Writes variables var1, var2, ... to file filename. The file mode
    can be chosen according to the h5py documentation. The key-value
    arguments have precedence on the provided dictionary.

    supported variable types are:
    * scalars
    * numpy arrays
    * strings
    * lists
    * dictionaries

    (Setting the option UNSUPPORTED equal to 'ignore' eliminates
    unsupported types. Default is 'fail', which raises an error.)

    The file mode can be chosen according to the h5py documentation.
    It defaults to overwriting an existing file.
    """

    filename = os.path.abspath(os.path.expanduser(filename))

    ctime = time.asctime()
    mtime = ctime

    # Update input dictionary
    if args:
        d = args[0].copy()  # shallow copy
    else:
        d = {}
    d.update(kwargs)

    # List of object ids to make sure we are not saving something twice.
    ids = []

    # This is needed to store strings
    #dt = h5py.new_vlen(str) # deprecated
    dt = h5py.special_dtype(vlen = str)

    def check_id(id):
        if id in ids:
            raise RuntimeError('Circular reference detected! Aborting save.')
        else:
            ids.append(id)

    def pop_id(id):
        ids[:] = [x for x in ids if x != id]

    # @sdebug
    def _store_numpy(group, a, name, compress=True):
        if compress:
            dset = group.create_dataset(name, data=a, compression='gzip')
        else:
            dset = group.create_dataset(name, data=a)
        dset.attrs['type'] = 'array'
        return dset

    # @sdebug
    def _store_string(group, s, name):
        dset = group.create_dataset(name, data=np.asarray(s.encode('utf8')), dtype=dt)
        dset.attrs['type'] = 'string'
        return dset

    # @sdebug
    def _store_list(group, l, name):
        check_id(id(l))
        arrayOK = len(set([type(x) for x in l])) == 1
        if arrayOK:
            try:
                # Try conversion to a numpy array
                la = np.array(l)
                if la.dtype.type is np.string_:
                    arrayOK = False
                else:
                    dset = _store_numpy(group, la, name)
                    dset.attrs['type'] = 'arraylist'
            except:
                arrayOK = False
        if not arrayOK:
            # inhomogenous list. Store all elements individually
            dset = group.create_group(name)
            for i, v in enumerate(l):
                _store(dset, v, '%05d' % i)
            dset.attrs['type'] = 'list'
        pop_id(id(l))
        return dset

    # @sdebug
    def _store_tuple(group, t, name):
        dset = _store_list(group, list(t), name)
        dset_type = dset.attrs['type']
        dset.attrs['type'] = 'arraytuple' if dset_type == 'arraylist' else 'tuple'
        return dset

    # @sdebug
    def _store_dict(group, d, name):
        check_id(id(d))
        if any([type(k) not in [str,] for k in d.keys()]):
            raise RuntimeError('Only dictionaries with string keys are supported.')
        dset = group.create_group(name)
        dset.attrs['type'] = 'dict'
        for k, v in d.items():
            if k.find('/') > -1:
                k = k.replace('/', h5options['SLASH_ESCAPE'])
                ndset = _store(dset, v, k)
                if ndset is not None:
                    ndset.attrs['escaped'] = '1'
            else:
                _store(dset, v, k)
        pop_id(id(d))
        return dset

    # # @sdebug
    # def _store_ordered_dict(group, d, name):
    #     check_id(id(d))
    #     if any([type(k) not in [str,] for k in d.keys()]):
    #         raise RuntimeError('Only dictionaries with string keys are supported.')
    #     dset = group.create_group(name)
    #     dset.attrs['type'] = 'ordered_dict'
    #     for k, v in d.items():
    #         if k.find('/') > -1:
    #             k = k.replace('/', h5options['SLASH_ESCAPE'])
    #             ndset = _store(dset, v, k)
    #             if ndset is not None:
    #                 ndset.attrs['escaped'] = '1'
    #         else:
    #             _store(dset, v, k)
    #     pop_id(id(d))
    #     return dset

    # @sdebug
    def _store_param(group, d, name):
        # call _to_dict method
        dset = _store_dict(group, d._to_dict(), name)
        dset.attrs['type'] = 'param'
        return dset

    def _store_dict_new(group, d, name):
        check_id(id(d))
        dset = group.create_group(name)
        dset.attrs['type'] = 'dict'
        for i, kv in enumerate(d.items()):
            _store(dset, kv, '%05d' % i)
        pop_id(id(d))
        return dset

        # @sdebug

    def _store_pickle(group, a, name):
        apic = pickle.dumps(a)
        group[name] = np.string_(apic)
        group[name].attrs['type'] = 'pickle'
        return group[name]

    # @sdebug
    def _store_None(group, a, name):
        dset = group.create_dataset(name, data=np.zeros((1,)))
        dset.attrs['type'] = 'None'
        return dset

    # @sdebug
    def _store_numpy_record_array(group, a, name):
        dumped_array = a.dumps()
        group[name] =np.string_(dumped_array)
        group[name].attrs['type'] = 'record_array'
        return group[name]

    # @sdebug
    def _store(group, a, name):
        if type(a) is str:
            dset = _store_string(group, a, name)
        elif type(a) is dict:
            dset = _store_dict(group, a, name)
        elif type(a) is OrderedDict:
            dset = _store_dict(group, a, name)
        elif type(a) is Param:
            dset = _store_param(group, a, name)
        elif type(a) is list:
            dset = _store_list(group, a, name)
        elif type(a) is tuple:
            dset = _store_tuple(group, a, name)
        elif type(a) is np.ndarray:
            dset = _store_numpy(group, a, name)
        elif isinstance(a, (np.record, np.recarray)): # h5py can't handle this.
            dset = _store_numpy_record_array(group, a, name)
        elif np.isscalar(a):
            dset = _store_numpy(group, np.asarray(a), name, compress=False)
            dset.attrs['type'] = 'scalar'
        elif a is None:
            dset = _store_None(group, a, name)
        elif type(a) in STR_CONVERT:
            dset = _store_string(group, str(a), name)
        else:
            if h5options['UNSUPPORTED'] == 'fail':
                raise RuntimeError('Unsupported data type : %s' % type(a))
            elif h5options['UNSUPPORTED'] == 'pickle':
                dset = _store_pickle(group, a, name)
            else:
                dset = None
        return dset

    # generate all parent directories
    base = os.path.split(filename)[0]
    if not os.path.exists(base):
        os.makedirs(base)
    # Open the file and save everything
    with h5py.File(filename, mode) as f:
        f.attrs['h5rw_version'] = h5options['H5RW_VERSION']
        f.attrs['ctime'] = ctime
        f.attrs['mtime'] = mtime
        for k, v in d.items():
            # if the first group key exists, make an overwrite, i.e. delete group `k`
            # Otherwise it was not possible in this framework to write
            # into an existing file, where a key is already occupied,
            # i.e. a replace operation. On the other hand we are violating
            # the pure 'appending' nature of h5append
            if k in f.keys():
                del f[k]
            _store(f, v, k)
    return


def h5write(filename, *args, **kwargs):
    """\
    h5write(filename, {'var1'=..., 'var2'=..., ...})
    h5write(filename, var1=..., var2=..., ...)
    h5write(filename, dict, var1=..., var2=...)

    Writes variables var1, var2, ... to file filename. The key-value
    arguments have precedence on the provided dictionary.

    supported variable types are:
    * scalars
    * numpy arrays
    * strings
    * lists
    * dictionaries

    (Setting the option UNSUPPORTED equal to 'ignore' eliminates
    unsupported types. Default is 'fail', which raises an error.)

    The file mode can be chosen according to the h5py documentation.
    It defaults to overwriting an existing file.
    """

    _h5write(filename, 'w', *args, **kwargs)
    return


def h5append(filename, *args, **kwargs):
    """\
    h5append(filename, {'var1'=..., 'var2'=..., ...})
    h5append(filename, var1=..., var2=..., ...)
    h5append(filename, dict, var1=..., var2=...)

    Appends variables var1, var2, ... to file filename. The
    key-value arguments have precedence on the provided dictionary.

    supported variable types are:
    * scalars
    * numpy arrays
    * strings
    * lists
    * dictionaries

    (Setting the option UNSUPPORTED equal to 'ignore' eliminates
    unsupported types. Default is 'fail', which raises an error.)

    The file mode can be chosen according to the h5py documentation.
    It defaults to overwriting an existing file.
    """

    _h5write(filename, 'a', *args, **kwargs)
    return


def h5read(filename, *args, **kwargs):
    """\
    h5read(filename)
    h5read(filename, s1, s2, ...)
    h5read(filename, (s1,s2, ...))

    Read variables from a hdf5 file created with h5write and returns them as
    a dictionary.

    If specified, only variable named s1, s2, ... are loaded.

    Variable names support slicing and group access. For instance, provided
    that the file contains the appropriate objects, the following syntax is
    valid:

    a = h5read('file.h5', 'myarray[2:4]')
    a = h5read('file.h5', 'adict.thekeyIwant')

    Another way of slicing, is with the slice keyword argument, which will take
    the provided slice object and apply it on the last variable name:

    a = h5read('file.h5', 'array1', 'array2', slice=slice(1,2))
    # Will read array2[1:2]

    h5read(filename_with_wildcard, ... , doglob=True)
    Reads sequentially all globbed filenames.

    """
    doglob = kwargs.pop('doglob', None)
    depth = kwargs.pop('depth', None)
    depth = 99 if depth is None else depth + 1

    # Used if we read a list of files
    fnames = []
    if not isinstance(filename, str):
        # We have a list
        fnames = filename
    else:
        if doglob is None:
            # glob only if there is a wildcard in the filename
            doglob = glob.has_magic(filename)
        if doglob:
            fnames = sorted(glob.glob(filename))
            if not fnames:
                raise IOError('%s : no match.' % filename)

    if fnames:
        # We are here only if globbing was allowed.
        dl = []
        # Loop over file names
        for f in fnames:
            # Call again, but this time without globbing.
            d = h5read(f, *args, doglob=False, **kwargs)
            dl.append(d)
        return dl

    # We are here only if there was no globbing (fnames is empty)
    filename = os.path.abspath(os.path.expanduser(filename))

    # Define helper functions
    def _load_dict_new(dset):
        d = {}
        keys = list(dset.keys())
        keys.sort()
        for k in keys:
            dk, dv = _load(dset[k])
            d[dk] = dv
        return d

    def _load_dict(dset, depth):
        d = {}
        if depth > 0:
            for k, v in dset.items():
                if v.attrs.get('escaped', None) is not None:
                    k = k.replace(h5options['SLASH_ESCAPE'], '/')
                d[str(k)] = _load(v, depth - 1)
        return d

    def _load_list(dset, depth):
        l = []
        if depth > 0:
            keys = list(dset.keys())
            keys.sort()
            for k in keys:
                l.append(_load(dset[k], depth - 1))
        return l

    def _load_numpy(dset, sl=None):
        if sl is not None:
            return dset[sl]
        else:
            return dset[...]

    def _load_scalar(dset):
        try:
            return dset[...].item()
        except:
            return dset[...]

    # def _load_ordered_dict(dset, depth):
    #     d = OrderedDict()
    #     if depth > 0:
    #         for k, v in dset.items():
    #             if v.attrs.get('escaped', None) is not None:
    #                 k = k.replace(h5options['SLASH_ESCAPE'], '/')
    #             d[k] = _load(v, depth - 1)
    #     return d

    def _load_str(dset):
        if h5py.version.version_tuple[0]>2:
            return dset[()].decode('utf-8')
        else:
            return str(dset[()])
        
    def _load_unicode(dset):
        return dset[()].decode('utf-8')

    def _load_pickle(dset):
        #return cPickle.loads(dset.value)
        return pickle.loads(dset[()])

    def _load_numpy_record_array(dset):
        d = dset[()]
        if isinstance(d, str):
            d = d.encode()
        return pickle.loads(d)

    def _load(dset, depth, sl=None):
        dset_type = dset.attrs.get('type', None)
        if isinstance(dset_type, bytes):
            dset_type = dset_type.decode()

        # Treat groups as dicts
        if (dset_type is None) and (type(dset) is h5py.Group):
            dset_type = 'dict'

        if dset_type == 'dict' or dset_type == 'param':
            if sl is not None:
                raise RuntimeError('Dictionaries or ptypy.Param do not support slicing')
            val = _load_dict(dset, depth)
            if dset_type == 'param':
                val = Param(val)
        elif dset_type == 'list':
            val = _load_list(dset, depth)
            if sl is not None:
                val = val[sl]
        elif dset_type == 'ordered_dict':
            val = _load_dict(dset, depth)
        elif dset_type == 'array':
            val = _load_numpy(dset, sl)
        elif dset_type == 'arraylist':
            val = [x for x in _load_numpy(dset)]
            if sl is not None:
                val = val[sl]
        elif dset_type == 'tuple':
            val = tuple(_load_list(dset, depth))
            if sl is not None:
                val = val[sl]
        elif dset_type == 'arraytuple':
            val = tuple(_load_numpy(dset).tolist())
            if sl is not None:
                val = val[sl]
        elif dset_type == 'string':
            val = _load_str(dset)
            if sl is not None:
                val = val[sl]
        elif dset_type == 'record_array':
            val = _load_numpy_record_array(dset)
        elif dset_type == 'unicode':
            val = _load_unicode(dset)
            if sl is not None:
                val = val[sl]
        elif dset_type == 'scalar':
            val = _load_scalar(dset)
        elif dset_type == 'None':
            # 24.4.13 : B.E. commented due to hr5read not being able to return None type
            # try:
            #   val = _load_numpy(dset)
            # except:
            #    val = None
            val = None
        elif dset_type == 'pickle':
            val = _load_pickle(dset)
        elif dset_type is None:
            val = _load_numpy(dset, sl)
        else:
            raise RuntimeError('Unsupported data type : %s' % dset_type)
        return val

    # Read file content
    outdict = {}

    slice = kwargs.get('slice', None)

    try:
        f = h5py.File(filename, 'r')
    except:
        print('Error when opening file %s.' % filename)
        raise
    else:
        with f:
            # h5rw_version = f.attrs.get('h5rw_version',None)
            # if h5rw_version is None:
            #     print('Warning: this file does not seem to follow h5read format.')
            ctime = f.attrs.get('ctime', None)
            if ctime is not None:
                logger.debug('File created : ' + str(ctime))
            if len(args) == 0:
                # no input arguments - load everything
                if slice is not None:
                    raise RuntimeError('A variable name must be given when slicing.')
                key_list = list(f.keys())
            else:
                if (len(args) == 1) and (type(args[0]) is list):
                    # input argument is a list of object names
                    key_list = args[0]
                else:
                    # arguments form a list
                    key_list = list(args)

            last_k = key_list[-1]
            for k in key_list:
                if k == last_k and slice is not None:
                    sl = slice
                else:
                    # detect slicing
                    if '[' in k:
                        k, slice_string = k.split('[')
                        slice_string = slice_string.split(']')[0]
                        sl = str_to_slice('[' + slice_string + ']')
                    else:
                        sl = None

                # detect group access
                if '.' in k:
                    glist = k.split('.')
                    k = glist[-1]
                    gr = f[glist[0]]
                    for gname in glist[1:-1]:
                        gr = gr[gname]
                    outdict[k] = _load(gr[k], depth, sl=sl)
                else:
                    outdict[k] = _load(f[k], depth, sl=sl)

    return outdict


def h5info(filename, path='', output=None, depth=8):
    """\
    h5info(filename)

    Prints out a tree structure of given h5 file.
    """
    depth = 8 if depth is None else depth
    indent = 4
    filename = os.path.abspath(os.path.expanduser(filename))

    def _format_dict(d, key, dset, isParam=False):
        ss = 'Param' if isParam else 'dict'
        stringout = ' ' * key[0] + ' * %s [%s %d]:\n' % (key[1], ss, len(dset))
        if d > 0:
            for k, v in dset.items():
                if v is not None and v.attrs.get('escaped', None) is not None:
                    k = k.replace(h5options['SLASH_ESCAPE'], '/')
                stringout += _format(d - 1, (key[0] + indent, k), v)
        return stringout

    def _format_list(d, key, dset):
        stringout = ' ' * key[0] + ' * %s [list %d]:\n' % (key[1], len(dset))
        if d > 0:
            keys = list(dset.keys())
            keys.sort()
            for k in keys:
                stringout += _format(d - 1, (key[0] + indent, ''), dset[k])
        return stringout

    def _format_tuple(d, key, dset):
        stringout = ' ' * key[0] + ' * %s [tuple]:\n' % key[1]
        if d > 0:
            keys = list(dset.keys())
            keys.sort()
            for k in keys:
                stringout += _format(d - 1, (key[0] + indent, ''), dset[k])
        return stringout

    def _format_arraytuple(key, dset):
        a = dset[...]
        if len(a) < 5 and a.ndim==1:
            stringout = ' ' * key[0] + ' * ' + key[1] + ' [tuple = ' + str(tuple(a.ravel())) + ']\n'
        else:
            stringout = ' ' * key[0] + ' * ' + key[1] + \
                        ' [tuple = ' + str(len(a)) + 'x[' + (('%dx' * (a[0].ndim - 1) + '%d') % a[0].shape) + \
                        ' ' + str(a.dtype) + ' array]]\n'
        return stringout

    def _format_arraylist(key, dset):
        a = dset[...]
        if len(a) < 5:
            stringout = ' ' * key[0] + ' * ' + key[1] + ' [list = ' + str(a.tolist()) + ']\n'
        else:
            try:
                float(a.ravel()[0])
                stringout = ' ' * key[0] + ' * ' + key[1] + ' [list = [' + (
                            ('%f, ' * 4) % tuple(a.ravel()[:4])) + ' ...]]\n'
            except ValueError:
                stringout = ' ' * key[0] + ' * ' + key[1] + ' [list = [%d x %s objects]]\n' % (a.size, str(a.dtype))
        return stringout

    def _format_numpy(key, dset):
        a = dset[...]
        if len(a) < 5 and a.ndim == 1:
            stringout = ' ' * key[0] + ' * ' + key[1] + ' [array = ' + str(a.ravel()) + ']\n'
        else:
            stringout = ' ' * key[0] + ' * ' + key[1] + ' [' + (('%dx' * (a.ndim - 1) + '%d') % a.shape) + ' ' + str(
                a.dtype) + ' array]\n'
        return stringout

    def _format_scalar(key, dset):
        stringout = ' ' * key[0] + ' * ' + key[1] + ' [scalar = ' + str(dset[...]) + ']\n'
        return stringout

    def _format_str(key, dset):
        s = str(dset[...])
        if len(s) > 40:
            s = s[:40] + '...'
        stringout = ' ' * key[0] + ' * ' + key[1] + ' [string = "' + s + '"]\n'
        return stringout

    def _format_unicode(key, dset):
        s = str(dset[...]).decode('utf8')
        if len(s) > 40:
            s = s[:40] + '...'
        stringout = ' ' * key[0] + ' * ' + key[1] + ' [unicode = "' + s + '"]\n'
        return stringout

    def _format_None(key, dset):
        stringout = ' ' * key[0] + ' * ' + key[1] + ' [None]\n'
        return stringout

    def _format_unknown(key, dset):
        stringout = ' ' * key[0] + ' * ' + key[1] + ' [unknown]\n'
        return stringout

    def _format(d, key, dset):
        dset_type = 'None' if dset is None else dset.attrs.get('type', None)

        # Treat groups as dicts
        if (dset_type is None) and (type(dset) is h5py.Group):
            dset_type = 'dict'

        if dset_type == 'dict' or dset_type == 'ordered_dict':
            stringout = _format_dict(d, key, dset, False)
        elif dset_type == 'param':
            stringout = _format_dict(d, key, dset, True)
        elif dset_type == 'list':
            stringout = _format_list(d, key, dset)
        elif dset_type == 'array':
            stringout = _format_numpy(key, dset)
        elif dset_type == 'arraylist':
            stringout = _format_arraylist(key, dset)
        elif dset_type == 'tuple':
            stringout = _format_tuple(d, key, dset)
        elif dset_type == 'arraytuple':
            stringout = _format_arraytuple(key, dset)
        elif dset_type == 'string':
            stringout = _format_str(key, dset)
        elif dset_type == 'unicode':
            stringout = _format_unicode(key, dset)
        elif dset_type == 'scalar':
            stringout = _format_scalar(key, dset)
        elif dset_type == 'None':
            stringout = _format_None(key, dset)
        elif dset_type is None:
            stringout = _format_numpy(key, dset)
        else:
            stringout = _format_unknown(key, dset)
        return stringout

    with h5py.File(filename, 'r') as f:
        # h5rw_version = f.attrs.get('h5rw_version',None)
        # if h5rw_version is None:
        #     print('Warning: this file does not seem to follow h5read format.')
        ctime = f.attrs.get('ctime', None)
        if ctime is not None:
            print('File created : ' + ctime)
        if not path.endswith('/'): path += '/'
        key_list = list(f[path].keys())
        outstring = ''
        for k in key_list:
            outstring += _format(depth, (0, k), f[path + k])

    print(outstring)

    # return string if output variable passed as option
    if output != None:
        return outstring
