# -*- coding: utf-8 -*-
"""
Wrapper to store nearly anything in a file using JSON encoding.

This file is part of the PTYPY package.

    :copyright: Copyright 2014 by the PTYPY team, see AUTHORS.
    :license: see LICENSE for details.
"""

import json
import numpy as np
import time
import os
import glob
import base64

__all__ = ['jwrite', 'jread']

VERSION = 0.1


class NumpyEncoder(json.JSONEncoder):
    """
    This class is adapted from http://stackoverflow.com/questions/3488934/simplejson-and-numpy-array.
    """

    def default(self, obj):
        """
        base64 encoding of array objects.
        """
        if hasattr(obj, '__array_interface__'):
            data_b64 = base64.b64encode(obj.data)
            return dict(__ndarray__=data_b64,
                        dtype=str(obj.dtype),
                        shape=obj.shape)

        # Pass unsupported objects to base class
        return json.JSONEncoder(self, obj)


def json_numpy_obj_hook(dct):
    """
    Decode an encoded numpy ndarray
    :param dct: (dict) json encoded ndarray
    :return: (ndarray) if input was an encoded ndarray
    """
    if isinstance(dct, dict) and '__ndarray__' in dct:
        return np.frombuffer(base64.b64decode(dct['__ndarray__']), dct['dtype']).reshape(dct['shape'])
    return dct

# Helper functions to load slices
class Str_to_Slice(object):
    def __getitem__(self, x):
        return x
    def __call__(self, s):
        return eval('self' + s)
str_to_slice = Str_to_Slice()

def jwrite(filename, *args, **kwargs):
    """
    jwrite(filename, {'var1'=..., 'var2'=..., ...})
    jwrite(filename, var1=..., var2=..., ...)
    jwrite(filename, dict, var1=..., var2=...)

    Writes variables var1, var2, ... to file filename using JSON encoding.
    The key-value arguments have precedence on the provided dictionary.

    All JSON supported types are supported, as well as numpy arrays, which
    are replaced by a base64 encoding.
    """

    filename = os.path.abspath(os.path.expanduser(filename))

    # Update input dictionnary
    if args:
        d = args[0].copy() # shallow copy
    else:
        d = {}
    d.update(kwargs)

    # Add meta information
    if '__meta__' in d:
        raise KeyError("reserved key '__meta__' already exist.")
    d['__meta__'] = {'json_rw_version': VERSION, 'ctime':time.asctime()}

    # Prepare file
    base= os.path.split(filename)[0]
    if not os.path.exists(base):
        os.makedirs(base)
    # Open the file and save everything
    with open(filename, 'w') as f:
        json.dump(d, f, cls=NumpyEncoder)

def jread(filename, *args, **kwargs):
    """\
    h5read(filename)
    h5read(filename, s1, s2, ...)
    h5read(filename, (s1,s2, ...))

    Read variables from a JSON file created with jwrite and returns them as
    a dictionary.

    The following features are meant to have an interface identical to h5read.
    Note however that no time of memory is saved by "loading" only a subset
    of variables or slices.

    If specified, only variable named s1, s2, ... are loaded.

    Variable names support slicing and group access. For instance, provided
    that the file contains the appropriate objects, the following syntax is
    valid:

    a = jread('file.h5', 'myarray[2:4]')
    a = jread('file.h5', 'adict.thekeyIwant')

    Another way of slicing is with the slice keyword argument which will take
    the provided slice object and apply it on the last variable name:

    a = jread('file.json', 'array1', 'array2', slice=slice(1,2))
    # Will read array2[1:2]

    jread(filename_with_wildcard, ... , doglob=True)
    Reads sequentially all globbed filenames.

    """

    doglob = kwargs.pop('doglob', None)

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
            d = jread(f, *args, doglob=False, **kwargs)
            dl.append(d)
        return dl

    # We are here only if there was no globbing (fnames is empty)
    filename = os.path.abspath(os.path.expanduser(filename))

    # Check if a slice was requested
    slice = kwargs.pop('slice', None)

    outdict = {}
    try:
        f = open(filename, 'r')
    except:
        print('Error when opening file %s.' % filename)
        raise
    else:
        with f:
            content = json.load(f, object_hook=json_numpy_obj_hook)

            # Get meta data if the file had been saved with jwrite.
            meta = content.pop('__meta__', {})
            json_rw_version = meta.get('json_rw_version',None)
            if json_rw_version is None:
                print('Warning: this file does not seem to follow json_rw format.')
            ctime = meta.get('ctime', None)
            if ctime is not None:
                print('File created : ' + ctime)

            if len(args) == 0:
                # The simplest case: no input argument.
                if slice is not None:
                    raise RuntimeError('A variable name must be given when slicing.')
                outdict = content
            else:
                # A list of keys was passed.
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
                        gr = content[glist[0]]
                        for gname in glist[1:-1]:
                            gr = gr[gname]
                        if sl is None:
                            outdict[k] = gr[k]
                        else:
                            outdict[k] = gr[k][sl]
                    else:
                        if sl is None:
                            outdict[k] = content[k]
                        else:
                            outdict[k] = content[k][sl]

    return outdict
