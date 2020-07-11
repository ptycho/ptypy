
import numpy as np
import time
from weakref import WeakValueDictionary as WVD

from .. import utils as u
from ..io import h5write, h5read
from .classes import *
from .geometry import Geo
from .manager import ModelManager
from .classes import get_class
from ..utils.verbose import logger
from .ptycho import Ptycho

__all__ = ['link', 'unlink', 'to_h5', 'from_h5']

_dict_like = [dict, u.Param, WVD]
_list_like = [list, tuple]
_ptypy = [Geo, View, Container, Storage, Base, Ptycho, POD,
          u.Param, ModelManager, Geo]


def unlink(obj):
    """
    Looks into all references of obj. Places references in a pool with
    string labels resembling their python ID plus an object dependent prefix.
    Reduces all objects and dicts to dictionaries.
    Reduces all tuples to list.
    All references are replaced by the labels in the pool.

    Original references in 'obj' are preserved
    """
    pool = {}

    calls = []
    t = time.time()

    def _pool(obj):
        calls.append(None)
        # Recursively labels references and shifts objects to the pool
        try:
            prefix = obj._PREFIX
        except:
            prefix = 'I'

        # In case of an immutable type, just return that
        if str(obj) == obj or np.isscalar(obj) or type(obj) is np.ndarray:
            return obj
        # Assign a label
        ID = prefix + str(hex(id(obj)))
        if ID in pool.keys():
            # This object has been labeled already. No further action required
            return ID
        else:
            # If object contains references, make shallow copy and
            # recursively iterate over the copy

            if hasattr(obj, 'items'):
                # pool[ID] = {}
                nobj = {}
                pool[ID] = nobj
                for k, v in obj.items():
                    # pool[ID][k] = _pool(v)
                    nobj[k] = _pool(v)

            elif type(obj) in _list_like:
                # pool[ID] = list(obj)
                nobj = list(obj)  # pool[ID]
                pool[ID] = nobj
                for k, v in enumerate(nobj):
                    nobj[k] = _pool(v)

            elif type(obj) in _ptypy:
                try:
                    nobj = obj._to_dict().copy()
                except:
                    nobj = obj.__dict__.copy()
                pool[ID] = nobj
                # nobj = pool[ID]
                for k, v in nobj.items():
                    nobj[k] = _pool(v)

            else:
                pool[ID] = obj

            return ID

    ID = _pool(obj)
    pool['root'] = ID
    t = time.time() - t
    logger.info(
        "Converted cross-linked structure flat pool in %.3f sec.\n"
        "%d recursions were needed,\n"
        "%d objects were found. \n" % (t, len(calls), len(pool)-1))

    return pool


def link(pool, replace_objects_only=False, preserve_input_pool=True):
    """
    Reverse operation to unlink.
    """
    used = []
    t = time.time()
    # First a shallow copy
    pool = pool.copy()
    # First replace all occurrences of object dictionaries with their
    # respective objects. Since all objects appear only once, this is a safe.
    for k, v in pool.items():
        # At this point we can make copies of objects,
        # since they are uniquely referenced here
        if preserve_input_pool:
            if type(v) is dict or type(v) is np.ndarray:
                pool[k] = v.copy()
            elif type(v) is list:
                pool[k] = list(v)

        cls = get_class(k)
        if cls is not None:
            logger.debug('Found %s' % cls)
            # Check if the value is a dict.
            if type(v) is dict:
                # try:
                logger.debug('Try calling class "_from_dict()" method')
                pool[k] = cls._from_dict(pool[k])
                # except:
                #     logger.debug('Attempt failed. Invoking class instance '
                #                  'without arguments')
                #     inst = cls()
                #     inst.__dict__ = pool[k]

    if replace_objects_only:
        return pool

    calls = []
    keys = list(pool.keys())
    def _unpool(obj):
        calls.append(None)
        if str(obj) in keys:
            # Replace key by object. As these keys ALWAYS refer to objects
            # and not to other keys, no further checking is needed
            obj = pool[obj]
            if type(obj) in _dict_like and id(obj) not in used:
                used.append(id(obj))
                for k, v in obj.items():
                    obj[k] = _unpool(v)
            elif type(obj) in _list_like and id(obj) not in used:
                used.append(id(obj))
                for k, v in enumerate(obj):
                    obj[k] = _unpool(v)
            elif type(obj) in _ptypy and id(obj) not in used:
                used.append(id(obj))
                for k, v in obj.__dict__.items():
                    obj.__dict__[k]= _unpool(v)

            return obj
        else:
            return obj

    # Reestablish references, start from root
    out = _unpool(pool['root'])
    t = time.time() - t

    logger.info(
        'Converted flat pool to cross-linked structure in %.3f sec.\n '
        '%d recursions were needed,\n '
        '%d objects were used\n'
        '%d objects are mutable.' % (t, len(calls), len(pool)-1, len(used)))

    return out


def to_h5(filename, obj):
    h5write(filename, unlink(obj))


def from_h5(filename):
    return link(h5read(filename))
