# -*- coding: utf-8 -*-
"""\
Utility functions and classes to support MPI computing.

This file is part of the PTYPY package.

    :copyright: Copyright 2014 by the PTYPY team, see AUTHORS.
    :license: see LICENSE for details.
"""
import numpy as np

from .. import __has_mpi4py__ as hmpi

size = 1
rank = 0
MPI = None
comm = None
if hmpi:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
del hmpi

MPIenabled = not (size == 1)
master = (rank == 0)

__all__ = ['MPIenabled', 'comm', 'MPI', 'master','barrier',
           'LoadManager', 'loadmanager','allreduce','send','receive','bcast',
           'bcast_dict', 'gather_dict', 'gather_list', 
           'MPIrand_normal', 'MPIrand_uniform','MPInoise2d']


def useMPI(do=None):
    """\
    Toggle using MPI or not. Is this useful? YES!
    """
    global MPIenabled
    if do is None:
        return MPIenabled
    if MPI is None:
        MPIenabled = False
    else:
        MPIenabled = do
        if do is False:
            global size
            global master
            global loadmanager
            size = 1
            master = True
            loadmanager = LoadManager()




###################################
# Helper functions, wrapper to mpi4py
###################################

class LoadManager(object):
    def __init__(self):
        """
        Keeps track of the amount of data managed by each
        process and helps keeping it balanced.

        Note
        ----
        A LoadManager should always be given the full range of ids, as
        the `all-knowing` power stems from the fact that it always knows
        how many processes there are, what rank it has and the full range
        of ids to distribute. Hence, please *"assign"* always the same list
        across all nodes.
        """

        self.load = np.zeros((size,), dtype=int)
        """ Represents the number of elements assigned to this process. """

        self.rank_of = {}
        """ Rank of specific *id*, i.e. element of *idlist*. """

    def assign(self, idlist=None):
        """
        Subdivide the provided list of ids into contiguous blocks to balance
        the load among the processes.

        The elements of idlist are used as keys to subsequently identify
        the rank of a given id, through the dict self.rank_of. No check
        is done whether ids passed are unique or even hashable (they
        better are)

        If idlist is None, increase by one the load of the least busy
        process. :any:`rank_of` is not updated in this case.

        Parameters
        ----------
        idlist : list
            List of objects that can also be keys in a `dict`, i.e. hashable
            objects.

        Returns
        -------
        R : list
            A nested list, (a list of lists) such that ``R[rank]=list``
            of elements of `idlist` managed by process of given `rank`.
        """
        # Simplest case
        if idlist is None:
            r = size - 1 - self.load[::-1].argmin()
            self.load[r] += 1
            return [[0] if rr == r else [] for rr in range(size)]

        # Total load
        Nid = len(idlist)
        total_load = (self.load.sum() + Nid)

        # Eliminate nodes that are too busy already
        li = total_load > self.load * size

        # Recompute total load among available nodes
        nsize = li.sum()
        total_load = (self.load[li].sum() + Nid)

        # Numerator part of the number of elements to assign to each node
        partition = total_load - self.load[li] * nsize

        # Integer part
        ipart = partition // nsize

        # Spread the fractional remainder among the nodes, starting from the
        # last one.
        rem = (partition % nsize).sum() // nsize
        ipart[:-int(1 + rem):-1] += 1

        # Update the loads
        part = np.zeros_like(self.load)
        part[li] = ipart

        self.load += part

        # Cumulative sum give the index boundaries between the ranks
        cumpart = np.cumsum(part)

        # Now assign the rank
        rlist = np.arange(size)

        out = [[] for x in range(size)]
        for i, k in enumerate(idlist):
            r = rlist[i < cumpart][0]
            out[r].append(i)
            self.rank_of[k] = r
        return out

    def reset(self):
        """
        Resets :any:`LoadManager` to initial state.
        """
        self.__init__()

# Create one instance - typically only this one should be used
loadmanager = LoadManager()


def allreduce(a, op=None):
    """
    Wrapper for comm.Allreduce, always in place.

    Parameters
    ----------
    a : numpy-ndarray
        The array to operate on.

    op : operation
        MPI operation to execute. None or one of MPI.BAND, MPI.BOR,
        MPI.BXOR, MPI.LAND, MPI.LOR, MPI.LXOR, MPI.MAX, MPI.MAXLOC,
        MPI.MIN, MPI.MINLOC, MPI.OP_NULL, MPI.PROD, MPI.REPLACE or MPI.SUM.
        If None, uses MPI.SUM.

    Note
    ----
    *Explanation* : If process #1 has ndarray ``a`` and process #2 has
    ndarray ``b``. After calling allreduce, the new arrays after allreduce
    are ``a'=op(a,b)`` and ``b'=op(a,b)`` on process #1 and #2 respectively
    """

    if not MPIenabled:
        return a
    isscalar = np.isscalar(a)
    if isscalar:
        a = np.array(a)
    if op is None:
        # print a.shape
        comm.Allreduce(MPI.IN_PLACE, a)
    else:
        comm.Allreduce(MPI.IN_PLACE, a, op=op)
    if isscalar:
        return a.item()
    else:
        return a

def allreduceC(c):
    """
    Performs MPI parallel ``allreduce`` with a sum as reduction
    for all :any:`Storage` instances held by :any:`Container` *c*

    :param Container c: Input

    See also
    --------
    ptypy.utils.parallel.allreduce
    """
    for s in c.S.values():
        allreduce(s.data)

def _MPIop(a, op, axis=None):
    """
    Apply operation op on accross a list of arrays distributed between
    processes. Supported operations are SUM, MAX, MIN, and PROD.
    """
    
    if MPIenabled:
        MPIop = {'SUM': MPI.SUM, 'MAX': MPI.MAX, 'MIN': MPI.MIN, 'PROD': MPI.PROD}[op.upper()]

    npop = {'SUM': np.sum, 'MAX':  np.max, 'MIN': np.min, 'PROD': np.prod}[op.upper()]
    
    # Total op
    if axis is None:
        # Very special case: calling with an empty object might make sense in a few situations.
        if len(a) == 0:
            if op.upper() == 'MAX':
                s = np.array([-np.inf])
            if op.upper() == 'MIN':
                s = np.array([np.inf])
            elif op.upper() == 'SUM':
                s = np.array([0.])
            else:
                s = np.array([1.])
        else:
            # Apply op on locally owned data (and wrap the scalar result in a numpy array
            s = np.array([npop([npop(ai) for ai in a if ai is not None])])

        # Reduce and return scalar
        if MPIenabled:
            # Use lower-case reduce to allow for empty list.
            #comm.Allreduce(MPI.IN_PLACE, s, op=MPIop)
            s = comm.allreduce(s, op=MPIop)
        return s[0]
    elif len(a) == 0:
        raise RuntimeError('MPIop cannot be called with empty arrays.')
    # Axis across the processes
    elif axis == 0:
        # Apply op on locally owned arrays
        s = npop(ai for ai in a if ai is not None)

        # Reduce and return result
        if MPIenabled:
            comm.Allreduce(MPI.IN_PLACE, s, op=MPIop)
        return s

    else:
        # No cross-talk needed
        return [npop(ai, axis=axis - 1) if ai is not None else None for ai in a]


def MPIsum(a, axis=None):
    """
    Compute the sum of list of arrays distributed over multiple processes.
    """
    return _MPIop(a, op='SUM', axis=axis)


def MPImin(a, axis=None):
    """
    Compute the minimum over a list of arrays distributed over multiple processes.
    """
    return _MPIop(a, op='MIN', axis=axis)


def MPImax(a, axis=None):
    """
    Compute the maximum over a list of arrays distributed over multiple processes.
    """
    return _MPIop(a, op='MAX', axis=axis)


def MPIprod(a, axis=None):
    """
    Compute the product over a list of arrays distributed over multiple processes.
    """
    return _MPIop(a, op='PROD', axis=axis)


def barrier():
    """
    Wrapper for comm.Barrier.
    """
    if not MPIenabled:
        return
    comm.Barrier()

def send(data, dest=0, tag=0):
    """
    Wrapper for `comm.Send` and `comm.send`.
    If data is a `numpy.ndarray`, a header will be sent first with
    `comm.send` that contains information on array shape and data type.
    Afterwards the array will be sent with `comm.Send`.
    If data is not a `numpy.ndarray`, the whole object will be pickled
    and sent with `comm.send` in one go.

    Parameters
    ----------
    data : ndarray or other
        Object to send
    dest : int
        The rank of the destination node / process. Defaults to 0 (master).
    tag : int
        Defaults to 0.

    See also
    --------
    receive
    bcast
    """
    if type(data) is np.ndarray:
        # Sends info that array is coming and array dimensions
        comm.send(('npy', data.shape, data.dtype.str), dest=dest, tag=1)

        # Send array with faster numpy interface
        # mpi4py has in issue sending booleans. we convert to uint8 (same size)
        if data.dtype.str == '|b1':
            comm.Send(data.astype('|u1'), dest=dest, tag=tag)
        else:
            try:
                comm.Send(data, dest=dest, tag=tag)
            except KeyError:
                # mpi4py may complain for non-native byte order
                data = data.astype(data.dtype.newbyteorder('='))
                comm.Send(data, dest=dest, tag=tag)
    else:
        # Send pickled whatever thing
        comm.send(data, dest=dest, tag=1)




def receive(source=None, tag=0):
    """
    Wrapper for `comm.Recv`. Probes first with `comm.recv`. If the
    unpickled is a 3-tuple with first entry=='npy', prepares buffer and
    waits with `comm.Recv`

    Parameters
    ----------
    source : int or None
        The rank of the node / process sending data. If None, this is set
        to MPI.ANY_SOURCE

    tag : int
        Not really useful here - defaults to `0` all the time

    Returns
    -------
    out : ndarray or other
    """

    if source is None:
        source = MPI.ANY_SOURCE

    # Receive thing
    thing = comm.recv(source=source, tag=1)

    try:
        check,shape,dtypestr = thing
        if check=='npy':

            # prepare uint8 in case of booleans as buffer
            newdtype = '|u1' if dtypestr == '|b1' else dtypestr

            # Create array
            out = np.empty(shape, dtype=newdtype)

            # Receive raw data
            comm.Recv(out, source=source, tag=tag)

            return out.astype(dtypestr)
        else:
            return thing
    except:
        return thing
    else:
        return thing

def bcast(data, source=0):
    """
    Wrapper for `comm.bcast` and `comm.Bcast`.
    If data is a `numpy.ndarray`, a header will be broadcasted first with
    `comm.bcast` that contains information on array shape and data type.
    Afterwards the array will be sent with `comm.Bcast`.
    If data is not a `numpy.ndarray`, the whole object will be pickled
    and broadcasted with `comm.bcast` in one go.

    Parameters
    ----------
    data : ndarray or other
        Object to send
    source : int
        The rank of the source node / process. Defaults to 0 (master).
    tag : int
        Defaults to 0.

    See also
    --------
    receive
    send
    """
    if not MPIenabled:
        return data

    # Communicate if array or pickle.
    if rank == source:
        if type(data) is np.ndarray:
            msg = comm.bcast('array', source)
        else:
            msg = comm.bcast('pickle', source)
    else:
        msg = comm.bcast(None, source)

    if str(msg) == 'array':
        # Communicate size before sending array
        if rank == source:
            shape, dtypestr = comm.bcast((data.shape, data.dtype.str), source)
        else:
            shape, dtypestr = comm.bcast(None, source)

        newdtype = '|u1' if dtypestr == '|b1' else dtypestr

        if rank == source:
            buf = data.astype(newdtype)
        else:
            buf = np.empty(shape, dtype=newdtype)

        # Send
        comm.Bcast(buf, source)

        if dtypestr == '|b1':
            buf = buf.astype('bool')

        return buf
    else:
        # Send pickled thing directly.
        if rank == source:
            thing = comm.bcast(data, source)
        else:
            thing = comm.bcast(None, source)

        return thing

def bcast_dict(dct, keys='all', source=0):
    """
    Broadcasts or scatters a dict `dct` from ``rank==source``.

    Parameters
    ----------
    keys : list or 'all'
        List of keys whose values are accepted at each node. In case
        of ``keys=all``, every node accepts all items and :any:`bcast_dict`
        acts as broadcast.

    source : int
        Rank of node / process which broadcasts / scatters.

    Returns
    -------
    dct : dict
        A smaller dictionary with values to `keys` if that key
        was in source dictionary.

    Note
    ----
    There is no guarantee that each *key,value* pair is accepted at other
    nodes, except for ``keys='all'``. Also in this implementation
    the input `dct` from source is *completely* transmitted to every node,
    potentially creating a large overhead for may nodes and huge dictionarys

    Deleting reference in input dictionary may result in data loss at
    ``rank==source``

    See also
    --------
    gather_dict

    """
    if not MPIenabled:
        out = dict(dct)
        return out

    # Broadcast all keys (the full dict)
    if str(keys) == 'all':
        out = comm.bcast(dct)
        return out

    # Broadcast only given keys of dict
    if rank == source:
        out = {}
        length = comm.bcast(len(dct), source)
        for k, v in dct.items():
            comm.bcast(k,source)
            bcast(v,source)
            if k in keys:
                out[k] = v
        return out
    else:
        out = {}
        length = comm.bcast(None, source)
        for k in range(length):
            k = comm.bcast(None,source)
            v = bcast(None,source)
            if k in keys:
                out[k] = v
        return out

def allgather_dict(dct):
    """
    Allgather dict in place.
    """
    gdict = gather_dict(dct)
    gdict = bcast_dict(gdict)
    dct.update(gdict)

def gather_dict(dct, target=0):
    """
    Gathers broadcasted or scattered dict `dct` at rank `target`.

    Parameters
    ----------
    dct : dict
        Input dictionary. Remains unaltered
    target : int
        Rank of process where the `dct`'s are gathered

    Returns
    -------
    out : dict
        Gathered dict at ``rank==target``, Empty dict at ``rank!=target``

    Note
    ----
    If the same `key` exists on different nodes, the corresponding values
    will be consecutively overidden in the order of the ranks at the
    gathering node without complain or notification.

    See also
    --------
    bcast_dict
    """
    out = {}
    if not MPIenabled:
        out.update(dct)
        return out
    ret = comm.gather(dct, root=target)
    if rank == target:
        for d in ret:
            out.update(d)
    return out

    # for r in range(size):
    #     if r == target:
    #         if rank == target:
    #             #print rank,dct
    #             out.update(dct)
    #         continue

    #     if rank == target:
    #         l = comm.recv(source=r,tag=9999)
    #         for i in range(l):
    #             #k = receive(r)
    #             k = comm.recv(source=r,tag=9999)
    #             v = receive(r)
    #             #print rank,str(k),v
    #             out[k] = v
    #     elif r == rank:
    #         # your turn to send
    #         l = len(dct)
    #         comm.send(l, dest=target,tag=9999)
    #         for k,v in dct.items():
    #             #print rank,str(k),v
    #             #send(k, dest=target)
    #             comm.send(k, dest=target,tag=9999)
    #             send(v, dest=target)
    #     barrier()
    # return out

def _send(data, dest=0, tag=0):
    """
    Wrapper for comm.Send

    Parameters
    ----------
    data : numpy array or 2 tuple (key,numpy array)
           The array to send with optional key
    dest : int
           The rank of the destination process. Defaults to 0 (master).
    tag : int
          Defaults to 0.
    """

    # Send array info
    key, npdata = _check(data)
    header = (npdata.shape, npdata.dtype.str, key)
    comm.send(header, dest=dest, tag=1)

    # Send data
    # mpi4py has in issue sending booleans. we convert to uint8 (same size)
    if npdata.dtype.str == '|b1':
        comm.Send(npdata.astype('uint8'), dest=dest, tag=tag)
    else:
        comm.Send(npdata, dest=dest, tag=tag)


def _receive(source=None, tag=0, out=None):
    """
    Wrapper for comm.Recv

    Parameters
    ----------
    source : int or None
             The rank of the process sending data. If None, this is set
             to MPI.ANY_SOURCE
    tag : int
          Not really useful here - default to 0 all the time
    out : numpy array or None
          If a numpy array, the transfered data will be stored in out. If
          None, a new array is created.

    Returns
    -------
    out : numpy array or 2tuple (key, numpy array)
    """

    if source is None:
        source = MPI.ANY_SOURCE

    # Receive array info
    shape, dtypestr, key = comm.recv(source=source, tag=1)

    newdtype = '|u1' if dtypestr == '|b1' else dtypestr
    # Create array if none is provided
    if out is None:
        out = np.empty(shape, dtype=newdtype)

    # Receive raw data
    comm.Recv(out, source=source, tag=tag)

    if dtypestr == '|b1':
        out = out.astype('bool')

    if str(key) != "":
        return (key, out)
    else:
        return out


def _bcast(data, source=0, key=""):
    """
    Wrapper for comm.bcast
    """
    # FIXME: what is this function supposed to do? Is the following non-parallel case ok?
    if not MPIenabled:
        key, npdata = _check(data)
        if str(key) == "":
            return npdata
        else:
            return (key, npdata)

    # Communicate size
    if rank == source:
        key, npdata = _check(data)
        shape, dtypestr, key = comm.bcast((npdata.shape, npdata.dtype.str, key), source)
    else:
        shape, dtypestr, key = comm.bcast(None, source)

    # Prepare buffers
    newdtype = '|u1' if (dtypestr == '|b1') else dtypestr
    if rank == source:
        buf = npdata.astype(newdtype)
    else:
        buf = np.empty(shape, dtype=newdtype)

    # Send
    comm.Bcast(buf, source)

    if str(key) == "":
        return buf.astype(dtypestr)
    else:
        return (key, buf)


def _check(data):
    """
    Check if data is compatible for MPI broadcast.

    FIXME: is the following true?
    data can be either a numpy array or a pair (key, array)
    """
    if type(data) is np.ndarray:
        key = ""
        if not data.flags.contiguous:
            data = np.ascontiguousarray(data)
    elif type(data) is tuple:
        key = data[0]
        data = data[1] #_check(data[1])[1]
    else:
        raise TypeError("Input data %s incompatible for broadcast" % str(type(data)))
    return key, data


def MPIrand_normal(loc=0.0, scale=1.0, size=(1)):
    """
        **Wrapper** for ``np.random.normal`` for same random sample across all nodes.
        *See numpy/scipy documentation below.*
    """
    if master:
        sample = np.array(np.random.normal(loc=loc, scale=scale, size=size))
    else:
        sample = np.zeros(size)
    allreduce(sample)
    return sample

MPIrand_normal.__doc__+=np.random.normal.__doc__

def MPIrand_uniform(low=0.0, high=1.0, size=(1)):
    """
        **Wrapper** for ``np.random.uniform`` for same random sample across all nodes.
        *See numpy/scipy documentation below.*
    """
    if master:
        sample = np.array(np.random.uniform(low=low, high=high, size=size))
    else:
        sample = np.zeros(size)
    allreduce(sample)
    return sample

MPIrand_uniform.__doc__+=np.random.uniform.__doc__

if MPI is not None:
    # local rank
    hosts_ranks = {}
    host = MPI.Get_processor_name()   
    rank_host = gather_dict({rank : host})
    for k,v in rank_host.items():
        if v not in hosts_ranks:
            hosts_ranks[v]=[k]
        else:
            hosts_ranks[v].append(k)
            
    hosts_ranks = bcast_dict(hosts_ranks)
    rank_local = hosts_ranks[host].index(rank)
    del rank_host
else:
    rank_local = 0
    hosts_ranks={'localhost':[0]}

def MPInoise2d(sh,rms=1.0, mfs=2,rms_mod=None, mfs_mod=2):
    """
    Creates complex-valued statistical noise in the shape of `sh`
    consistent across all nodes.

    Parameters
    ----------
    sh : tuple
        Output shape.

    rms: float or None
        Root mean square of noise in phase. If None, only ones are
        returned.

    mfs: float
        Minimum feature size [in pixel] of noise in phase.

    rms_mod: float or None
        Root mean square of noise in amplitude / modulus.

    mfs_mod:
        Minimum feature size [in pixel] of noise in amplitude / modulus.

    Returns
    -------
    noise : ndarray
        Numpy array filled with noise

    See also
    --------
    MPIrand_uniform
    MPIrand_normal
    """
    from ..utils import gf_2d
    sh = tuple(sh)
    A=np.ones(sh,dtype=complex)
    if rms is not None and float(rms)!=0.:
        mfs /= 2.35
        phnoise = MPIrand_normal(0.0,rms,sh)
        phnoise[:] = gf_2d(phnoise,mfs)
        A *= np.exp(1j*phnoise)
    if rms_mod is not None and float(rms_mod)!=0.:
        ampnoise = MPIrand_normal(1.0,rms_mod,sh)
        mfs_mod /=2.35
        ampnoise[:] = gf_2d(ampnoise,mfs_mod)
        A *= ampnoise
    return A


def gather_list(lst, length, indices):
    """
    gathers list `lst` of all processes to a list of length `length`
    according to order given by `indices`. definitely not foolproof
    The user has to make sure that no index appears twice in all processes

    return list of length `length`. has only meaning in master process
    """
    new = [None] * length
    for index, item in zip(indices, lst):
        if index < length:
            new[index] = item
    if MPIenabled:
        for i in range(length):
            if master:
                # Root receives the data if it doesn't have it yet
                if new[i] is None:
                    new[i] = receive()
                barrier()
            else:
                if new[i] is not None:
                    # Send data to root.
                    send(new[i])
                barrier()
        barrier()

    return new


def _scatter_list(lst, length, indices):
    """
    master process scatters a list `lst` of length `length`
    to non-masters that have the respective index in their `indeces` list

    functional, but maybe not 100% foolproof

    return list of length len(indices) for all processes.
    indices that extend boyond length are filled with None
    """
    new = [None] * len(indices)
    if MPIenabled:
        for i in range(length):
            if master:
                data = bcast(lst[i])
                # Root broadcasts the data
            else:
                data = bcast(None)

            # data = pp.bcast(data)
            try:
                new[indices.index(i)] = data
            except:
                pass

    return new


def _gather_list(lst, target=0):
    out = []
    for r in range(size):
        if r == target:
            if rank == target:
                out += lst
            continue

        if rank == target:
            l = comm.recv(source=r)
            for i in range(l):
                out.append(receive(r))
                # out[k] = v
        elif r == rank:
            # your turn to send
            l = len(lst)
            comm.send(l, dest=target)
            for item in lst:
                send(item, dest=target)
        barrier()

    return out


def _gather_dict(dct, target=0):
    """
    Gathers broadcasted dict `dct` at rank `target`.
    Input dictionaries remain unaltered
    Double key access will cause an overwrite
    without complains.

    returns : gathered dict at rank==target
              empty dict at other ranks
    """
    out = {}
    if not MPIenabled:
        out.update(dct)
        return out

    for r in range(size):
        if r == target:
            if rank == target:
                out.update(dct)
            continue

        if rank == target:
            l = comm.recv(source=r)
            for i in range(l):
                k, v = receive(r)
                out[k] = v
        elif r == rank:
            # your turn to send
            l = len(dct)
            comm.send(l, dest=target)
            for item in dct.items():
                send(item, dest=target)
        barrier()

    return out


def _bcast_dict(dct, keys_accepted='all', source=0):
    """
    Broadcasts a dict where all values are numpy arrays
    Fills dict `dct` in place for receiving nodes

    There is no guarantee that each key,value pair is accepted at other
    nodes. Deleting reference to input dictionary may result in data loss.

    Returns
    -------
    dct : dict
        A smaller dictionary with only accepted keys
    """
    if not MPIenabled:
        out = dict(dct)
        return out

    # communicate the dict length
    if rank == source:
        out = {}
        length = comm.bcast(len(dct), source)
        for k, v in dct.items():
            bcast((k, v))
            if str(keys_accepted) == 'all' or k in keys_accepted:
                out[k] = v

        return out
    else:
        if dct is None:
            dct = {}
        length = comm.bcast(None, source)
        for k in range(length):
            k, v = bcast(None)
            if str(keys_accepted) == 'all' or k in keys_accepted:
                dct[k] = v

        return dct
