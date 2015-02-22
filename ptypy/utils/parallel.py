# -*- coding: utf-8 -*-
"""\
Utility functions and classes to support MPI computing.

This file is part of the PTYPY package.

    :copyright: Copyright 2014 by the PTYPY team, see AUTHORS.
    :license: GPLv2, see LICENSE for details.
"""
__all__ = ['MPIenabled', 'psize', 'prank', 'comm', 'MPI', 'master',
           'LoadManager', 'loadmanager', 'MPIrand_normal', 'MPIrand_uniform',
           'gather_list', 'scatter_list', 'bcast_dict', 'gather_dict','MPInoise2d']

import numpy as np

size = 1
rank = 0
MPI = None
comm = None

try:
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
except:
    print 'MPI initialization failed. Proceeding with one processor'

MPIenabled = not (size == 1)
master = (rank == 0)


def useMPI(do=None):
    """\
    Toggle using MPI or not. Is this useful?
    """
    global MPIenabled
    if do is None:
        return MPIenabled
    if MPI is None:
        MPIenabled = False
    else:
        MPIenabled = do


###################################
# Helper functions, wrapper to mpi4py
###################################

class LoadManager(object):
    def __init__(self):
        """
        LoadManager: keep track of the amount of data managed by each
        process and help keeping it balanced.
        """

        self.load = np.zeros((size,), dtype=int)
        self.rank_of = {}

    def assign(self, idlist=None):
        """
        
        Subdivide the provided list of ids into contiguous blocks to balance
        the load among the processes. 
        
        The elements of idlist are used as keys to subsequently identify
        the rank of a given id, through the dict self.rank_of. No check is done
        whether ids passed are unique or even hashable.
        
        If idlist is None, increase by one the load of the least busy process.
        self.rank_of is not updated in this case.
        
        return R, a list of list such that
        R[rank] = list of indices of idlist managed by process of given rank. 
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

# Create one instance - typically only this one should be used
loadmanager = LoadManager()


def allreduce(a, op=None):
    """
    Wrapper for comm.Allreduce, always in place.
    
    Parameters
    ----------
    a : numpy array
        The array to operate on.
    op : None or one of MPI.BAND, MPI.BOR, MPI.BXOR, MPI.LAND, MPI.LOR, 
         MPI.LXOR, MPI.MAX, MPI.MAXLOC, MPI.MIN, MPI.MINLOC, MPI.OP_NULL,
         MPI.PROD, MPI.REPLACE or MPI.SUM. 
         If None, use MPI.SUM.
    """

    if not MPIenabled:
        return
    if op is None:
        # print a.shape
        comm.Allreduce(MPI.IN_PLACE, a)
    else:
        comm.Allreduce(MPI.IN_PLACE, a, op=op)
    return


def _MPIop(a, op, axis=None):
    """
    Apply operation op on accross a list of arrays distributed between
    processes. Supported operations are SUM, MAX, MIN, and PROD. 
    """

    MPIop, npop = \
        {'SUM': (MPI.SUM, np.sum), 'MAX': (MPI.MAX, np.max), 'MIN': (MPI.MIN, np.min), 'PROD': (MPI.PROD, np.prod)}[
            op.upper()]

    # Total op
    if axis is None:
        # Apply op on locally owned data (and wrap the scalar result in a numpy array
        s = np.array([npop([npop(ai) for ai in a if ai is not None])])

        # Reduce and return scalar
        if MPIenabled:
            comm.Allreduce(MPI.IN_PLACE, s, op=MPIop)
        return s[0]

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


def receive(source=None, tag=0, out=None):
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


def bcast(data, source=0, key=""):
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
    elif np.iterable(data):
        key = data[0]
        data = _check(data[1])[1]
    else:
        raise TypeError("Input data %s incompatible for broadcast" % str(type(data)))
    return key, data


def MPIrand_normal(loc=0.0, scale=1.0, size=(1)):
    """
    wrapper for np.random.normal for same random sample across all nodes.
    """
    if master:
        sample = np.array(np.random.normal(loc=loc, scale=scale, size=size))
    else:
        sample = np.zeros(size)
    allreduce(sample)
    return sample


def MPIrand_uniform(low=0.0, high=1.0, size=(1)):
    """
     wrapper for np.random.uniform for same random sample across all nodes.
    """
    if master:
        sample = np.array(np.random.uniform(low=low, high=high, size=size))
    else:
        sample = np.zeros(size)
    allreduce(sample)
    return sample

def MPInoise2d(sh,rms=1.0, mfs=2,rms_mod=None, mfs_mod=2):
    """
    creates noise in the shape of `sh` consistent across all nodes.
    
    :param sh: output shape
    :param rms: root mean square of noise in phase
    :param mfs: minimum feature [pixel] of noise in phase    
    :param rms_mod: root mean square of noise in amplitude / modulus
    :param mfs_mod: minimum feature [pixel]  of noise in amplitude / modulus  
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


def scatter_list(lst, length, indices):
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


def gather_list(lst, target=0):
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


def gather_dict(dct, target=0):
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
            for item in dct.iteritems():
                send(item, dest=target)
        barrier()

    return out


def bcast_dict(dct, keys_accepted='all', source=0):
    """
    Broadcasts a dict of where all values are numpy arrays
    Fills dict `dct` in place for receiving nodes
    
    There is no guarantee that each key,value pair is accepted at other 
    nodes. Deleting reference to input dictionary may result in data loss.
    
    returns:
        dict, length : a reduced dictionary with only accepted keys 
                      and the length of the original dictionary
    """
    if not MPIenabled:
        out = dict(dct)
        return out, len(out)

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
