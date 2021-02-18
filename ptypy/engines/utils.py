# -*- coding: utf-8 -*-
"""\
Engine-specific utilities.
This could be compiled, or GPU accelerated.

This file is part of the PTYPY package.

    :copyright: Copyright 2014 by the PTYPY team, see AUTHORS.
    :license: GPLv2, see LICENSE for details.
"""
import numpy as np
from .. import utils as u
from .. import parallel
from scipy.sparse.linalg import eigsh

# This dynamic loas could easily be generalized to other types.
def dynamic_load(path, baselist, fail_silently = True):
    """
    Load an derivatives `of baselist` dynamically from .py files in the given path.

    :param path: Path to Python files.
    """
    
    import os
    import glob
    import re
    from importlib.machinery import SourceFileLoader

    # Loop through paths
    engine_path = {}
    
    try:
        # Complete directory path
        directory = os.path.abspath(os.path.expanduser(path))
    
        if not os.path.exists(directory):
            # Continue silently
            raise IOError('Engine path %s does not exist.'
                           % str(directory))
    
        # Get list of python files
        py_files = glob.glob(directory + '/*.py')
        if not py_files:
            raise IOError('Directory %s does not contain Python files,' 
                                            % str(directory))
    
        # Loop through files to find engines
        for filename in py_files:
            modname = os.path.splitext(os.path.split(filename)[-1])[0]
    
            # Find classes
            res = re.findall(
                '^class (.*)\((.*)\)', file(filename, 'r').read(), re.M)
    
            for classname, basename in res:
                if (basename in baselist) and classname not in baselist:
                    # Match!
                    engine_path[classname] = (modname, filename)
                    u.logger.info("Found Engine '%s' in file '%s'"
                                  % (classname, filename))
    
        # Load engines that have been found
        for classname, mf in engine_path.items():
    
            # Import module
            modname, filename = mf
            print(modname, filename)
            engine_module = SourceFileLoader(modname, filename).load_module()

    except Exception as e:
        if not fail_silently:
            u.logger.warning("Attempt at loading Engines dynamically failed")
            import sys
            import traceback
            ex_type, ex, tb = sys.exc_info()
            u.logger.warning(traceback.format_exc(tb))


def basic_fourier_update(diff_view, pbound=None, alpha=1., LL_error=True):
    """\
    Fourier update a single view using its associated pods.
    Updates on all pods' exit waves.

    Parameters
    ----------
    diff_view : View
        View to diffraction data

    alpha : float, optional
        Mixing between old and new exit wave. Valid interval ``[0, 1]``

    pbound : float, optional
        Power bound. Fourier update is bypassed if the quadratic deviation
        between diffraction data and `diff_view` is below this value.
        If ``None``, fourier update always happens.

    LL_error : bool
        If ``True``, calculates log-likelihood and puts it in the last entry
        of the returned error vector, else puts in ``0.0``

    Returns
    -------
    error : ndarray
        1d array, ``error = np.array([err_fmag, err_phot, err_exit])``.

        - `err_fmag`, Fourier magnitude error; quadratic deviation from
          root of experimental data
        - `err_phot`, quadratic deviation from experimental data (photons)
        - `err_exit`, quadratic deviation of exit waves before and after
          Fourier iteration
    """
    # Prepare dict for storing propagated waves
    f = {}

    # Buffer for accumulated photons
    af2 = np.zeros_like(diff_view.data)
    # Get measured data
    I = diff_view.data

    # Get the mask
    fmask = diff_view.pod.mask

    # For log likelihood error
    if LL_error is True:
        LL = np.zeros_like(diff_view.data)
        for name, pod in diff_view.pods.items():
            LL += pod.downsample(u.abs2(pod.fw(pod.probe * pod.object)))
        err_phot = (np.sum(fmask * (LL - I)**2 / (I + 1.))
                    / np.prod(LL.shape))
    else:
        err_phot = 0.

    # Propagate the exit waves
    for name, pod in diff_view.pods.items():
        if not pod.active:
            continue
        f[name] = pod.fw((1 + alpha) * pod.probe * pod.object
                         - alpha * pod.exit)
        af2 += pod.downsample(u.abs2(f[name]))

    fmag = np.sqrt(np.abs(I))
    af = np.sqrt(af2)

    # Fourier magnitudes deviations
    fdev = af - fmag
    err_fmag = np.sum(fmask * fdev**2) / fmask.sum()
    err_exit = 0.

    if pbound is None:
        # No power bound
        fm = (1 - fmask) + fmask * fmag / (af + 1e-10)
        for name, pod in diff_view.pods.items():
            if not pod.active:
                continue
            df = pod.bw(pod.upsample(fm) * f[name]) - alpha * pod.probe * pod.object + (alpha - 1) * pod.exit
            pod.exit += df
            err_exit += np.mean(u.abs2(df))
    elif err_fmag > pbound:
        # Power bound is applied
        renorm = np.sqrt(pbound / err_fmag)
        fm = (1 - fmask) + fmask * (fmag + fdev * renorm) / (af + 1e-10)
        for name, pod in diff_view.pods.items():
            if not pod.active:
                continue
            df = pod.bw(pod.upsample(fm) * f[name]) - alpha * pod.probe * pod.object + (alpha - 1) * pod.exit
            pod.exit += df
            err_exit += np.mean(u.abs2(df))
    else:
        # Within power bound so no constraint applied.
        for name, pod in diff_view.pods.items():
            if not pod.active:
                continue
            df = (pod.probe * pod.object - pod.exit)
            pod.exit += df
            err_exit += np.mean(u.abs2(df))

    return np.array([err_fmag, err_phot, err_exit])


def reduce_dimension(a, dim, local_indices=None):
    """
    Apply a low-rank approximation on a.

    :param a:
     3D numpy array.

    :param dim:
     The number of dimensions to retain. The case dim=0 (which would
     just reduce all layers to a mean) is not implemented.

    :param local_indices:
     Used for Containers distributed across nodes. Local indices of
     the current node.

    :return: [reduced array, modes, coefficients]
     Where:
      - reduced array is the result of dimensionality reduction
         (same shape as a)
      - modes: 3D array of length dim containing eigenmodes
         (aka singular vectors)
      - coefficients: 2D matrix representing the decomposition of a.
    """
    if local_indices is None:  # No MPI - generate a list of indices
        Nl = len(a)
        local_indices = range(Nl)
    # Distributed array - share info between nodes to compute
    # totalsize of matrix
    else:
        assert len(a) == len(local_indices)
        Nl = parallel.allreduce(len(local_indices))

    # Create the matrix to diagonalise
    M = np.zeros((Nl, Nl), dtype=complex)

    size = parallel.size
    rank = parallel.rank

    # Communication takes a different form if size is even or odd
    size_is_even = (size == 2 * (size // 2))

    # Using Round-Robin pairing to optimise parallelism
    if size_is_even:
        peer_nodes = np.roll(np.arange(size - 1), rank)
        peer_nodes[peer_nodes == rank] = size - 1
        if rank == size - 1:
            peer_nodes = ((size // 2) * np.arange(size - 1)) % (size - 1)
    else:
        peer_nodes = np.roll(np.arange(size), rank)

    # Even size means that local scalar product have all
    # to be done in parallel
    if size_is_even:
        for l0, i0 in enumerate(local_indices):
            for l1, i1 in enumerate(local_indices):
                if i0 > i1:
                    continue
                M[i0, i1] = np.vdot(a[l0], a[l1])
                M[i1, i0] = np.conj(M[i0, i1])

    # Fill matrix by looping through peers and communicate info
    # for scalar products
    for other_rank in peer_nodes:
        if other_rank == rank:
            # local scalar product
            for l0, i0 in enumerate(local_indices):
                for l1, i1 in enumerate(local_indices):
                    if i0 > i1:
                        continue
                    M[i0, i1] = np.vdot(a[l0], a[l1])
                    M[i1, i0] = np.conj(M[i0, i1])
        elif other_rank > rank:
            # Send layer indices
            parallel.send(local_indices, other_rank, tag=0)
            # Send data
            parallel.send(a, other_rank, tag=1)
        else:
            # Receive layer indices
            other_indices = parallel.receive(source=other_rank, tag=0)
            b = parallel.receive(source=other_rank, tag=1)
            # Compute matrix elements
            for l0, i0 in enumerate(local_indices):
                for l1, i1 in enumerate(other_indices):
                    M[i0, i1] = np.vdot(a[l0], b[l1])
                    M[i1, i0] = np.conj(M[i0, i1])

    # Finally group all matrix info
    parallel.allreduce(M)

    # Diagonalise the matrix
    eigval, eigvec = eigsh(M, k=dim + 2, which='LM')

    # Generate the modes
    modes = np.array([sum(a[l] * eigvec[i, k]
        for l, i in enumerate(local_indices)) for k in range(dim)])

    parallel.allreduce(modes)

    # Reconstruct the array
    eigvecc = eigvec.conj()[:,:-2]
    output = np.zeros_like(a)
    for l, i in enumerate(local_indices):
        output[l] = sum(modes[k] * eigvecc[i, k] for k in range(dim))

    return output, modes, eigvecc


def Cnorm2(c):
    """\
    Computes a norm2 on whole container `c`.

    :param Container c: Input
    :returns: The norm2 (*scalar*)

    See also
    --------
    ptypy.utils.math_utils.norm2
    """
    r = 0.
    for name, s in c.storages.items():
        r += u.norm2(s.data)
    return r


def Cdot(c1, c2):
    """\
    Compute the dot product on two containers `c1` and `c2`.
    No check is made to ensure they are of the same kind.

    :param Container c1, c2: Input
    :returns: The dot product (*scalar*)
    """
    r = 0.
    for name, s in c1.storages.items():
        r += np.vdot(c1.storages[name].data.flat, c2.storages[name].data.flat)
    return r

