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

# This dynamic loas could easily be generalized to other types.
def dynamic_load(path, baselist, fail_silently = True):
    """
    Load an derivatives `of baselist` dynamically from .py files in the given path.

    :param path: Path to Python files.
    """
    
    import os
    import glob
    import re
    import imp

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
        for classname, mf in engine_path.iteritems():
    
            # Import module
            modname, filename = mf
            engine_module = imp.load_source(modname, filename)

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
    I = diff_view.data

    af2 = np.zeros_like(I)
    # Get measured data


    # Get the mask
    fmask = diff_view.pod.mask

    # For log likelihood error
    if LL_error is True:
        LL = np.zeros_like(I)
        for name, pod in diff_view.pods.iteritems():
            LL += u.abs2(pod.fw(pod.probe * pod.object))
        err_phot = (np.sum(fmask * (LL - I)**2 / (I + 1.))
                    / np.prod(LL.shape))
    else:
        err_phot = 0.

    # Propagate the exit waves
    for name, pod in diff_view.pods.iteritems():
        if not pod.active:
            continue
        f[name] = pod.fw((1 + alpha) * pod.probe * pod.object
                         - alpha * pod.exit)

        af2 += u.abs2(f[name])

    fmag = np.sqrt(np.abs(I))
    af = np.sqrt(af2)

    # Fourier magnitudes deviations
    fdev = af - fmag
    err_fmag = np.sum(fmask * fdev**2) / fmask.sum()
    err_exit = 0.

    if pbound is None:
        # No power bound
        fm = (1 - fmask) + fmask * fmag / (af + 1e-10)
        for name, pod in diff_view.pods.iteritems():
            if not pod.active:
                continue
            df = pod.bw(fm * f[name]) - pod.probe * pod.object
            pod.exit += df
            err_exit += np.mean(u.abs2(df))
    elif err_fmag > pbound:
        # Power bound is applied
        renorm = np.sqrt(pbound / err_fmag)
        fm = (1 - fmask) + fmask * (fmag + fdev * renorm) / (af + 1e-10)
        for name, pod in diff_view.pods.iteritems():
            if not pod.active:
                continue
            df = pod.bw(fm * f[name]) - pod.probe * pod.object
            pod.exit += df
            err_exit += np.mean(u.abs2(df))
    else:
        # Within power bound so no constraint applied.
        for name, pod in diff_view.pods.iteritems():
            if not pod.active:
                continue
            df = alpha * (pod.probe * pod.object - pod.exit)
            pod.exit += df
            err_exit += np.mean(u.abs2(df))

    if pbound is not None:
        # rescale the fmagnitude error to some meaning !!!
        # PT: I am not sure I agree with this.
        err_fmag /= pbound

    return np.array([err_fmag, err_phot, err_exit])


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
    for name, s in c.storages.iteritems():
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
    for name, s in c1.storages.iteritems():
        r += np.vdot(c1.storages[name].data.flat, c2.storages[name].data.flat)
    return r
