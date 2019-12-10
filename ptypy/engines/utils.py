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


# Hook for subpixel shift
def hook_subpixel_shift_fourier(self, data, sp):
    """
    Fourier space shift
    """
    return u.shift_fourier(data, sp)


def hook_subpixel_shift_linear(self, data, sp):
    """
    Bilinear shift
    """
    return u.shift_interp(data, sp, order=1)


def hook_subpixel_shift_interp(self, data, sp):
    """
    Spline 3 shift
    """
    return u.shift_interp(data, sp)


def hook_subpixel_shift_null(self, data, sp):
    """
    No shift
    """
    return data
