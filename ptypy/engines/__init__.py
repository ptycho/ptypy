# -*- coding: utf-8 -*-
"""
Engines module.

Implements the difference map (DM) and maximum likelihood (ML) reconstruction
algorithms for ptychography.

This file is part of the PTYPY package.

    :copyright: Copyright 2014 by the PTYPY team, see AUTHORS.
    :license: GPLv2, see LICENSE for details.
"""
from .. import utils as u
from base import BaseEngine, DEFAULT_iter_info
from base import DEFAULT as COMMON
import DM
import DM_simple
import ML
import dummy
import ePIE

__all__ = ['DM', 'ML', 'ePIE', 'BaseEngine']

# List of supported engines
engine_names = ['Dummy', 'DM_simple', 'DM', 'ML', 'ML_new',
                            'ePIE']

# Supported engines defaults
DEFAULTS = u.Param(
    common=COMMON,
    Dummy=dummy.DEFAULT,
    DM_simple=DM_simple.DEFAULT,
    DM=DM.DEFAULT,
    ML=ML.DEFAULT,
    ePIE=ePIE.DEFAULT
)

# Engine objects
ENGINES = u.Param(
    Dummy=dummy.Dummy,
    DM_simple=DM_simple.DM_simple,
    DM=DM.DM,
    ML=ML.ML,
    ePIE=ePIE.EPIE
)


def by_name(name):
    if name not in ENGINES.keys():
        raise RuntimeError('Unknown engine: %s' % name)
    return ENGINES[name]

try:
    import os
    import glob
    import re
    import imp

    # Default search paths for engines
    DEFAULT_ENGINE_PATHS = ['./', '~/.ptypy/']

    def dynamic_load(path=None):
        """
        Load an engine dynamically from the given paths

        :param path: Path or list of paths
        """

        # Update list of paths to search for
        if path is not None:
            if str(path) == path:
                path_list = [path] + DEFAULT_ENGINE_PATHS
            else:
                path_list = path + DEFAULT_ENGINE_PATHS
        else:
            path_list = DEFAULT_ENGINE_PATHS

        # List of base classes an engine could derive from
        baselist = ['BaseEngine'] + engine_names

        # Loop through paths
        engine_path = {}
        for path in path_list:
            # Complete directory path
            directory = os.path.abspath(os.path.expanduser(path))

            if not os.path.exists(directory):
                # Continue silently
                continue
                # raise IOError('Engine path %s does not exist.'
                #               % str(directory))

            # Get list of python files
            py_files = glob.glob(directory + '/*.py')
            if not py_files:
                continue

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

            # Update list
            DEFAULTS[classname] = getattr(engine_module, 'DEFAULT')
            ENGINES[classname] = getattr(engine_module, classname)
            engine_names.append(classname)

    dynamic_load()
except Exception as e:
    u.logger.warning("Attempt at loading Engines dynamically failed")
    import sys
    import traceback
    ex_type, ex, tb = sys.exc_info()
    u.logger.warning(traceback.format_exc(tb))
