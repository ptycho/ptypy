# -*- coding: utf-8 -*-
"""\
Scan loading plugin. Meant to make easier user-generated, problem-specific
data preparation.

This file is part of the PTYPY package.

    :copyright: Copyright 2014 by the PTYPY team, see AUTHORS.
    :license: see LICENSE for details.


"""

import os
from importlib.machinery import SourceFileLoader
import inspect
from .. import utils as u
from ..core.data import PtyScan

logger = u.verbose.logger

# Recipe defaults
RECIPE = u.Param()
RECIPE.file = None             # Mandatory: plugin filename
RECIPE.classname = None    # Optional: PtyScan subclass. If None, such a class will be found dynamically.
RECIPE.recipe = u.Param()        # Recipe parameters to pass to the preparation plugin


def makeScanPlugin(pars=None):
    """
    Factory wrapper that provides a PtyScan object.

    """

    # Recipe defaults
    rinfo = RECIPE.copy()
    rinfo.update(pars.recipe)

    # Sanity checks (for user-friendliness)
    filename = os.path.abspath(os.path.expanduser(rinfo.file))
    if not os.path.exists(filename):
        raise IOError('Plugin file %s does not exist.' % str(filename))

    plugin_name, file_ext = os.path.splitext(os.path.split(filename)[-1])
    if file_ext.lower() != '.py':
        raise IOError('Plugin file %s is not a python file.' % str(filename))

    # Load plugin
    plugin = SourceFileLoader(plugin_name, filename).load_module()

    # Find the PtyScan class
    if rinfo.classname is None:
        # We try to find the class
        ptyscan_objects = {}
        for name, obj in plugin.__dict__.items():
            if inspect.isclass(obj) and issubclass(obj, PtyScan) and obj is not PtyScan:
                ptyscan_objects[name] = obj
        if not ptyscan_objects:
            raise RuntimeError('Failed to find a PtyScan subclass in plugin %s' % plugin_name)
        elif len(ptyscan_objects) > 1:
            raise RuntimeError('Multiple PtyScan subclasses in plugin %s: %s' % (plugin_name, str(list(ptyscan_objects.keys()))))
        # Class found
        ptyscan_obj_name = list(ptyscan_objects.keys())[0]
        ptyscan_obj = ptyscan_objects[ptyscan_obj_name]
    else:
        ptyscan_obj_name = rinfo.classname
        ptyscan_obj = getattr(plugin, ptyscan_obj_name)

    logger.info('Using plugin preparation class "%s.%s".' % (plugin_name, ptyscan_obj_name))

    # Replace the plugin recipe structure with the plugin parameters
    pars.recipe = rinfo.recipe

    # Create the object and return it
    return ptyscan_obj(pars)
