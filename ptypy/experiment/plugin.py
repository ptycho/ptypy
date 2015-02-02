# -*- coding: utf-8 -*-
"""\
Scan loading plugin. Meant to make easier user-generated, problem-specific
data preparation.

This file is part of the PTYPY package.

    :copyright: Copyright 2014 by the PTYPY team, see AUTHORS.
    :license: GPLv2, see LICENSE for details.


"""

import os
import imp
import inspect
from .. import utils as u
from ..core.data import PtyScan

logger = u.verbose.logger
DEFAULT = u.Param()
DEFAULT.file = None             # Mandatory: plugin filename
DEFAULT.classname = None    # Optional: PtyScan subclass. If None, such a class will be found dynamically.
DEFAULT.pars = u.Param()        # Parameters to pass to the preparation plugin


def makeScanPlugin(pars=None, **kwargs):
    """
    Factory wrapper that provides a PtyScan object.

    """
    p = DEFAULT.copy()
    p.update(pars)
    p.update(kwargs)

    # Sanity checks (for user-friendliness)
    filename = os.path.abspath(os.path.expanduser(p.file))
    if not os.path.exists(p.file):
        raise IOError('Plugin file %s does not exist.' % str(filename))

    plugin_name, file_ext = os.path.splitext(os.path.split(filename)[-1])
    if file_ext.lower() != '.py':
        raise IOError('Plugin file %s is not a python file.' % str(filename))

    # Load plugin
    plugin = imp.load_source(plugin_name, filename)

    # Find and instantiate the PtyScan class
    if p.classname is None:
        # We try to find the class
        ptyscan_objects = {}
        for name, obj in plugin.__dict__.iteritems():
            if inspect.isclass(obj) and issubclass(obj, PtyScan):
                ptyscan_objects[name] = obj
        if not ptyscan_objects:
            raise RuntimeError('Failed to find a PtyScan subclass in plugin %s' % plugin_name)
        elif len(ptyscan_objects) > 1:
            raise RuntimeError('Multiple PtyScan subclasses in plugin %s: %s' % (plugin_name, str(ptyscan_objects.keys())))
        # Class found
        ptyscan_obj_name = ptyscan_objects.keys()[0]
        ptyscan_obj = ptyscan_objects[ptyscan_obj_name]
    else:
        ptyscan_obj_name = p.classname
        ptyscan_obj = getattr(plugin, ptyscan_obj_name)

    logger.info('Using plugin preparation class "%s.%s".' % (plugin_name, ptyscan_obj_name))

    # Create the object and return it
    return ptyscan_obj(p.pars)