#!/usr/bin/env python2
# -*- coding: utf-8 -*-
from .version import short_version, version, release

__doc__ = \
"""
PTYPY(v%(short)s): A ptychography reconstruction package.

To cite PTYPY in publications, use
 @article{ ... }

    :copyright: Copyright 2018 by the PTYPY team, see AUTHORS.
    :license: GPLv2, see LICENSE for details.
    :version: %(version)s
    :status: %(devrel)s
""" % {'version': version, 'devrel': 'release' if release else 'development', 'short': short_version}

del short_version, release

try:
    import zmq
except ImportError as ie:
    __has_zmq__ = False
else:
    __has_zmq__ = True
    del zmq

try:
    import mpi4py
except ImportError as ie:
    __has_mpi4py__ = False
else:
    __has_mpi4py__ = True
    del mpi4py

try:
    import matplotlib
except ImportError as ie:
    __has_matplotlib__ = False
else:
    __has_matplotlib__ = True
    del matplotlib

# Initialize MPI (eventually GPU)
from .utils import parallel

# Logging
from .utils import verbose

# Log immediately dependency information
if not __has_zmq__:
    __zmq_msg = 'ZeroMQ not found.\nInteraction server & client disabled.\n\
    Install python-zmq via the package repositories or with `pip install --user pyzmq`'
    verbose.logger.warning(__zmq_msg)
if not __has_mpi4py__:
    __mpi_msg = 'Message Passaging for Python (mpi4py) not found.\n\
    CPU-parallelization disabled.\n\
    Install python-mpi4py via the package repositories or with `pip install --user mpi4py`'
    verbose.logger.warning(__mpi_msg)
if not __has_matplotlib__:
    __mpl_msg = 'Plotting for Python (matplotlib) not found.\n\
    Plotting disabled.\n\
    Install python-matplotlib via the package repositories or with `pip install --user matplotlib`'
    verbose.logger.warning(__mpl_msg)

# Start a parameter tree
from .utils.descriptor import EvalDescriptor
defaults_tree = EvalDescriptor('root')
del EvalDescriptor


# Import core modules
from . import utils
from . import io
from . import experiment
from . import core
from . import simulations
from . import resources

