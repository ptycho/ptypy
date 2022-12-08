#!/usr/bin/env python2
# -*- coding: utf-8 -*-
from .version import short_version, version, release

__doc__ = \
"""
PTYPY(v%(short)s): A ptychography reconstruction package.

To cite PTYPY in publications, use
 @article{ ... }

    :copyright: Copyright 2018 by the PTYPY team, see AUTHORS.
    :license: see LICENSE for details.
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
from . import core
from . import simulations
from . import resources

# Convenience loader for GPU engines
def load_gpu_engines(arch='cuda'):
    if arch=='cuda':
        from .accelerate.cuda_pycuda.engines import projectional_pycuda
        from .accelerate.cuda_pycuda.engines import projectional_pycuda_stream
        from .accelerate.cuda_pycuda.engines import stochastic
        from .accelerate.cuda_pycuda.engines import ML_pycuda
    if arch=='serial':
        from .accelerate.base.engines import projectional_serial
        from .accelerate.base.engines import projectional_serial_stream
        from .accelerate.base.engines import stochastic
        from .accelerate.base.engines import ML_serial
    if arch=='ocl':
        from .accelerate.ocl_pyopencl.engines import DM_ocl, DM_ocl_npy

from importlib import import_module
from .utils.verbose import log
ptyscan_modules = ['hdf5_loader',
                   'cSAXS',
                   'savu', 
                   'plugin', 
                   'ID16Anfp', 
                   'AMO_LCLS', 
                   'DiProI_FERMI', 
                   'optiklabor', 
                   'UCL', 
                   'nanomax', 
                   'nanomax_streaming', 
                   'ALS_5321', 
                   'Bragg3dSim']

# Convenience loader for ptyscan modules
def load_ptyscan_module(module):
    try:
        lib = import_module("."+module, 'ptypy.experiment')
    except ImportError as exception:
        log(2, 'Could not import ptyscan module %s, Reason: %s' % (module, exception))
        pass

# Convenience loader for all ptyscan modules
def load_all_ptyscan_modules():
    for m in ptyscan_modules:
        load_ptyscan_module(m)

# Convenience loader for all ptyscan modules and all gpu engines
def load_all():
    load_gpu_engines("cuda")
    load_gpu_engines("serial")
    #load_gpu_engines("ocl")
    load_all_ptyscan_modules()