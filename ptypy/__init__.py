#!/usr/bin/python
# -*- coding: utf-8 -*-
from version import short_version, version, release
"""
PTYPY(v%(short)s): A ptychography reconstruction package.

To cite PTYPY in publications, use
 @article{ ... }
 
    :copyright: Copyright 2014 by the PTYPY team, see AUTHORS.
    :license: GPLv2, see LICENSE for details.
    :version: %(version)s
    :status: %(devrel)s
""" % {'version':version,'devrel':'release' if release else 'development','short':short_version}

del short_version, release

try:
    import zmq
    __has_zmq__= True
    del zmq
except ImportError('ZeroMQ not found.\nInteraction server & client disabled.\n\
Install python-zmq via the package repositories or with `pip install --user pyzmq`'): 
    __has_zmq__= False

try:
    import mpi4py
    __has_mpi4py__= True
    del mpi4py
except ImportError('Message Passaging for Python (mpi4py) not found.\n\
CPU-parallelization disabled.\n\
Install python-mpi4py via the package repositories or with `pip install --user mpi4py`'): 
    __has_mpi4py__= False

try:
    import matplotlib
    __has_matplotlib__= True
    del matplotlib
except ImportError('Plotting for Python (matplotlib) not found.\n\
Plotting disabled.\n\
Install python-matplotlib via the package repositories or with `pip install --user matplotlib`'):
    __has_matplotlib__= False
       
# Initialize MPI (eventually GPU)
from utils import parallel

# Logging
from utils import verbose
#verbose.set_level(2)

import utils
# Import core modules
import io 
import experiment
import core


#from core import *
import simulations 
import resources

# Import libraries
import lib

if __name__ == "__main__":
    # TODO: parse command line arguments for extra options
    import sys
    # Get parameters from command line argument
    param_filename = sys.argv[1]
    p = parameters.load(param_filename)
    
    # Initialize Ptycho object
    pt = Ptycho(p)
    
    # Start reconstruction
    pt.run()
