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
#from modules import *
import simulations 
import resources

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
