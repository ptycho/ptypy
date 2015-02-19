#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
PTYPY: A ptychographic reconstruction package.

To cite PTYPY in publications, use
 @article{ ... }
 
    :copyright: Copyright 2014 by the PTYPY team, see AUTHORS.
    :license: GPLv2, see LICENSE for details.
"""
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
