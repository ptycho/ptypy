# -*- coding: utf-8 -*-
"""
Dummy reconstruction engine - for testing purposes.

This file is part of the PTYPY package.

    :copyright: Copyright 2014 by the PTYPY team, see AUTHORS.
    :license: GPLv2, see LICENSE for details.
"""

import time
import numpy as np

import ptypy
from .. import utils as u
from ..utils import parallel
from . import BaseEngine

#import utils

__all__ = ['Dummy']

DEFAULT = u.Param(
    itertime = 2.,    # Sleep time for a single iteration (in seconds)
)

class Dummy(BaseEngine):
    
    """
    Minimum implementation of BaseEngine
    """
    
    DEFAULT = DEFAULT

    def __init__(self, ptycho_parent, pars=None):
        """
        Dummy reconstruction engine.
        """
        super(Dummy,self).__init__(ptycho_parent,pars)
        
    def engine_initialize(self):
        """
        Prepare for reconstruction.
        """
        self.itertime = self.p.itertime / parallel.size
        self.error = np.ones((100,3))*1e6
        
    def engine_prepare(self):
        """
        Last-minute preparation before iterating.
        """
        pass
            
    def engine_iterate(self):
        """
        Compute one iteration.
        Should return a per-view-error-array of size ($number_of_views,3)
        """
        ############################
        # Simulate hard work
        ############################
        time.sleep(self.itertime)
        # virtual error reduces 10%
        self.error *= 0.9
        self.curiter +=1
        return self.error
        
    def engine_finalize(self):
        """
        Clean up after iterations are done
        """
        pass
        
        


        
