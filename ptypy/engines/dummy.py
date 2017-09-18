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
from base import BaseEngine
from ..utils.descriptor import defaults_tree

__all__ = ['Dummy']

@defaults_tree.parse_doc('engine.Dummy')
class Dummy(BaseEngine):
    """
    Dummy reconstruction engine.


    Defaults:

    [itertime]
    default = .2
    type = float
    help = Sleep time for a single iteration (in seconds)

    """

    def __init__(self, ptycho_parent, pars=None):
        """
        Dummy reconstruction engine.
        """
        super(Dummy,self).__init__(ptycho_parent)

        p = self.DEFAULTS.copy()
        if pars is not None:
            p.update(pars)
        self.p = p

        self.ntimescalled  = 0

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
            
    def engine_iterate(self,numiter):
        """
        Compute one iteration.
        Should return a per-view-error-array of size ($number_of_views,3)
        """
        ############################
        # Simulate hard work
        ############################
        time.sleep(self.itertime)
        # virtual error reduces 10%
        error_dct = error = {}
        for dname, diff_view in self.di.views.iteritems():
            error_dct[dname] = [0., 0.9**self.ntimescalled, 0.]
        self.ntimescalled+=1
        return error_dct
     
    def engine_finalize(self):
        """
        Clean up after iterations are done
        """
        pass
        
        


        
