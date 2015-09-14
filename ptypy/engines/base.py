# -*- coding: utf-8 -*-
"""
Base engine. Used to define reconstruction parameters that are shared
by all engines.

This file is part of the PTYPY package.

    :copyright: Copyright 2014 by the PTYPY team, see AUTHORS.
    :license: GPLv2, see LICENSE for details.
"""
import numpy as np
import time

import ptypy
from .. import utils as u
from ..utils import parallel
from ..utils.verbose import logger, headerline

__all__ = ['BaseEngine','DEFAULT_iter_info']


DEFAULT = u.Param(
    numiter = 10,             # Total number of iterations
    numiter_contiguous = 1,    # number of iterations until next interactive release   
    probe_support = 0.8,       # relative area of probe array to which the probe is constraint if probe support becomes of more complex nature, consider putting it in illumination.py
    subpix_start = 0,         # Number of iterations before starting subpixel interpolation
    subpix = 'linear',  # 'fourier','linear' subpixel interpolation or None for no interpolation
    probe_inertia = 0.001,             # (88) Probe fraction kept from iteration to iteration
    object_inertia = 0.1,              # (89) Object fraction kept from iteration to iteration
    clip_object = None,                # (91) Clip object amplitude into this intrervall
    obj_smooth_std = 20,               # (90) Gaussian smoothing (pixel) of kept object fraction
    probe_update_start =0,
)

DEFAULT_iter_info = u.Param(
    iteration = 0,
    iterations = 0,
    engine = 'None',
    duration = 0.,
    error = np.zeros((3,))
    )

class BaseEngine(object):
    """
    Base reconstruction engine.
    
    In child classes, overwrite the following methods for custom behavior :
    
    engine_initialize
    engine_prepare
    engine_iterate
    engine_finalize
    """
    
    DEFAULT= DEFAULT.copy()

    def __init__(self, ptycho, pars=None):
        """
        Base reconstruction engine.
        
        Parameters
        ----------
        ptycho : Ptycho 
            The parent :any:`Ptycho` object.
            
        pars: Param or dict
            Initialization parameters
        """
        self.ptycho = ptycho
        p = u.Param(self.DEFAULT)

        if pars is not None: p.update(pars)
        self.p = p
        #self.itermeta = []
        #self.meta = u.Param()
        self.finished = False
        self.numiter = self.p.numiter
        #self.initialize()
               
    def initialize(self):
        """
        Prepare for reconstruction.
        """
        logger.info('\n'+headerline('Starting %s-algoritm.' % str(self.__class__.__name__),'l','=')+'\n')
        logger.info('Parameter set:')
        logger.info(u.verbose.report(self.p,noheader=True).strip()) 
        logger.info(headerline('','l','='))
        
        self.curiter = 0
        if self.ptycho.runtime.iter_info:
            self.alliter = self.ptycho.runtime.iter_info[-1]['iterations']
        else:
            self.alliter = 0
        
        # common attributes for all reconstructions
        self.di = self.ptycho.diff
        self.ob = self.ptycho.obj
        self.pr = self.ptycho.probe
        self.ma = self.ptycho.mask
        self.ex = self.ptycho.exit
        self.pods = self.ptycho.pods

        self.probe_support = {}
        # call engine specific initialization
        self.engine_initialize()
        
    def prepare(self):
        """
        Last-minute preparation before iterating.
        """
        self.finished = False
        ### Calculate probe support 
        # an individual support for each storage is calculated in saved
        # in the dict self.probe_support
        supp = self.p.probe_support
        if supp is not None:
            for name, s in self.pr.S.iteritems():
                sh = s.data.shape
                ll,xx,yy = u.grids(sh,FFTlike=False)
                support = (np.pi*(xx**2 + yy**2) < supp * sh[1] * sh[2])
                self.probe_support[name] = support
                
        # call engine specific preparation
        self.engine_prepare()
            
    def iterate(self, num=None):
        """
        Compute one or several iterations.
        
        num : None,int number of iterations. If None or num<1, a single iteration is performed
        """
        # several iterations 
        N = self.p.numiter_contiguous
        
        # overwrite default parameter
        if num is not None:
            N = num
        
        if self.finished: return 

        # for benchmarking
        self.t = time.time()
        
        # call engine specific iteration routine 
        # and collect the per-view error.      
        self.error = self.engine_iterate(N)
        
        # Increment the iterate number
        self.curiter += N
        self.alliter += N
        
        if self.curiter >= self.numiter: self.finished = True
        
        # Prepare runtime
        self._fill_runtime()

        parallel.barrier()

    def _fill_runtime(self):
        local_error = u.parallel.gather_dict(self.error)
        if local_error:
            error = np.array(local_error.values()).mean(0)
        else:
            error = np.zeros((1,))
        info = dict(
            iteration = self.curiter,
            iterations = self.alliter,
            engine = self.__class__.__name__,
            duration = time.time()-self.t,
            error = error
            )
        
        self.ptycho.runtime.iter_info.append(info)
        self.ptycho.runtime.error_local = local_error
        
    def finalize(self):
        """
        Clean up after iterations are done
        """
        self.engine_finalize()
        pass
    
    def engine_initialize(self):
        """
        Engine-specific initialization.
        Called at the end of self.initialize().
        """
        raise NotImplementedError()
        
    def engine_prepare(self):
        """
        Engine-specific preparation. Last-minute initialization providing up-to-date
        information for reconstruction.
        Called at the end of self.prepare()
        """
        raise NotImplementedError()
    
    def engine_iterate(self, num):
        """
        Engine single-step iteration. All book-keeping is done in self.iterate(),
        so this routine only needs to implement the "core" actions.
        """
        raise NotImplementedError()

    def engine_finalize(self):
        """
        Engine-specific finalization. Used to wrap-up engine-specific stuff.
        Called at the end of self.finalize()
        """
        raise NotImplementedError()

