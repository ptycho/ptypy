# -*- coding: utf-8 -*-
"""
Difference Map reconstruction engine.

This file is part of the PTYPY package.

    :copyright: Copyright 2014 by the PTYPY team, see AUTHORS.
    :license: GPLv2, see LICENSE for details.
"""

#from .. import core
from .. import utils as u
from utils import basic_fourier_update, C_allreduce
import numpy as np
import time

parallel = u.parallel

DEFAULT = u.Param(
    fourier_relax_factor = 0.01,
    alpha = 1.0,    
    probe_inertia = 0.001,             
    object_inertia = 0.1,              
)

__all__=['DM_pub']

class DM_pub(object):
    
    DEFAULT = DEFAULT

    def __init__(self, ptycho_parent, pars=None):
        """
        Base reconstruction engine.
        
        Parameters:
        ----------
        ptycho_parent: ptycho object
                       The parent ptycho class.
        pars: Param or dict
              Initialization parameters
        """
        self.ptycho = ptycho_parent
        p = u.Param(self.DEFAULT)

        if pars is not None: p.update(pars)
        self.p = p
        self.finished = False
        self.numiter = self.p.numiter
        
    def initialize(self):
        """
        Prepare for reconstruction.
        """
        self.curiter = 0
        
        # common attributes for all reconstructions
        self.di = self.ptycho.diff
        self.ob = self.ptycho.obj
        self.pr = self.ptycho.probe
        self.ma = self.ptycho.mask
        self.ex = self.ptycho.exit
        self.pods = self.ptycho.pods

        # generate container copies
        self.ob_nrm = self.ob.copy(self.ob.ID+'nrm',fill=0.)
        self.pr_nrm = self.pr.copy(self.pr.ID+'nrm',fill=0.)

    
    def prepare(self):
        """
        Last-minute preparation before iterating.
        """
        self.finished = False
        for s in self.di.S.values():
            s.pbound = .25 *  self.p.fourier_relax_factor**2 * s.max_power / s.shape[-1]**2
            
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
        
        for n in range(N):
            if self.finished: break 
            # for benchmarking
            self.t = time.time()
            
            self.errors = {}
            
            ###### fourier update #################  
            for name,di_view in self.di.V.iteritems():
                if not di_view.active: continue
                pbound = di_view.storage.pbound
                self.errors[name] = basic_fourier_update(di_view,pbound,alpha=self.p.alpha)
            
            ###### probe update  ##################
            self.probe_update(self.pr_nrm,self.pr, self.p.probe_inertia)
            
            ###### object update ##################
            self.object_update(self.ob_nrm,self.ob,self.p.object_inertia)
                       
            self.curiter += 1
            self.finished = (self.curiter >= self.numiter)
            
            self.fill_runtime()
            parallel.barrier()
        
    def object_update(self,ob_nrm,ob_buf,cfact=0.5):
        """
        DM object update. Requires 2 buffer deck copies
        ob_nrm, ob_buf
        """
        # Object update
        ob_buf *= cfact
        ob_nrm << cfact + 1e-10
            
        for pod in self.pods.itervalues():
            if not pod.active: continue
            ob_buf[pod.ob_view] += pod.probe.conj() * pod.exit
            ob_nrm[pod.ob_view] += pod.probe * pod.probe.conj()
            
        C_allreduce(ob_buf)
        C_allreduce(ob_nrm)
        
        ob_buf /= ob_nrm


    def probe_update(self,pr_nrm,pr_buf,cfact=0.5):
        """
        DM probe update. Requires 2 buffer deck copies
        pr_nrm, pr_buf
        """
        pr_buf *= cfact
        pr_nrm << cfact + 1e-10
            
        for name,pod in self.pods.iteritems():
            if not pod.active: continue
            pr_buf[pod.pr_view] += pod.object.conj() * pod.exit
            pr_nrm[pod.pr_view] += pod.object * pod.object.conj()

        C_allreduce(pr_buf)
        C_allreduce(pr_nrm)
        
        pr_buf /= pr_nrm

    def fill_runtime(self):
        error = parallel.gather_dict(self.errors)
        
        info = dict(
            iteration = self.curiter,
            iterations = len(self.ptycho.runtime.iter_info),
            engine = self.__class__.__name__,
            duration = time.time()-self.t,
            error = error
            )

        self.ptycho.runtime.iter_info.append(info)
    
    def finalize(self):
        pass
