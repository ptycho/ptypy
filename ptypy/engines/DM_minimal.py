# -*- coding: utf-8 -*-
"""
Difference Map reconstruction engine.

This file is part of the PTYPY package.

    :copyright: Copyright 2014 by the PTYPY team, see AUTHORS.
    :license: GPLv2, see LICENSE for details.
"""

from .. import utils as u
from utils import basic_fourier_update
import numpy as np
import time

DEFAULT = u.Param(
    numiter = 100,
    numiter_contiguous = 1,
    fourier_relax_factor = 0.01,
    alpha = 1.0,    
    probe_inertia = 0.001,             
    object_inertia = 0.1,              
)

__all__=['DM_minimal']

class DM_minimal(object):
    
    DEFAULT = DEFAULT

    def __init__(self, ptycho, pars=None):
        """
        Simple difference map reconstruction engine.
        
        Parameters
        ----------
        ptycho : Ptycho 
            The parent :any:`Ptycho` object.
            
        pars: Param or dict
            Initialization parameters
        """
        self.ptycho = ptycho_parent
        p = u.Param(self.DEFAULT)

        if pars is not None: p.update(pars)
        self.p = p
        
    def initialize(self):
        """
        Prepare for reconstruction.
        """
        self.curiter = 0
        self.finished = False
                
        # common attributes for all reconstructions
        self.ob = self.ptycho.obj
        self.pr = self.ptycho.probe

        # generate container copies
        self.ob_nrm = self.ob.copy(self.ob.ID+'nrm',fill=0.)
        self.pr_nrm = self.pr.copy(self.pr.ID+'nrm',fill=0.)

    
    def prepare(self):
        """
        Last-minute preparation before iterating.
        """
        self.finished = False
        for s in self.ptycho.diff.S.values():
            s.pbound = .25 *  self.p.fourier_relax_factor**2 * s.max_power / s.shape[-1]**2
            
    def iterate(self, num=None):
        """
        Compute one or several iterations.
        
        :param num: None,int number of iterations. 
        """
        # several iterations 
        N = self.p.numiter_contiguous if num is None else num
        
        for n in range(N):
            if self.finished: break 
            # for benchmarking
            self.t = time.time()
            
            self.errors = {}
            
            ###### fourier update #################  
            for name,di_view in self.ptycho.diff.V.iteritems():
                if not di_view.active: continue
                pbound = di_view.storage.pbound
                self.errors[name] = basic_fourier_update(di_view,pbound,alpha=self.p.alpha)
            
            ###### probe update  ##################
            self.pr *= self.p.probe_inertia
            self.pr_nrm << self.p.probe_inertia + 1e-10
                
            for name,pod in self.ptycho.pods.iteritems():
                if not pod.active: continue
                self.pr[pod.pr_view] += pod.object.conj() * pod.exit
                self.pr_nrm[pod.pr_view] += pod.object * pod.object.conj()
            
            # Mpi sum
            self.pr.allreduce()
            self.pr_nrm.allreduce()
            
            self.pr /= self.pr_nrm
            
            ###### object update ##################
            self.ob *= self.p.object_inertia
            self.ob_nrm << self.p.object_inertia + 1e-10
                
            for pod in self.ptycho.pods.itervalues():
                if not pod.active: continue
                self.ob[pod.ob_view] += pod.probe.conj() * pod.exit
                self.ob_nrm[pod.ob_view] += pod.probe * pod.probe.conj()
                
            # Mpi sum
            self.ob.allreduce()
            self.ob_nrm.allreduce()
            
            self.ob /= self.ob_nrm
            
                       
            self.curiter += 1
            self.finished = (self.curiter >= self.p.numiter)
            
            self._fill_runtime()
            u.parallel.barrier()

    def finalize(self):
        """ cleanup """
        pass
        
    def _fill_runtime(self):
        error = u.parallel.gather_dict(self.errors)
        
        info = dict(
            iteration = self.curiter,
            iterations = len(self.ptycho.runtime.iter_info),
            engine = self.__class__.__name__,
            duration = time.time()-self.t,
            error = error
            )

        self.ptycho.runtime.iter_info.append(info)
    

                
    def _object_update(self,ob,norm,cfact=0.5):
        """
        DM object update. 
        
        :param ob: the object (sample) container
        :param norm: a copy of `ob`
        :param cfact: fraction of `ob` to keep between iterations 
        """
        # Object update
        ob *= cfact
        norm << cfact + 1e-10
            
        for pod in self.pods.itervalues():
            if not pod.active: continue
            ob[pod.ob_view] += pod.probe.conj() * pod.exit
            norm[pod.ob_view] += pod.probe * pod.probe.conj()
            
        # Mpi sum
        C_allreduce(ob)
        C_allreduce(norm)
        
        ob /= norm

    def _probe_update(self,pr,norm,cfact=0.5):
        """
        DM probe update. 
        
        :param pr: the probe (illumination) container
        :param norm: a copy of `pr`
        :param cfact: fraction of `pr` to keep between iterations 
        """
        pr *= cfact
        norm << cfact + 1e-10
            
        for name,pod in self.pods.iteritems():
            if not pod.active: continue
            pr[pod.pr_view] += pod.object.conj() * pod.exit
            norm[pod.pr_view] += pod.object * pod.object.conj()

        # Mpi sum
        C_allreduce(pr)
        C_allreduce(norm)
        
        pr /= nrm


