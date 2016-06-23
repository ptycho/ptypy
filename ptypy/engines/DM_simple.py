# -*- coding: utf-8 -*-
"""
Difference Map reconstruction engine.

This file is part of the PTYPY package.

    :copyright: Copyright 2014 by the PTYPY team, see AUTHORS.
    :license: GPLv2, see LICENSE for details.
"""
import numpy as np
#from .. import core
from .. import utils as u
from engine_utils import basic_fourier_update
from dummy import BaseEngine

parallel = u.parallel

DEFAULT = u.Param(
    fourier_relax_factor = 0.01,
    alpha = 1.0
)

__all__=['DM_simple']

class DM_simple(BaseEngine):
    
    DEFAULT = DEFAULT

    def __init__(self, ptycho_parent, pars=None):
        """
        Simplest possible Difference map reconstruction engine.
        """
        super(DM_simple,self).__init__(ptycho_parent,pars)
        
    def engine_initialize(self):
        """
        Prepare for reconstruction.
        """
        self.error =  []

        # generate container copies
        self.ob_nrm = self.ob.copy(self.ob.ID+'nrm',fill=0.)
        self.pr_nrm = self.pr.copy(self.pr.ID+'nrm',fill=0.)

        self.max_power = parallel.MPImax([np.sum(diview.data) if diview.active else None for diview in self.di.V.values()]) 
        self.pbound = .25 *  self.p.fourier_relax_factor**2 * self.max_power / self.di.S.values()[0].shape[-1]**2
        
    def engine_iterate(self):
        """
        Compute one iteration.
        """
        
        error_dct={}
        
        #fourier update  
        for name,di_view in self.di.V.iteritems():
            ma_view = di_view.pods.values()[0].ma_view # a bit ugly
            error_dct[name] = basic_fourier_update(di_view,ma_view,pbound=self.pbound,alpha=self.p.alpha)
        # make a sorted error array for MPI reduction
        error = np.array([error_dct[k] for k in np.sort(error_dct.keys())])
        #print error.shape
        error[error<0]=0.
        parallel.allreduce(error)
        # store error. maybe better in runtime?
        self.error = error
        #self.pty.runtime.error.append(error)
        # probe update
        self.probe_update(self.pr_nrm,self.pr_buf)
        # object update
        self.object_update(self.ob_nrm,self.ob_buf)
        
        return error
        
    def object_update(self,ob_nrm,ob_buf):
        """
        DM object update. Requires 2 buffer deck copies
        ob_nrm, ob_buf
        """
        # Object update
        for name,s in self.ob.S.iteritems():
            ob_buf.S[name].fill(0.) 
            ob_nrm.S[name].fill(0.) 
            
        for name,pod in self.pods.iteritems():
            if not pod.active: continue
            # pod.ob_view[ob1] += pod.pr * pod.psi.conj()
            ob_buf[pod.ob_view] += pod.probe.conj() * pod.exit
            ob_nrm[pod.ob_view] += u.abs2(pod.probe)
            
        for name,s in self.ob.S.iteritems():
            # get the np arrays
            d1=ob_buf.S[name].data
            d2=ob_nrm.S[name].data
            parallel.allreduce(d1)
            parallel.allreduce(d2)
            s.data[:] = d1/(d2+1e-10)

    def probe_update(self,pr_nrm,pr_buf):
        """
        DM probe update. Requires 2 buffer deck copies
        pr_nrm, pr_buf
        """
        for name,s in self.pr.S.iteritems():
            pr_buf.S[name].fill(0.0) 
            pr_nrm.S[name].fill(0.)
            
        for name,pod in self.pods.iteritems():
            if not pod.active: continue
            pr_buf[pod.pr_view] += pod.object.conj() * pod.exit
            pr_nrm[pod.pr_view] += u.abs2(pod.object)

        for name,s in self.pr.S.iteritems():
            d1=pr_buf.S[name].data
            d2=pr_nrm.S[name].data
            parallel.allreduce(d1)
            parallel.allreduce(d2)
            s.data[:] = d1/(d2+1e-10)

