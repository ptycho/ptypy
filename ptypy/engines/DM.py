# -*- coding: utf-8 -*-
"""
Difference Map reconstruction engine.

This file is part of the PTYPY package.

    :copyright: Copyright 2014 by the PTYPY team, see AUTHORS.
    :license: GPLv2, see LICENSE for details.
"""

#from .. import core
from .. import utils as u
from ..utils.verbose import logger
from ..utils import parallel
from utils import basic_fourier_update
from . import BaseEngine
import numpy as np
import time

__all__=['DM']

parallel = u.parallel

DEFAULT = u.Param(
    fourier_relax_factor = 0.05,
    alpha = 1,
    update_object_first = True,
    overlap_converge_factor = .1,
    overlap_max_iterations = 10,
    #probe_inertia = 1e-9,               # Portion of probe that is kept from iteraiton to iteration, formally cfact
    #object_inertia = 1e-4,              # Portion of object that is kept from iteraiton to iteration, formally DM_smooth_amplitude
    obj_smooth_std = None,              # Standard deviation for smoothing of object between iterations
    clip_object = None,                 # None or tuple(min,max) of desired limits of the object modulus
)


class DM(BaseEngine):
    
    DEFAULT = DEFAULT

    def __init__(self, ptycho_parent, pars=None):
        """
        Difference map reconstruction engine.
        """
        if pars is None:
            pars = DEFAULT.copy()
            
        super(DM,self).__init__(ptycho_parent,pars)
        
    def engine_initialize(self):
        """
        Prepare for reconstruction.
        """
        self.error = []

        # generate container copies
        self.ob_buf = self.ob.copy(self.ob.ID+'_alt',fill=0.) 
        self.ob_nrm = self.ob.copy(self.ob.ID+'_nrm',fill=0.)
        self.ob_viewcover = self.ob.copy(self.ob.ID+'_vcover',fill=0.)

        self.pr_buf = self.pr.copy(self.pr.ID+'_alt',fill=0.)
        self.pr_nrm = self.pr.copy(self.pr.ID+'_nrm',fill=0.)
  
            
    def engine_prepare(self):
        """
        last minute initialization, everything, that needs to be recalculated, when new data arrives
        """
        mxlist = [np.sum(diview.data) if diview.active else self.ptycho.FType(0.0) for diview in self.di.V.values()]
        self.max_power = parallel.MPImax(mxlist) 
        self.pbound = .25 *  self.p.fourier_relax_factor**2 * self.max_power / self.di.S.values()[0].shape[-1]**2
        
        # fill object with coverage of views
        for name,s in self.ob_viewcover.S.iteritems():
            s.fill(s.get_view_coverage())

    def engine_iterate(self):
        """
        Compute one iteration.
        """
        
        error_dct={}
        t = time.time() 
        # Fourier update  
        for name,di_view in self.di.V.iteritems():
            ma_view = di_view.pod.ma_view 
            error_dct[name] = basic_fourier_update(di_view, ma_view, pbound=self.pbound)
        logger.info('Time spent in Fourier update: %.2f' % (time.time()-t))    

        ## make a sorted error array for MPI reduction
        ## error is sorted after the di_view IDs. This is needed for local error analysis later.
        error = np.array([error_dct[k] for k in np.sort(error_dct.keys())])
        error[error<0]=0.
        parallel.allreduce(error)
        
        # store error. maybe better in runtime?
        self.error = error
        self.overlap_update()
       
        return error

    def engine_finalize(self):
        """
        try deleting ever helper contianer
        """
        containers = [
        self.ob_buf, 
        self.ob_nrm, 
        self.ob_viewcover, 
        self.pr_buf,
        self.pr_nrm ]
        #IDM = self.ptycho.IDM_container
        for c in containers:
            logger.debug('Attempt to remove container %s' %c.ID)
            del self.ptycho.containers[c.ID]
        #    IDM.used.remove(c.ID)
        
        del self.ob_buf
        del self.ob_nrm 
        del self.ob_viewcover 
        del self.pr_buf
        del self.pr_nrm

        del containers
        
    def overlap_update(self):
        """
        DM overlap constraint update.
        """
        change = 1.
        # Condition to update probe
        do_update_probe = (self.p.probe_update_start <= self.curiter)
         
        for inner in range(self.p.overlap_max_iterations):
            prestr = 'Iteration (Overlap) #%02d:  ' % inner
            
            # Update object first
            if self.p.update_object_first or (inner > 0):
                # Update object
                logger.debug(prestr + '----- object update -----')
                self.object_update()
                               
            # Exit if probe should not yet be updated
            if not do_update_probe: break
            
            # Update probe
            logger.debug(prestr + '----- probe update -----')
            change = self.probe_update()
            logger.info(prestr + 'change in probe is %.3f' % change)
            
            # stop iteration if probe change is small
            if change < self.p.overlap_converge_factor: break
            


    def object_update(self):
        """
        DM object update.
        """
        ob = self.ob
        ob_nrm = self.ob_nrm
        
        ### prefill container 
        if not parallel.master:
            ob.fill(0.0)
            ob_nrm.fill(0.)
        else:
            for name,s in self.ob.S.iteritems():
                # in original code:
                # DM_smooth_amplitude = (p.DM_smooth_amplitude * max_power * p.num_probes * Ndata) / np.prod(asize)
                # using the number of views here, but don't know if that is good.
                cfact = self.p.object_inertia * len(s.views)
                #cfact = self.ob_viewcover.S[name].data +1e-10
                
                if self.p.obj_smooth_std is not None:
                    logger.info('Smoothing object, cfact is %.2f' % cfact)
                    smooth_mfs = [0,self.p.obj_smooth_std,self.p.obj_smooth_std]
                    s.data[:] = cfact * u.c_gf(s.data,smooth_mfs) 
                else:
                    s.data[:] =  s.data * cfact
                    
                ob_nrm.S[name].fill(cfact)
        
        ### DM update per node
        for name,pod in self.pods.iteritems():
            if not pod.active: continue
            pod.object += pod.probe.conj() * pod.exit
            ob_nrm[pod.ob_view] += u.cabs2(pod.probe)
        
        ### distribute result with MPI
        for name,s in self.ob.S.iteritems():
            # get the np arrays
            nrm = ob_nrm.S[name].data
            parallel.allreduce(s.data)
            parallel.allreduce(nrm)
            s.data /= nrm
                
            # clip object
            if self.p.clip_object is not None:
                clip_min,clip_max = self.p.clip_object
                aobj = np.abs(s.data);
                phobj = np.exp(1j* np.angle(s.data))
                too_high = (aobj > clip_max)
                too_low = (aobj < clip_min)
                s.data[too_high] = clip_max*phobj[too_high]
                s.data[too_low] = clip_min*phobj[too_low]
                
    def probe_update(self):
        """
        DM probe update.
        """
        pr = self.pr
        pr_nrm = self.pr_nrm
        pr_buf = self.pr_buf
        
        ### prefill container 
        # "cfact" fill
        # BE: was this asymmetric in original code only because of the number of MPI nodes ?
        if parallel.master:
            for name,s in pr.S.iteritems():
                # instead of Npts_scan, the number of views should be considered
                # please note that a call to s.views maybe slow for many views in the probe.
                cfact = self.p.probe_inertia * len(s.views) / s.data.shape[0]
                s.data[:]= cfact * s.data
                pr_nrm.S[name].fill(cfact)
        else:
            pr.fill(0.0)
            pr_nrm.fill(0.0)

        ### DM update per node
        for name,pod in self.pods.iteritems():
            if not pod.active: continue
            pod.probe += pod.object.conj() * pod.exit
            pr_nrm[pod.pr_view] += u.cabs2(pod.object)

        change = 0.
        
        ### distribute result with MPI
        for name,s in pr.S.iteritems():
            # MPI reduction of results
            nrm = pr_nrm.S[name].data
            parallel.allreduce(s.data)
            parallel.allreduce(nrm)
            s.data /= nrm
            
            # apply probe support if requested
            support = self.probe_support.get(name)
            if support is not None: 
                s.data *= self.probe_support[name]

            # compute relative change in probe
            buf = pr_buf.S[name].data
            change += u.norm2(s.data-buf) / u.norm2(s.data)

            # fill buffer with new probe
            buf[:] = s.data

            
        return np.sqrt(change/len(pr.S))

    


