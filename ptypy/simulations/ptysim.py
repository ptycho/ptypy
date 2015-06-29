# -*- coding: utf-8 -*-
"""
Simulation of ptychographic datasets.

This file is part of the PTYPY package.

    :copyright: Copyright 2014 by the PTYPY team, see AUTHORS.
    :license: GPLv2, see LICENSE for details.
"""
import numpy as np
import os
import time

if __name__ == "__main__":
    from ptypy import utils as u
    from ptypy import resources
    from detector import Detector, conv
    from ptysim_utils import *
    from ptypy.core.data import PtyScan
    from ptypy.core.ptycho import Ptycho
else:
    from .. import utils as u
    from .. import resources
    from detector import Detector, conv
    from ptysim_utils import *
    from ..core.data import PtyScan
    from ..core.ptycho import Ptycho
    
DEFAULT = u.Param(
    pos_noise = 1e-10,  # (float) unformly distributed noise in xy experimental positions
    pos_scale = 0,      # (float, list) amplifier for noise. Will be extended to match number of positions. Maybe used to only put nois on individual points  
    pos_drift = 0,      # (float, list) drift or offset paramter. Noise independent drift. Will be extended like pos_scale.
    detector = 'PILATUS_300K',
    frame_size = None ,   # (None, or float, 2-tuple) final frame size when saving if None, no cropping/padding happens
    psf = None,          # (None or float, 2-tuple, array) Parameters for gaussian convolution or convolution kernel after propagation
                        # use it for simulating partial coherence
)

__all__ = ['SimScan','simulate_basic_with_pods']

def simulate_basic_with_pods(ptypy_pars_tree=None,sim_pars=None,save=False):
    """
    Basic Simulation
    
    DEPRECATED - cannot produce the right .ptyd format
    """
    p = DEFAULT.copy()
    ppt = ptypy_pars_tree
    if ppt is not None:
        p.update(ppt.get('simulation'))
    if sim_pars is not None:
        p.update(sim_pars)
        
    P = Ptycho(ppt,level=1)

    # make a data source that has is basicaly empty
    P.datasource = make_sim_datasource(P.modelm,p.pos_drift,p.pos_scale,p.pos_noise)
    
    P.modelm.new_data()
    u.parallel.barrier()
    P.print_stats()
    
    # Propagate and apply psf for simulationg partial coherence (if not done so with modes)
    for name,pod in P.pods.iteritems():
        if not pod.active: continue
        pod.diff += conv(u.abs2(pod.fw(pod.exit)),p.psf)
    
    # Filter storage data similar to a detector.
    if p.detector is not None:
        Det = Detector(p.detector)
        save_dtype = Det.dtype
        for ID,Sdiff in P.diff.S.items():
            # get the mask storage too although their content will be overriden
            Smask = P.mask.S[ID]
            dat, mask = Det.filter(Sdiff.data)
            if p.frame_size is not None:
                hplanes = u.expect2(p.frame_size)-u.expect2(dat.shape[-2:])
                dat = u.crop_pad(dat,hplanes,axes=[-2,-1]).astype(dat.dtype)
                mask = u.crop_pad(mask,hplanes,axes=[-2,-1]).astype(mask.dtype)
            Sdiff.fill(dat)
            Smask.fill(mask) 
    else:
        save_dtype = None        
    
    if save:
        P.modelm.collect_diff_mask_meta(save=save,dtype=save_dtype)
           
    u.parallel.barrier()
    return P
    


class SimScan(data.PtyScan):
    """
    Test Ptyscan class producing a romantic ptychographic dataset of a moon
    illuminating flowers.
    """
    
    def __init__(self, pars = None,scan_pars=None,**kwargs):

        # Initialize parent class
        super(SimScan, self).__init__(pars, **kwargs)
        
       
        # we will use ptypy to figure out everything
        pp = u.Param()
        
        # we don't want a server
        pp.interaction = None
        
        # be as silent as possible
        pp.verbose_level = 2
       
        # get scan parameters
        if scan_pars is None:
            pp.model = scan_DEFAULT.copy()
        else:
            pp.model = scan_pars.copy()
            
        # note that shape cannot be None
        if self.info.shape is None:
            self.info.shape = pp.model.geometry.shape
        
        rinfo = DEFAULT.copy()
        rinfo.update(self.info.recipe)
        self.rinfo = rinfo
        self.info.recipe = rinfo
        
        # update changes specified in recipe
        pp.model.update(rinfo)

        # Create a Scan that will deliver empty diffraction patterns
        # FIXME: This may be obsolete if the dry_run switch works.
        
        pp.scans=u.Param()
        pp.scans.sim = u.Param()
        pp.scans.sim.data=u.Param()
        pp.scans.sim.data.source ='empty'
        pp.scans.sim.data.shape = pp.model.geometry.shape
        pp.scans.sim.data.auto_center = False
        
        # Now we let Ptycho sort out things
        P=Ptycho(pp,level=2)
        P.modelm.new_data()
        
        u.parallel.barrier()
        
        #############################################################
        # Place here additional manipulation on position and sample #
        
        P = self.manipulate_ptycho(P)
        #############################################################        
        
        # Simulate diffraction signal
        for name,pod in P.pods.iteritems():
            if not pod.active: continue
            pod.diff += conv(u.abs2(pod.fw(pod.exit)),rinfo.psf)

        # Simulate detector reponse
        if rinfo.detector is not None:
            Det = Detector(rinfo.detector)
            save_dtype = Det.dtype
            acquire = Det.filter
        else:
            save_dtype = None
            acquire = lambda x: x
                    
        # create dictionaries for 'raw' data
        self.diff = {}
        self.mask = {}
        self.pos = {}
        
        
        ID,Sdiff = P.diff.S.items()[0]
        for view in Sdiff.views:
            ind = view.layer
            dat, mask = acquire(view.data) 
            view.data = dat
            view.mask = mask
            pos = view.pod.ob_view.physcoord
            dat = dat.astype(save_dtype) if save_dtype is not None else dat
            self.diff[ind] = dat
            self.mask[ind] = mask
            self.pos[ind] = pos

        self.P=P
        
        # Create 'raw' ressource buffers. We will let the master node keep them
        # as memary may be short (Not that this is the most efficient type)
        self.diff = u.parallel.gather_dict(self.diff)
        self.mask = u.parallel.gather_dict(self.mask)
        self.pos = u.parallel.gather_dict(self.pos)
        
        # we have to avoid loading in parallel now
        self.load_in_parallel = False
        
        # Fix the number of available frames
        self.num_frames = np.min(len(self.diff),self.num_frames)
        
        # RESET THE loadmanager
        u.parallel.loadmanager.reset()
        
    
    def load(self,indices):
        """
        Load data, weights and positions from internal dictionarys
        """
        raw = {}
        pos = {}
        weight = {}
        for ind in indices:
            raw[ind] = self.diff[ind]
            pos[ind] = self.pos[ind]
            weight[ind] = self.mask[ind]
        return raw, pos, weight
      
    def manipulate_ptycho(self, ptycho):
        """
        Overwrite in child class for inline manipulation 
        of the ptycho instance that is created by the Simulation
        """
        ptycho.print_stats()
        
        return ptycho
    
if __name__ == "__main__":
    s = scan_DEFAULT.copy()
    s.xy.scan_type = "round_roi"                # (25) None,'round', 'raster', 'round_roi','custom'
    s.xy.dr = 1e-6                             # (26) round,round_roi :width of shell
    s.xy.nr = 10                                # (27) round : number of intervals (# of shells - 1) 
    s.xy.lx = 5e-6                            # (29) round_roi: Width of ROI
    s.xy.ly = 5e-6                            # (30) round_roi: Height of ROI
    shape = 256
    s.geometry.energy = 6.2                    # (17) Energy (in keV)
    s.geometry.lam = None                       # (18) wavelength
    s.geometry.distance = 7                        # (19) distance from object to screen
    s.geometry.psize = 172e-6                # (20) Pixel size in Detector plane
    s.geometry.shape = shape                          # (22) Number of detector pixels
    s.geometry.propagation = "farfield"           # (23) propagation type
    s.illumination.probe = resources.moon_pr((shape,shape))*1e2
    s.sample = u.Param()
    s.sample.obj = resources.flower_obj((shape*2,shape*2))


    u.verbose.set_level(3)
    MS = SimScan(None,s)
    MS.P.plot_overview()
    u.verbose.set_level(3)
    u.pause(10)
    MS.initialize()
    for i in range(20):
        msg = MS.auto(10)
        u.verbose.logger.info(u.verbose.report(msg), extra={'allprocesses': True})
        u.parallel.barrier()
