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
    from detector import Detector, conv
    from ptypy.core.data import PtyScan
    from ptypy.core.ptycho import Ptycho
    from ptypy.core.manager import scan_DEFAULT
else:
    from .. import utils as u
    from detector import Detector, conv
    from ..core.data import PtyScan
    from ..core.ptycho import Ptycho
    from ..core.manager import scan_DEFAULT

logger = u.verbose.logger

DEFAULT = u.Param(
    pos_noise = 1e-10,  # (float) unformly distributed noise in xy experimental positions
    pos_scale = 0,      # (float, list) amplifier for noise. Will be extended to match number of positions. Maybe used to only put nois on individual points  
    pos_drift = 0,      # (float, list) drift or offset paramter. Noise independent drift. Will be extended like pos_scale.
    detector = 'PILATUS_300K',
    frame_size = None ,   # (None, or float, 2-tuple) final frame size when saving if None, no cropping/padding happens
    psf = None,          # (None or float, 2-tuple, array) Parameters for gaussian convolution or convolution kernel after propagation
                        # use it for simulating partial coherence
    verbose_level = 1, # verbose level when simulating
    plot = True,
)

__all__ = ['SimScan']
  

class SimScan(PtyScan):
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
       
        # get scan parameters
        if scan_pars is None:
            pp.scan = scan_DEFAULT.copy(depth =4)
        else:
            pp.scan = scan_pars.copy(depth =4)
            
        # note that shape cannot be None
        if self.info.shape is None:
            self.info.shape = pp.scan.geometry.shape
        
        rinfo = DEFAULT.copy()
        rinfo.update(self.info.recipe, in_place_depth = 4)
        self.rinfo = rinfo
        self.info.recipe = rinfo
        
        # be as silent as possible
        self.verbose_level = u.verbose.get_level()
        pp.verbose_level = rinfo.verbose_level
        
        # update changes specified in recipe
        pp.scan.update(rinfo, in_place_depth = 4)

        # Create a Scan that will deliver empty diffraction patterns
        # FIXME: This may be obsolete if the dry_run switch works.
        
        pp.scans=u.Param()
        pp.scans.sim = u.Param()
        pp.scans.sim.data=u.Param()
        pp.scans.sim.data.source = 'empty'
        pp.scans.sim.data.shape = pp.scan.geometry.shape
        pp.scans.sim.data.auto_center = False
        # deactivate sharing since we create a seperate Ptycho instance fro each scan
        pp.scans.sim.sharing = None
        
        # Now we let Ptycho sort out things
        logger.info('Generating simulating Ptycho instance for scan `%s`.' % str(self.info.get('label')))
        P=Ptycho(pp,level=2)
        P.modelm.new_data()
        u.parallel.barrier()
        
        # Be now as verbose as before
        u.verbose.set_level(self.verbose_level )
        
        #############################################################
        # Place here additional manipulation on position and sample #
        logger.info('Calling inline manipulation function.')
        P = self.manipulate_ptycho(P)
        #############################################################        
        
        # Simulate diffraction signal
        logger.info('Propagating exit waves.')
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
        logger.info('Collectiong simulated `raw` data.')
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

        # plot overview
        if self.rinfo.plot and u.parallel.master:
            logger.info('Plotting simulation overview')
            P.plot_overview(200)
            u.pause(5.)
        u.parallel.barrier()
        
        #self.P=P
        # Fix the number of available frames
        num = np.array([len(self.diff)])
        u.parallel.allreduce(num)
        self.num_frames = np.min((num[0],self.num_frames)) if self.num_frames is not None else num[0]
        logger.info('Setting frame count to %d.' %self.num_frames)
        # Create 'raw' ressource buffers. We will let the master node keep them
        # as memary may be short (Not that this is the most efficient type)
        logger.debug('Gathering data at master node.')
        self.diff = u.parallel.gather_dict(self.diff)
        self.mask = u.parallel.gather_dict(self.mask)
        self.pos = u.parallel.gather_dict(self.pos)
        
        # we have to avoid loading in parallel now
        self.load_in_parallel = False
        
        
        # RESET THE loadmanager
        logger.debug('Resetting loadmanager().')
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
        #ptycho.print_stats()
        
        return ptycho
    
if __name__ == "__main__":
    from ptypy import resources

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
