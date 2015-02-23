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

from ptypy import utils as u
import ptypy
from detector import Detector, conv
from ptysim_utils import *
from ptypy.core.data import Ptyscan

DEFAULT = u.Param(
    pos_noise = 1e-10,  # (float) unformly distributed noise in xy experimental positions
    pos_scale = 0,      # (float, list) amplifier for noise. Will be extended to match number of positions. Maybe used to only put nois on individual points  
    pos_drift = 0,      # (float, list) drift or offset paramter. Noise independent drift. Will be extended like pos_scale.
    detector = 'PILATUS_300K',
    frame_size = None ,   # (None, or float, 2-tuple) final frame size when saving if None, no cropping/padding happens
    psf = None,          # (None or float, 2-tuple, array) Parameters for gaussian convolution or convolution kernel after propagation
                        # use it for simulating partial coherence
)

__all__ = ['simulate_basic_with_pods']

def simulate_basic_with_pods(ptypy_pars_tree=None,sim_pars=None,save=False):
    """
    Basic Simulation
    """
    p = DEFAULT.copy()
    ppt = ptypy_pars_tree
    if ppt is not None:
        p.update(ppt.get('simulation'))
    if sim_pars is not None:
        p.update(sim_pars)
        
    P = ptypy.core.Ptycho(ppt,level=1)

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
    
class SimulatedScan(Ptyscan):
    """
    Test Ptyscan class producing a romantic ptychographic dataset of a moon
    illuminating flowers.
    """
    
    def __init__(self, pars = None, recipe=None **kwargs):

        p = u.Param(self.DEFAULT.copy()
        if pars is not None:
            p.update(pars)
        # Initialize parent class
        super(SimulatedScan, self).__init__(pars, **kwargs)
        
        from ptypy.core.manager import scan_DEFAULT
        pp = u.Param()
        pp.interaction = None
        pp.verbose_level = 1
        pp.data_type = 'single'
        pp.model = scan_DEFAULT.copy(depth=4)
        pp.model.update(recipe)
        pp.scans=u.Param()
        pp.scans.sim = u.Param()
        pp.scans.sim.data=u.Param()
        pp.scans.sim.data.soure ='empty'
        P=Ptycho(pp,level=2)
        P.modelm.new_data()
        u.parallel.barrier()
        P.print_stats()
                
        self.P=P
        
    def load_common(self):
        """
        Transmit positions
        """
        return {'weight2d': np.ones(self.pr.shape),
                'positions_scan': self.pos}

    def load(self, indices):
        """
        Forward propagation
        """
        # dummy fill
        p=self.pixel
        s=self.G.shape
        raw = {}
        for i in indices:
            raw[i]=np.random.poisson(u.abs2(self.G.propagator.fw(self.pr * self.obj[p[i][0]:p[i][0]+s[0],p[i][1]:p[i][1]+s[1]]))).astype(np.int32)
        return raw, {}, {}

