# -*- coding: utf-8 -*-
"""
Simulation of ptychographic datasets.

This file is part of the PTYPY package.

    :copyright: Copyright 2014 by the PTYPY team, see AUTHORS.
    :license: see LICENSE for details.
"""
import numpy as np
import os
import time

if __name__ == "__main__":
    from ptypy import utils as u
    from .detector import Detector, conv
    from ptypy.core.data import PtyScan
    from ptypy.core.ptycho import Ptycho
    from ptypy.core.manager import Full as ScanModel
    from ptypy.core.sample import sample_desc
    from ptypy.core.illumination import illumination_desc
    from ptypy.core import xy
    from ptypy import defaults_tree
else:
    from .. import utils as u
    from .detector import Detector, conv
    from ..core.data import PtyScan
    from ..core.ptycho import Ptycho
    from ..core.manager import Full as ScanModel
    from .. import defaults_tree
    from ..core.sample import sample_desc
    from ..core.illumination import illumination_desc
    from ..core import xy

logger = u.verbose.logger

__all__ = ['SimScan']


defaults_tree['scandata'].add_child(u.descriptor.EvalDescriptor('SimScan'))
defaults_tree['scandata.SimScan'].add_child(illumination_desc, copy=True)
defaults_tree['scandata.SimScan'].add_child(sample_desc, copy=True)
defaults_tree['scandata.SimScan'].add_child(xy.xy_desc, copy=True)
@defaults_tree.parse_doc('scandata.SimScan')
class SimScan(PtyScan):
    """
    Simulates a ptychographic scan and acts as Virtual data source.

    Defaults:

    [name]
    default = 'SimScan'
    type = str
    help =

    [pos_noise]
    default =  1e-10
    type = float
    help = Uniformly distributed noise in xy experimental positions

    [pos_scale]
    default = 0.
    type = float, list
    help = Amplifier for noise.
    doc = Will be extended to match number of positions. Maybe used to only put nois on individual points

    [pos_drift]
    default = 0.
    type = float, list
    help = Drift or offset paramter
    doc = Noise independent drift. Will be extended like pos_scale.

    [detector]
    default = 'PILATUS_300K'
    type = str, Param
    help =

    [frame_size]
    default =
    type = float, tuple
    help = Final frame size when saving
    doc = If None, no cropping/padding happens.

    [psf]
    default =
    type = float, tuple, array
    help = Parameters for gaussian convolution or convolution kernel after propagation
    doc = Use it for simulating partial coherence.

    [verbose_level]
    default = 1
    type = int
    help = Verbose level when simulating

    [plot]
    default = True
    type = bool
    help =

    [propagation]
    default = farfield
    type = str
    help = farfield or nearfield

    """

    def __init__(self, pars=None, **kwargs):
        """
        Parameters
        ----------
        pars : Param
            PtyScan parameters. See :py:data:`scandata.SimScan`.

        """

        p = self.DEFAULT.copy(99)
        p.update(pars)

        # Initialize parent class
        super(SimScan, self).__init__(p, **kwargs)

        # we will use ptypy to figure out everything
        pp = u.Param()

        # we don't want a server
        pp.io = u.Param()
        pp.io.interaction = None

        # be as silent as possible
        self.verbose_level = u.verbose.get_level()
        pp.verbose_level = p.verbose_level

        # Create a Scan that will deliver empty diffraction patterns
        # FIXME: This may be obsolete if the dry_run switch works.

        pp.scans=u.Param()
        pp.scans.sim = u.Param()
        pp.scans.sim.name = 'Full'
        pp.scans.sim.propagation = self.info.propagation
        pp.scans.sim.data=u.Param()
        pp.scans.sim.data.positions_theory = xy.from_pars(self.info.xy)
        pp.scans.sim.data.name = 'PtyScan'
        pp.scans.sim.data.shape = self.info.shape
        pp.scans.sim.data.psize = self.info.psize
        pp.scans.sim.data.energy = self.info.energy
        pp.scans.sim.data.distance = self.info.distance
        pp.scans.sim.sample = self.info.sample
        pp.scans.sim.illumination = self.info.illumination

        pp.scans.sim.data.auto_center = False

        # Now we let Ptycho sort out things
        logger.info('Generating simulating Ptycho instance for scan `%s`.' % str(self.info.get('label')))
        P=Ptycho(pp,level=2)
        P.model.new_data()
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
        for name,pod in P.pods.items():
            if not pod.active: continue
            pod.diff += conv(u.abs2(pod.fw(pod.exit)), self.info.psf)

        # Simulate detector reponse
        if self.info.detector is not None:
            Det = Detector(self.info.detector)
            save_dtype = Det.dtype
            acquire = Det.filter
        else:
            save_dtype = None
            acquire = lambda x: (x, np.ones(x.shape).astype(bool))

        # create dictionaries for 'raw' data
        self.diff = {}
        self.mask = {}
        self.pos = {}


        ID,Sdiff = list(P.diff.S.items())[0]
        logger.info('Collecting simulated `raw` data.')
        for view in Sdiff.views:
            ind = view.layer
            dat, mask = acquire(view.data)
            view.data = dat
            #view.mask = mask
            pos = np.array(view.pod.ob_view.coord)
            dat = dat.astype(save_dtype) if save_dtype is not None else dat
            self.diff[ind] = dat
            self.mask[ind] = mask
            self.pos[ind] = pos

        # plot overview
        if self.info.plot and u.parallel.master:
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

# if __name__ == "__main__":
#     from ptypy import resources
    
#     s = scan_DEFAULT.copy()
#     s.xy.model = "round"
#     s.xy.spacing = 1e-6
#     s.xy.steps = 8
#     s.xy.extent = 5e-6 
#     shape = 256
#     s.geometry.energy = 6.2
#     s.geometry.lam = None
#     s.geometry.distance = 7
#     s.geometry.psize = 172e-6
#     s.geometry.shape = shape
#     s.geometry.propagation = "farfield"
#     s.illumination = resources.moon_pr((shape,shape))*1e2
#     s.sample =  resources.flower_obj((shape*2,shape*2))


#     u.verbose.set_level(3)
#     MS = SimScan(None,s)
#     #MS.P.plot_overview()
#     u.verbose.set_level(3)
#     u.pause(10)
#     MS.initialize()
#     for i in range(20):
#         msg = MS.auto(10)
#         u.verbose.logger.info(u.verbose.report(msg), extra={'allprocesses': True})
#         u.parallel.barrier()
