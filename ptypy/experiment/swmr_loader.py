# -*- coding: utf-8 -*-
"""\
Scan loading recipe for the Diamond beamlines.

This file is part of the PTYPY package.

    :copyright: Copyright 2014 by the PTYPY team, see AUTHORS.
    :license: GPLv2, see LICENSE for details.
"""
import h5py as h5
import numpy as np

from ptypy.experiment import register
from ptypy.experiment.hdf5_loader import Hdf5Loader
from ptypy.utils.verbose import log

try:
    from swmr_tools import KeyFollower
except ImportError:
    log(3,"The SWMR loader requires swmr_tools to be installed, try pip install swmr_tools")

@register()
class SwmrLoader(Hdf5Loader):
    """
    This is an attempt to load data from a live SWMR file that is still being written to.

    Defaults:

    [name]
    default = 'SwmrLoader'
    type = str
    help =
    
    [intensities.live_key]
    default = None
    type = str
    help = Key to live keys inside the intensities file
    doc = Live_keys indicate where the data collection has progressed to. They are zero at the 
          scan start, but non-zero when the position is complete.

    [positions.live_fast_key]
    default = None
    type = str
    help = Key to live keys inside the positions file
    doc = Live_keys indicate where the data collection has progressed to. They are zero at the 
          scan start, but non-zero when the position is complete.

    [positions.live_slow_key]
    default = None
    type = str
    help = Key to live keys inside the positions file
    doc = Live_keys indicate where the data collection has progressed to. They are zero at the 
          scan start, but non-zero when the position is complete.

    """
    def __init__(self, pars=None, **kwargs):
        super().__init__(pars=pars, **kwargs)

    def setup(self, *args, **kwargs):
        self._is_swmr = True
        super().setup(*args, **kwargs)

        # Check if we have been given the live keys
        if None in [self.p.intensities.live_key,
                    self.p.positions.live_slow_key,
                    self.p.positions.live_fast_key]:
            raise RuntimeError("Missing the live keys to intensities and positions!")

        # Check that intensities and positions (and their live keys) are loaded from the same file
        if self.p.intensities.file != self.p.positions.file:
            raise RuntimeError("Intensities and positions file should be the same")

        # Initialize KeyFollower
        self.kf = KeyFollower(h5.File(self.p.intensities.file, 'r', swmr=self._is_swmr), 
                             [self.p.intensities.live_key, 
                              self.p.positions.live_fast_key, 
                              self.p.positions.live_slow_key], 
                             timeout = 5)

    @property
    def num_frames_available(self):
        self.kf.refresh()
        return self.kf.get_current_max() + 1

    def load_unmapped_raster_scan(self, *args, **kwargs):
        self.intensities.refresh()
        self.slow_axis.refresh()
        self.fast_axis.refresh()
        return super().load_unmapped_raster_scan(*args, **kwargs)

    def load_mapped_and_raster_scan(self, *args, **kwargs):
        self.intensities.refresh()
        self.slow_axis.refresh()
        self.fast_axis.refresh()
        return super().load_mapped_and_raster_scan(*args, **kwargs)

    def load_mapped_and_arbitrary_scan(self, *args, **kwargs):
        self.intensities.refresh()
        self.slow_axis.refresh()
        self.fast_axis.refresh()
        return super().load_mapped_and_arbitrary_scan(*args, **kwargs)

    
    def check(self, frames=None, start=None):
        """
        Check the live SWMR file for available frames.
        """
        if start is None:
            start = self.framestart

        if frames is None:
            frames = self.min_frames

        _nframes = self.num_frames_available

        new_frames = _nframes - start
        if new_frames < frames: 
            # not reached expected nr. of frames, 
            # but its last chunk of scan so load it anyway
            if _nframes >= self.num_frames:
                frames_accessible = new_frames
                end_of_scan = 1
            # not reached expected nr. of frames, do nothing
            else:
                end_of_scan = 0
                frames_accessible = 0
        # reached expected nr. of frames
        else: 
            frames_accessible = frames
            end_of_scan = 0

        return frames_accessible, end_of_scan
        


