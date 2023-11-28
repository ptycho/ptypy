# -*- coding: utf-8 -*-
"""\
Scan loading recipe for the Diamond beamlines.

This file is part of the PTYPY package.

    :copyright: Copyright 2014 by the PTYPY team, see AUTHORS.
    :license: see LICENSE for details.
"""
import h5py as h5
import numpy as np

from ptypy.experiment import register
from ptypy.experiment.hdf5_loader import Hdf5Loader
from ptypy.utils.verbose import log

try:
    from swmr_tools import KeyFollower

except ImportError:
    log(3, "The SWMR loader requires swmr_tools to be installed,"
           " try pip install swmr_tools")
    raise ImportError


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
    doc = Live_keys indicate where the data collection has progressed to.
          They are zero at the scan start, but non-zero when the position
          is complete.

    [positions.live_fast_key]
    default = None
    type = str
    help = Key to live key for fast axis inside the positions file
    doc = Live_keys indicate where the data collection has progressed to.
          They are zero at the scan start, but non-zero when the position
          is complete.

    [positions.live_slow_key]
    default = None
    type = str
    help = Key to live key for slow axis inside the positions file
    doc = Live_keys indicate where the data collection has progressed to.
          They are zero at the scan start, but non-zero when the position
          is complete.

    [positions.fast_key_with_expected_shape]
    default = None
    type = str
    help = Key for fast axis inside the positions file with expected shape
    doc = The shape of this key entry is used to estimate the expected 
          scan trajectory mapping and the total nr. of expected frames. 

    [positions.slow_key_with_expected_shape]
    default = None
    type = str
    help = Key for slow axis inside the positions file with expected shape
    doc = The shape of this key entry is used to estimate the expected 
          scan trajectory mapping and the total nr. of expected frames. 
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, swmr=True, **kwargs)

    def _params_check(self):
        super()._params_check()

        # Check if we have been given the live keys
        if None in [self.p.intensities.live_key,
                    self.p.positions.live_slow_key,
                    self.p.positions.live_fast_key]:
            raise RuntimeError("Missing live keys to intensities or positions")

        # Check that intensities and positions (and their live keys)
        # are loaded from the same file
        if self.p.intensities.file != self.p.positions.file:
            raise RuntimeError("Intensities and positions file should be same")
        
    def _prepare_intensity_and_positions(self):
        super()._prepare_intensity_and_positions()
        self.positions_slow_shape = self.fhandle_positions_slow[self.p.positions.slow_key_with_expected_shape].shape
        self.positions_fast_shape = self.fhandle_positions_slow[self.p.positions.fast_key_with_expected_shape].shape
        if len(self.data_shape[:-2]) == 2:
            self.data_shape = self.positions_slow_shape + self.positions_fast_shape + tuple(np.array(self.data_shape)[-2:])
        elif len(self.data_shape[:-2]) == 1:
            self.data_shape = (self.positions_slow_shape[0],) + tuple(np.array(self.data_shape)[-2:])
        print("self.data_shape", self.data_shape)
        print("self.positions_slow_shape", self.positions_slow_shape)
        print("self.positions_fast_shape", self.positions_fast_shape)
        self.kf = KeyFollower((self.fhandle_intensities[self.p.intensities.live_key],
                               self.fhandle_positions_slow[self.p.positions.live_slow_key],
                               self.fhandle_positions_fast[self.p.positions.live_fast_key]),
                               timeout=5)
        
    def compute_scan_mapping_and_trajectory(self,*args):
        super().compute_scan_mapping_and_trajectory(*args)
        #assert isinstance(self.slow_axis, h5.Dataset), "Scantype = {:s} and mapped={:} is not compatible with the SwmrLoader".format(self._scantype, self._ismapped)

    def get_data_chunk(self, *args, **kwargs):
        self.kf.refresh()
        self.intensities.refresh()
        try:
            self.slow_axis.refresh()
            self.fast_axis.refresh()
        except AttributeError:
            print("Can't refresh position keys")
        # refreshing here to update before Ptyscan.get_data_chunk calls check and load
        return super().get_data_chunk(*args, **kwargs)

    def check(self, frames=None, start=None):
        """
        Check the live SWMR file for available frames.
        """
        if start is None:
            start = self.framestart

        if frames is None:
            frames = self.min_frames

        available = min(self.kf.get_current_max() + 1, self.num_frames)
        new_frames = available - start
        # not reached expected nr. of frames
        if new_frames <= frames:
            # but its last chunk of scan so load it anyway
            if available == self.num_frames:
                frames_accessible = new_frames
                end_of_scan = 1
            # otherwise, do nothing
            else:
                end_of_scan = 0
                frames_accessible = 0
        # reached expected nr. of frames
        else:
            frames_accessible = frames
            end_of_scan = 0

        return frames_accessible, end_of_scan
