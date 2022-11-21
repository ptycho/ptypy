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
from ptypy.utils import parallel

try:
    from swmr_tools import KeyFollower

except ImportError:
    log(3,"The SWMR loader requires swmr_tools to be installed, try pip install swmr_tools")

class KeyFollowerV2(KeyFollower):
    def __init__(self, *args, **kwargs):
        self.missing_frames = set()
        self.nonzeros_after_zeros = set()
        self.skipped = set()
        self.block_start = 0

        super().__init__(*args, **kwargs)

    def get_number_frames(self):
        if len(self.skipped):
            print(f"skipping {len(self.skipped)} frames between {min(self.skipped)} and {max(self.skipped)}")
        return self.get_current_max() + 1 - len(self.skipped)

    def get_max_possible(self):
        # this should be the highest number of frames we can expect 
        # once we remove all the known skipped frames
        max_size = max([x.size for x in self._get_keys()])
        return max_size - len(self.skipped)

    def get_framefilter(self, shape = None):
        max_size = max([x.size for x in self._get_keys()])
        flat_array = np.array([i not in self.skipped for i in range(max_size)])
        if shape:
            return flat_array.reshape(shape)
        else:
            return flat_array
        # should be 1d array of bool of skipped keys. 

    def _is_next(self):
        karray = self._get_keys()
        if not karray:
            return False

        if len(karray) == 1:
            merged = karray[0]
        else:
            max_size = max([x.size for x in karray])
            merged = np.zeros(max_size)
            first = karray[0]
            merged[: first.size] = merged[: first.size] + first

            for k in karray[1:]:
                padded = np.zeros(max_size)
                padded[: k.size] = k
                merged = merged * padded

        self.block_start = max(self.block_start, self.get_current_max())
        # each block starts where first element is nonzero???
        remaining = merged[self.block_start:]
        # skips over already checked indices. Might not be good if they are filled in late

        new_block_start = -1
        for idx, m in enumerate(remaining):
            if m != 0 and (remaining[idx - 1] == 0 or idx == 0):
                new_block_start = idx + self.block_start
                # gets set to highest case of nonzero element appearing directly after a zero element
        new_block_start = max(new_block_start, self.get_current_max())
        skipped = np.argwhere(merged[self.block_start:new_block_start] == 0).flatten() + self.block_start
        self.skipped.update(skipped)
        new_nonzeros = np.argwhere(merged[new_block_start:] != 0).flatten() + new_block_start
        new_max = new_nonzeros[-1] if len(new_nonzeros) else self.current_max # bit of a hack, change
        self.block_start = new_block_start


        if new_max < 0 and merged[0] != 0:
            # all keys non zero
            new_max = merged.size - 1

        if self.current_max == new_max:
            return False

        self.current_max = new_max
        return True



@register()
class SwmrLoader(Hdf5Loader):
    """
    This is an attempt to load data from a live SWMR file that is still being written to.

    Defaults:

    [name]
    default = 'SwmrLoader'
    type = str
    help =

    [max_frames]
    default = None
    type = int
    help = Maximum number of frames to load before marking end of scan. Mainly for debugging
    
    
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
        self._use_keyfilter = False if self.p.framefilter else True
        # if no framefilter passed, use key follower to filter frames by index
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

        intensity_file = h5.File(self.p.intensities.file, 'r', swmr=self._is_swmr)
        positions_file = h5.File(self.p.positions.file, 'r', swmr=self._is_swmr)
        self.kf = KeyFollowerV2((intensity_file[self.p.intensities.live_key],
                               positions_file[self.p.positions.live_slow_key],
                               positions_file[self.p.positions.live_fast_key]),
                               timeout=5)

        # Get initial value of maximum number of frames to be loaded before
        # marking scan finished

        self.max_frames = min(f for f in [self.p.max_frames,
                                          self.num_frames,
                                          self.kf.get_max_possible()] if f is not None)

    def get_data_chunk(self, *args, **kwargs):
        self.kf.refresh()
        self.intensities.refresh()
        self.slow_axis.refresh()
        self.fast_axis.refresh()
        # refreshing here to update before Ptyscan.get_data_chunk calls check
        # and load
        self.max_frames = min(self.kf.get_max_possible(), self.max_frames)
        return super().get_data_chunk(*args, **kwargs)


    def load_unmapped_raster_scan(self, *args, **kwargs):
        raise NotImplementedError("framefilter not supported for unmapped raster scans (see hdf5 loader)")
        return super().load_unmapped_raster_scan(*args, **kwargs)

    def load_mapped_and_raster_scan(self, *args, **kwargs):
        if self._use_keyfilter:
            filter_shape = self.intensities.shape[:-2] # bit hacky
            filter = self.kf.get_framefilter(filter_shape)
            skip = self.p.positions.skip
            self.preview_indices = self.unfiltered_indices[:, filter[::skip,::skip].flatten()]
        return super().load_mapped_and_raster_scan(*args, **kwargs)

    def load_mapped_and_arbitrary_scan(self, *args, **kwargs):
        if self._use_keyfilter:
            filter = self.kf.get_framefilter()
            self.preview_indices = self.unfiltered_indices[filter[::self.p.positions.skip]]
        return super().load_mapped_and_arbitrary_scan(*args, **kwargs)

    def check(self, frames=None, start=None):
        """
        Check the live SWMR file for available frames.
        """
        if start is None:
            start = self.framestart

        if frames is None:
            frames = self.min_frames

        available = min(self.kf.get_number_frames(), self.max_frames)
        new_frames = available - start

        if new_frames <= frames:
            # not reached expected nr. of frames,
            # but its last chunk of scan so load it anyway
            if available == self.max_frames:
                frames_accessible = new_frames
                end_of_scan = 1
            # reached expected nr. of frames
            # but first block must be of maximum size (given by frames argument)
            elif start != 0 and new_frames >= self.min_frames:
                frames_accessible = self.min_frames
                end_of_scan = 0
            # not reached required nr. of frames, do nothing
            else:
                end_of_scan = 0
                frames_accessible = 0
        # reached expected nr. of frames
        else:
            frames_accessible = frames
            end_of_scan = 0

        # print(f"{start+frames_accessible}/{self.max_frames} ({frames_accessible} new frames)")
        return frames_accessible, end_of_scan
        


