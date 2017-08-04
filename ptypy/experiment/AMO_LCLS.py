# -*- coding: utf-8 -*-
"""\
Scan loading recipe for the AMO beamline, LCLS. (Data preprocessed by hummingbird)

This file is part of the PTYPY package.

    :copyright: Copyright 2014 by the PTYPY team, see AUTHORS.
    :license: GPLv2, see LICENSE for details.
"""

import numpy as np
import os
from .. import utils as u
from .. import io
from .. import core
from ..utils import parallel
from ..core.data import PtyScan
from ..utils.verbose import log
from ..core.paths import Paths
from ..core import DEFAULT_io as io_par

logger = u.verbose.logger

# Parameters for the h5 file saved by hummingbird
H5_PATHS = u.Param()
H5_PATHS.frame_pattern = 'data/photons'

# Recipe defaults
RECIPE = u.Param()
RECIPE.experimentID = None      # Experiment identifier
RECIPE.scan_number = None       # scan number
RECIPE.dark_number = None
RECIPE.flat_number = None
RECIPE.energy = None
RECIPE.lam = None               # 1.2398e-9 / RECIPE.energy
RECIPE.z = None                 # Distance from object to screen
RECIPE.detector_name = None     # Name of the detector as specified in the nexus file
RECIPE.motors = ['t1_sx', 't1_sy']      # Motor names to determine the sample translation
RECIPE.motors_multiplier = 1e-3 #for AMO #1e-6 #for CXI         # Motor conversion factor to meters
RECIPE.base_path = './'
RECIPE.data_file_pattern = '%(base_path)s' + 'input/r%(scan_number)04d.h5'
RECIPE.dark_file_pattern = '%(base_path)s' + 'input/r%(dark_number)04d.h5'
RECIPE.flat_file_pattern = '%(base_path)s' + 'input/r%(flat_number)04d.h5'
RECIPE.mask_file = None # '%(base_path)s' + 'processing/mask.h5'
RECIPE.averaging_number = 1 # Number of frames to be averaged


# Generic defaults
AMODEFAULT = core.data.PtyScan.DEFAULT.copy()
AMODEFAULT.recipe = RECIPE
AMODEFAULT.auto_center = False
AMODEFAULT.orientation = (False, False, False)


class AMOScan(core.data.PtyScan):
    DEFAULT = AMODEFAULT

    def __init__(self, pars=None, **kwargs):
        """
        AMO (Atomic Molecular and Optical Science, LCLS) data preparation class.
        """
        # Initialise parent class
        RDEFAULT = RECIPE.copy()
        RDEFAULT.update(pars.recipe)
        pars.recipe.update(RDEFAULT)

        super(AMOScan, self).__init__(pars, **kwargs)

        # Try to extract base_path to access data files
        if self.info.recipe.base_path is None:
            d = os.getcwd()
            base_path = None
            while True:
                if 'raw' in os.listdir(d):
                    base_path = d
                    break
                d, rest = os.path.split(d)
                if not rest:
                    break
            if base_path is None:
                raise RuntimeError('Could not guess base_path.')
            else:
                self.info.recipe.base_path = base_path

        # Default scan label
        # if self.info.label is None:
        #    self.info.label = 'S%5d' % rinfo.scan_number

        # Construct file names
        self.data_file = self.info.recipe.data_file_pattern % self.info.recipe
        log(3, 'Will read data from file %s' % self.data_file)
        if self.info.recipe.dark_number is None:
            self.dark_file = None
            log(3, 'No data for dark')
        else:
            self.dark_file = self.info.recipe.dark_file_pattern % self.info.recipe
            log(3, 'Will read dark from file %s' % self.dark_file)
        if self.info.recipe.flat_number is None:
            self.flat_file = None
            log(3, 'No data for flat')
        else:
            self.flat_file = self.info.recipe.flat_file_pattern % self.info.recipe
            log(3, 'Will read flat from file %s' % self.flat_file)

        # Create the ptyd file name if not specified
        if self.info.dfile is None:
            home = Paths(io_par).home
            self.info.dfile = '%s/prepdata/data_%05d.ptyd' % (home, self.info.recipe.scan_number)
            log(3, 'Save file is %s' % self.info.dfile)
        log(4, u.verbose.report(self.info))

    def load_common(self):
        """
        Load dark, flat, and mask file.
        """
        common = u.Param()

        # FIXME: do something better here. (detector-dependent)
        # Load mask
        # common.weight2d = None
        if self.info.recipe.mask_file is not None:
            common.weight2d = io.h5read(self.info.recipe.mask_file, 'mask')['mask'].astype(float)

        return common

    def load_positions(self):
        """
        Load the positions and return as an (N,2) array
        """

        # Load positions from file if possible.
        mmult = u.expect2(self.info.recipe.motors_multiplier)
        x = mmult[0] * io.h5read(self.data_file, 'data.posx')['posx']
        y = mmult[1] * io.h5read(self.data_file, 'data.posy')['posy']

        # Valid posiitons and frames
        try:
            v = io.h5read(self.data_file, 'data.valid')['valid']
        except:
            v = np.ones(len(x)).astype(np.bool)
        self.validframes = np.arange(len(x))[v]
        x,y = x[v], y[v]

        pos_list = []
        for i in range(0,len(x),self.info.recipe.averaging_number):
            pos_list.append([y[i],x[i]])
        positions = np.array(pos_list)

        return positions

    def check(self, frames, start=0):
        """
        Returns the number of frames available from starting index `start`, and whether the end of the scan
        was reached.

        :param frames: Number of frames to load
        :param start: starting point
        :return: (frames_available, end_of_scan)
        - the number of frames available from a starting point `start`
        - bool if the end of scan was reached (None if this routine doesn't know)
        """
        npos = self.num_frames
        frames_accessible = min((frames, npos-start))
        stop = self.frames_accessible + start
        return frames_accessible, (stop >= npos)
        
    def load(self, indices):
        """
        Load frames given by the indices.

        :param indices:
        :return:
        """
        raw = {}
        pos = {}
        weights = {}

        i = 0
        h = 0
        key = H5_PATHS.frame_pattern % self.info.recipe

        for j in indices:

            # NEW
            h = self.validframes[j]
            mean = io.h5read(self.data_file, H5_PATHS.frame_pattern % self.info.recipe, slice=h)[key].astype(np.float32)
            raw[j] = np.array(mean).T

            # OLD
            #mean = []
            #while h < (i+self.info.recipe.averaging_number):
            #    mean.append(io.h5read(self.data_file, H5_PATHS.frame_pattern % self.info.recipe, slice=h)[key].astype(np.float32))
            #    h+=1
            #raw[j] = np.array(mean).mean(0).T
            #i+=self.info.recipe.averaging_number

        log(3, 'Data loaded successfully.')
        return raw, pos, weights
        
    def correct(self, raw, weights, common):
        """
        Apply (eventual) corrections to the frames. Convert from "raw" frames to usable data.
        :param raw:
        :param weights:
        :param common:
        :return:
        """
        # Apply corrections to frames
        data = raw
        for k in data.keys():
            #data[k][data[k] < 0] = 0
            data[k][data[k] < 1] = 0
        weights = weights
        return data, weights
