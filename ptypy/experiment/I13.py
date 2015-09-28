# -*- coding: utf-8 -*-
"""\
Scan loading recipe for the I13 beamline, Diamond.

This file is part of the PTYPY package.

    :copyright: Copyright 2014 by the PTYPY team, see AUTHORS.
    :license: GPLv2, see LICENSE for details.
"""

import numpy as np
import os
from .. import utils as u
from .. import io
from .. import core
from ..core.data import PtyScan
from ..utils.verbose import log
from ..core.paths import Paths
from ..core import DEFAULT_io as io_par

logger = u.verbose.logger

# Parameters for the nexus file saved by GDA
NEXUS_PATHS = u.Param()
NEXUS_PATHS.frame_pattern = 'entry1/instrument/pco1_sw_hdf_nochunking/data' # 'entry1/instrument/%(detector_name)s/data'
NEXUS_PATHS.exposure = 'entry1/instrument/%(detector_name)s/count_time' # 'entry1/instrument/%(detector_name)s/count_time'
NEXUS_PATHS.motors = 'entry1/instrument/t1_sxy'
NEXUS_PATHS.command = 'entry1/scan_command'
NEXUS_PATHS.label = 'entry1/entry_identifier'
NEXUS_PATHS.experiment = 'entry1/experiment_identifier'

# Recipe defaults
RECIPE = u.Param()
RECIPE.experimentID = None   # Experiment identifier
RECIPE.scan_number = None      # scan number
RECIPE.dark_number = None
RECIPE.flat_number = None
RECIPE.energy = None
RECIPE.lam = None # 1.2398e-9 / RECIPE.energy
RECIPE.z = None                                          # Distance from object to screen
RECIPE.motors = ['t1_sy', 't1_sx']     # 'Motor names to determine the sample translation'
RECIPE.motors_multiplier = 1e-6       # 'Motor conversion factor to meters'
RECIPE.base_path = './'
RECIPE.data_file_pattern = '%(base_path)s' + 'raw/%(scan_number)05d.nxs'
RECIPE.dark_file_pattern = '%(base_path)s' + 'raw/%(dark_number)05d.nxs'
RECIPE.flat_file_pattern = '%(base_path)s' + 'raw/%(flat_number)05d.nxs'
RECIPE.mask_file = None # '%(base_path)s' + 'processing/mask.h5'

# Generic defaults
I13DEFAULT = core.data.PtyScan.DEFAULT.copy()
I13DEFAULT.recipe = RECIPE
I13DEFAULT.auto_center = False
I13DEFAULT.orientation = (False, False, False)


class I13Scan(core.data.PtyScan):
    DEFAULT = I13DEFAULT

    def __init__(self, pars=None, **kwargs):
        """
        I13 (Diamond Light Source) data preparation class.
        """
        # Initialise parent class
        RDEFAULT = RECIPE.copy()
        RDEFAULT.update(pars.recipe)
        pars.recipe.update(RDEFAULT)

        super(I13Scan, self).__init__(pars, **kwargs)

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

        # Attempt to extract experiment ID
        if self.info.recipe.experimentID is None:
            try:
                experimentID = io.h5read(self.data_file, NEXUS_PATHS.experiment)[NEXUS_PATHS.experiment][0]
            except:
                experimentID = os.path.split(self.info.recipe.base_path[:-1])[1]
                logger.debug('Could not find experiment ID from nexus file %s. Using %s instead.' %
                             (self.data_file, experimentID))
            self.info.recipe.experimentID = experimentID

        # Create the ptyd file name if not specified
        if self.info.dfile is None:
            home = Paths(io_par).home
            self.info.dfile = '%s/prepdata/data_%d.ptyd' % (home, self.info.recipe.scan_number)
            log(3, 'Save file is %s' % self.info.dfile)
        log(4, u.verbose.report(self.info))

    def load_common(self):
        """
        Load scanning positions and mask file.
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
        motor_positions = io.h5read(self.data_file, NEXUS_PATHS.motors)[NEXUS_PATHS.motors]
        mmult = u.expect2(self.info.recipe.motors_multiplier)
        pos_list = [mmult[i] * np.array(motor_positions[motor_name]) for i, motor_name in enumerate(self.info.recipe.motors)]
        positions = 1. * np.array(pos_list).T
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
        key = NEXUS_PATHS.frame_pattern % self.info.recipe
        print (key)
        for j in indices:
            key = NEXUS_PATHS.frame_pattern % self.info.recipe
            raw[j] = io.h5read(self.data_file, NEXUS_PATHS.frame_pattern % self.info.recipe, slice=j)[key].astype(np.float32)
            
        return raw, pos, weights
        
    def correct(self, raw, weights, common):
        """
        Apply (eventual) corrections to the frames. Convert from "raw" frames to usable data.
        :param raw:
        :param weights:
        :param common:
        :return:
        """
        # FIXME: this will depend on the detector type used.
        data = raw
        weights = weights
        return data, weights
