# -*- coding: utf-8 -*-
"""\
Scan loading recipe for the I13 beamline, Diamond.

This file is part of the PTYPY package.

    :copyright: Copyright 2014 by the PTYPY team, see AUTHORS.
    :license: GPLv2, see LICENSE for details.
"""

import numpy as np
import os
from .. import io
from .. import utils as u
from .. import core
from ..core.data import PtyScan

logger = u.verbose.logger

NEXUS_PATHS = u.Param()
NEXUS_PATHS.frame_pattern = 'entry1/instrument/%(detector_name)s/data'
NEXUS_PATHS.exposure = 'entry1/instrument/%(detector_name)s/count_time'
NEXUS_PATHS.motors = 'entry1/instrument/lab_sxy'
NEXUS_PATHS.command = 'entry1/scan_command'
NEXUS_PATHS.label = 'entry1/entry_identifier'
NEXUS_PATHS.experiment = 'entry1/experiment_identifier'

# Generic defaults
PREP_DEFAULT = u.Param()
PREP_DEFAULT.psize_det = 55e-6    # Camera pixel size
PREP_DEFAULT.orientation = (False, True, False)

# Recipe defaults
DEFAULT = u.Param()
DEFAULT.experimentID = None   # Experiment identifier
DEFAULT.scan_number = None      # scan number
DEFAULT.dark_number = None
DEFAULT.flat_number = None
DEFAULT.energy = None
DEFAULT.lam = None # 1.2398e-9 / DEFAULT.energy
DEFAULT.z = None                                          # Distance from object to screen
DEFAULT.motors = ['lab_sy', 'lab_sx']     # 'Motor names to determine the sample translation'
DEFAULT.motors_multiplier = 1e-6       # 'Motor conversion factor to meters'
DEFAULT.base_path = './'
DEFAULT.data_file_pattern = '%(base_path)s' + 'raw/%(scan_number)05d.nxs'
DEFAULT.dark_file_pattern = '%(base_path)s' + 'raw/%(dark_number)05d.nxs'
DEFAULT.flat_file_pattern = '%(base_path)s' + 'raw/%(flat_number)05d.nxs'
DEFAULT.mask_file = '%(base_path)s' + 'processing/mask.h5'


class I13Scan(PtyScan):
    """
    Subclass of PtyScan.
    """

    DEFAULT = PtyScan.DEFAULTS.copy().update(PREP_DEFAULT)

    def __init__(self, pars=None, **kwargs):
        """
        Create a PtyScan object that will load I13 data.

        :param pars: preparation parameters
        :param kwargs: Additive parameters
        """
        # Initialise parent class with input parameters
        super(I13Scan, self).__init__(pars, **kwargs)

        # Update recipe information
        rinfo = DEFAULT.copy()
        rinfo.update(pars.recipe)

        # Default scan label
        if self.info.label is None:
            self.info.label = 'S%5d' % rinfo.scan_number

        # Store recipe parameters in self.info
        self.info.recipe = rinfo

        # Attempt to extract base_path if missing
        if rinfo.base_path is None:
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
                raise RuntimeError('Could guess base_path.')
            else:
                rinfo.base_path = base_path

        rinfo.data_file = rinfo.data_file_pattern % rinfo
        rinfo.dark_file = None if rinfo.dark_number is None else rinfo.dark_file_pattern % rinfo
        rinfo.flat_file = None if rinfo.flat_number is None else rinfo.flat_file_pattern % rinfo

        # Attempt to extract experiment ID
        if rinfo.experimentID is None:
            try:
                experimentID = io.h5read(rinfo.data_file, NEXUS_PATHS.experiment)[NEXUS_PATHS.experiment]
            except:
                experimentID = os.path.split(rinfo.base_path[:-1])[1]
                logger.debug('Could not find experiment ID from nexus file %s. Using %s instead.' %
                             (rinfo.data_file, experimentID))
            rinfo.experimentID = experimentID

        self.rinfo = rinfo
        self.info.recipe = rinfo

        logger.info(u.verbose.report(self.info))

    def load_common(self):
        """
        Load scanning positions and mask file.
        """
        common = u.Param()

        # Get positions
        motor_positions = io.h5read(self.rinfo.data_file, NEXUS_PATHS.motors)[NEXUS_PATHS.motors]
        mmult = u.expect2(self.rinfo.motors_multiplier)
        pos_list = [mmult[i] * np.array(motor_positions[motor_name]) for i, motor_name in enumerate(self.rinfo.motors)]
        common.positions_scan = np.array(pos_list).T

        # FIXME: do something better here. (detector-dependent)
        # Load mask
        # common.weight2d = None
        if self.rinfo.mask_file is not None:
            common.weight2d = io.h5read(self.rinfo.mask_file, 'mask')['mask'].astype(float)

        return common._to_dict()
        
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
        for j in indices:
            key = NEXUS_PATHS.frame_pattern % self.rinfo
            raw[j] = io.h5read(self.rinfo.data_file, NEXUS_PATHS.frame_pattern % self.rinfo, slice=j)[key].astype(np.float32)
            
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
