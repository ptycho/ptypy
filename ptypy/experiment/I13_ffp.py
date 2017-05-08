# -*- coding: utf-8 -*-
"""\
FFP scan loading recipe for the I13 beamline, Diamond.

This file is part of the PTYPY package.

    :copyright: Copyright 2014 by the PTYPY team, see AUTHORS.
    :license: GPLv2, see LICENSE for details.
"""
import numpy as np
import os
from .. import utils as u
from .. import io
from ..core.data import PtyScan
from ..core.paths import Paths
from ..core import DEFAULT_io as IO_par

# Parameters for the nexus file saved by GDA
NEXUS_PATHS = u.Param()
NEXUS_PATHS.instrument = 'entry1/instrument'
NEXUS_PATHS.frame_pattern = 'entry1/instrument/%(detector_name)s/data'
NEXUS_PATHS.exposure = 'entry1/instrument/%(detector_name)s/count_time'
NEXUS_PATHS.motors = ['t1_sxy', 't1_sxyz', 'lab_sxy']
NEXUS_PATHS.command = 'entry1/scan_command'
NEXUS_PATHS.label = 'entry1/entry_identifier'
NEXUS_PATHS.experiment = 'entry1/experiment_identifier'

# Recipe defaults
RECIPE = u.Param()
# Experiment identifier
RECIPE.experimentID = None
# Scan number
RECIPE.scan_number = None
RECIPE.dark_number = None
RECIPE.flat_number = None
RECIPE.energy = None
RECIPE.lam = None
# Distance from object to screen
RECIPE.z = None
# Name of the detector as specified in the nexus file
RECIPE.detector_name = None
# Motor names to determine the sample translation
RECIPE.motors = ['t1_sx', 't1_sy']
RECIPE.theta = 'entry1/before_scan/t1_theta/t1_theta'
# Motor conversion factor to meters
RECIPE.motors_multiplier = 1e-6
RECIPE.base_path = './'
RECIPE.data_file_pattern = '%(base_path)s' + 'raw/%(scan_number)05d.nxs'
RECIPE.dark_file_pattern = '%(base_path)s' + 'raw/%(dark_number)05d.nxs'
RECIPE.flat_file_pattern = '%(base_path)s' + 'raw/%(flat_number)05d.nxs'
RECIPE.mask_file = None

# Generic defaults
I13DEFAULT = PtyScan.DEFAULT.copy()
I13DEFAULT.recipe = RECIPE
I13DEFAULT.auto_center = False
I13DEFAULT.orientation = (False, False, False)


class I13ScanFFP(PtyScan):
    """
    I13 (Diamond Light Source) data preparation class for FFP.
    """
    DEFAULT = I13DEFAULT

    def __init__(self, pars=None, **kwargs):
        """
        Initializes parent class.

        :param pars: dict
            - contains parameter tree.
        :param kwargs: key-value pair
            - additional parameters.
        """
        recipe_default = RECIPE.copy()
        recipe_default.update(pars.recipe, in_place_depth=1)
        pars.recipe.update(recipe_default)

        super(I13ScanFFP, self).__init__(pars, **kwargs)

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

        # Construct file names
        self.data_file = self.info.recipe.data_file_pattern % self.info.recipe
        u.log(3, 'Will read data from file %s' % self.data_file)

        # Load data information
        self.instrument = io.h5read(self.data_file, NEXUS_PATHS.instrument)[
            NEXUS_PATHS.instrument]

        # Extract detector name if not set or wrong
        if (self.info.recipe.detector_name is None
                or self.info.recipe.detector_name
                not in self.instrument.keys()):
                detector_name = None
                for k in self.instrument.keys():
                    if 'data' in self.instrument[k]:
                        detector_name = k
                        break

                if detector_name is None:
                    raise RuntimeError(
                        'Not possible to extract detector name. '
                        'Please specify in recipe instead.')
                elif (self.info.recipe.detector_name is not None
                      and detector_name
                      is not self.info.recipe.detector_name):
                    u.log(2, 'Detector name changed from %s to %s.'
                          % (self.info.recipe.detector_name, detector_name))
        else:
            detector_name = self.info.recipe.detector_name

        self.info.recipe.detector_name = detector_name

        # Attempt to extract experiment ID
        if self.info.recipe.experimentID is None:
            try:
                experiment_id = io.h5read(
                    self.data_file, NEXUS_PATHS.experiment)[
                    NEXUS_PATHS.experiment][()] #[0]
            except (AttributeError, KeyError):
                experiment_id = os.path.split(
                    self.info.recipe.base_path[:-1])[1]
                u.logger.debug(
                    'Could not find experiment ID from nexus file %s. '
                    'Using %s instead.' % (self.data_file, experiment_id))
            self.info.recipe.experimentID = experiment_id

        # Create the ptyd file name if not specified
        if self.info.dfile is None:
            home = Paths(IO_par).home
            self.info.dfile = ('%s/prepdata/data_%d.ptyd'
                               % (home, self.info.recipe.scan_number))
            u.log(3, 'Save file is %s' % self.info.dfile)

        u.log(4, u.verbose.report(self.info))

        # Instance attributes
        self.theta = None

    def load_weight(self):
        """
        For now, this function will be used to load the mask.

        Function description see parent class.

        :return: weight2d
            - np.array: Mask or weight if provided from file
        """
        # FIXME: do something better here. (detector-dependent)
        # Load mask as weight
        if self.info.recipe.mask_file is not None:
            return io.h5read(
                self.info.recipe.mask_file, 'mask')['mask'].astype(float)

    def load_positions(self):
        """
        Load the positions and return as an (N,2) array.

        :return: positions
            - np.array: contains scan positions.
        """
        motor_positions = None
        for k in NEXUS_PATHS.motors:
            if k in self.instrument:
                motor_positions = self.instrument[k]
                break

        # Apply motor conversion factor and create transposed position array
        if len(self.info.recipe.motors) == 3:
            self.theta = io.h5read(self.data_file, self.info.recipe.theta)[
                self.info.recipe.theta]
            # Convert from degree to radians
            self.theta *= np.pi / 180.
            mmult = u.expect3(self.info.recipe.motors_multiplier)
            pos_list = [mmult[i] * np.array(motor_positions[motor_name])
                        for i, motor_name in enumerate(self.info.recipe.motors)]
            positions = 1. * np.array([np.cos(self.theta) * pos_list[0] -
                                       np.sin(self.theta) * pos_list[2],
                                       pos_list[1]]).T
        else:
            mmult = u.expect2(self.info.recipe.motors_multiplier)
            pos_list = [mmult[i] * np.array(motor_positions[motor_name])
                        for i, motor_name in enumerate(self.info.recipe.motors)]
            positions = 1. * np.array(pos_list).T

        return positions

    def load_common(self):
        """
        Loads anything that is common to all frames and stores it in dict.

        :return: common
            - dict: contains information common to all frames.
        """
        common = u.Param()

        return common

    def check(self, frames=None, start=None):
        """
        Returns number of frames available and if end of scan was reached.

        :param frames: int
            Number of frames to load.
        :param start: int
            Starting point.
        :return: frames_available, end_of_scan
            - int: number of frames available from a starting point `start`.
            - bool: if the end of scan was reached.
                    (None if this routine doesn't know)
        """
        npos = self.num_frames
        frames_accessible = min((frames, npos - start))
        stop = self.frames_accessible + start
        return frames_accessible, (stop >= npos)

    def load(self, indices):
        """
        Load frames given by the indices.

        :param indices: list
            Frame indices available per node.
        :return: raw, pos, weight
            - dict: index matched data frames (np.array).
            - dict: new positions.
            - dict: new weights.
        """
        pos = {}
        weights = {}
        raw = {j: self.instrument[
            self.info.recipe.detector_name]['data'][j].astype(np.float32)
               for j in indices}

        u.log(3, 'Data loaded successfully.')

        return raw, pos, weights

    def correct(self, raw, weights, common):
        """
        Apply corrections to frames.

        :param raw: dict
            - dict containing index matched data frames (np.array).
        :param weights: dict
            - dict containing possible weights.
        :param common: dict
            - dict containing possible dark and flat frames.
        :return: data, weights
            - dict: contains index matched corrected data frames (np.array).
            - dict: contains modified weights.
        """
        # No corrections implemented for now.
        data = raw

        # FIXME: this will depend on the detector type used.

        weights = weights

        return data, weights
