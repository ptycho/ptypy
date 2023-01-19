# -*- coding: utf-8 -*-
"""\
FFP scan loading recipe for the I13 beamline, Diamond.

This file is part of the PTYPY package.

    :copyright: Copyright 2014 by the PTYPY team, see AUTHORS.
    :license: see LICENSE for details.
"""
import numpy as np
import os

from ptypy import utils as u
from ptypy.experiment import register
from ptypy import io
from ptypy.core.data import PtyScan
from ptypy.core.paths import Paths
from ptypy.core import Ptycho
from ptypy.utils.verbose import log

IO_par = Ptycho.DEFAULT['io']

# Parameters for the nexus file saved by GDA
NEXUS_PATHS = u.Param()
NEXUS_PATHS.instrument = 'entry1/instrument'
NEXUS_PATHS.frame_pattern = 'entry1/instrument/%(detector_name)s/data'
NEXUS_PATHS.exposure = 'entry1/instrument/%(detector_name)s/count_time'
NEXUS_PATHS.motors = ['t1_sxy', 't1_sxyz']
NEXUS_PATHS.command = 'entry1/scan_command'
NEXUS_PATHS.label = 'entry1/entry_identifier'
NEXUS_PATHS.experiment = 'entry1/experiment_identifier'


@register()
class I13ScanFFP(PtyScan):
    """
    I13 (Diamond Light Source) data preparation class for FFP.

    Defaults:

    [name]
    default = 'I13ScanFFP'
    type = str
    help =

    [experimentID]
    default = None

    [scan_number]
    default = None
    type = int
    help = Scan number

    [dark_number]
    default = None
    type = int
    help = 

    [flat_number]
    default = None
    type = int
    help = 

    [detector_name]
    default = None
    type = str
    help = Name of the detector 
    doc = As specified in the nexus file.

    [motors]
    default = ['t1_sx', 't1_sy']
    type = list
    help = Motor names to determine the sample translation

    [motors_multiplier]
    default = 1e-6
    type = float
    help = Motor conversion factor to meters

    [base_path]
    default = './'
    type = str
    help = 

    [data_file_pattern]
    default = '%(base_path)sraw/%(scan_number)05d.nxs'
    type = str
    help = 

    [dark_file_pattern]
    default = '%(base_path)sraw/%(dark_number)05d.nxs'
    type = str
    help = 

    [flat_file_pattern]
    default = '%(base_path)sraw/%(flat_number)05d.nxs'
    type = str
    help = 

    [mask_file]
    default = None
    type = str
    help = 

    [theta]
    default = 0.0
    type = float
    help = Angle of rotation

    [auto_center]
    default = False

    [orientation]
    default = (False, False, False)

    """

    def __init__(self, pars=None, **kwargs):
        """
        Initializes parent class.

        :param pars: dict
            - contains parameter tree.
        :param kwargs: key-value pair
            - additional parameters.
        """
        log(2, "The DLS loader will be deprecated in the next release. Please use the Hdf5Loader.")
        p = self.DEFAULT.copy(99)
        p.update(pars)
        super(I13ScanFFP, self).__init__(p, **kwargs)

        # Try to extract base_path to access data files
        if self.info.base_path is None:
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
                self.info.base_path = base_path

        # Construct file names
        self.data_file = self.info.data_file_pattern % self.info
        u.log(3, 'Will read data from file %s' % self.data_file)

        # Load data information
        self.instrument = io.h5read(self.data_file, NEXUS_PATHS.instrument)[
            NEXUS_PATHS.instrument]

        # Extract detector name if not set or wrong
        if (self.info.detector_name is None
                or self.info.detector_name
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
                elif (self.info.detector_name is not None
                      and detector_name
                      is not self.info.detector_name):
                    u.log(2, 'Detector name changed from %s to %s.'
                          % (self.info.detector_name, detector_name))
        else:
            detector_name = self.info.detector_name

        self.info.detector_name = detector_name

        # Attempt to extract experiment ID
        if self.info.experimentID is None:
            try:
                experiment_id = io.h5read(
                    self.data_file, NEXUS_PATHS.experiment)[
                    NEXUS_PATHS.experiment][0]
            except (AttributeError, KeyError):
                experiment_id = os.path.split(
                    self.info.base_path[:-1])[1]
                u.logger.debug(
                    'Could not find experiment ID from nexus file %s. '
                    'Using %s instead.' % (self.data_file, experiment_id))
            self.info.experimentID = experiment_id

        # Create the ptyd file name if not specified
        if self.info.dfile is None:
            home = Paths(IO_par).home
            self.info.dfile = ('%s/prepdata/data_%d.ptyd'
                               % (home, self.info.scan_number))
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
        if self.info.mask_file is not None:
            return io.h5read(
                self.info.mask_file, 'mask')['mask'].astype(float)

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
        if len(self.info.motors) == 3:
            self.theta = io.h5read(self.data_file, self.info.theta)[
                self.info.theta]
            # Convert from degree to radians
            self.theta *= np.pi / 180.
            mmult = u.expect3(self.info.motors_multiplier)
            pos_list = [mmult[i] * np.array(motor_positions[motor_name])
                        for i, motor_name in enumerate(self.info.motors)]
            positions = 1. * np.array([np.cos(self.theta) * pos_list[0] -
                                       np.sin(self.theta) * pos_list[2],
                                       pos_list[1]]).T
        else:
            mmult = u.expect2(self.info.motors_multiplier)
            pos_list = [mmult[i] * np.array(motor_positions[motor_name])
                        for i, motor_name in enumerate(self.info.motors)]
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
            self.info.detector_name]['data'][j].astype(np.float32)
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
