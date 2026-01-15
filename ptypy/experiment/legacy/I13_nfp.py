# -*- coding: utf-8 -*-
"""\
NFP scan loading recipe for the I13 beamline, Diamond.

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
from ptypy.utils.verbose import log
from ptypy.core import Ptycho

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
class I13ScanNFP(PtyScan):
    """
    I13 (Diamond Light Source) data preparation class for NFP.

    Defaults:

    [name]
    default = 'I13ScanNFP'
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

    [correct_positions_Oct14]
    default = False
    type = bool
    help = 

    [use_EP]
    default = False
    type = bool
    help = Use flat as Empty Probe (EP) for probe sharing
    doc = Needs to be set to True in the recipe of the scan that will act as EP.

    [max_scan_points]
    default = 100000
    type = int
    help = Maximum number of scan points to be loaded from origin

    [theta]
    default = 0.0
    type = float
    help = Angle of rotation (as used in NFP beamtime Jul 2015)

    [remove_hot_pixels]
    default = 
    type = Param
    help = Apply hot pixel correction

    [remove_hot_pixels.apply]
    default = False
    type = bool
    help = 

    [remove_hot_pixels.size]
    default = 3
    type = int
    help = Size of the window
    doc = The median filter will be applied around every data point.

    [remove_hot_pixels.tolerance]
    default = 3
    type = int
    help =
    doc = Tolerance multiplied with the standard deviation of the data array subtracted by the blurred array (difference array) yields the threshold for cutoff.

    [remove_hot_pixels.ignore_edges]
    default = False
    type = bool
    help = Ignore edges of the array
    doc = Enabling speeds up the code.

    [auto_center]
    default = False

    [orientation]
    default = (False, False, False)

    [rl_deconvolution]
    default =
    type = Param
    help = Apply Richardson Lucy deconvolution

    [rl_deconvolution.apply]
    default = False
    type = bool
    help = Initiate by setting to True

    [rl_deconvolution.numiter]
    default = 5
    type = int
    help = Number of iterations

    [rl_deconvolution.dfile]
    default = None
    type = str
    help = Provide MTF from file; no loading procedure present for now, loading through recon script required

    [rl_deconvolution.gaussians]
    default =
    type = Param
    help = Create fake psf as a sum of gaussians if no MTF provided

    [rl_deconvolution.gaussians.g1]
    default =
    type = Param
    help = list of gaussians for Richardson Lucy deconvolution

    [rl_deconvolution.gaussians.g1.std_x]
    default = 1.0
    type = float
    help = Standard deviation in x direction

    [rl_deconvolution.gaussians.g1.std_y]
    default = 1.0
    type = float
    help = Standard deviation in y direction

    [rl_deconvolution.gaussians.g1.off_x]
    default = 0.0
    type = float
    help = Offset / shift in x direction

    [rl_deconvolution.gaussians.g1.off_y]
    default = 0.0
    type = float
    help = Offset / shift in y direction

    """

    def __init__(self, pars=None, **kwargs):
        """
        Initializes parent class.

        :param pars: dict
            - contains parameter tree.
        :param kwargs: key-value pair
            - additional parameters.
        """
        log(2, "The I13ScanNFP loader will be deprecated in the next release. Please use the Hdf5Loader.")
        p = self.DEFAULT.copy(99)
        p.update(pars)
        super(I13ScanNFP, self).__init__(pars, **kwargs)

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

        if self.info.dark_number is None:
            self.dark_file = None
            u.log(3, 'No data for dark')
        else:
            self.dark_file = (self.info.dark_file_pattern
                              % self.info)
            u.log(3, 'Will read dark from file %s' % self.dark_file)

        if self.info.flat_number is None:
            self.flat_file = None
            u.log(3, 'No data for flat')
        else:
            self.flat_file = (self.info.flat_file_pattern
                              % self.info)
            u.log(3, 'Will read flat from file %s' % self.flat_file)

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

        # Set up dimensions for cropping
        try:
            # Switch for attributes which are set to None
            # Will be removed once None attributes are removed
            center = p.center
        except AttributeError:
            center = 'unset'

        # Check if dimension tuple is provided
        if type(center) == tuple:
            offset_x = p.center[0]
            offset_y = p.center[1]
        # If center unset, extract offset from raw data
        elif center == 'unset':
            raw_shape = self.instrument[
                self.info.detector_name]['data'].shape
            offset_x = raw_shape[-1] // 2
            offset_y = raw_shape[-2] // 2
        else:
            raise RuntimeError(
                'Center provided is not of type tuple or set to "unset". '
                'Please correct input parameters.')

        xdim = (offset_x - p.shape // 2, offset_x + p.shape // 2)
        ydim = (offset_y - p.shape // 2, offset_y + p.shape // 2)

        self.info.array_dim = [xdim, ydim]

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

        # If Empty Probe sharing is enabled, assign pseudo center position to
        # scan and skip the rest of the function. If no positions are found at
        # all, raise error.
        if motor_positions is None and self.info.use_EP:
            positions = 1. * np.array([[0., 0.]])
            return positions
        elif motor_positions is None:
            raise RuntimeError('Could not find motors (tried %s)'
                               % str(NEXUS_PATHS.motors))

        # Apply motor conversion factor and create transposed position array
        mmult = u.expect2(self.info.motors_multiplier)
        pos_list = [mmult[i] * np.array(motor_positions[motor_name])[
                               :self.info.max_scan_points]
                    for i, motor_name in enumerate(self.info.motors)]
        positions = 1. * np.array(pos_list).T

        # Correct positions for angle of rotation if necessary
        positions[:, 1] *= np.cos(np.pi * self.info.theta / 180.)

        # Position corrections for NFP beamtime Oct 2014.
        if self.info.correct_positions_Oct14:
            r = np.array([[0.99987485, 0.01582042], [-0.01582042, 0.99987485]])
            p0 = positions.mean(axis=0)
            positions = np.dot(r, (positions - p0).T).T + p0
            u.log(3, 'Original positions corrected by array provided.')

        return positions

    def load_common(self):
        """
        Load dark and flat.

        :return: common
            - dict: contains averaged dark and flat (np.array).
        """
        common = u.Param()

        # Load dark.
        if self.info.dark_number is not None:
            key = NEXUS_PATHS.frame_pattern % self.info
            dark_indices = list(range(len(
                io.h5read(self.dark_file, NEXUS_PATHS.frame_pattern
                          % self.info)[key])))

            dark = [io.h5read(self.dark_file, NEXUS_PATHS.frame_pattern
                              % self.info, slice=j)[key][
                    self.info.array_dim[1][0]:
                    self.info.array_dim[1][1],
                    self.info.array_dim[0][0]:
                    self.info.array_dim[0][1]].astype(np.float32)
                    for j in dark_indices]

            common.dark = np.array(dark).mean(0)
            u.log(3, 'Dark loaded successfully.')

        # Load flat.
        if self.info.flat_number is not None:
            key = NEXUS_PATHS.frame_pattern % self.info
            flat_indices = list(range(len(
                io.h5read(self.flat_file, NEXUS_PATHS.frame_pattern
                          % self.info)[key])))

            flat = [io.h5read(self.flat_file, NEXUS_PATHS.frame_pattern
                              % self.info, slice=j)[key][
                    self.info.array_dim[1][0]:
                    self.info.array_dim[1][1],
                    self.info.array_dim[0][0]:
                    self.info.array_dim[0][1]].astype(np.float32)
                    for j in flat_indices]

            common.flat = np.array(flat).mean(0)
            u.log(3, 'Flat loaded successfully.')

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
        raw = {j: self.instrument[self.info.detector_name]['data'][j][
                  self.info.array_dim[1][0]:
                  self.info.array_dim[1][1],
                  self.info.array_dim[0][0]:
                  self.info.array_dim[0][1]].astype(np.float32)
               for j in indices}

        u.log(3, 'Data loaded successfully.')

        return raw, pos, weights

    def correct(self, raw, weights, common):
        """
        Apply corrections to frames. See below for possible options.

        Options for corrections:
        - Hot pixel removal:
            Replace outlier pixels in frames by median.
        - Richardson–Lucy deconvolution:
            Deconvolve frames from detector psf.
        - Dark subtraction:
            Subtract dark from frames.
        - Flat division:
            Divide frames by flat.

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
        # Apply hot pixel removal
        if self.info.remove_hot_pixels.apply:
            u.log(3, 'Applying hot pixel removal...')
            for j in raw:
                raw[j] = u.remove_hot_pixels(
                    raw[j],
                    self.info.remove_hot_pixels.size,
                    self.info.remove_hot_pixels.tolerance,
                    self.info.remove_hot_pixels.ignore_edges)[0]

            if self.info.flat_number is not None:
                    common.dark = u.remove_hot_pixels(
                        common.dark,
                        self.info.remove_hot_pixels.size,
                        self.info.remove_hot_pixels.tolerance,
                        self.info.remove_hot_pixels.ignore_edges)[0]

            if self.info.flat_number is not None:
                common.flat = u.remove_hot_pixels(
                    common.flat,
                    self.info.remove_hot_pixels.size,
                    self.info.remove_hot_pixels.tolerance,
                    self.info.remove_hot_pixels.ignore_edges)[0]

            u.log(3, 'Hot pixel removal completed.')

        # Apply deconvolution
        if self.info.rl_deconvolution.apply:
            u.log(3, 'Applying deconvolution...')

            # Use mtf from a file if provided in recon script
            if self.info.rl_deconvolution.dfile is not None:
                mtf = self.info.rl_deconvolution.dfile
            # Create fake psf as a sum of gaussians from parameters
            else:
                gau_sum = 0
                for k in self.info.rl_deconvolution.gaussians.items():
                    gau_sum += u.gaussian2D(raw[0].shape[0],
                                            k[1].std_x,
                                            k[1].std_y,
                                            k[1].off_x,
                                            k[1].off_y)

                # Compute mtf
                mtf = np.abs(np.fft.fft2(gau_sum))

            for j in raw:
                raw[j] = u.rl_deconvolution(
                    raw[j],
                    mtf,
                    self.info.rl_deconvolution.numiter)

            u.log(3, 'Deconvolution completed.')

        # Apply flat and dark, only dark, or no correction
        if (self.info.flat_number is not None
                and self.info.dark_number is not None):
            for j in raw:
                raw[j] = (raw[j] - common.dark) / (common.flat - common.dark)
                raw[j][raw[j] < 0] = 0
            data = raw
        elif self.info.dark_number is not None:
            for j in raw:
                raw[j] = raw[j] - common.dark
                raw[j][raw[j] < 0] = 0
            data = raw
        else:
            data = raw

        # FIXME: this will depend on the detector type used.
        weights = weights

        return data, weights
