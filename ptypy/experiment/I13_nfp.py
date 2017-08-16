# -*- coding: utf-8 -*-
"""\
NFP scan loading recipe for the I13 beamline, Diamond.

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
#from ..core import DEFAULT_io as IO_par
from ..core import Ptycho
IO_par = Ptycho.DEFAULTS['io']

# Parameters for the nexus file saved by GDA
NEXUS_PATHS = u.Param()
NEXUS_PATHS.instrument = 'entry1/instrument'
NEXUS_PATHS.frame_pattern = 'entry1/instrument/%(detector_name)s/data'
NEXUS_PATHS.exposure = 'entry1/instrument/%(detector_name)s/count_time'
NEXUS_PATHS.motors = ['t1_sxy', 't1_sxyz']
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
# Motor conversion factor to meters
RECIPE.motors_multiplier = 1e-6
RECIPE.base_path = './'
RECIPE.data_file_pattern = '%(base_path)s' + 'raw/%(scan_number)05d.nxs'
RECIPE.dark_file_pattern = '%(base_path)s' + 'raw/%(dark_number)05d.nxs'
RECIPE.flat_file_pattern = '%(base_path)s' + 'raw/%(flat_number)05d.nxs'
RECIPE.mask_file = None
# Position corrections for NFP beamtime Oct 2014
RECIPE.correct_positions_Oct14 = False
# Use flat as Empty Probe (EP) for probe sharing;
# needs to be set to True in the recipe of the scan that will act as EP
RECIPE.use_EP = False
# Maximum number of scan points to be loaded from origin
RECIPE.max_scan_points = 100000
# Angle of rotation (as used in NFP beamtime Jul 2015)
RECIPE.theta = 0
# Apply hot pixel correction
RECIPE.remove_hot_pixels = u.Param(
    # Initiate by setting to True;
    # DEFAULT parameters will be used if not specified otherwise
    apply=False,
    # Size of the window on which the median filter will be applied
    # around every data point
    size=3,
    # Tolerance multiplied with the standard deviation of the data array
    # subtracted by the blurred array (difference array)
    # yields the threshold for cutoff.
    tolerance=3,
    # If True, edges of the array are ignored, which speeds up the code
    ignore_edges=False,
)

# Apply Richardson Lucy deconvolution
RECIPE.rl_deconvolution = u.Param(
    # Initiate by setting to True;
    # DEFAULT parameters will be used if not specified otherwise
    apply=False,
    # Number of iterations
    numiter=5,
    # Provide MTF from file; no loading procedure present for now,
    # loading through recon script required
    dfile=None,
    # Create fake psf as a sum of gaussians if no MTF provided
    gaussians=u.Param(
        # DEFAULT list of gaussians for Richardson Lucy deconvolution
        g1=u.Param(
            # Standard deviation in x direction
            std_x=1.0,
            # Standard deviation in y direction
            std_y=1.0,
            # Offset / shift in x direction
            off_x=0.,
            # Offset / shift in y direction
            off_y=0.,
            )
        ),
)

# Generic defaults
I13DEFAULT = PtyScan.DEFAULT.copy()
I13DEFAULT.recipe = RECIPE
I13DEFAULT.auto_center = False
I13DEFAULT.orientation = (False, False, False)


class I13ScanNFP(PtyScan):
    """
    I13 (Diamond Light Source) data preparation class for NFP.
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

        super(I13ScanNFP, self).__init__(pars, **kwargs)

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

        if self.info.recipe.dark_number is None:
            self.dark_file = None
            u.log(3, 'No data for dark')
        else:
            self.dark_file = (self.info.recipe.dark_file_pattern
                              % self.info.recipe)
            u.log(3, 'Will read dark from file %s' % self.dark_file)

        if self.info.recipe.flat_number is None:
            self.flat_file = None
            u.log(3, 'No data for flat')
        else:
            self.flat_file = (self.info.recipe.flat_file_pattern
                              % self.info.recipe)
            u.log(3, 'Will read flat from file %s' % self.flat_file)

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

        # Set up dimensions for cropping
        try:
            # Switch for attributes which are set to None
            # Will be removed once None attributes are removed
            center = pars.center
        except AttributeError:
            center = 'unset'

        # Check if dimension tuple is provided
        if type(center) == tuple:
            offset_x = pars.center[0]
            offset_y = pars.center[1]
        # If center unset, extract offset from raw data
        elif center == 'unset':
            raw_shape = self.instrument[
                self.info.recipe.detector_name]['data'].shape
            offset_x = raw_shape[-1] // 2
            offset_y = raw_shape[-2] // 2
        else:
            raise RuntimeError(
                'Center provided is not of type tuple or set to "unset". '
                'Please correct input parameters.')

        xdim = (offset_x - pars.shape // 2, offset_x + pars.shape // 2)
        ydim = (offset_y - pars.shape // 2, offset_y + pars.shape // 2)

        self.info.recipe.array_dim = [xdim, ydim]

        # Attempt to extract experiment ID
        if self.info.recipe.experimentID is None:
            try:
                experiment_id = io.h5read(
                    self.data_file, NEXUS_PATHS.experiment)[
                    NEXUS_PATHS.experiment][0]
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

        # If Empty Probe sharing is enabled, assign pseudo center position to
        # scan and skip the rest of the function. If no positions are found at
        # all, raise error.
        if motor_positions is None and self.info.recipe.use_EP:
            positions = 1. * np.array([[0., 0.]])
            return positions
        elif motor_positions is None:
            raise RuntimeError('Could not find motors (tried %s)'
                               % str(NEXUS_PATHS.motors))

        # Apply motor conversion factor and create transposed position array
        mmult = u.expect2(self.info.recipe.motors_multiplier)
        pos_list = [mmult[i] * np.array(motor_positions[motor_name])[
                               :self.info.recipe.max_scan_points]
                    for i, motor_name in enumerate(self.info.recipe.motors)]
        positions = 1. * np.array(pos_list).T

        # Correct positions for angle of rotation if necessary
        positions[:, 1] *= np.cos(np.pi * self.info.recipe.theta / 180.)

        # Position corrections for NFP beamtime Oct 2014.
        if self.info.recipe.correct_positions_Oct14:
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
        if self.info.recipe.dark_number is not None:
            key = NEXUS_PATHS.frame_pattern % self.info.recipe
            dark_indices = range(len(
                io.h5read(self.dark_file, NEXUS_PATHS.frame_pattern
                          % self.info.recipe)[key]))

            dark = [io.h5read(self.dark_file, NEXUS_PATHS.frame_pattern
                              % self.info.recipe, slice=j)[key][
                    self.info.recipe.array_dim[1][0]:
                    self.info.recipe.array_dim[1][1],
                    self.info.recipe.array_dim[0][0]:
                    self.info.recipe.array_dim[0][1]].astype(np.float32)
                    for j in dark_indices]

            common.dark = np.array(dark).mean(0)
            u.log(3, 'Dark loaded successfully.')

        # Load flat.
        if self.info.recipe.flat_number is not None:
            key = NEXUS_PATHS.frame_pattern % self.info.recipe
            flat_indices = range(len(
                io.h5read(self.flat_file, NEXUS_PATHS.frame_pattern
                          % self.info.recipe)[key]))

            flat = [io.h5read(self.flat_file, NEXUS_PATHS.frame_pattern
                              % self.info.recipe, slice=j)[key][
                    self.info.recipe.array_dim[1][0]:
                    self.info.recipe.array_dim[1][1],
                    self.info.recipe.array_dim[0][0]:
                    self.info.recipe.array_dim[0][1]].astype(np.float32)
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
        raw = {j: self.instrument[self.info.recipe.detector_name]['data'][j][
                  self.info.recipe.array_dim[1][0]:
                  self.info.recipe.array_dim[1][1],
                  self.info.recipe.array_dim[0][0]:
                  self.info.recipe.array_dim[0][1]].astype(np.float32)
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
        if self.info.recipe.remove_hot_pixels.apply:
            u.log(3, 'Applying hot pixel removal...')
            for j in raw:
                raw[j] = u.remove_hot_pixels(
                    raw[j],
                    self.info.recipe.remove_hot_pixels.size,
                    self.info.recipe.remove_hot_pixels.tolerance,
                    self.info.recipe.remove_hot_pixels.ignore_edges)[0]

            if self.info.recipe.flat_number is not None:
                    common.dark = u.remove_hot_pixels(
                        common.dark,
                        self.info.recipe.remove_hot_pixels.size,
                        self.info.recipe.remove_hot_pixels.tolerance,
                        self.info.recipe.remove_hot_pixels.ignore_edges)[0]

            if self.info.recipe.flat_number is not None:
                common.flat = u.remove_hot_pixels(
                    common.flat,
                    self.info.recipe.remove_hot_pixels.size,
                    self.info.recipe.remove_hot_pixels.tolerance,
                    self.info.recipe.remove_hot_pixels.ignore_edges)[0]

            u.log(3, 'Hot pixel removal completed.')

        # Apply deconvolution
        if self.info.recipe.rl_deconvolution.apply:
            u.log(3, 'Applying deconvolution...')

            # Use mtf from a file if provided in recon script
            if self.info.recipe.rl_deconvolution.dfile is not None:
                mtf = self.info.rl_deconvolution.dfile
            # Create fake psf as a sum of gaussians from parameters
            else:
                gau_sum = 0
                for k in (
                        self.info.recipe.rl_deconvolution.gaussians.iteritems()):
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
                    self.info.recipe.rl_deconvolution.numiter)

            u.log(3, 'Deconvolution completed.')

        # Apply flat and dark, only dark, or no correction
        if (self.info.recipe.flat_number is not None
                and self.info.recipe.dark_number is not None):
            for j in raw:
                raw[j] = (raw[j] - common.dark) / (common.flat - common.dark)
                raw[j][raw[j] < 0] = 0
            data = raw
        elif self.info.recipe.dark_number is not None:
            for j in raw:
                raw[j] = raw[j] - common.dark
                raw[j][raw[j] < 0] = 0
            data = raw
        else:
            data = raw

        # FIXME: this will depend on the detector type used.
        weights = weights

        return data, weights
