# -*- coding: utf-8 -*-
"""\
Data preparation for the DiProI beamline, FERMI.

Written by S. Sala, August 2016.

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

# Parameters for the h5 file
H5_PATHS = u.Param()
H5_PATHS.frame_pattern = 'image/ccd1'
H5_PATHS.motor_x = 'DPI/SampleX'
H5_PATHS.motor_y = 'DPI/SampleY'
# H5_PATHS.energy = '??/??/' #it seems like this hasn't been stored
FLAT_PATHS = u.Param()
FLAT_PATHS.key = "flat"

# DiProI recipe default parameters
RECIPE = u.Param()
RECIPE.base_path = None
RECIPE.scan_name = None             # this has to be a string (e.g. 'Cycle001')
RECIPE.run_ID = None                # this has to be a string (e.g. 'Scan018')
RECIPE.dark_name = None             # this has to be a string (e.g. 'Dark')
RECIPE.dark_value = 200.            # Used if dark_number is None
RECIPE.detector_flat_file = None
RECIPE.h5_file_pattern = '%(base_path)s/imported/%(run_ID)s/%(scan_name)s/rawdata/'
RECIPE.dark_h5_file_pattern = '%(base_path)s/imported/%(run_ID)s/%(dark_name)s/rawdata/'
RECIPE.date = None
RECIPE.motors = ['sample_x', 'sample_y']  # check orientation
RECIPE.energy = None
RECIPE.lam = None
RECIPE.z = None
RECIPE.motors_multiplier = 1e-3     # DiProI-specific
RECIPE.mask_file = None             # Mask file name
RECIPE.use_refined_positions = False
RECIPE.refined_positions_pattern = '%(base_path)s/imported/%(run_ID)s/%(scan_name)s/'
RECIPE.flat_division = False        # Switch for flat division
RECIPE.dark_subtraction = False     # Switch for dark subtraction

# Default generic parameter set from
DiProI_FERMIDEFAULT = PtyScan.DEFAULT.copy()
DiProI_FERMIDEFAULT.recipe = RECIPE
DiProI_FERMIDEFAULT.auto_center = False


class DiProIFERMIScan(PtyScan):
    DEFAULT = DiProI_FERMIDEFAULT

    def __init__(self, pars=None, **kwargs):
        """
        DiProI (FERMI) data preparation class.
        """
        # Initialize parent class. All updated parameters are now in self.info
        recipe_default = RECIPE.copy()
        recipe_default.update(pars.recipe, in_place_depth=1)
        pars.recipe.update(recipe_default)

        super(DiProIFERMIScan, self).__init__(pars, **kwargs)

        # Check whether base_path exists
        if self.info.recipe.base_path is None:
            raise RuntimeError('Base path missing.')

        # Construct the file names
        self.h5_filename_list = sorted([i for i in os.listdir(
            self.info.recipe.h5_file_pattern % self.info.recipe)
                                        if not i.startswith('.')])

        # Path to data files
        self.data_path = (self.info.recipe.h5_file_pattern %
                          self.info.recipe)

        u.log(3, 'Will read data from h5 files in {data_path}'.format(
            data_path=self.data_path))

        # Path to data files
        self.dark_path = (self.info.recipe.dark_h5_file_pattern %
                          self.info.recipe)

        u.log(3, 'Will read dark from h5 files in {dark_path}'.format(
            dark_path=self.dark_path))

        # Check whether ptyd file name exists
        if self.info.dfile is None:
            raise RuntimeError('Save path (dfile) missing.')

    def load_weight(self):
        """
        Function description see parent class. For now, this function will be
        used to load the mask.
        """
        # FIXME: do something better here. (detector-dependent)
        # Load mask as weight
        if self.info.recipe.mask_file is not None:
            return io.h5read(self.info.recipe.mask_file, 'mask')['mask'].astype(
                np.float32)

    def load_positions(self):
        """
        Load the positions and return as an (N, 2) array.
        """
        mmult = u.expect2(self.info.recipe.motors_multiplier)

        # Load positions
        if self.info.recipe.use_refined_positions:
            # From prepared .h5 file
            positions = io.h5read(self.info.recipe.refined_positions_pattern %
                                  self.info.recipe + '/Fermi_reconstruction.h5',
                                  'data.probe_positions')['probe_positions']

            positions = [(positions[0, i], positions[1, i])
                         for i in range(positions.shape[-1])]
            positions = np.array(positions)
        else:
            # From raw data
            key_x = H5_PATHS.motor_x
            key_y = H5_PATHS.motor_y
            positions = [(io.h5read(self.data_path + i, key_x)[key_x].tolist(),
                         (io.h5read(self.data_path + i, key_y)[key_y].tolist()))
                         for i in self.h5_filename_list]

            positions = np.array(positions) * mmult[0]

        # load the positions => check required structure vs currently dict with (x,y)
        ### is this less efficient than loading the whole file and calling individually raw and pos when needed?

        return positions

    def load_common(self):
        """
        loading dark and flat
        """
        common = u.Param()
        key = H5_PATHS.frame_pattern

        if self.info.recipe.dark_name is not None:
            dark = [io.h5read(self.dark_path + i, key)[key].astype(np.float32)
                    for i in os.listdir(self.dark_path) if i.startswith('Dark')]
        else:
            dark = self.info.recipe.dark_value

        if self.info.recipe.detector_flat_file is not None:
            flat = io.h5read(self.info.recipe.detector_flat_file,
                             FLAT_PATHS.key)[FLAT_PATHS.key]
        else:
            flat = 1.

        common.dark = np.array(dark).mean(0)
        common.flat = flat

        return common

    def load(self, indices):
        """
        Load data frames.

        :param indices:
        :return:
        """
        #pos not actually necessary at this stage: left empty
        raw = {}  # Container for the frames
        pos = {}  # Container for the positions
        weights = {}  # Container for the weights
        key = H5_PATHS.frame_pattern

        for i in range(len(indices)):
            raw[i] = io.h5read(self.data_path + self.h5_filename_list[i],
                               key)[key].astype(np.float32)

        return raw, pos, weights

    def correct(self, raw, weights, common):
        """
        Apply (eventual) corrections to the raw frames. Convert from "raw"
        frames to usable data.
        :param raw:
        :param weights:
        :param common:
        :return:
        """
        # Apply flat and dark, only dark, or no correction
        if self.info.recipe.flat_division and self.info.recipe.dark_subtraction:
            for j in raw:
                raw[j] = (raw[j] - common.dark) / (common.flat - common.dark)
                raw[j][raw[j] < 0] = 0
            data = raw
        elif self.info.recipe.dark_subtraction:
            for j in raw:
                raw[j] = raw[j] - common.dark
                raw[j][raw[j] < 0] = 0
            data = raw
        else:
            data = raw

        # FIXME: this will depend on the detector type used.

        weights = weights

        return data, weights
