# -*- coding: utf-8 -*-
"""\
Data preparation for the DiProI beamline, FERMI.

Written by S. Sala, August 2016.

This file is part of the PTYPY package.

    :copyright: Copyright 2014 by the PTYPY team, see AUTHORS.
    :license: GPLv2, see LICENSE for details.

"""
import numpy as np
from .. import utils as u
from .. import io
from ..core.data import PtyScan
#from ..core.paths import Paths
#from ..core import DEFAULT_io as IO_par

# Parameters for the h5 file
H5_PATHS = u.Param()
H5_PATHS.frame_pattern = 'image/ccd1'
H5_PATHS.motor_x = 'DPI/SampleX'
H5_PATHS.motor_y = 'DPI/SampleY'
#H5_PATHS.energy = '??/??/' #it seems like this hasn't been stored
FLAT_PATHS = u.Param()
FLAT_PATHS.key = "flat"

# DiProI recipe default parameters
RECIPE = u.Param()
RECIPE.base_path = '../'
RECIPE.scan_name = None             # this has to be a string (e.g. 'Scan018')
RECIPE.data_file = '%(base_path)s/raw/%(scan_name)s.hdf'
RECIPE.frame_key = H5_PATHS.frame_pattern  # added to allow for Michal's raws to be loaded
RECIPE.dark_file = None
RECIPE.dark_value = None  # 400.            # Used if dark_file is None
RECIPE.date = None
RECIPE.energy = None
RECIPE.lam = None
RECIPE.z = None
RECIPE.motors_multiplier = 1e-3     # DiProI-specific
RECIPE.flat_file = None
RECIPE.mask_file = None             # Mask file name
RECIPE.position_file = None #'%(base_path)s/processing/recons_by_Michal.h5'
RECIPE.position_key = None          # used to select a subset of indices
RECIPE.refined_positions_multiplier = 1.68396935*1e-4

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

        # Path to data files
        self.data_path = (self.info.recipe.data_file %
                          self.info.recipe)

        u.log(3, 'Will read data from h5 files in {data_path}'.format(
                                               data_path=self.data_path))

        if self.info.recipe.dark_file is not None:
            u.log(3, 'Will read dark from {dark_file}'.format(
                            dark_file=self.info.recipe.dark_file))

        # Check whether ptyd file name exists
        if self.info.dfile is None:
            raise RuntimeError('Save path (dfile) missing.')


    def load_weight(self):
        """
        Function description see parent class. For now, this function will be
        used to load the mask.
        """
        # Load mask as weight
        if self.info.recipe.mask_file is not None:
            return io.h5read(self.info.recipe.mask_file, 'mask')['mask'].astype(
                np.float32)


    def load_positions(self):
        """
        Load the positions and return as an (N, 2) array.
        """
        mmult = u.expect2(self.info.recipe.motors_multiplier)
        if self.info.recipe.position_file is None:
            key_x = H5_PATHS.motor_x
            key_y = H5_PATHS.motor_y
            positions = np.array(
                        [(io.h5read(self.data_path, key_x)[key_x].tolist()),
                         (io.h5read(self.data_path, key_y)[key_y].tolist())]).T
        else:
            positions = io.h5read(self.info.recipe.position_file %
                self.info.recipe, 'data.probe_positions')['probe_positions'].T
            positions *= self.info.recipe.refined_positions_multiplier

        positions *= mmult

        if self.info.recipe.position_key is not None:
            indices_used = io.h5read(self.info.recipe.position_file %
                      self.info.recipe, self.info.recipe.position_key)[
                      self.info.recipe.position_key][0].astype(int) - 1
            positions = positions[indices_used]

        return positions


    def load_common(self):
        """
        loading dark and flat
        """
        common = u.Param()

        if self.info.recipe.dark_file is not None:
            dark = io.h5read(self.info.recipe.dark_file)['data']
            common.dark = np.array(dark).mean(0)
            common.dark_std = np.array(dark).std(0)
        else:
            common.dark = self.info.recipe.dark_value

        if self.info.recipe.flat_file is not None:
            common.flat = io.h5read(self.info.recipe.flat_file,
                            FLAT_PATHS.key)[FLAT_PATHS.key]

        return common


    def load(self, indices):
        """
        Load data frames.

        :param indices:
        :return:
        """
        raw = {}  # Container for the frames
        pos = {}  # Container for the positions
        weights = {}  # Container for the weights
        key = self.info.recipe.frame_key
        key_pos = self.info.recipe.position_key

        if self.info.recipe.position_key is None:
            indices_used = indices
        else:
            indices_used = io.h5read(self.info.recipe.position_file
                % self.info.recipe, key_pos)[key_pos][0].astype(int) - 1

        raw_temp = io.h5read(self.info.recipe.data_file, key)[key].astype(np.float32)
        for i in range(len(indices)):
            raw[i] = raw_temp[indices_used[i]]

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

        if self.info.recipe.flat_file and common.dark is not None:
            for j in raw:
                raw[j] = (raw[j] - common.dark) / (common.flat - common.dark)
                raw[j][raw[j] < 0] = 0

        elif common.dark is not None:

            for j in raw:

                raw[j] = raw[j] - common.dark
                if common.dark_std is not None:
                    raw[j] = raw[j] - (1.2*common.dark_std)

        data = raw

        weights = weights

        return data, weights
