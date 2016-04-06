# -*- coding: utf-8 -*-
"""\
Scan loading recipe for the laser imaging setup, UCL.

This file is part of the PTYPY package.

	:copyright: Copyright 2014 by the PTYPY team, see AUTHORS.
	:license: GPLv2, see LICENSE for details.
"""

import numpy as np
import os
from .. import utils as u
from .. import io
# from ..utils import parallel
from ..core.data import PtyScan
from ..utils.verbose import log
from ..core.paths import Paths
from ..core import DEFAULT_io as IO_par

logger = u.verbose.logger

# Recipe defaults
RECIPE = u.Param()
RECIPE.experimentID = None  # Experiment identifier
RECIPE.scan_number = None  # scan number
RECIPE.dark_number = None
RECIPE.flat_number = None
RECIPE.energy = None
RECIPE.lam = None  # 1.2398e-9 / RECIPE.energy
RECIPE.z = None  # Distance from object to screen
RECIPE.detector_name = None  # Name of the detector as specified in the nexus file
RECIPE.motors = ['t1_sx', 't1_sy']  # Motor names to determine the sample translation
RECIPE.motors_multiplier = 1e-3  # Motor conversion factor to meters
RECIPE.base_path = './'
RECIPE.data_file_path = '%(base_path)s' + 'raw/%(scan_number)06d'
RECIPE.dark_file_path = '%(base_path)s' + 'raw/%(dark_number)06d'
RECIPE.flat_file_path = '%(base_path)s' + 'raw/%(flat_number)06d'
RECIPE.mask_file = None  # '%(base_path)s' + 'processing/mask.h5'
RECIPE.NFP_correct_positions = False  # Position corrections for NFP beamtime Oct 2014
RECIPE.use_EP = False  # Use flat as Empty Probe (EP) for probe sharing; needs to be set to True in the recipe of the scan that will act as EP
RECIPE.remove_hot_pixels = u.Param(  # Apply hot pixel correction
    apply=False,  # Initiate by setting to True; DEFAULT parameters will be used if not specified otherwise
    size=3,  # Size of the window on which the median filter will be applied around every data point
    tolerance=10,  # Tolerance multiplied with the standard deviation of the data array subtracted by the blurred array
    # (difference array) yields the threshold for cutoff.
    ignore_edges=False,  # If True, edges of the array are ignored, which speeds up the code
)

# Generic defaults
UCLDEFAULT = PtyScan.DEFAULT.copy()
UCLDEFAULT.recipe = RECIPE
UCLDEFAULT.auto_center = False
UCLDEFAULT.orientation = (False, False, False)


class UCLLaserScan(PtyScan):
    DEFAULT = UCLDEFAULT

    def __init__(self, pars=None, **kwargs):
        """
		Laser imaging setup (UCL) data preparation class.
		"""
        # Initialise parent class
        recipe_default = RECIPE.copy()
        recipe_default.update(pars.recipe, in_place_depth=1)
        pars.recipe.update(recipe_default)

        super(UCLLaserScan, self).__init__(pars, **kwargs)

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

        # Construct path names
        self.data_path = self.info.recipe.data_file_path % self.info.recipe
        log(3, 'Will read data from directory %s' % self.data_path)
        if self.info.recipe.dark_number is None:
            self.dark_file = None
            log(3, 'No data for dark')
        else:
            self.dark_path = self.info.recipe.dark_file_path % self.info.recipe
            log(3, 'Will read dark from directory %s' % self.dark_path)
        if self.info.recipe.flat_number is None:
            self.flat_file = None
            log(3, 'No data for flat')
        else:
            self.flat_path = self.info.recipe.flat_file_path % self.info.recipe
            log(3, 'Will read flat from file %s' % self.flat_path)

        # Create the ptyd file name if not specified
        if self.info.dfile is None:
            home = Paths(IO_par).home
            self.info.dfile = '%s/prepdata/data_%d.ptyd' % (home, self.info.recipe.scan_number)
            log(3, 'Save file is %s' % self.info.dfile)
        log(4, u.verbose.report(self.info))

    def load_weight(self):
        """
		Function description see parent class. For now, this function will be used to load the mask.
		"""
        # FIXME: do something better here. (detector-dependent)
        # Load mask as weight
        if self.info.recipe.mask_file is not None:
            return io.h5read(self.info.recipe.mask_file, 'mask')['mask'].astype(float)

    def load_positions(self):
        """
		Load the positions and return as an (N,2) array
		"""
        # Load positions from file if possible.
        motor_positions = io.h5read(self.info.recipe.base_path + '/raw/%06d/%06d_metadata.h5'
                                    % (self.info.recipe.scan_number, self.info.recipe.scan_number),
                                    'positions')['positions']

        # If no positions are found at all, raise error.
        if motor_positions is None:
            raise RuntimeError('Could not find motors.')

        # Apply motor conversion factor and create transposed array of positions.
        mmult = u.expect2(self.info.recipe.motors_multiplier)
        positions = motor_positions * mmult[0]

        return positions

    def load_common(self):
        """
		Load dark and flat.
		"""
        common = u.Param()

        # Load dark.
        if self.info.recipe.dark_number is not None:
            dark = []

            for j in np.arange(1, len(os.listdir(self.dark_path))):
                data = io.h5read(self.dark_path + '/%06d_%04d.nxs' % (self.info.recipe.dark_number, j),
                                 'entry.instrument.detector.data')['data'][0].astype(np.float32)
                data = data[10:-10, 10:-10]
                if self.info.recipe.remove_hot_pixels.apply:
                    data = u.remove_hot_pixels(data, self.info.recipe.remove_hot_pixels.size,
                                               self.info.recipe.remove_hot_pixels.tolerance,
                                               self.info.recipe.remove_hot_pixels.ignore_edges)[0]
                dark.append(data)
            dark = np.array(dark).mean(0)
            common.dark = dark
            log(3, 'Dark loaded successfully.')

        # Load flat.
        if self.info.recipe.flat_number is not None:
            flat = []

            for j in np.arange(1, len(os.listdir(self.flat_path))):
                data = io.h5read(self.flat_path + '/%06d_%04d.nxs' % (self.info.recipe.flat_number, j),
                                 'entry.instrument.detector.data')['data'][0].astype(np.float32)
                data = data[10:-10, 10:-10]
                if self.info.recipe.remove_hot_pixels.apply:
                    data = u.remove_hot_pixels(data, self.info.recipe.remove_hot_pixels.size,
                                               self.info.recipe.remove_hot_pixels.tolerance,
                                               self.info.recipe.remove_hot_pixels.ignore_edges)[0]
                flat.append(data)
            flat = np.array(flat).mean(0)
            common.flat = flat
            log(3, 'Flat loaded successfully.')

        return common

    def load(self, indices):
        """
		Load frames given by the indices.

		:param indices:
		:return:
		"""
        raw = {}
        pos = {}
        weights = {}

        for j in np.arange(1, len(indices) + 1):
            data = io.h5read(self.data_path + '/%06d_%04d.nxs' % (self.info.recipe.scan_number, j),
                             'entry.instrument.detector.data')['data'][0].astype(np.float32)
            data = data[10:-10, 10:-10]
            if self.info.recipe.remove_hot_pixels.apply:
                data = u.remove_hot_pixels(data, self.info.recipe.remove_hot_pixels.size,
                                           self.info.recipe.remove_hot_pixels.tolerance,
                                           self.info.recipe.remove_hot_pixels.ignore_edges)[0]
            raw[j - 1] = data
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

        # Apply flat and dark, only dark, or no correction
        if self.info.recipe.flat_number is not None and self.info.recipe.dark_number is not None:
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
