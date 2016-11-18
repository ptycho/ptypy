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
# testing commits again
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
RECIPE.dark_value = 400.            # Used if dark_number is None
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
RECIPE.positions_version = None #can be 'original', 'refined'
RECIPE.positions_indices = None #can be 'all', 'good', 'minimal'
RECIPE.use_new_hdf_files = False
RECIPE.refined_positions_multiplier = 1.68396935*1e-4
RECIPE.refined_positions_pattern = '%(base_path)s/processing/'
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
        if not self.info.recipe.use_new_hdf_files:
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

        if self.info.recipe.use_new_hdf_files:
            u.log(3, 'Will read dark from h5 files in {data_path}'.format(
                                                   data_path=self.data_path))
        else:
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

    def load_positions_all(self):
        """
        Load all positions (regardless of specific index selection)
        """
        mmult = u.expect2(self.info.recipe.motors_multiplier)
        if (self.info.recipe.positions_version == 'original' or
            self.info.recipe.positions_version == None):
            key_x = H5_PATHS.motor_x
            key_y = H5_PATHS.motor_y
            if self.info.recipe.use_new_hdf_files:
                positions = [(io.h5read(self.data_path + self.info.recipe.run_ID
                                        + '.hdf', key_x)[key_x].tolist()),
                             (io.h5read(self.data_path + self.info.recipe.run_ID
                                        + '.hdf', key_y)[key_y].tolist())]
            else:
                positions = [(io.h5read(self.data_path + i, key_x)[key_x].tolist()
                              for i in self.h5_filename_list),
                             (io.h5read(self.data_path + i, key_y)[key_y].tolist()
                              for i in self.h5_filename_list)]
            positions = np.array(positions).T

        elif self.info.recipe.positions_version == 'refined':
            positions = io.h5read(self.info.recipe.refined_positions_pattern %
                            self.info.recipe + '/recons_by_Michal.h5',
                            'data.probe_positions')['probe_positions'].T
            positions *= self.info.recipe.refined_positions_multiplier
        else:
            raise RuntimeError('positions_version can only be None/original or refined.')

        positions *= mmult
        u.ipshell()
        return positions

    def load_positions(self):
        """
        Load the positions and return as an (N, 2) array.
        """
        positions = self.load_positions_all()

        if (self.info.recipe.positions_indices == 'all' or
            self.info.recipe.positions_indices == None):
            indices_used = positions.shape[0]
        elif self.info.recipe.positions_indices == 'good':
            indices_used = io.h5read(self.info.recipe.refined_positions_pattern %
                self.info.recipe + '/recons_by_Michal.h5', 'data.reconstruct_ind'
                                                    )['reconstruct_ind'][0].astype(int)-1
        elif self.info.recipe.positions_indices == 'minimal':
            indices_used = io.h5read(self.info.recipe.refined_positions_pattern %
                self.info.recipe + '/recons_by_Michal.h5', 'data.reconstruct_ind_minimal'
                                            )['reconstruct_ind_minimal'][0].astype(int)-1
        else:
            raise RuntimeError('positions_indices can only be None/all, good or minimal.')

        if not self.info.recipe.use_new_hdf_files:
            for i in range(indices_used.shape[0]):
                if indices_used[i] > len(self.h5_filename_list):
                    indices_used = indices_used[:i]
                    break
        u.ipshell()
        positions = positions[indices_used]
        return positions

    def load_common(self):
        """
        loading dark and flat
        """
        common = u.Param()
        key = H5_PATHS.frame_pattern

        if self.info.recipe.dark_name is not None:
            if self.info.recipe.use_new_hdf_files:
                dark = io.h5read(self.data_path + self.info.recipe.run_ID
                                            + '_dark.hdf')['data']
            else:
                u.log(3, 'Loading darks: one frame per file.')
                dark = [io.h5read(self.dark_path + i, key)[key].astype(np.float32)
                       for i in os.listdir(self.dark_path) if i.startswith('Dark')]
            common.dark = np.array(dark).mean(0)
            common.dark_std = np.array(dark).std(0)
        else:
            dark = self.info.recipe.dark_value
            common.dark = dark

        if self.info.recipe.detector_flat_file is not None:
            flat = io.h5read(self.info.recipe.detector_flat_file,
                             FLAT_PATHS.key)[FLAT_PATHS.key]
        else:
            flat = 1.
        common.flat = flat

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
        key = H5_PATHS.frame_pattern

        if (self.info.recipe.positions_indices == 'all' or
            self.info.recipe.positions_indices == None):
            indices_used = indices
        elif self.info.recipe.positions_indices == 'good':
            indices_used = io.h5read(self.info.recipe.refined_positions_pattern %
                    self.info.recipe + '/recons_by_Michal.h5','data.reconstruct_ind'
                                                )['reconstruct_ind'][0].astype(int)-1
        elif self.info.recipe.positions_indices == 'minimal':
            indices_used = io.h5read(self.info.recipe.refined_positions_pattern %
                self.info.recipe + '/recons_by_Michal.h5','data.reconstruct_ind_minimal'
                                            )['reconstruct_ind_minimal'][0].astype(int)-1
        else:
            raise RuntimeError('positions_indices can only be None/all, good or minimal.')

        if self.info.recipe.use_new_hdf_files:
            raw_temp = io.h5read(self.data_path + self.info.recipe.run_ID + '.hdf',
                                     key)[key].astype(np.float32)
            for i in range(len(indices)):
            #for i in indices:
                raw[i] = raw_temp[indices_used[i]]
        else:
            u.log(3, 'Loading frames: one frame per file.')
            for i in range(len(indices)):
            # for i in indices:
                raw[i] = io.h5read(self.data_path + self.h5_filename_list[
                              indices_used[i]],key)[key].astype(np.float32)
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
        elif self.info.recipe.dark_subtraction:

            ##raw_medians = []
            ##raw_means   = []

            for j in raw:

                # average dark subtraction
                raw[j] = raw[j] - common.dark

                # thresholding
                raw[j][raw[j] < (3.*common.dark_std)] = 0.

                # normalizing to centre of frame
                #raw_means.append(raw[j][447:509, 456:513].mean())
                #raw[j] = raw[j] / raw_means[j]

                # normalizing to corner of frame
                #raw[j] = raw[j] / np.median(raw[j][-160:,:160])

                # normalizing to full frame
                ##raw_medians.append(np.median(raw[j][raw[j]>0.]))
                ##raw_means.append(raw[j][raw[j] > 0.].mean())
            #min_value  = np.array( raw_medians).min()
            #norm_value = np.median(raw_means) # - (min_value-1.)
            ##for j in raw:
            ##    raw[j] = raw[j] / ( raw_medians[j] )# - (min_value-1.) )
            ##    #raw[j] = raw[j] / raw_means[j]
            ##    raw[j] *= norm_value

            #for j in raw:

                # thresholding
                #raw[j][raw[j] < (1.*common.dark_std)] = 0.

                # signal to photons conversion
                #raw[j] = raw[j] /6.

        data = raw
        #u.log(3,'you are in data, i.e. after correction')
        #u.ipshell()

        # FIXME: this will depend on the detector type used.
        weights = weights

        return data, weights
