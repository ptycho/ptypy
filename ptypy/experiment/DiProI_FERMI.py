# -*- coding: utf-8 -*-
"""\
Data preparation for the DiProI beamline, FERMI.

Started by S. Sala, August 2016.

This file is part of the PTYPY package.

    :copyright: Copyright 2014 by the PTYPY team, see AUTHORS.
    :license: GPLv2, see LICENSE for details.

"""

import numpy as np
import os

from .. import utils as u
from . import register
from .. import io
from ..core.data import PtyScan
from ..core import Ptycho

IO_par = Ptycho.DEFAULT['io']
# testing commits again
# Parameters for the h5 file
H5_PATHS = u.Param()
#H5_PATHS.frame_pattern = 'image/ccd1' # no longer used
# functionality moved to data_key so different key can be given as input
H5_PATHS.motor_x = 'DPI/SampleX'
H5_PATHS.motor_y = 'DPI/SampleY'
# H5_PATHS.energy = '??/??/' #it seems like this hasn't been stored
DARK_PATHS = u.Param()
DARK_PATHS.key = 'data'
FLAT_PATHS = u.Param()
FLAT_PATHS.key = 'data'


@register()
class DiProIFERMIScan(PtyScan):
    """
    DiProI (FERMI) data preparation class.

    Defaults:

    [name]
    default = DiProIFERMIScan
    type = str
    help =

    [base_path]
    default = None
    type = str
    help = 

    [date]
    default = None
    type = str
    help = timestamp-like date for documenting recons

    [scan_name]
    default = None
    type = str
    help = has to be a string (e.g. 'Cycle001')

    [run_ID]
    default = None
    type = str
    help = has to be a string (e.g. 'Scan018')

    [data_file]
    default = '%(base_path)s/raw/%(scan_name)s.hdf'
    type = str
    help = path to data file

    [data_key]
    default = 'image/ccd1'
    type = str
    help = key to load raws from data file

    [mask_file]
    default = None
    type = str
    help = path to mask file

    [posi_file]
    default = None
    type = str
    help = path to positions file

    [posi_key]
    default = None
    type = str
    help = position key referring to subset of indices

    [motors_multiplier]
    default = 1e-3
    type = float
    help = specific to DiProI

    [motors_multiplier_refined]
    default = 1.0
    type = float
    help = specific to refined positions

    [dark_file]
    default = None
    type = str
    help = path do dark file

    [dark_value]
    default = 200.0
    type = float
    help = Used for dark subtraction if dark_number is None

    [dark_std]
    default = None
    type = float
    help = Multiplying dark_std within dark subtraction

    [dark_subtraction]
    default = False
    type = bool
    help = Switch for dark subtraction

    [flat_file]
    default = None
    type = str
    help = path to flat file

    [flat_division]
    default = False
    type = bool
    help = Switch for flat division

    """

    def __init__(self, pars=None, **kwargs):
        """
        DiProI (FERMI) data preparation class.
        """

        p = self.DEFAULT.copy(99)
        p.update(pars)

        # Initialize parent class. All updated parameters are now in self.info
        super(DiProIFERMIScan, self).__init__(p, **kwargs)

        # Check whether base_path exists
        if self.info.base_path is None:
            raise RuntimeError('Base path missing.')

        u.log(3, 'Will read data from file {data_file}'.format(
            data_file=self.info.data_file))

        u.log(3, 'Will read dark from file {dark_file}'.format(
            dark_file=self.info.dark_file))

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
        if self.info.mask_file is not None:
            return io.h5read(self.info.mask_file)['mask'].astype(np.float32)
        elif self.info.shape is not None:
            return np.ones(u.expect2(self.info.shape), dtype='bool')
        else:
            return None

    def load_positions(self):
        """
        Load the positions and return as an (N, 2) array.
        """
        mmult = u.expect2(self.info.motors_multiplier)

        # Load positions
        if self.info.posi_file is None:
            # From raw data
            key_x = H5_PATHS.motor_x
            key_y = H5_PATHS.motor_y
            positions = np.array(
                [(io.h5read(self.info.data_file, key_x)[key_x].tolist()),
                 (io.h5read(self.info.data_file, key_y)[key_y].tolist())]).T
        else:
            positions = io.h5read(self.info.posi_file % self.info,
                                  'data.probe_positions')['probe_positions'].T
            positions *= self.info.motors_multiplier_refined

        positions *= mmult[0]

        # load the positions => check required structure vs currently dict with (x,y)
        ### is this less efficient than loading the whole file and calling individually
        # raw and pos when needed?
        # ---> 20180512 SS: yes, it is; positions might be loaded from a different file than raws
        # (e.g. '2016_06_FERMIDiProI_ptycho' required separate position refinement step)

        if self.info.posi_key is not None:
            # use subset of indices
            indices_used = io.h5read(self.info.posi_file % self.info,
                                     self.info.posi_key)[
                                     self.info.posi_key][0].astype(int) - 1
            positions = positions[indices_used]

        return positions

    def load_common(self):
        """
        loading dark and flat
        """
        common = u.Param()

        if self.info.dark_file is not None:
            dark = io.h5read(self.info.dark_file,
                             DARK_PATHS.key)[DARK_PATHS.key]
        else:
            dark = [self.info.dark_value]

        if self.info.flat_file is not None:
            flat = io.h5read(self.info.flat_file,
                             FLAT_PATHS.key)[FLAT_PATHS.key]
        else:
            flat = 1.

        common.dark = np.array(dark).mean(0)
        common.dark_std = np.array(dark).std(0)
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
        key = self.info.data_key

        if self.info.posi_key is None:
            indices_used = indices
        else:
            # use subset of indices
            key_pos = self.info.posi_key
            indices_used = io.h5read(self.info.posi_file
                % self.info, key_pos)[key_pos][0].astype(int) - 1

        raw_temp = io.h5read(self.info.data_file % self.info,
                             key)[key].astype(np.float32)

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
        if self.info.flat_division and self.info.dark_subtraction:
            for j in raw:
                raw[j] = (raw[j] - common.dark) / (common.flat - common.dark)
                raw[j][raw[j] < 0] = 0
        elif self.info.dark_subtraction:
            for j in raw:
                raw[j] = raw[j] - common.dark
                if self.info.dark_std is not None:
                    #raw[j] = raw[j] - (self.info.dark_std*common.dark_std)
                    raw[j][raw[j]<self.info.dark_std*common.dark_std] = 0.
                raw[j][raw[j]<0] = 0.

        data = raw

        # FIXME: this will depend on the detector type used.
        ### 20180516 SS: this script is specific to DiProI@FERMI
        # => unlikely to frequently change detector

        weights = weights

        return data, weights
