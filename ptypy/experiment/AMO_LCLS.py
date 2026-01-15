# -*- coding: utf-8 -*-
"""\
Scan loading recipe for the AMO beamline, LCLS. (Data preprocessed by hummingbird)

This file is part of the PTYPY package.

    :copyright: Copyright 2014 by the PTYPY team, see AUTHORS.
    :license: see LICENSE for details.
"""

import numpy as np

from .. import utils as u
from .. import io
from . import register
from ..core.data import PtyScan
from ..utils.verbose import log
from ..core.paths import Paths
from ..core import Ptycho

IO_par = Ptycho.DEFAULT['io']
logger = u.verbose.logger

# Parameters for the h5 file saved by hummingbird
H5_PATHS = u.Param()
H5_PATHS.frame_pattern = 'data/photons'

@register()
class AMOScan(PtyScan):
    """
    Defaults:

    [name]
    default = 'AMOScan'
    type = str
    help =

    [date]
    default = None
    type = str
    help = timestamp-like date for documenting recons

    [scan_number]
    default = None
    type = int
    help = indicate scan number (label)

    [motors_multiplier]
    default = 1e-3
    type = float
    help = Motor conversion factor to meters
    doc = 1e-3 AMO, 1e-6 for CXI

    [data_file]
    default = None
    type = str
    help = path to data file

    [dark_file]
    default = None
    type = str
    help = path to dark file

    [flat_file]
    default = None
    type = str
    help = path to flat file

    [mask_file]
    default = None
    type = str
    help = path to mask file

    [threshold_correct]
    default = 0.
    type = float
    help = value used for thresholding raws (values below set to 0.)

    """

    def __init__(self, pars=None, **kwargs):
        """
        AMO (Atomic Molecular and Optical Science, LCLS) data preparation class.
        """
        # Initialise parent class
        p = self.DEFAULT.copy(99)
        p.update(pars)

        super(AMOScan, self).__init__(p, **kwargs)

        # Construct file names
        if self.info.data_file is None:
            raise RuntimeError('No file for data provided.')
        self.data_file = self.info.data_file
        log(3, 'Will read data from file %s' % self.data_file)
        if self.info.dark_file is None:
            self.dark_file = None
            log(3, 'No data for dark')
        else:
            self.dark_file = self.info.dark_file
            log(3, 'Will read dark from file %s' % self.dark_file)
        if self.info.flat_file is None:
            self.flat_file = None
            log(3, 'No data for flat')
        else:
            self.flat_file = self.info.flat_file
            log(3, 'Will read flat from file %s' % self.flat_file)

        # Create the ptyd file name if not specified
        if self.info.dfile is None:
            home = Paths(IO_par).home
            self.info.dfile = '%s/prepdata/data_%05d.ptyd' % (home, self.info.scan_number)
            log(3, 'Save file is %s' % self.info.dfile)
        log(4, u.verbose.report(self.info))

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

    def load_common(self):
        """
        Load dark and flat file.
        """
        common = u.Param()

        return common

    def load_positions(self):
        """
        Load the positions and return as an (N,2) array
        """

        # Load positions from file if possible.
        mmult = u.expect2(self.info.motors_multiplier)
        x = mmult[0] * io.h5read(self.data_file, 'data.posx')['posx']
        y = mmult[1] * io.h5read(self.data_file, 'data.posy')['posy']

        pos_list = []
        for i in range(len(x)):
            pos_list.append([y[i],x[i]])
        positions = np.array(pos_list)

        return positions

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
            raw[j] = io.h5read(self.data_file, 'data.photons')[
                'photons'][j].astype(np.float32)
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
        # Apply corrections to frames
        data = raw
        for k in data.keys():
            data[k][data[k] < self.info.threshold_correct] = 0
        weights = weights
        return data, weights
