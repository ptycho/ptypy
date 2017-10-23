# -*- coding: utf-8 -*-
"""\
Scan loading recipe for the AMO beamline, LCLS. (Data preprocessed by hummingbird)

This file is part of the PTYPY package.

    :copyright: Copyright 2014 by the PTYPY team, see AUTHORS.
    :license: GPLv2, see LICENSE for details.
"""

import numpy as np
import os
from .. import utils as u
from .. import io
from .. import core
from ..utils import parallel
from ..core.data import PtyScan
from ..utils.verbose import log
from ..core.paths import Paths
#from ..core import DEFAULT_io as IO_par
from ..core import Ptycho
from ..utils.descriptor import defaults_tree
IO_par = Ptycho.DEFAULT['io']

logger = u.verbose.logger

# Parameters for the h5 file saved by hummingbird
H5_PATHS = u.Param()
H5_PATHS.frame_pattern = 'data/photons'


@defaults_tree.parse_doc('scandata.AMOScan')
class AMOScan(core.data.PtyScan):
    """
    Defaults:

    [name]
    default = AMOScan
    type = str
    help =

    [experimentID]
    default = None
    type = str
    help = Experiment identifier

    [scan_number]
    default = None
    type = int
    help = 

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
    help = Name of the detector as specified in the nexus file

    [motors]
    default = ['t1_sx', 't1_sy']
    type = list
    help = Motor names to determine the sample translation

    [motors_multiplier]
    default = 1e-3
    type = float
    help = Motor conversion factor to meters
    doc = 1e-3 AMO, 1e-6 for CXI

    [base_path]
    default = './'
    type = str
    help = 

    [data_file_pattern]
    default = '%(base_path)sinput/r%(scan_number)04d.h5'
    type = str
    help = 

    [dark_file_pattern]
    default = '%(base_path)sinput/r%(dark_number)04d.h5'
    type = str
    help = 

    [flat_file_pattern]
    default = '%(base_path)sinput/r%(flat_number)04d.h5'
    type = str
    help = 

    [mask_file]
    default = None
    type = str
    help = 

    [averaging_number]
    default = 1
    type = int
    help = Number of frames to be averaged

    [auto_center]
    default = False
    type = bool
    help = Overrides PtyScan default

    [orientation]
    default = (False, False, False)
    type = tuple
    help = Overrides PtyScan default

    """

    def __init__(self, pars=None, **kwargs):
        """
        AMO (Atomic Molecular and Optical Science, LCLS) data preparation class.
        """
        # Initialise parent class
        p = self.DEFAULT.copy(99)
        p.update(pars)

        super(AMOScan, self).__init__(p, **kwargs)

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

        # Default scan label
        # if self.info.label is None:
        #    self.info.label = 'S%5d' % rinfo.scan_number

        # Construct file names
        self.data_file = self.info.data_file_pattern % self.info
        log(3, 'Will read data from file %s' % self.data_file)
        if self.info.dark_number is None:
            self.dark_file = None
            log(3, 'No data for dark')
        else:
            self.dark_file = self.info.dark_file_pattern % self.info
            log(3, 'Will read dark from file %s' % self.dark_file)
        if self.info.flat_number is None:
            self.flat_file = None
            log(3, 'No data for flat')
        else:
            self.flat_file = self.info.flat_file_pattern % self.info
            log(3, 'Will read flat from file %s' % self.flat_file)

        # Create the ptyd file name if not specified
        if self.info.dfile is None:
            home = Paths(IO_par).home
            self.info.dfile = '%s/prepdata/data_%05d.ptyd' % (home, self.info.scan_number)
            log(3, 'Save file is %s' % self.info.dfile)
        log(4, u.verbose.report(self.info))

    def load_common(self):
        """
        Load dark, flat, and mask file.
        """
        common = u.Param()

        # FIXME: do something better here. (detector-dependent)
        # Load mask
        # common.weight2d = None
        if self.info.mask_file is not None:
            common.weight2d = io.h5read(self.info.mask_file, 'mask')['mask'].astype(float)

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
        for i in range(0,len(x),self.info.averaging_number):
            pos_list.append([y[i],x[i]])
        positions = np.array(pos_list)

        return positions

    def check(self, frames, start=0):
        """
        Returns the number of frames available from starting index `start`, and whether the end of the scan
        was reached.

        :param frames: Number of frames to load
        :param start: starting point
        :return: (frames_available, end_of_scan)
        - the number of frames available from a starting point `start`
        - bool if the end of scan was reached (None if this routine doesn't know)
        """
        npos = self.num_frames
        frames_accessible = min((frames, npos-start))
        stop = self.frames_accessible + start
        return frames_accessible, (stop >= npos)

    def load(self, indices):
        """
        Load frames given by the indices.

        :param indices:
        :return:
        """
        raw = {}
        pos = {}
        weights = {}

        i = 0
        h = 0
        key = H5_PATHS.frame_pattern % self.info
        for j in indices:
            mean = []
            while h < (i+self.info.averaging_number):
                mean.append(io.h5read(self.data_file, H5_PATHS.frame_pattern % self.info, slice=h)[key].astype(np.float32))
                h+=1
            raw[j] = np.array(mean).mean(0).T
            i+=self.info.averaging_number
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
            data[k][data[k] < 1] = 0
        weights = weights
        return data, weights
