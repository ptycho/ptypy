# -*- coding: utf-8 -*-
"""\
Scan loading recipe for the I13 beamline, Diamond.

This file is part of the PTYPY package.

    :copyright: Copyright 2014 by the PTYPY team, see AUTHORS.
    :license: GPLv2, see LICENSE for details.
"""

import numpy as np
import os
from .. import utils as u
from .. import io
from ..utils import parallel
from ..core.data import PtyScan
from ..utils.verbose import log
from ..core.paths import Paths
#from ..core import DEFAULT_io as IO_par
from ..core import Ptycho
IO_par = Ptycho.DEFAULT['io']
import h5py as h5

logger = u.verbose.logger

# Parameters for the nexus file saved by GDA
NEXUS_PATHS = u.Param()
NEXUS_PATHS.instrument = 'entry1/%(detector_name)s'
NEXUS_PATHS.frame_pattern = 'entry1/%(detector_name)s/data'
NEXUS_PATHS.live_key_pattern = 'entry1/%(detector_name)s/live_key'
NEXUS_PATHS.finished_pattern = 'entry1/live/finished'
NEXUS_PATHS.exposure = 'entry1/%(detector_name)s/count_time'
NEXUS_PATHS.motors = ['lab_sy', 'lab_sx']
# NEXUS_PATHS.motors = ['t1_sy', 't1_sx']
# NEXUS_PATHS.motors = ['lab_sy', 'lab_sx']
#NEXUS_PATHS.motors = ['lab_sy', 'lab_sx']
NEXUS_PATHS.command = 'entry1/scan_command'
NEXUS_PATHS.label = 'entry1/entry_identifier'
NEXUS_PATHS.experiment = 'entry1/experiment_identifier'

# Recipe defaults
RECIPE = u.Param()
RECIPE.is_swmr = False
RECIPE.israster = 0
RECIPE.experimentID = None      # Experiment identifier
RECIPE.scan_number = None       # scan number
RECIPE.dark_number = None
RECIPE.flat_number = None
RECIPE.energy = None
RECIPE.lam = None               # 1.2398e-9 / RECIPE.energy
RECIPE.z = None                 # Distance from object to screen
RECIPE.detector_name = 'merlin_sw_hdf'     # Name of the detector as specified in the nexus file
RECIPE.motors = ['t1_sx', 't1_sy']      # Motor names to determine the sample translation
# RECIPE.motors_multiplier = 1e-6         # Motor conversion factor to meters
RECIPE.motors_multiplier = [1e-6,-1e-6]         # Motor conversion factor to meters
RECIPE.base_path = './'
RECIPE.data_file_pattern = '%(base_path)s' + 'raw/%(scan_number)05d.nxs'
RECIPE.dark_file_pattern = '%(base_path)s' + 'raw/%(dark_number)05d.nxs'
RECIPE.flat_file_pattern = '%(base_path)s' + 'raw/%(flat_number)05d.nxs'
RECIPE.mask_file = None                 # '%(base_path)s' + 'processing/mask.h5'
RECIPE.NFP_correct_positions = False    # Position corrections for NFP beamtime Oct 2014
RECIPE.use_EP = False                   # Use flat as Empty Probe (EP) for probe sharing; needs to be set to True in the recipe of the scan that will act as EP'
RECIPE.remove_hot_pixels = u.Param(         # Apply hot pixel correction
    apply = False,                          # Initiate by setting to True; DEFAULT parameters will be used if not specified otherwise
    size = 3,                               # Size of the window on which the median filter will be applied around every data point
    tolerance = 10,                         # Tolerance multiplied with the standard deviation of the data array subtracted by the blurred array
                                            # (difference array) yields the threshold for cutoff.
    ignore_edges = False,                   # If True, edges of the array are ignored, which speeds up the code
)

# Generic defaults
I13DEFAULT = PtyScan.DEFAULT.copy()
I13DEFAULT.recipe = RECIPE
I13DEFAULT.auto_center = False
I13DEFAULT.orientation = (False, False, False)


class DlsScan(PtyScan):
    DEFAULT = I13DEFAULT

    def __init__(self, pars=None, **kwargs):
        """
        I13 (Diamond Light Source) data preparation class.
        """
        # Initialise parent class
        recipe_default = RECIPE.copy()
        recipe_default.update(pars.recipe, in_place_depth=5)
        pars.recipe.update(recipe_default)

        super(DlsScan, self).__init__(pars, **kwargs)
        self.data_file = self.info.recipe.data_file_pattern  % self.info.recipe


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
            return io.h5read(self.info.recipe.mask_file % self.info.recipe, 'mask')['mask'].astype(float)

    def load_positions(self):
        """
        Load the positions and return as an (N,2) array
        """
        # Load positions from file if possible.
        if self.info.recipe.is_swmr:
            instrument = h5.File(self.data_file, 'r', libver='latest', swmr=True)[NEXUS_PATHS.instrument % self.info.recipe]
        else:
            instrument = h5.File(self.data_file, 'r')[NEXUS_PATHS.instrument % self.info.recipe]
        if self.info.recipe.israster:
            self.position_shape = instrument[0].shape
        motor_positions = []
        i=0
        mmult = u.expect2(self.info.recipe.motors_multiplier)

        for k in NEXUS_PATHS.motors:
            if not self.info.recipe.israster:
                motor_positions.append(instrument[k]*mmult[i])
            else:
                motor_positions.append((instrument[k]*mmult[i]).ravel())
            i+=1

        positions = np.array(motor_positions).T
        return positions

    def check(self, frames, start):
        """
        Returns the number of frames available from starting index `start`, and whether the end of the scan
        was reached.

        :param frames: Number of frames to load
        :param start: starting point
        :return: (frames_available, end_of_scan)
        - the number of frames available from a starting point `start`
        - bool if the end of scan was reached (None if this routine doesn't know)
        """
        if not self.info.recipe.is_swmr:
            npos = self.num_frames
            frames_accessible = min((frames, npos - start))
            stop = self.frames_accessible + start
            return frames_accessible, (stop >= npos)
        else:
            f = h5.File(self.data_file, 'r', libver='latest', swmr=True)
            dset= f[NEXUS_PATHS.live_key_pattern % self.info.recipe]
            dset.id.refresh()
            num_avail = len(dset)-start
            frames_accessible = min((frames, num_avail))
            stop = f[NEXUS_PATHS.finished_pattern][0] and (self.num_frames == start)
            f.close()
            print "HERE",frames_accessible, stop
            return frames_accessible,stop

    def load(self, indices):
        """
        Load frames given by the indices.

        :param indices:
        :return:
        """
        raw = {}
        pos = {}
        weights = {}
        key = NEXUS_PATHS.frame_pattern % self.info.recipe
        if not self.info.recipe.israster:
            for j in indices:
                if not self.info.recipe.is_swmr:
#                     print "frame number "+str(j)
                    data = io.h5read(self.data_file, key, slice=j)[key].astype(np.float32)
                    raw[j] = data
                else:

                    #print "frame number "+str(j)
                    dset= h5.File(self.data_file, 'r', libver='latest', swmr=True)[key]
                    dset.id.refresh()
                    #print dset.shape
                    raw[j] = dset[j]
                    dset.file.close()
        else:
            if not self.info.recipe.is_swmr:
                data = h5.File(self.data_file)[key]
                sh = data.shape
                for j in indices:
                    raw[j]=data[j % sh[0], j // sh[1]] # or the other way round???
            else:
                dset= h5.File(self.data_file, 'r', libver='latest', swmr=True)[key]
                dset.id.refresh()
                sh = self.position_shape
                for j in indices:
                    raw[j]=data[j % sh[0], j // sh[1]]
        log(3, 'Data loaded successfully.')
        return raw, pos, weights



