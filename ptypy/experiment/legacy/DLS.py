# -*- coding: utf-8 -*-
"""\
Scan loading recipe for the I13 beamline, Diamond.

This file is part of the PTYPY package.

    :copyright: Copyright 2014 by the PTYPY team, see AUTHORS.
    :license: see LICENSE for details.
"""

import numpy as np
import h5py as h5

from ptypy import utils as u
from ptypy import io
from ptypy.experiment import register
from ptypy.core.data import PtyScan
from ptypy.utils.verbose import log
from ptypy.core.paths import Paths
from ptypy.core import Ptycho

IO_par = Ptycho.DEFAULT['io']

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


@register()
class DlsScan(PtyScan):
    """
    I13 (Diamond Light Source) data preparation class.

    Defaults:

    [name]
    default = 'DlsScan'
    type = str
    help =

    [is_swmr]
    default = False
    type = bool
    help = 

    [israster]
    default = 0
    type = int
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
    default = 'merlin_sw_hdf'
    type = str
    help = Name of the detector 
    doc = As specified in the nexus file.

    [motors]
    default = ['t1_sx', 't1_sy']
    type = list
    help = Motor names to determine the sample translation

    [motors_multiplier]
    default = [1e-6,-1e-6]
    type = list
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

    [NFP_correct_positions]
    default = False
    type = bool
    help = Position corrections for NFP beamtime Oct 2014

    [use_EP]
    default = False
    type = bool
    help = Use flat as Empty Probe (EP) for probe sharing
    doc = Needs to be set to True in the recipe of the scan that will act as EP.

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
    default = 10
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
    type = int, tuple, list

    """

    def __init__(self, pars=None, **kwargs):
        """
        I13 (Diamond Light Source) data preparation class.
        """
        log(2, "The DLS loader will be deprecated in the next release. Please use the Hdf5Loader.")
        # Initialise parent class
        p = self.DEFAULT.copy(99)
        p.update(pars)

        super(DlsScan, self).__init__(p, **kwargs)
        self.data_file = self.info.data_file_pattern  % self.info


        # Create the ptyd file name if not specified
        if self.info.dfile is None:
            home = Paths(IO_par).home
            self.info.dfile = '%s/prepdata/data_%d.ptyd' % (home, self.info.scan_number)
            log(3, 'Save file is %s' % self.info.dfile)
        log(4, u.verbose.report(self.info))

    def load_weight(self):
        """
        Function description see parent class. For now, this function will be used to load the mask.
        """
        # FIXME: do something better here. (detector-dependent)
        # Load mask as weight
        if self.info.mask_file is not None:
            return io.h5read(self.info.mask_file % self.info, 'mask')['mask'].astype(float)

    def load_positions(self):
        """
        Load the positions and return as an (N,2) array
        """
        # Load positions from file if possible.
        if self.info.is_swmr:
            instrument = h5.File(self.data_file, 'r', libver='latest', swmr=True)[NEXUS_PATHS.instrument % self.info]
        else:
            instrument = h5.File(self.data_file, 'r')[NEXUS_PATHS.instrument % self.info]
        if self.info.israster:
            self.position_shape = instrument[0].shape
        motor_positions = []
        i=0
        mmult = u.expect2(self.info.motors_multiplier)

        for k in NEXUS_PATHS.motors:
            if not self.info.israster:
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
        if not self.info.is_swmr:
            npos = self.num_frames
            frames_accessible = min((frames, npos - start))
            stop = self.frames_accessible + start
            return frames_accessible, (stop >= npos)
        else:
            f = h5.File(self.data_file, 'r', libver='latest', swmr=True)
            dset= f[NEXUS_PATHS.live_key_pattern % self.info]
            dset.id.refresh()
            num_avail = len(dset)-start
            frames_accessible = min((frames, num_avail))
            stop = f[NEXUS_PATHS.finished_pattern][0] and (self.num_frames == start)
            f.close()
            print("HERE",frames_accessible, stop)
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
        key = NEXUS_PATHS.frame_pattern % self.info
        if not self.info.israster:
            for j in indices:
                if not self.info.is_swmr:
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
            if not self.info.is_swmr:
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



