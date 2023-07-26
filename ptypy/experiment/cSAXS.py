# -*- coding: utf-8 -*-
"""
Data preparation class for cSAXS at the SLS
This scan file was correct as of the end of 2017. It's not yet maintained by cSAXS,
 but if you raise a ticket if you notice any oddities, we will try to fix it up.
"""

import numpy as np
import os
import fabio
from collections import OrderedDict
from scipy.io import loadmat

from .. import utils as u
from ..core.data import PtyScan
from ..utils.verbose import log
from . import register
logger = u.verbose.logger



@register()
class cSAXSScan(PtyScan):
    def __init__(self, pars=None, **kwargs):
        '''
        Defaults:
        [name]
        default = cSAXS
        type = str
        help = the reference name of this loader
        doc =

        [base_path]
        default = None
        type = str
        help = the base path for reconstruction
        doc =

        [visit]
        default = None
        type = str
        help = the visit number
        doc =

        [detector]
        default = pilatus_1
        type = str
        help = the detector used for acquisition
        doc =

        [scan_number]
        default = None
        type = int
        help = the scan number for reconstruction
        doc =

        [motors]
        default = ('Target_y', 'Target_x')
        type = tuple
        help = the motor names, this should be a 2-tuple
        doc =


        [mask_path]
        default = None
        type = str
        help = the motor names, this should be a 2-tuple
        doc =

        [mask_file]
        default = None
        type = str
        help = the mask mat file
        doc =

        '''

        # Initialise parent class
        self.p = self.DEFAULT.copy(99)
        self.p.update(pars)
        self.data_object = None
        super(cSAXSScan, self).__init__(pars, **kwargs)

        #         if parallel.master: # populate the fabio object
        self.data_object = get_data_object(self.info.recipe)
        self.num_frames = self.data_object.shape[0]
        log(4, u.verbose.report(self.info))

    def load_weight(self):
        return loadmat(get_mask_path(self.info.recipe))['mask']

    def load_positions(self):
        return get_positions(self.info.recipe)

    def check(self, frames, start):
        frames_accessible = min((frames, self.num_frames - start))
        stop = self.frames_accessible + start
        return frames_accessible, (stop >= self.num_frames)

    def load(self, indices):
        raw = {}
        for i in indices:
            raw[i] = self.data_object.getframe(i).data.astype(float)
        return raw, {}, {}


def get_position_path(inargs):
    '''
    returns a validated data path to the first file for a given scan number. If it's not valid, it will return None.
    '''
    pathargs = inargs.copy()  # this is ok here
    file_path = '%(base_path)s/specES1/scan_positions/' % pathargs
    positions_path = file_path + 'scan_%(scan_number)05d.dat' % pathargs
    # check that it exists
    if not os.path.isfile(positions_path):
        print("File:%s does not exist." % positions_path)
        exists = False
    else:
        exists = True
    return positions_path, exists


def get_positions(inargs):
    path, exists = get_position_path(inargs)
    k = 0
    if exists:
        with open(path) as f:
            for line in f:
                if k == 1:
                    keys = line.split()
                    break
                k += 1

        _DataDict = OrderedDict.fromkeys(keys)  #

        for i, key in enumerate(_DataDict.keys()):
            _DataDict[key] = np.loadtxt(path, skiprows=2, usecols=[i])  # data[i]
        positions = np.zeros((len(_DataDict[inargs.motors[0]]), 2))
        positions[:, 0] = _DataDict[inargs.motors[0]] * inargs.motors_multiplier[0]
        positions[:, 1] = _DataDict[inargs.motors[1]] * inargs.motors_multiplier[1]
        return positions
    else:
        raise IOError('File:%s does not exist.' % path)


def get_mask_path(inargs):
    pathargs = inargs.copy()  # this is ok here
    if pathargs.mask_path is None:
        file_path = '%(base_path)s/matlab/ptycho/%(mask_file)s' % pathargs
    else:
        file_path = pathargs.mask_path + pathargs.mask_file
    return file_path


def get_data_path(inargs):
    '''
    returns a validated data path to the first file for a given scan number. If it's not valid, it will return None.
    '''
    pathargs = inargs.copy()  # this is ok here
    pathargs['inner_number'] = 0
    pathargs['frame_number'] = 0
    scan_int = pathargs['scan_number'] // 1000
    pathargs['group_folder'] = "S%02d000-%02d999" % (scan_int, scan_int)
    file_path = '%(base_path)s/%(detector)s/%(group_folder)s/S%(scan_number)05d/' % pathargs
    frame_path = file_path + '%(visit)s_1_%(scan_number)05d_%(inner_number)05d_%(frame_number)05d.cbf' % pathargs
    num_frames = len([name for name in os.listdir(file_path) if os.path.isfile(file_path + name)])
    # check that it exists
    if not os.path.isfile(frame_path):
        print("File:%s does not exist." % frame_path)
        exists = False
    else:
        exists = True
    return frame_path, num_frames, exists


def get_data_object(inargs):
    path, numframes, exists = get_data_path(inargs)
    if not exists:
        raise IOError('File:%s does not exist.' % path)
    else:
        _data = fabio.open(path, 0)
        _data.shape = (numframes, _data.dim1, _data.dim2)
        _data.scan_number = inargs['scan_number']
        return _data
