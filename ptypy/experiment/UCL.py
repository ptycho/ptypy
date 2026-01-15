# -*- coding: utf-8 -*-
"""
Scan loading recipe for the laser imaging setup, UCL.

This file is part of the PTYPY package.

    :copyright: Copyright 2014 by the PTYPY team, see AUTHORS.
    :license: see LICENSE for details.
"""
import numpy as np
import os

from .. import utils as u
from .. import io
from ..core.data import PtyScan
from ..utils.verbose import log
from ..core.paths import Paths
from ..core import Ptycho
from . import register

IO_par = Ptycho.DEFAULT['io']

logger = u.verbose.logger


@register()
class UCLLaserScan(PtyScan):
    """
    Laser imaging setup (UCL) data preparation class.

    Defaults:

    [name]
    default = UCLLaserScan
    type = str
    help = 

    [auto_center]
    default = False

    [orientation]
    default = (False, False, False)

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

    [energy]
    default = None

    [lam]
    default = None
    type = float
    help =

    [z]
    default = None
    type = float
    help = Distance from object to screen

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

    [base_path]
    default = './'
    type = str
    help = 

    [data_file_path]
    default = '%(base_path)s' + 'raw/%(scan_number)06d'
    type = str
    help = 

    [dark_file_path]
    default = '%(base_path)s' + 'raw/%(dark_number)06d'
    type = str
    help =

    [flat_file_path]
    default = '%(base_path)s' + 'raw/%(flat_number)06d'
    type = str
    help =

    [mask_file]
    default = None
    type = str
    help =

    [use_EP]
    default = False
    type = bool
    help = Use flat as Empty Probe (EP) for probe sharing needs to be set to True in the recipe of the scan that will act as EP

    [remove_hot_pixels]
    default =
    type = Param
    help = Apply hot pixel correction

    [remove_hot_pixels.apply]
    default = False 
    type = bool
    help = Initiate by setting to True

    [remove_hot_pixels.size]
    default = 3
    type = int
    help = Size of the window on which the median filter will be applied around every data point

    [remove_hot_pixels.tolerance]
    default = 10
    type = int
    help = Tolerance multiplied with the standard deviation of the data array subtracted by the blurred array (difference array) yields the threshold for cutoff.

    [remove_hot_pixels.ignore_edges]
    default = False
    type = bool
    help = If True, edges of the array are ignored, which speeds up the code

    [rl_deconvolution]
    default =
    type = Param
    help = Apply Richardson Lucy deconvolution

    [rl_deconvolution.apply]
    default = False
    type = bool
    help = Initiate by setting to True

    [rl_deconvolution.numiter]
    default = 5
    type = int
    help = Number of iterations

    [rl_deconvolution.dfile]
    default = None
    type = str
    help = Provide MTF from file; no loading procedure present for now, loading through recon script required

    [rl_deconvolution.gaussians]
    default =
    type = Param
    help = Create fake psf as a sum of gaussians if no MTF provided

    [rl_deconvolution.gaussians.g1]
    default =
    type = Param
    help = list of gaussians for Richardson Lucy deconvolution

    [rl_deconvolution.gaussians.g1.std_x]
    default = 1.0
    type = float
    help = Standard deviation in x direction

    [rl_deconvolution.gaussians.g1.std_y]
    default = 1.0
    type = float
    help = Standard deviation in y direction

    [rl_deconvolution.gaussians.g1.off_x]
    default = 0.0
    type = float
    help = Offset / shift in x direction

    [rl_deconvolution.gaussians.g1.off_y]
    default = 0.0
    type = float
    help = Offset / shift in y direction

    """

    def __init__(self, pars=None, **kwargs):
        """
        Initializes parent class.
        """

        p = self.DEFAULT.copy(99)
        p.update(pars)
        pars = p

        super(UCLLaserScan, self).__init__(pars, **kwargs)

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

        # Construct path names
        self.data_path = self.info.data_file_path % self.info
        log(3, 'Will read data from directory %s' % self.data_path)
        if self.info.dark_number is None:
            self.dark_file = None
            log(3, 'No data for dark')
        else:
            self.dark_path = self.info.dark_file_path % self.info
            log(3, 'Will read dark from directory %s' % self.dark_path)
        if self.info.flat_number is None:
            self.flat_file = None
            log(3, 'No data for flat')
        else:
            self.flat_path = self.info.flat_file_path % self.info
            log(3, 'Will read flat from file %s' % self.flat_path)

        # Load data information
        self.instrument = io.h5read(self.data_path + '/%06d_%04d.nxs'
                                    % (self.info.scan_number, 1),
                                    'entry.instrument')['instrument']

        # Extract detector name if not set or wrong
        if (self.info.detector_name is None
                or self.info.detector_name
                not in self.instrument.keys()):
            detector_name = None
            for k in self.instrument.keys():
                if 'data' in self.instrument[k]:
                    detector_name = k
                    break

            if detector_name is None:
                raise RuntimeError(
                    'Not possible to extract detector name. '
                    'Please specify in recipe instead.')
            elif (self.info.detector_name is not None
                  and detector_name is not self.info.detector_name):
                u.log(2, 'Detector name changed from %s to %s.'
                      % (self.info.detector_name, detector_name))
        else:
            detector_name = self.info.detector_name

        self.info.detector_name = detector_name

        # Set up dimensions for cropping
        try:
            # Switch for attributes which are set to None
            # Will be removed once None attributes are removed
            center = pars.center
        except AttributeError:
            center = 'unset'

        # Check if dimension tuple is provided
        if type(center) == tuple:
            offset_x = pars.center[0]
            offset_y = pars.center[1]
        # If center unset, extract offset from raw data
        elif center == 'unset':
            raw_shape = self.instrument[
                self.info.detector_name]['data'].shape
            offset_x = raw_shape[-1] // 2
            offset_y = raw_shape[-2] // 2
        else:
            raise RuntimeError(
                'Center provided is not of type tuple or set to "unset". '
                'Please correct input parameters.')

        xdim = (offset_x - pars.shape // 2, offset_x + pars.shape // 2)
        ydim = (offset_y - pars.shape // 2, offset_y + pars.shape // 2)

        self.info.array_dim = [xdim, ydim]

        # Create the ptyd file name if not specified
        if self.info.dfile is None:
            home = Paths(IO_par).home
            self.info.dfile = ('%s/prepdata/data_%d.ptyd'
                               % (home, self.info.scan_number))
            log(3, 'Save file is %s' % self.info.dfile)

        log(4, u.verbose.report(self.info))

    def load_weight(self):
        """
        For now, this function will be used to load the mask.

        Function description see parent class.

        """
        # FIXME: do something better here. (detector-dependent)
        # Load mask as weight
        if self.info.mask_file is not None:
            return io.h5read(
                self.info.mask_file, 'mask')['mask'].astype(float)

    def load_positions(self):
        """
        Load the positions and return as an (N,2) array.

        """
        # Load positions from file if possible.
        motor_positions = io.h5read(
            self.info.base_path + '/raw/%06d/%06d_metadata.h5'
            % (self.info.scan_number, self.info.scan_number),
            'positions')['positions']

        # If no positions are found at all, raise error.
        if motor_positions is None:
            raise RuntimeError('Could not find motors.')

        # Apply motor conversion factor and create transposed array.
        mmult = u.expect2(self.info.motors_multiplier)
        positions = motor_positions * mmult[0]

        return positions

    def load_common(self):
        """
        Load dark and flat.

        """
        common = u.Param()

        # Load dark.
        if self.info.dark_number is not None:
            dark = [io.h5read(self.dark_path + '/%06d_%04d.nxs'
                              % (self.info.dark_number, j),
                              'entry.instrument.detector.data')['data'][0][
                    self.info.array_dim[1][0]:
                    self.info.array_dim[1][1],
                    self.info.array_dim[0][0]:
                    self.info.array_dim[0][1]].astype(np.float32)
                    for j in np.arange(1, len(os.listdir(self.dark_path)))]

            dark = np.array(dark).mean(0)
            common.dark = dark
            log(3, 'Dark loaded successfully.')

        # Load flat.
        if self.info.flat_number is not None:
            flat = [io.h5read(self.flat_path + '/%06d_%04d.nxs'
                              % (self.info.flat_number, j),
                              'entry.instrument.detector.data')['data'][0][
                    self.info.array_dim[1][0]:
                    self.info.array_dim[1][1],
                    self.info.array_dim[0][0]:
                    self.info.array_dim[0][1]].astype(np.float32)
                    for j in np.arange(1, len(os.listdir(self.flat_path)))]

            flat = np.array(flat).mean(0)
            common.flat = flat
            log(3, 'Flat loaded successfully.')

        return common

    def load(self, indices):
        """
        Load frames given by the indices.
        """
        raw = {}
        pos = {}
        weights = {}

        for j in np.arange(1, len(indices) + 1):
            data = io.h5read(self.data_path + '/%06d_%04d.nxs'
                             % (self.info.scan_number, j),
                             'entry.instrument.detector.data')['data'][0][
                   self.info.array_dim[1][0]:
                   self.info.array_dim[1][1],
                   self.info.array_dim[0][0]:
                   self.info.array_dim[0][1]].astype(np.float32)
            raw[j - 1] = data
        log(3, 'Data loaded successfully.')

        return raw, pos, weights

    def correct(self, raw, weights, common):

        # Apply hot pixel removal
        if self.info.remove_hot_pixels.apply:
            u.log(3, 'Applying hot pixel removal...')
            for j in raw:
                raw[j] = u.remove_hot_pixels(
                    raw[j],
                    self.info.remove_hot_pixels.size,
                    self.info.remove_hot_pixels.tolerance,
                    self.info.remove_hot_pixels.ignore_edges)[0]

            if self.info.flat_number is not None:
                common.dark = u.remove_hot_pixels(
                    common.dark,
                    self.info.remove_hot_pixels.size,
                    self.info.remove_hot_pixels.tolerance,
                    self.info.remove_hot_pixels.ignore_edges)[0]

            if self.info.flat_number is not None:
                common.flat = u.remove_hot_pixels(
                    common.flat,
                    self.info.remove_hot_pixels.size,
                    self.info.remove_hot_pixels.tolerance,
                    self.info.remove_hot_pixels.ignore_edges)[0]

            u.log(3, 'Hot pixel removal completed.')

        # Apply deconvolution
        if self.info.rl_deconvolution.apply:
            u.log(3, 'Applying deconvolution...')

            # Use mtf from a file if provided in recon script
            if self.info.rl_deconvolution.dfile is not None:
                mtf = self.info.rl_deconvolution.dfile
            # Create fake psf as a sum of gaussians from parameters
            else:
                gau_sum = 0
                for k in (
                        self.info.rl_deconvolution.gaussians.items()):
                    gau_sum += u.gaussian2D(raw[0].shape[0],
                                            k[1].std_x,
                                            k[1].std_y,
                                            k[1].off_x,
                                            k[1].off_y)

                # Compute mtf
                mtf = np.abs(np.fft.fft2(gau_sum))

            for j in raw:
                raw[j] = u.rl_deconvolution(
                    raw[j],
                    mtf,
                    self.info.rl_deconvolution.numiter)

            u.log(3, 'Deconvolution completed.')

        # Apply flat and dark, only dark, or no correction
        if (self.info.flat_number is not None
                and self.info.dark_number is not None):
            for j in raw:
                raw[j] = (raw[j] - common.dark) / (common.flat - common.dark)
                raw[j][raw[j] < 0] = 0
            data = raw
        elif self.info.dark_number is not None:
            for j in raw:
                raw[j] = raw[j] - common.dark
                raw[j][raw[j] < 0] = 0
            data = raw
        else:
            data = raw

        # FIXME: this will depend on the detector type used.
        weights = weights

        return data, weights
