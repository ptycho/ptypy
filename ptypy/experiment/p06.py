# -*- coding: utf-8 -*-
"""
Data preparation class for P06 at Petra III.
"""
import os
import os.path

from ..core.data import PtyScan
from .. import utils as u
from . import register
logger = u.verbose.logger

import numpy as np
from PIL import Image
try:
    import hdf5plugin
except ImportError:
    logger.warning("Couldn't find hdf5plugin - better hope your h5py has bitshuffle!")
import h5py

__all__ = ["P06Scan"]


@register()
class P06Scan(PtyScan):
    """
    This class loads data at the P06 beamline.

    Defaults:

    [name]
    default = P06Scan
    type = str
    help =

    [scan_path_raw]
    default = None
    type = str
    help = Path to where the raw data is. Detector data is found in scan_path_raw/{detector}
    doc =

    [positions_path]
    default = None
    type = str
    help = Path to hdf5 file where positions are stored. The motor positions should be found under, for example, "/axes/{x_motor}"
    doc =

    [scanNumber]
    default = None
    type = int, list, tuple
    help = Scan number or list of scan numbers
    doc =

    [energy]
    default = None
    type = float
    help = photon energy in keV, if None it will be read from the nexus file
    doc =

    [energy]
    default = None
    type = float
    help = photon energy in keV, if None it will be read from the nexus file
    doc =

    [energy]
    default = None
    type = float
    help = photon energy in keV, if None it will be read from the nexus file
    doc =

    [nexus_path]
    default = None
    type = str
    help = Path to nexus file.
    doc =

    [cropOnLoad]
    default = True
    type = bool
    help = Only load the used bits of each detector frame
    doc =

    [cropOnLoad_y_lower]
    default = None
    type = int, list, tuple
    help = y-axis lower limit
    doc =

    [cropOnLoad_y_upper]
    default = None
    type = int, list, tuple
    help = y-axis upper limit
    doc =

    [cropOnLoad_x_lower]
    default = None
    type = int, list, tuple
    help = x-axis lower limit
    doc =

    [cropOnLoad_x_upper]
    default = None
    type = int, list, tuple
    help = x-axis upper limit
    doc =

    [tmp_center]
    default = None
    type = int, list, tuple
    help = x-axis upper limit
    doc =

    [xMotor]
    default = samy
    type = str
    help = Which x motor to use
    doc =

    [yMotor]
    default = samz
    type = str
    help = Which y motor to use
    doc =

    [xMotor_unit]
    default = None
    type = float, int
    help = Motor unit, i.e. 1 for meters, 1e-3 for mm, 1e-6 for microns, etc. \
    Lookup table will be used if value is None
    doc =

    [yMotor_unit]
    default = None
    type = float, int
    help = Motor unit, i.e. 1 for meters, 1e-3 for mm, 1e-6 for microns, etc. \
    Lookup table will be used if value is None
    doc =

    [xMotorFlipped]
    default = False
    type = bool
    help = Flip detector x positions
    doc =

    [yMotorFlipped]
    default = False
    type = bool
    help = Flip detector y positions
    doc =

    [xMotorAngle]
    default = 0.0
    type = float
    help = Angle of the motor x axis relative to the lab x axis
    doc =

    [yMotorAngle]
    default = 0.0
    type = float
    help = Angle of the motor y axis relative to the lab y axis
    doc =

    [xyAxisSkewOffset]
    default = 0.0
    type = float
    help = Relative rotation angle beyond the expected 90 degrees between the motor x and y axes in degree
    doc = If for example the scanner is damaged and x and y end up not beeing perfectly under 90 degrees, this value can can be used to correct for that.

    [zDetectorAngle]
    default = 0.0
    type = float
    help = Relative rotation angle between the motor x and y axes and the detector pixel rows and columns in degree
    doc = If the Detector is mounted rotated around the beam axis relative to the scanning motors, use this angle to rotate the motor position into the detector frame of reference. The rotation angle is in mathematical positive sense from the motors to the detector pixel grid.

    [detector]
    default = 'eiger_4m_01'
    type = str
    help = Which detector to use, can be pil100k or merlin

    [maskfile]
    default = None
    type = str
    help = Arbitrary mask file
    doc = Tiff file containing the mask or Hdf5 file containing an array called 'mask' at the root level.

    [I0]
    default = None
    type = str
    help = Normalization channel, like alba2/1 for example
    doc =

    """
    # Lookup table for motor units.
    UNITS = {
        'scanu': 1e-6,
        'scanv': 1e-6,
        'scanw': 1e-6,
        'scanx': 1e-6,
        'scany': 1e-6,
        'scanz': 1e-6,
        'samx': 1e-3,
        'samy': 1e-3,
        'samz': 1e-3,
        'hexx': 1e-3,
        'hexy': 1e-3,
        'hexz': 1e-3,
        'cenx': 1e-3,
        'ceny': 1e-3,
    }

    def __init__(self, pars=None, **kwargs):
        self.p = self.DEFAULT.copy()
        if pars is not None:
            self.p.update(pars)
        self.p.update(kwargs)
        super(P06Scan, self).__init__(self.p)

    def clean_mask(self, mask):
        mask[mask>=0.5] = 1
        mask[mask<0.5] = 0
        return mask

    def load_mask_h5(self):
        with h5py.File(self.info.maskfile, 'r') as hf:
            mask = np.array(hf.get('mask'))
        return self.clean_mask(mask)

    def load_mask_tiff(self):
        with Image.open(self.info.maskfile) as im:
            mask = np.array(im)
        return self.clean_mask(mask)

    def _check_unit(self, motor_name, motor_unit):
        """
        Checks that units are ok.

        Parameters
        ----------
        motor_name : str
        motor_unit : float, None
            The given motor unit parameter for the parameter tree. If None,
            the UNITS lookup table will be used.

        Returns
        -------
        float
            The unit, i.e. the number to multiply with in order to convert
            to meters.
        """
        if motor_name in self.UNITS:
            lookup_unit = self.UNITS[motor_name]
        else:
            lookup_unit = None

        if motor_unit is not None:
            unit = motor_unit
            if lookup_unit is not None and unit != lookup_unit:
                raise Warning(
                    f"The given unit for {motor_name}, is not equal to the unit defined in the lookup table, which is {lookup_unit}.")
        elif lookup_unit is not None:
            unit = lookup_unit
        else:
            raise NotImplementedError(
                f'There is no unit defined for {self.info.xMotor}. Please define xMotor_unit in the parameter tree')
        return unit

    def _apply_transformations(self, positions):
        xFlipper, yFlipper = 1, 1
        if self.info.xMotorFlipped:
            xFlipper = -1
            logger.warning("note: x motor is specified as flipped")
        if self.info.yMotorFlipped:
            yFlipper = -1
            logger.warning("note: y motor is specified as flipped")

        # if the x/y axis is tilted with respect to the beam axis, take that into account.
        from pprint import pprint
        print(self.info)
        xCosFactor = np.cos(self.info.xMotorAngle / 180.0 * np.pi)
        yCosFactor = np.cos(self.info.yMotorAngle / 180.0 * np.pi)
        logger.info(
            "x and y motor angles result in multiplication by %.2f, %.2f" % (
                xCosFactor, yCosFactor))

        x = positions[:, 1] * xFlipper * xCosFactor
        y = positions[:, 0] * yFlipper * yCosFactor

        chi_rad_x = 0
        chi_rad_y = 0
        # if the detector and motor frame of reference are rotated around the beam axis
        if self.info.zDetectorAngle != 0:
            chi_rad_x = self.info.zDetectorAngle / 180.0 * np.pi
            chi_rad_y = 1. * chi_rad_x
            logger.info(
                "x and y motor positions were rotated by %.4f degree to align with the detector pixel grid" % (
                    self.info.zDetectorAngle))

        # if x and y are not under 90 degrees to each other
        if self.info.xyAxisSkewOffset != 0:
            chi_rad_x += -0.5 * self.info.xyAxisSkewOffset / 180.0 * np.pi
            chi_rad_y += +0.5 * self.info.xyAxisSkewOffset / 180.0 * np.pi
            logger.info(
                "x and y motor positions were skewed by %.4f degree to each other" % (
                    self.info.xyAxisSkewOffset))
        x, y = np.cos(chi_rad_x) * x - np.sin(chi_rad_y) * y, np.sin(
            chi_rad_x) * x + np.cos(chi_rad_y) * y

        return np.vstack((y, x)).T

    def load_positions(self):
        positions_path = self.info.positions_path
        x_unit = self._check_unit(self.info.xMotor, self.info.xMotor_unit)
        y_unit = self._check_unit(self.info.yMotor, self.info.yMotor_unit)
        pos = {}
        with h5py.File(positions_path, 'r') as f:
            pos['x'] = np.array(f[f'/axes/{self.info.xMotor}'][:]) * x_unit
            pos['y'] = np.array(f[f'/axes/{self.info.yMotor}'][:]) * y_unit
        nan_mask = np.logical_not(np.isnan(pos['x']))
        pos['x'] = pos['x'][nan_mask]  # necessary for cmesh
        pos['y'] = pos['y'][nan_mask]  # necessary for cmesh

        # put the two arrays together
        positions = -np.vstack((pos['y'], pos['x'])).T

        # make transformations
        positions = self._apply_transformations(positions)

        # set minimum to zero so ptypy can work out the proper object size
        positions = positions - np.min(positions, axis=0)

        # This may or may not be needed.
        self.frames_per_scan = {}

        return positions



    def load(self, indices):
        raw, weights, positions = {}, {}, {}

        detector_directory = os.path.join(
            self.info.scan_path_raw, self.info.detector
        )
        detector_file_list = sorted([
            os.path.join(detector_directory, x) for x in os.listdir(detector_directory) if not ('master' in x)
        ])
        with h5py.File(detector_file_list[0], 'r') as fp:
            frames = fp['entry/data/data'][:, :5, :5]
            eiger_mod = len(frames)

        # crop on load is requested, but the actual indices to crop are not yet defined
        if self.info.cropOnLoad and self.info.cropOnLoad_y_lower == None:

            # center of the diffraction patterns is not explicitly given
            if self.info.center == None:
                # requires to load the first frame and to find the center of mass there
                with h5py.File(detector_file_list[0], 'r') as fp:
                    frame = fp['entry/data/data'][0]
                # and to mask the hot pixels ... sadly this will have double with self.load_weight
                mask = np.ones_like(frame)
                if self.info.detector == 'pilatus_1m_01':
                    mask[np.where(frame < 0)] = 0
                if self.info.detector == 'eiger_4m_01':
                    mask[np.where(frame == 2 ** 32 - 1)] = 0
                    mask[np.where(frame == 2 ** 16 - 1)] = 0
                if self.info.maskfile:
                    if self.info.maskfile.endswith('.h5'):
                        mask2 = self.load_mask_h5()
                    else:
                        mask2 = self.load_mask_tiff()
                    mask = mask * mask2
                # now find the center of mass can be estimated using the ptypy internal function and make it integers
                self.info.center = u.scripts.mass_center(frame * mask)
                self.info.center = [int(x) for x in self.info.center]
                logger.info(
                    f'Estimated the center of the (first) diffraction pattern to be {self.info.center}')

            # the center of the full frames is (now) known, and thus the indices for the cropping can be defined
            cy, cx = self.info.center
            dy, dx = self.info.shape
            logger.info(
                f'Found the center of the full frames at {self.info.center}')
            logger.info(
                f'Will crop all diffraction patterns on load to a size of {self.info.shape}')
            self.info.cropOnLoad_y_lower, self.info.cropOnLoad_x_lower = int(
                cy) - dy // 2, int(cx) - dy // 2
            self.info.cropOnLoad_y_upper, self.info.cropOnLoad_x_upper = self.info.cropOnLoad_y_lower + dy, self.info.cropOnLoad_x_lower + dx

            # the (temporary) center needs to be redefined for the cropped frames
            tmp_center_y, tmp_center_x = dy // 2, dx // 2

            # if the lower crop indices are negative, set them zero
            if self.info.cropOnLoad_y_lower < 0:
                tmp_center_y += self.info.cropOnLoad_y_lower
                self.info.cropOnLoad_y_lower = 0
            if self.info.cropOnLoad_x_lower < 0:
                tmp_center_x += self.info.cropOnLoad_x_lower
                self.info.cropOnLoad_x_lower = 0
                # no need to have something similar for too large upper indices due to the way python slices arrays

            # now fix the new center
            self.info.tmp_center = (tmp_center_y, tmp_center_x)
            self.info.center = (dy // 2, dx // 2)

        # set the photon energy
        path_options = ['scan/data/energy']
        if self.info.energy == None:
            with h5py.File(self.info.nexus_path, 'r') as fp:
                existing_paths = [x for x in path_options if x in fp.keys()]
                self.meta.energy = fp[existing_paths[0]][:] * 1e-3
        else:
            self.meta.energy = self.info.energy

        # actually loading the detector frames
        for ind in indices:
            i_file = ind // eiger_mod
            i_frame = ind % eiger_mod

            with h5py.File(detector_file_list[i_file], 'r') as fp:
                # load only a cropped bit of the full frame
                if self.info.cropOnLoad:
                    frame = fp['entry/data/data'][i_frame,
                            self.info.cropOnLoad_y_lower:self.info.cropOnLoad_y_upper,
                            self.info.cropOnLoad_x_lower:self.info.cropOnLoad_x_upper]
                    # print('--', ind, np.shape(frame))
                    raw[ind] = self.pad_to_size(frame, -1)
                # load the full raw frame
                else:
                    raw[ind] = fp['entry/data/data' % self.info.detector][ind]
                # if there is I0 information, use it to normalize the just loaded frame
                if self.info.I0 != None:
                    self.normdata = self.normdata.flatten()
                    # logger.info('normalizing frame %u by %f' % (ind, self.normdata[ind]))
                    # logger.info('hack! assuming mask = 2**32-1 when I0-normalizing')
                    msk = np.where(raw[ind] == 2 ** 32 - 1)
                    raw[ind] = np.round(raw[ind] / self.normdata[ind]).astype(
                        raw[ind].dtype)
                    raw[ind][msk] = 2 ** 32 - 1

        return raw, positions, weights

    def load_weight(self):
        """
        Provides the mask used for every single diffraction pattern of the whole scan.
        """

        r, w, p = self.load(indices=(0,))
        data = r[0]
        mask = np.ones_like(data)
        if self.info.detector == 'pilatus':
            mask[np.where(data < 0)] = 0
        if 'eiger' in self.info.detector:
            bit_depth = int(''.join(filter(str.isdigit, str(data.dtype))))
            logger.info(f"found bit depth of {bit_depth} in the eiger frames")
            logger.info(f"    -> masking all pixels with values of {2**(bit_depth) -1} and above")
            mask[np.where(data < 0)] = 0
            mask[np.where(data >= ((2**bit_depth)-1))] = 0

        logger.info("took account of the built-in mask, %u x %u, sum %u, so %u masked pixels" %
                    (mask.shape + (np.sum(mask), np.prod(mask.shape)-np.sum(mask))))

        if self.info.maskfile:
            if self.info.maskfile.endswith('.h5'):
                mask2 = self.load_mask_h5()
            else:
                mask2 = self.load_mask_tiff()

            if self.info.cropOnLoad:
                mask2 = mask2[
                        self.info.cropOnLoad_y_lower:self.info.cropOnLoad_y_upper,
                        self.info.cropOnLoad_x_lower:self.info.cropOnLoad_x_upper]
                mask2 = self.pad_to_size(mask2, 0)

            logger.info(
                "loaded additional mask, %u x %u, sum %u, so %u masked pixels" %
                (mask2.shape + (np.sum(mask2),
                                np.prod(mask2.shape) - np.sum(mask2))))
            mask = mask * mask2
            logger.info("total mask, %u x %u, sum %u, so %u masked pixels" %
                        (mask.shape + (np.sum(mask),
                                       np.prod(mask.shape) - np.sum(mask))))
        return mask

    def pad_to_size(self, frame, value):
        ny, nx = np.shape(frame)
        cy, cx = self.info.tmp_center
        dy, dx = self.info.shape
        ry, rx = dy//2 , dx//2
        pad_xl   = rx - cx
        pad_xu   = rx + cx - nx
        pad_yl   = ry - cy
        pad_yu   = ry + cy - ny
        return np.pad(frame, [[pad_yl,pad_yu],[pad_xl,pad_xu]], mode='constant', constant_values=[value])
