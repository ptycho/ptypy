# -*- coding: utf-8 -*-
"""
Data preparation class for P06 at Petra III.
"""
import os
import os.path
import itertools

from IPython.utils.wildcard import is_type

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


__all__ = ["P06Scan", "P06Scan_scanning_mirror"]


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

    [position_bounds]
    default =
    type = float, list, tuple
    help = Omit data outside bounding box. Given as ((xmin, xmax), (ymin, ymax))

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

        # CRITICAL: Pre-calculate filtered frame count during initialization
        self.all_positions = self.load_positions()
        self.all_selected_inds = self.filter_by_position(self.all_positions)
        self.num_frames = len(self.all_selected_inds)
        logger.info(f"Set num_frames to {self.num_frames} for PtyPy")

    def filter_by_position(self, all_positions):
        # Apply position bounds filtering if specified
        if self.info.position_bounds:
            (xmin, xmax), (ymin, ymax) = self.info.position_bounds
            in_box = np.logical_and.reduce([
                all_positions[:, 0] >= ymin,
                all_positions[:, 0] <= ymax,
                all_positions[:, 1] >= xmin,
                all_positions[:, 1] <= xmax,
            ])
            all_selected_inds = np.nonzero(in_box)[0]
        else:
            all_selected_inds = np.arange(len(all_positions))

        return all_selected_inds

    def __len__(self):
        """Return the number of available frames after filtering"""
        if hasattr(self, 'num_frames'):
            return self.num_frames
        else:
            # Fallback to calculating on the fly
            return len(self.load_positions())

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
            #logger.warning("note: x motor is specified as flipped")
        if self.info.yMotorFlipped:
            yFlipper = -1
            #logger.warning("note: y motor is specified as flipped")

        # if the x/y axis is tilted with respect to the beam axis, take that into account.
        xCosFactor = np.cos(self.info.xMotorAngle / 180.0 * np.pi)
        yCosFactor = np.cos(self.info.yMotorAngle / 180.0 * np.pi)
        #logger.info("x and y motor angles result in multiplication by %.2f, %.2f" % (xCosFactor, yCosFactor))

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

    def load(self, indices, disable_frame_filtering=False):
        """
        Loads data from P06 beamtime.

        Parameters
        ----------
        indices : list
            The frame indices (in the filtered frame stack) to be loaded.

        """
        logger.info(
            f"loading frames in index range ({indices[0]} - {indices[-1]})")
        raw, weights, positions = {}, {}, {}

        detector_directory = os.path.join(
            self.info.scan_path_raw, self.info.detector
        )
        detector_file_list = sorted([
            os.path.join(detector_directory, x) for x in os.listdir(detector_directory) if not ('master' in x)
        ])
        n_files = len(detector_file_list)
        with h5py.File(detector_file_list[0], 'r') as fp:
            frames = fp['entry/data/data'][:, :5, :5]
            frames_per_file = len(frames)

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



        # filter indices according to position_bounds
        all_positions = self.all_positions
        all_selected_inds = self.all_selected_inds

        # Filter to keep only indices that are selected
        i_consecutive = indices
        i_scan = all_selected_inds[i_consecutive]
        if not is_type(i_scan, np.ndarray):
            i_scan = [i_scan]

        # To avoid opening and closing the file for each frame, list valid indices for each file
        per_file_inds = {i: {"i_scan":[], "i_consecutive":[], "i_in_file":[]} for i in range(n_files)}
        for i_c in i_consecutive:
            i_s = all_selected_inds[i_c]
            i_file, i_in_file = divmod(i_s, frames_per_file)
            per_file_inds[i_file]["i_scan"].append(i_s)
            per_file_inds[i_file]["i_consecutive"].append(i_c)
            per_file_inds[i_file]["i_in_file"].append(i_in_file)

        # actually loading the detector frames
        for i_file, valid_indices in per_file_inds.items():
            i_in_file = valid_indices["i_in_file"]
            with h5py.File(detector_file_list[i_file], 'r') as fp:
                # load only a cropped bit of the full frames
                if self.info.cropOnLoad:
                    frames = fp['entry/data/data'][i_in_file,
                            self.info.cropOnLoad_y_lower:self.info.cropOnLoad_y_upper,
                            self.info.cropOnLoad_x_lower:self.info.cropOnLoad_x_upper]
                # load the full raw frames
                else:
                    frames = fp['entry/data/data' % self.info.detector][i_in_file]

            # Put frames in raw dictionary
            # First, determine the index of the kept frames in the reduced
            # filtered frame stack
            for i, i_c in enumerate(valid_indices["i_consecutive"]):
                raw[i_c] = self.pad_to_size(frames[i], -1)
                positions[i_c] = all_positions[all_selected_inds[i_c]]
                weights[i_c] = np.ones(self.info.shape)

        logger.info(
            f"Loaded {len(raw)} frames. The rest was filtered out.")

        return raw, positions, weights

    def load_weight(self):
        """
        Provides the mask used for every single diffraction pattern of the whole scan.
        """
        # load the first non-masked frame
        for i in itertools.count():
            raw, _, _ = self.load(indices=(i,), disable_frame_filtering=True)
            print(i, raw)
            if i in raw:
                data = raw[i]
                break

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


@register()
class P06Scan_scanning_mirror(P06Scan):
    """
    This class loads scanning mirror data at the P06 beamline.
    """

    def __init__(self, pars=None, **kwargs):
        self.p = self.DEFAULT.copy()
        if pars is not None:
            self.p.update(pars)
        self.p.update(kwargs)

        super(P06Scan_scanning_mirror, self).__init__(self.p)

        if self.info.center is None:
            raise ValueError("center may not be None for P06Scan_scanning_mirror")

    def load_positions(self):
        positions = super(P06Scan_scanning_mirror, self).load_positions()
        center_of_mass = self.load_center_of_mass()

        if positions.shape != center_of_mass.shape:
            raise ValueError(f"positions.shape != center_of_mass.shape, i.e. {positions.shape} != {center_of_mass.shape}")
        positions = np.hstack([
            positions,
            center_of_mass
        ])
        print(positions)
        return positions

    def load_center_of_mass(self):
        positions_path = self.info.positions_path
        com = {}  # center of mass

        # fake center of mass data
        # positions = super(P06Scan_scanning_mirror, self).load_positions()
        # print(positions_path)
        # with h5py.File(positions_path, 'a') as f:
        #     f.create_dataset(name=f'detectors/{self.info.detector}/centre_of_mass/x', data=positions[:, 0]*0.5)
        #     f.create_dataset(name=f'detectors/{self.info.detector}/centre_of_mass/y', data=positions[:, 1]*0.5)
        # asdasd

        with h5py.File(positions_path, 'r') as f:
            com['x'] = np.array(f[f'detectors/{self.info.detector}/centre_of_mass/x'][:])  # unit assumed to be pixels
            com['y'] = np.array(f[f'detectors/{self.info.detector}/centre_of_mass/y'][:])  # unit assumed to be pixels
        nan_mask = np.logical_not(np.isnan(com['x']))
        com['x'] = com['x'][nan_mask]  # necessary for cmesh
        com['y'] = com['y'][nan_mask]  # necessary for cmesh

        # subtract cropping center
        com['x'] = com['x'] - self.info.center[1]  # order of center is (y, x)
        com['y'] = com['y'] - self.info.center[0]  # order of center is (y, x)

        # put the two arrays together
        center_of_mass = np.vstack((com['y'], com['x'])).T

        return center_of_mass
