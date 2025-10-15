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
    help =
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

        self.all_positions = self.load_positions()
        self.all_selected_inds = self.filter_by_position(self.all_positions)
        self.num_frames = len(self.all_selected_inds)
        self.frames_per_file, ni, nj = self.determine_data_shape()
        self.detector_shape = (ni, nj)


        if self.num_frames == 0:
            raise IOError(
                f"Aborting, because no frames were selected. Check position bounds: {self.info.position_bounds}"
            )
        logger.info(f"Set num_frames to {self.num_frames} for PtyPy")

        if not self.info.cropOnLoad:
            # supporting non cropped data requires to allow center to not be in the middle of the frame.
            raise ValueError("cropOnLoad = False is not yet supported")

        if self.info.center is None:
            self.info.center = self.auto_center()

        # center is not the center of mass on the detector, however, it is used like that in all our config files and scripts, so it is kept for now.
        # The following is a workaround.
        self.detector_center = self.info.center  # center used to determine where to crop the raw data
        self.info.center = (self.info.shape[0] // 2, self.info.shape[1] // 2)

    def filter_by_position(self, all_positions):
        # Apply position bounds filtering if specified

        # Set Nones to inf
        for axis in range(len(self.info.position_bounds)):
            for ind in range(len(self.info.position_bounds[axis])):
                if self.info.position_bounds[axis][ind] is None:
                    # Set to negative inf if ind is 0
                    self.info.position_bounds[axis][ind] = np.inf * np.sign(2 * ind - 1)

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

    def auto_center(self):
        # center of the diffraction patterns is not explicitly given
        frame = self.get_first_frame()
        center = u.scripts.mass_center(frame * self.load_weight(ignore_crop=True))
        center = [int(x) for x in center]
        logger.info(
            f'Estimated the center of the (first) diffraction pattern to be {center}')

        return center

    def load(self, indices, disable_frame_filtering=False):
        """
        Loads data from P06 beamtime.

        Parameters
        ----------
        indices : numpy.ndarray
            The frame indices (in the filtered frame stack) to be loaded.

        """
        logger.info(
            f"loading frames in index range ({indices[0]} - {indices[-1]})")

        detector_file_list = self.get_file_list()

        # set the photon energy
        path_options = ['scan/data/energy']
        if self.info.energy == None:
            with h5py.File(self.info.nexus_path, 'r') as fp:
                existing_paths = [x for x in path_options if x in fp.keys()]
                self.meta.energy = fp[existing_paths[0]][:] * 1e-3
        else:
            self.meta.energy = self.info.energy

        per_file_inds = self.create_per_file_inds(indices, self.frames_per_file, self.all_selected_inds)
        raw, positions, weights = self.loading_loop(per_file_inds, detector_file_list)
        return raw, positions, weights

    def loading_loop(self, per_file_inds, detector_file_list):
        # actually loading the detector frames
        raw, weights, positions = {}, {}, {}

        slice_i, slice_j, pad_args = self.crop_pad_params(self.detector_center, self.detector_shape, self.info.shape)
        for i_file, valid_indices in per_file_inds.items():
            # i_file: which file
            i_in_file = valid_indices["i_in_file"]  # which frames in the file should be loaded

            with h5py.File(detector_file_list[i_file], 'r') as fp:
                # load only a cropped bit of the full frames
                if self.info.cropOnLoad:
                    frames = fp['entry/data/data'][i_in_file, slice_i, slice_j]
                # load the full raw frames
                else:
                    frames = fp['entry/data/data' % self.info.detector][i_in_file]

            # Put frames in raw dictionary
            # First, determine the index of the kept frames in the reduced
            # filtered frame stack
            for i, i_c in enumerate(valid_indices["i_consecutive"]):
                raw[i_c] = np.pad(frames[i], pad_args, mode='constant', constant_values=-1)
                positions[i_c] = self.all_positions[self.all_selected_inds[i_c]]
                weights[i_c] = np.ones(self.info.shape)

        return raw, positions, weights

    @staticmethod
    def create_per_file_inds(i_consecutive, frames_per_file, all_selected_inds):
        """
        Utility function to link together file indices, scan indices, and consecutive indices.

        Parameters
        ----------
        i_consecutive : list
            Indices in the stack of valid frames. I.e. i_consecutive can
            contain any integer from 0 to the number of valid frames.

        frames_per_file : int
            The number of frames per file.

        all_selected_inds : list
            Selcted indices from the stack of all recorded frames, including
            filtered out frames.    [sub_pixel_beam_shift_active]
    default = True
    type = str
    help = activates
    doc =

        Returns
        -------
        per_file_inds : dict
            Dictionary of indices linking together file indices, scan indices, and consecutive indices.
        """
        # To avoid opening and closing the file for each frame, list valid indices for each file
        #per_file_inds = {i: {"i_scan":[], "i_consecutive":[], "i_in_file":[]} for i in range(n_files)}
        per_file_inds = {}
        for i_c in i_consecutive:
            i_s = all_selected_inds[i_c]
            i_file, i_in_file = divmod(i_s, frames_per_file)
            if i_file not in per_file_inds:
                per_file_inds[i_file] = {"i_scan": [i_s], "i_consecutive": [i_c], "i_in_file": [i_in_file]}
            else:
                per_file_inds[i_file]["i_scan"].append(i_s)
                per_file_inds[i_file]["i_consecutive"].append(i_c)
                per_file_inds[i_file]["i_in_file"].append(i_in_file)

        return per_file_inds

    @staticmethod
    def crop_pad_params(detector_center_pixel, detector_shape, crop_shape):
        """
        Calculates cropping and padding parameters arguments to make the
        cropped frame the right shape, while also keeping the center pixel in
        the center of the padded image.

        Parameters
        ----------
        detector_center_pixel : tuple
            A pair of integers

        detector_shape : tuple
            A pair of integers

        crop_shape : tuple
            A pair of integers

        Returns
        -------
        slice_i : slice
            The slice for the first image dimension.

        slice_j : slice
            The slice for the second image dimension.

        pad_args : tuple
            A 2x2 tuple containing arguments to be passed to np.pad
        """
        i_lower = detector_center_pixel[0] - crop_shape[0] // 2
        j_lower = detector_center_pixel[1] - crop_shape[1] // 2
        i_upper = i_lower + crop_shape[0]
        j_upper = j_lower + crop_shape[1]

        i_low_pad = 0
        i_up_pad = 0
        j_low_pad = 0
        j_up_pad = 0

        if i_lower < 0:
            i_low_pad = -i_lower
            i_lower = 0
        if j_lower < 0:
            j_low_pad = -j_lower
            j_lower = 0
        if i_upper > detector_shape[0]:
            i_up_pad = i_upper - detector_shape[0]
            i_upper = detector_shape[0]
        if j_upper > detector_shape[1]:
            j_up_pad = j_upper - detector_shape[1]
            j_upper = detector_shape[1]

        slice_i = slice(i_lower, i_upper)
        slice_j = slice(j_lower, j_upper)
        pad_args = ((i_low_pad, i_up_pad), (j_low_pad, j_up_pad))
        return slice_i, slice_j, pad_args

    def get_file_list(self):
        detector_directory = os.path.join(
            self.info.scan_path_raw, self.info.detector
        )
        detector_file_list = sorted([
            os.path.join(detector_directory, x) for x in os.listdir(detector_directory) if not ('master' in x)
        ])
        return detector_file_list

    def determine_data_shape(self):
        """
        Check the first frame to determine data shape.

        Returns
        -------
        frames_per_file : int
        ni : int
        nj : int
        """
        detector_file_list = self.get_file_list()
        with h5py.File(detector_file_list[0], 'r') as handle:
            frames_per_file, ni, nj = handle['entry/data/data'].shape

        return frames_per_file, ni, nj

    def get_first_frame(self):
        """
        Load and return the first frame.

        Returns
        -------
        numpy.ndarray
            The first frame
        """
        detector_file_list = self.get_file_list()
        with h5py.File(detector_file_list[0], 'r') as handle:
            frame0 = handle['entry/data/data'][0, :, :]

        return frame0

    def load_weight(self, ignore_crop=False):
        """
        Provides the mask used for every single diffraction pattern of the whole scan.
        """
        frame0 = self.get_first_frame()
        if self.info.cropOnLoad and not ignore_crop:
            slice_i, slice_j, pad_args = self.crop_pad_params(self.detector_center, frame0.shape, self.info.shape)
            frame0 = frame0[slice_i, slice_j]  # crop
        else:
            slice_i = slice(None, None)
            slice_j = slice(None, None)
            pad_args = ((0, 0), (0, 0))
        mask = np.ones_like(frame0)

        if 'pilatus' in self.info.detector:
            mask[np.where(frame0 < 0)] = 0
        if 'eiger' in self.info.detector:
            bit_depth = int(''.join(filter(str.isdigit, str(frame0.dtype))))
            logger.info(f"found bit depth of {bit_depth} in the eiger frames")
            logger.info(f"    -> masking all pixels with values of {2**(bit_depth) -1} and above")
            mask[np.where(frame0 < 0)] = 0
            mask[np.where(frame0 >= ((2**bit_depth)-1))] = 0

        logger.info("took account of the built-in mask, %u x %u, sum %u, so %u masked pixels" %
                    (mask.shape + (np.sum(mask), np.prod(mask.shape)-np.sum(mask))))

        if self.info.maskfile:
            if self.info.maskfile.endswith('.h5'):
                mask2 = self.load_mask_h5()
            else:
                mask2 = self.load_mask_tiff()

            logger.info(
                "loaded additional mask, %u x %u, sum %u, so %u masked pixels" %
                (mask2.shape + (np.sum(mask2),
                                np.prod(mask2.shape) - np.sum(mask2))))
            mask = mask * mask2[slice_i, slice_j]
            logger.info("total mask, %u x %u, sum %u, so %u masked pixels" %
                        (mask.shape + (np.sum(mask),
                                       np.prod(mask.shape) - np.sum(mask))))
        mask = np.pad(mask, pad_args)
        return mask


@register()
class P06Scan_scanning_mirror(P06Scan):
    """
    This class loads scanning mirror data at the P06 beamline.

    Defaults:

    [sub_pixel_beam_shift_active]
    default = True
    type = bool
    help = If true, sub pixel beam shifts will be active. If False, the shifts will be set to 0. This will not speed anything up, as all computation will still occur.
    doc =
    """

    def __init__(self, pars=None, **kwargs):
        self.p = self.DEFAULT.copy()
        if pars is not None:
            self.p.update(pars)
        self.p.update(kwargs)

        if self.p.center is None:
            raise ValueError("center may not be None for P06Scan_scanning_mirror")
            # Center will be assigned automatically during call to super(P06Scan_scanning_mirror, self).__init__(self.p), however, for scanning mirror data, this is invalid.

        super(P06Scan_scanning_mirror, self).__init__(self.p)
        self.full_mask = self.load_weight(ignore_crop=True).astype(bool)

        self.info.full_mask = self.full_mask
        self.info.loaded_center_of_mass = self.all_positions[:, 2:]  # center of mass before dynamic cropping

    def load_positions(self):
        positions = super(P06Scan_scanning_mirror, self).load_positions()
        center_of_mass = self.load_center_of_mass()

        if positions.shape != center_of_mass.shape:
            raise ValueError(f"positions.shape != center_of_mass.shape, i.e. {positions.shape} != {center_of_mass.shape}")
        positions = np.hstack([
            positions,
            center_of_mass
        ])

        return positions

    def load_center_of_mass(self):
        positions_path = self.info.positions_path
        com = {}  # center of mass

        with h5py.File(positions_path, 'r') as f:
            com['x'] = np.array(f[f'detectors/{self.info.detector}/centre_of_mass/x'][:])  # unit assumed to be pixels
            com['y'] = np.array(f[f'detectors/{self.info.detector}/centre_of_mass/y'][:])  # unit assumed to be pixels
        nan_mask = np.logical_not(np.isnan(com['x']))
        com['x'] = com['x'][nan_mask]  # necessary for cmesh
        com['y'] = com['y'][nan_mask]  # necessary for cmesh

        # put the two arrays together
        center_of_mass = np.vstack((com['y'], com['x'])).T

        return center_of_mass

    def loading_loop(self, per_file_inds, detector_file_list):
        """
        Modified loading loop that to allow dynamic cropping

        Parameters
        ----------
        per_file_inds
        detector_file_list

        Returns
        -------
            raw, positions, weights
        """
        # actually loading the detector frames
        raw, weights, positions = {}, {}, {}
        all_positions = self.all_positions[:, :2]  # only the actual posiitons
        centers_of_mass = self.all_positions[:, 2:]
        center_pixels = np.round(centers_of_mass).astype(int)
        # trim center of mass to be relative to center pixel.
        center_remainder = centers_of_mass - center_pixels
        if self.info.sub_pixel_beam_shift_active:
            pod_positions = np.hstack([all_positions, center_remainder])
        else:
            # set remainder shifts to 0 if not active.
            pod_positions = np.hstack([all_positions, np.zeros_like(center_remainder)])

        for i_file, valid_indices in per_file_inds.items():
              # which frames in the file should be loaded
            with h5py.File(detector_file_list[i_file], 'r') as fp:
                # load only a cropped bit of the full frames
                if self.info.cropOnLoad:
                    frames = np.zeros((self.frames_per_file, *self.info.shape))
                    masks = np.zeros((self.frames_per_file, *self.info.shape), dtype=bool)
                    for i_if, i_s in zip(valid_indices["i_in_file"], valid_indices["i_scan"]):
                        slice_i, slice_j, pad_args = self.crop_pad_params(center_pixels[i_s], self.detector_shape, self.info.shape)
                        frame = fp['entry/data/data'][i_if, slice_i, slice_j]
                        frames[i_if] = np.pad(frame, pad_args, mode='constant', constant_values=-1)
                        masks[i_if] = np.pad(self.full_mask[slice_i, slice_j], pad_args, mode='constant', constant_values=False)

                # load the full raw frames
                else:
                    i_in_file = valid_indices["i_in_file"]
                    frames = fp['entry/data/data' % self.info.detector][i_in_file]
                    masks = [self.full_mask] * len(frames)

            # Put frames in raw dictionary
            for i_if, i_c in zip(valid_indices["i_in_file"], valid_indices["i_consecutive"]):
                raw[i_c] = frames[i_if]
                weights[i_c] = masks[i_if]  # np.ones(self.info.shape)
                positions[i_c] = pod_positions[self.all_selected_inds[i_c]]

        return raw, positions, weights
