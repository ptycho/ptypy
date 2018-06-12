# -*- coding: utf-8 -*-
"""\
Scan loading recipe for the I13 beamline, Diamond.

This file is part of the PTYPY package.

    :copyright: Copyright 2014 by the PTYPY team, see AUTHORS.
    :license: GPLv2, see LICENSE for details.
"""

import h5py as h5
import numpy as np
import copy

from ptypy import utils as u
from ptypy.core.data import PtyScan
from ptypy.utils.descriptor import defaults_tree
from ptypy.utils.verbose import log
from ptypy.utils.array_utils import _translate_to_pix


@defaults_tree.parse_doc('scandata.Hdf5Loader')
class Hdf5Loader(PtyScan):
    """
    First attempt to make a generalised hdf5 loader for data. Please raise a ticket in github if changes are required
    so we can coordinate. There will be a Nexus and CXI subclass to this in the future.

    Defaults:

    [name]
    default = 'Hdf5'
    type = str
    help =

    [intensities]
    default =
    type = Param
    help = This parameter contains the diffraction data. Data shapes can be either (A, B, frame_size_m, frame_size_n) or
            (C, frame_size_m, frame_size_n). It's assumed in this latter case that the fast axis in the scan corresponds
            the fast axis on disc (i.e. C-ordered layout).

    [intensities.is_swmr]
    default = False
    type = bool
    help = If this is set to be true, then intensities are assumed to be a swmr dataset that is being written as processing
            is taking place

    [intensities.live_key]
    default = None
    type = str
    help = If intensities.is_swmr is true then we need a live_key to know where the data collection has progressed to.
            This is the key to these live keys inside the intensities.file. They are zero at the scan start, but non-zero
            when the position is complete.

    [intensities.file]
    default = None
    type = str
    help = This is the path to the file containing the diffraction intensities.

    [intensities.key]
    default = None
    type = str
    help = This is the key to the intensities entry in the hdf5 file.

    [positions]
    default =
    type = Param
    help = This parameter contains the position information data. Shapes for each axis that are currently covered and
            tested corresponding to the intensity shapes are:
                                 axis_data.shape (A, B) for data.shape (A, B, frame_size_m, frame_size_n),
                                 axis_data.shape (k,) for data.shape (k, frame_size_m, frame_size_n),
                                 axis_data.shape (C, D) for data.shape (C*D, frame_size_m, frame_size_n) ,
                                 axis_data.shape (C,) for data.shape (C, D, frame_size_m, frame_size_n) where D is the
                                 size of the other axis,
                            and  axis_data.shape (C,) for data.shape (C*D, frame_size_m, frame_size_n) where D is the
                            size of the other axis.

    [positions.is_swmr]
    default = False
    type = bool
    help = If this is set to be true, then positions are assumed to be swmr datasets that are being written as processing
            is taking place.

    [positions.live_key]
    default = None
    type = str
    help = If positions.is_swmr is true then we need a live_key to know where the data collection has progressed to.
            This is the key to these live keys inside the positions.file. If None, whilst positions.is_swmr is True,
            then we just assume the same keys work for both positions and intensities. They are zero at the scan start,
            but non-zero when the position is complete.

    [positions.file]
    default = None
    type = str
    help = This is the path to the file containing the  position information. If None then we try to find the information
            in the "intensities.file" location.

    [positions.slow_key]
    default = None
    type = str
    help = This is the key to the slow-axis positions entry in the hdf5 file.

    [positions.slow_multiplier]
    default = 1.0
    type = float
    help = This is a scaling factor to get the motor position into metres.

    [positions.fast_key]
    default = None
    type = str
    help = This is the key to the fast-axis positions entry in the hdf5 file.

    [positions.fast_multiplier]
    default = 1.0
    type = float
    help = This is a scaling factor to get the motor position into metres.

    [mask]
    default =
    type = Param
    help = This parameter contains the mask data. The  shape is assumed to be (frame_size_m, frame_size_n) or the same
            shape of the full intensities data.

    [mask.file]
    default = None
    type = str
    help = This is the path to the file containing the diffraction mask.

    [mask.key]
    default = None
    type = str
    help = This is the key to the mask entry in the hdf5 file.

    [flatfield]
    default =
    type = Param
    help = This parameter contains the flatfield data. The shape is assumed to be (frame_size_m, frame_size_n) or the same
            shape of the full intensities data.

    [flatfield.file]
    default = None
    type = str
    help = This is the path to the file containing the diffraction flatfield.

    [flatfield.key]
    default = None
    type = str
    help = This is the key to the flatfield entry in the hdf5 file.

    [darkfield]
    default =
    type = Param
    help = This parameter contains the darkfield data.The shape is assumed to be (frame_size_m, frame_size_n) or the same
            shape of the full intensities data.

    [darkfield.file]
    default = None
    type = str
    help = This is the path to the file containing the diffraction darkfield.

    [darkfield.key]
    default = None
    type = str
    help = This is the key to the darkfield entry in the hdf5 file.

    [normalisation]
    default =
    type = Param
    help = This parameter contains information about the per-point normalisation (i.e. ion chamber reading).
            It is assumed to have the same dimensionality as data.shape[:-2]

    [normalisation.is_swmr]
    default = False
    type = bool
    help = If this is set to be true, then normalisations are assumed to be swmr datasets that are being written as processing
            is taking place.

    [normalisation.live_key]
    default = None
    type = str
    help = If normalisation.is_swmr is true then we need a live_key to know where the data collection has progressed to.
            This is the key to these live keys inside the normalisation.file. If None, whilst normalisation.is_swmr is
            True, then we just assume the same keys work for both normalisation and intensities. They are zero at the
            scan start, but non-zero when the position is complete.

    [normalisation.file]
    default = None
    type = str
    help = This is the path to the file containing the normalisation information. If None then we try to find the information
            in the "intensities.file" location.

    [normalisation.key]
    default = None
    type = str
    help = This is the key to the normalisation entry in the hdf5 file.

    [recorded_energy]
    default =
    type = Param
    help = This parameter contains information if we are use the recorded energy rather than as a parameter.
            It should be a scalar value.
    
    [recorded_energy.file]
    default = None
    type = str
    help = This is the path to the file containing the recorded_energy.

    [recorded_energy.key]
    default = None
    type = str
    help = This is the key to the recorded_energy entry in the hdf5 file.

    [recorded_distance]
    default =
    type = Param
    help = This parameter contains information if we are use the recorded distance to the detector rather than as a parameter,
            It should be a scalar value.
    
    [recorded_distance.file]
    default = None
    type = str
    help = This is the path to the file containing the recorded_distance between sample and detector.

    [recorded_distance.key]
    default = None
    type = str
    help = This is the key to the recorded_distance entry in the hdf5 file.

    [recorded_psize]
    default =
    type = Param
    help = This parameter contains information if we are use the recorded psize to the detector rather than as a parameter,
            It should be a scalar value.
    
    [recorded_psize.file]
    default = None
    type = str
    help = This is the path to the file containing the recorded detector psize.

    [recorded_psize.key]
    default = None
    type = str
    help = This is the key to the recorded_psize entry in the hdf5 file.

    [shape]
    type = int, tuple
    default = None
    help = Shape of the region of interest cropped from the raw data.
    doc = Cropping dimension of the diffraction frame
      Can be None, (dimx, dimy), or dim. In the latter case shape will be (dim, dim).
    userlevel = 1

    """

    def __init__(self, pars=None, **kwargs):
        """
        hdf5 data loader
        """
        self.p = self.DEFAULT.copy(99)
        self.p.update(pars, in_place_depth=99)

        super(Hdf5Loader, self).__init__(self.p, **kwargs)

        self._scantype = None
        self._ismapped = None
        self.intensities = None
        self.slow_axis = None
        self.fast_axis = None
        self.darkfield = None
        self.flatfield = None
        self.normalisation = None
        self.normalisation_laid_out_like_positions = None
        self.darkfield_laid_out_like_data = None
        self.flatfield_field_laid_out_like_data = None
        # lets raise some exceptions here for the essentials
        if None in [self.p.intensities.file,
                    self.p.intensities.key,
                    self.p.positions.file,
                    self.p.positions.slow_key,
                    self.p.positions.fast_key]:
            raise RuntimeError("Missing some information about either the positions or the intensity mapping!")

        log(4, u.verbose.report(self.info))
        if True in [self.p.intensities.is_swmr,
                    self.p.positions.is_swmr,
                    self.p.normalisation.is_swmr]:
            raise NotImplementedError("Currently swmr functionality is not implemented! Coming soon...")

        self.intensities = h5.File(self.p.intensities.file, 'r')[self.p.intensities.key]
        data_shape = self.intensities.shape
        self.slow_axis = h5.File(self.p.positions.file, 'r')[self.p.positions.slow_key]
        positions_slow_shape = self.slow_axis.shape
        self.fast_axis = h5.File(self.p.positions.file, 'r')[self.p.positions.fast_key]
        positions_fast_shape = self.fast_axis.shape
        log(3, "The shape of the \n\tdiffraction intensities is: {}\n\tslow axis data:{}\n\tfast axis data:{}".format(data_shape,
                                                                                                                       positions_slow_shape,
                                                                                                                      positions_fast_shape))
        self.compute_scan_mapping_and_trajectory(data_shape, positions_fast_shape, positions_slow_shape)

        if None not in [self.p.darkfield.file, self.p.darkfield.key]:
            self.darkfield = h5.File(self.p.darkfield.file, 'r')[self.p.darkfield.key]
            log(3, "The darkfield has shape: {}".format(self.darkfield.shape))
            if self.darkfield.shape == data_shape:
                log(3, "The darkfield is laid out like the data.")
                self.darkfield_laid_out_like_data = True
            elif self.darkfield.shape == data_shape[-2:]:
                log(3, "The darkfield is not laid out like the data.")
                self.darkfield_laid_out_like_data = False
            else:
                raise RuntimeError("I have no idea what to do with this shape of darkfield data.")
        else:
            log(3, "No darkfield will be applied.")

        if None not in [self.p.flatfield.file, self.p.flatfield.key]:
            self.flatfield = h5.File(self.p.flatfield.file, 'r')[self.p.flatfield.key]
            log(3, "The flatfield has shape: {}".format(self.flatfield.shape))
            if self.flatfield.shape == data_shape:
                log(3, "The flatfield is laid out like the data.")
                self.flatfield_laid_out_like_data = True
            elif self.flatfield.shape == data_shape[-2:]:
                log(3, "The flatfield is not laid out like the data.")
                self.flatfield_laid_out_like_data = False
            else:
                raise RuntimeError("I have no idea what to do with this shape of flatfield data.")
        else:
            log(3, "No flatfield will be applied.")

        if None not in [self.p.normalisation.file, self.p.normalisation.key]:
            self.normalisation = h5.File(self.p.normalisation.file, 'r')[self.p.normalisation.key]
            if (self.normalisation.shape == self.fast_axis.shape == self.slow_axis.shape):
                log(3, "The normalisation is the same dimensionality as the axis information.")
                self.normalisation_laid_out_like_positions = True
            elif self.normalisation.shape[:2] == self.fast_axis.shape == self.slow_axis.shape:
                log(3, "The normalisation matches the axis information, but will average the other dimensions.")
                self.normalisation_laid_out_like_positions = False
            else:
                raise RuntimeError("I have no idea what to do with this is shape of normalisation data.")
        else:
            log(3, "No normalisation will be applied.")

        if None not in [self.p.recorded_energy.file, self.p.recorded_energy.key]:
            self.p.energy = h5.File(self.p.recorded_energy.file, 'r')[self.p.recorded_energy.key][0]
            log(3, "loading energy={} from file".format(self.p.energy))


        if None not in [self.p.recorded_distance.file, self.p.recorded_distance.key]:
            self.p.distance = h5.File(self.p.recorded_distance.file, 'r')[self.p.recorded_distance.key][0]
            log(3, "loading distance={} from file".format(self.p.distance))
        
        if None not in [self.p.recorded_psize.file, self.p.recorded_psize.key]:
            self.p.psize = h5.File(self.p.recorded_psize.file, 'r')[self.p.recorded_psize.key][0]
            log(3, "loading psize={} from file".format(self.p.psize))


        # now lets figure out the cropping and centering roughly so we don't load the full data in.
        frame_shape = np.array(data_shape[-2:])
        center = frame_shape // 2 if self.p.center is None else u.expect2(self.p.center)
        center = np.array([_translate_to_pix(frame_shape[ix], center[ix]) for ix in range(len(frame_shape))])

        if self.p.shape is None:
            self.frame_slices = (slice(None, None, 1), slice(None, None, 1))
            self.p.shape = frame_shape
        else:
            pshape = u.expect2(self.p.shape)
            low_pix = center - pshape // 2
            high_pix = low_pix + pshape
            self.frame_slices = (slice(low_pix[0], high_pix[0], 1), slice(low_pix[1], high_pix[1], 1))
            self.p.center = pshape // 2 #the  new center going forward
            self.info.center = self.p.center
            self.p.shape = pshape


    def load_weight(self):
        # Load mask as weight
        if (self.p.mask.key is not None) and (self.p.mask.file is not None):
            mask_dset = h5.File(self.p.mask.file)[self.p.mask.key]
            mask_slices = tuple([slice(0, ix, 1) for ix in mask_dset.shape[:-2]])
            mask_slices += self.frame_slices
            return mask_dset[mask_slices].astype(np.float)
        else:
            log(4, "No mask was loaded. mask.key was {} and mask.file was {}".format(self.p.mask.file, self.p.mask.key))

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

        return self.num_frames, True

    def load(self, indices):
        """
        Load frames given by the indices.
        indices:
        """
        intensities = {}
        positions = {}
        weights = {}
        # shouldn't be able to get here without having something that matches this logic.
        # move the following into a helper method.
        if (self._ismapped and (self._scantype is 'arb')):
            # easy peasy
            for jj in indices:
                intensities[jj] = self.get_corrected_intensities(jj)
                positions[jj] = np.array([self.slow_axis[jj]*self.p.positions.slow_multiplier, self.fast_axis[jj]*self.p.positions.fast_multiplier])

        if (self._ismapped and (self._scantype is 'raster')):
            sh = self.slow_axis.shape
            for jj in indices:
                intensities[jj] = self.get_corrected_intensities((jj % sh[0], jj // sh[1]))  # or the other way round???
                positions[jj] = np.array([self.slow_axis[jj % sh[0], jj // sh[1]]*self.p.positions.slow_multiplier, self.fast_axis[jj % sh[0], jj // sh[1]]*self.p.positions.fast_multiplier])


        if (self._scantype is 'raster') and not self._ismapped:
            sh = self.slow_axis.shape
            for jj in indices:
                intensities[jj] = self.get_corrected_intensities(jj)
                positions[jj] = np.array([self.slow_axis[jj % sh[0], jj // sh[1]]*self.p.positions.slow_multiplier, self.fast_axis[jj % sh[0], jj // sh[1]]*self.p.positions.fast_multiplier])
        log(3, 'Data loaded successfully.')
        return intensities, positions, weights

    def get_corrected_intensities(self, index):
        '''
        Corrects the intensities for darkfield, flatfield and normalisations if they exist.
        There is a lot of logic here, I wonder if there is a better way to get rid of it.
        Limited a bit by the MPI, adn thinking about extension to large data size.
        '''
        if isinstance(index, int):
            index = (index,)
        indexed_frame_slices = tuple([slice(ix, ix+1, 1) for ix in index])
        indexed_frame_slices += self.frame_slices

        intensity = self.intensities[indexed_frame_slices]
        if self.darkfield is not None:
            if self.darkfield_laid_out_like_data:
                intensity -= self.darkfield[indexed_frame_slices]
            else:
                intensity -= self.darkfield[self.frame_slices]

        if self.flatfield is not None:
            if self.flatfield_laid_out_like_data:
                intensity /= self.flatfield[indexed_frame_slices]
            else:
                intensity /= self.flatfield[self.frame_slices]

        if self.normalisation is not None:
            if self.normalisation_laid_out_like_positions:
                intensity /= self.normalisation[index]
            else:
                intensity /= self.normalisation
        return intensity

    def compute_scan_mapping_and_trajectory(self, data_shape, positions_fast_shape, positions_slow_shape):
        '''
        This horrendous block of logic is all to do with making a semi-intelligent guess at what the data looks like.
        '''
        if data_shape[:-2] == positions_slow_shape == positions_fast_shape:
            '''
            cases covered:
            axis_data.shape (A, B) for data.shape (A, B, frame_size_m, frame_size_n) or
            axis_data.shape (k,) for data.shape (k, frame_size_m, frame_size_n)
            '''
            log(3, "Everything is wonderful, each diffraction point has a co-ordinate.")
            self.num_frames = np.prod(positions_fast_shape)
            self._ismapped = True
            if len(data_shape) == 4:
                self._scantype = "raster"
            else:
                self._scantype = "arb"

        elif data_shape[0] == np.prod(positions_fast_shape) == np.prod(positions_slow_shape):
            '''
            cases covered:
            axis_data.shape (C, D) for data.shape (C*D, frame_size_m, frame_size_n) ,
            '''
            log(3, "Positions are raster, but data is a list of frames. Unpacking the data to match the positions...")
            self.num_frames = np.prod(positions_fast_shape)
            self._ismapped = False
            self._scantype = 'raster'

        elif (len(positions_slow_shape) == 1) and (len(positions_fast_shape) == 1):
            if data_shape[:-2] == (positions_slow_shape[0], positions_fast_shape[0]):
                '''
                cases covered:
                axis_data.shape (C,) for data.shape (C, D, frame_size_m, frame_size_n) where D is the size of the other axis,
                '''
                log(3, "Assuming the axes are 1D and need to be meshed to match the raster style data")
                self.num_frames = np.prod(data_shape[:-2])
                self.fast_axis, self.slow_axis = np.meshgrid(self.fast_axis[...], self.slow_axis[...])
                self._ismapped = True
                self._scantype = 'raster'
            elif data_shape[0] == (positions_slow_shape[0] * positions_fast_shape[0]):
                '''
                cases covered:
                axis_data.shape (C,) for data.shape (C*D, frame_size_m, frame_size_n) where D is the size of the other axis.
                '''
                self.num_frames = data_shape[0]
                self.fast_axis, self.slow_axis = np.meshgrid(self.fast_axis[...], self.slow_axis[...])
                self._ismapped = False
                self._scantype = 'raster'
            else:
                raise IOError("I don't know what to do with these positions/data shapes")
        else:
            raise IOError("I don't know what to do with these positions/data shapes")

