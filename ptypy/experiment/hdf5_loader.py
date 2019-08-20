# -*- coding: utf-8 -*-
"""\
Scan loading recipe for the I13 beamline, Diamond.

This file is part of the PTYPY package.

    :copyright: Copyright 2014 by the PTYPY team, see AUTHORS.
    :license: GPLv2, see LICENSE for details.
"""
from __future__ import print_function

from builtins import range
import h5py as h5
import numpy as np

from ptypy import utils as u
from ptypy.core.data import PtyScan
from ptypy.experiment import register
from ptypy.utils.verbose import log
from ptypy.utils.array_utils import _translate_to_pix


@register()
class Hdf5Loader(PtyScan):
    """
    First attempt to make a generalised hdf5 loader for data. Please raise a ticket in github if changes are required
    so we can coordinate. There will be a Nexus and CXI subclass to this in the future.

    Defaults:

    [name]
    default = 'Hdf5Loader'
    type = str
    help =

    [intensities]
    default =
    type = Param
    help = Parameters for the diffraction data.
    doc = Data shapes can be either (A, B, frame_size_m, frame_size_n) or (C, frame_size_m, frame_size_n).
          It is assumed in this latter case that the fast axis in the scan corresponds
          the fast axis on disc (i.e. C-ordered layout).

    [intensities.is_swmr]
    default = False
    type = bool
    help = If True, then intensities are assumed to be a swmr dataset that is being written as processing
           is taking place.

    [intensities.live_key]
    default = None
    type = str
    help = Key to live keys inside the intensities.file (used only if is_swmr is True)
    doc = Live_keys indicate where the data collection has progressed to. They are zero at the 
          scan start, but non-zero when the position is complete.

    [intensities.file]
    default = None
    type = str
    help = Path to the file containing the diffraction intensities.

    [intensities.key]
    default = None
    type = str
    help = Key to the intensities entry in the hdf5 file.

    [positions]
    default =
    type = Param
    help = Parameters for the position information data. 
    doc = Shapes for each axis that are currently covered and tested corresponding 
          to the intensity shapes are:
            * axis_data.shape (A, B) for data.shape (A, B, frame_size_m, frame_size_n),
            * axis_data.shape (k,) for data.shape (k, frame_size_m, frame_size_n),
            * axis_data.shape (C, D) for data.shape (C*D, frame_size_m, frame_size_n) ,
            * axis_data.shape (C,) for data.shape (C, D, frame_size_m, frame_size_n) where D is the
              size of the other axis, and 
            * axis_data.shape (C,) for data.shape (C*D, frame_size_m, frame_size_n) where D is the
              size of the other axis.

    [positions.is_swmr]
    default = False
    type = bool
    help = If True, positions are assumed to be a swmr dataset that is being written as processing
           is taking place.

    [positions.live_key]
    default = None
    type = str
    help = Live_keys indicate where the data collection has progressed to. They are zero at the 
           scan start, but non-zero when the position is complete. If None whilst positions.is_swmr 
           is True, use "intensities.live_key".

    [positions.file]
    default = None
    type = str
    help = Path to the file containing the position information. If None use "intensities.file".

    [positions.slow_key]
    default = None
    type = str
    help = Key to the slow-axis positions entry in the hdf5 file.

    [positions.slow_multiplier]
    default = 1.0
    type = float
    help = Multiplicative factor that converts motor positions to metres.

    [positions.fast_key]
    default = None
    type = str
    help = Key to the fast-axis positions entry in the hdf5 file.

    [positions.fast_multiplier]
    default = 1.0
    type = float
    help = Multiplicative factor that converts motor positions to metres.

    [positions.bounding_box]
    default =
    type = Param
    help = Bounding box (in array indices) to reconstruct a restricted area

    [positions.bounding_box.fast_axis_bounds]
    default = None
    type = None, int, tuple, list
    help = If an int, this is the lower bound only, if a tuple is (min, max)

    [positions.bounding_box.slow_axis_bounds]
    default =
    type = None, int, tuple, list
    help = If an int, this is the lower bound only, if a tuple is (min, max)

    [mask]
    default =
    type = Param
    help = Parameters for mask data. 
    doc = The shape of the loaded data is assumed to be (frame_size_m, frame_size_n) or the same
          shape of the full intensities data.

    [mask.file]
    default = None
    type = str
    help = Path to the file containing the diffraction mask.

    [mask.key]
    default = None
    type = str
    help = Key to the mask entry in the hdf5 file.

    [flatfield]
    default =
    type = Param
    help = Parameters for flatfield data.
    doc = The shape of the loaded data is assumed to be (frame_size_m, frame_size_n) or the same
            shape of the full intensities data.

    [flatfield.file]
    default = None
    type = str
    help = Path to the file containing the diffraction flatfield.

    [flatfield.key]
    default = None
    type = str
    help = Key to the flatfield entry in the hdf5 file.

    [darkfield]
    default =
    type = Param
    help = Parameters for darkfield data. 
    doc = The shape is assumed to be (frame_size_m, frame_size_n) or the same
          shape of the full intensities data.

    [darkfield.file]
    default = None
    type = str
    help = Path to the file containing the diffraction darkfield.

    [darkfield.key]
    default = None
    type = str
    help = Key to the darkfield entry in the hdf5 file.

    [normalisation]
    default =
    type = Param
    help = Parameters for per-point normalisation (i.e. ion chamber reading).
    doc = The shape of loaded data is assumed to have the same dimensionality as data.shape[:-2]

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

    [recorded_energy.multiplier]
    default = 1.0
    type = float
    help = This is the multiplier for the recorded energy.

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
        self.mask = None
        self.normalisation = None
        self.normalisation_laid_out_like_positions = None
        self.darkfield_laid_out_like_data = None
        self.flatfield_field_laid_out_like_data = None
        self.mask_laid_out_like_data = None
        self.preview_indices = None

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

        fast_axis = h5.File(self.p.positions.file, 'r')[self.p.positions.fast_key][...]
        self.fast_axis = np.squeeze(fast_axis) if fast_axis.ndim > 2 else fast_axis
        positions_fast_shape = self.fast_axis.shape


        slow_axis = h5.File(self.p.positions.file, 'r')[self.p.positions.slow_key][...]
        self.slow_axis = np.squeeze(slow_axis) if slow_axis.ndim > 2 else slow_axis
        positions_slow_shape = self.slow_axis.shape



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

        if None not in [self.p.mask.file, self.p.mask.key]:
            self.mask = h5.File(self.p.mask.file, 'r')[self.p.mask.key]
            log(3, "The mask has shape: {}".format(self.mask.shape))
            if self.mask.shape == data_shape:
                log(3, "The mask is laid out like the data.")
                self.mask_laid_out_like_data = True
            elif self.mask.shape == data_shape[-2:]:
                log(3, "The mask is not laid out like the data.")
                self.mask_laid_out_like_data = False
            else:
                raise RuntimeError("I have no idea what to do with this shape of mask data.")
        else:
            log(3, "No mask will be applied.")


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
            print(self.p.recorded_energy.multiplier)
            self.p.energy = np.float(h5.File(self.p.recorded_energy.file, 'r')[self.p.recorded_energy.key][()] * self.p.recorded_energy.multiplier)
            self.meta.energy  = self.p.energy
            log(3, "loading energy={} from file".format(self.p.energy))


        if None not in [self.p.recorded_distance.file, self.p.recorded_distance.key]:
            self.p.distance = h5.File(self.p.recorded_distance.file, 'r')[self.p.recorded_distance.key][()]
            self.meta.distance = self.p.distance
            log(3, "loading distance={} from file".format(self.p.distance))
        
        if None not in [self.p.recorded_psize.file, self.p.recorded_psize.key]:
            self.p.psize = h5.File(self.p.recorded_psize.file, 'r')[self.p.recorded_psize.key][()]
            self.meta.psize = self.p.psize
            log(3, "loading psize={} from file".format(self.p.psize))


        # now lets figure out the cropping and centering roughly so we don't load the full data in.
        frame_shape = np.array(data_shape[-2:])
        center = frame_shape // 2 if self.p.center is None else u.expect2(self.p.center)
        center = np.array([_translate_to_pix(frame_shape[ix], center[ix]) for ix in range(len(frame_shape))])

        if self.p.shape is None:
            self.frame_slices = (slice(None, None, 1), slice(None, None, 1))
            self.p.shape = frame_shape
            log(3, "Loading full shape frame.")
        elif self.p.shape is not None and not self.p.auto_center:
            pshape = u.expect2(self.p.shape)
            low_pix = center - pshape // 2
            high_pix = low_pix + pshape
            self.frame_slices = (slice(int(low_pix[0]), int(high_pix[0]), 1), slice(int(low_pix[1]), int(high_pix[1]), 1))
            self.p.center = pshape // 2 #the  new center going forward
            self.info.center = self.p.center
            self.p.shape = pshape
            log(3, "Loading in frame based on a center in:%i, %i" % tuple(center))
        else:
            self.frame_slices = (slice(None, None, 1), slice(None, None, 1))
            self.info.center = None
            self.info.auto_center = self.p.auto_center
            log(3, "center is %s, auto_center: %s" % (self.info.center, self.info.auto_center))

            log(3, "The loader will not do any cropping.")


        # it's much better to have this logic here than in load!
        if (self._ismapped and (self._scantype is 'arb')):
            # easy peasy
            log(3, "This scan looks to be a mapped arbitrary trajectory scan.")
            self.load = self.load_mapped_and_arbitrary_scan

        if (self._ismapped and (self._scantype is 'raster')):
            log(3, "This scan looks to be a mapped raster scan.")
            self.load = self.loaded_mapped_and_raster_scan

        if (self._scantype is 'raster') and not self._ismapped:
            log(3, "This scan looks to be an unmapped raster scan.")
            self.load = self.load_unmapped_raster_scan

    def load_unmapped_raster_scan(self, indices):
        intensities = {}
        positions = {}
        weights = {}
        sh = self.slow_axis.shape
        for ii in indices:
            slow_idx, fast_idx = self.preview_indices[:, ii]
            intensity_index = slow_idx * sh[1] + fast_idx
            weights[ii], intensities[ii] = self.get_corrected_intensities(intensity_index)
            positions[ii] = np.array([self.slow_axis[slow_idx, fast_idx] * self.p.positions.slow_multiplier,
                                      self.fast_axis[slow_idx, fast_idx] * self.p.positions.fast_multiplier])
        log(3, 'Data loaded successfully.')
        return intensities, positions, weights

    def loaded_mapped_and_raster_scan(self, indices):
        intensities = {}
        positions = {}
        weights = {}
        for jj in indices:
            slow_idx, fast_idx = self.preview_indices[:, jj]
            weights[jj], intensities[jj] = self.get_corrected_intensities((slow_idx, fast_idx))  # or the other way round???
            positions[jj] = np.array([self.slow_axis[slow_idx, fast_idx] * self.p.positions.slow_multiplier,
                                      self.fast_axis[slow_idx, fast_idx] * self.p.positions.fast_multiplier])
        log(3, 'Data loaded successfully.')
        return intensities, positions, weights

    def load_mapped_and_arbitrary_scan(self, indices):
        intensities = {}
        positions = {}
        weights = {}
        for ii in indices:
            jj = self.preview_indices[ii]
            weights[ii], intensities[jj] = self.get_corrected_intensities(jj)
            positions[ii] = np.array([self.slow_axis[jj] * self.p.positions.slow_multiplier,
                                      self.fast_axis[jj] * self.p.positions.fast_multiplier])

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


        intensity = self.intensities[indexed_frame_slices].squeeze()

        # TODO: Remove these logic blocks into something a bit more sensible.
        if self.darkfield is not None:
            if self.darkfield_laid_out_like_data:
                intensity -= self.darkfield[indexed_frame_slices].squeeze()
            else:
                intensity -= self.darkfield[self.frame_slices].squeeze()

        if self.flatfield is not None:
            if self.flatfield_laid_out_like_data:
                intensity /= self.flatfield[indexed_frame_slices].squeeze()
            else:
                intensity /= self.flatfield[self.frame_slices].squeeze()

        if self.normalisation is not None:
            if self.normalisation_laid_out_like_positions:
                intensity /= self.normalisation[index]
            else:
                intensity /= self.normalisation

        if self.mask is not None:
            if self.mask_laid_out_like_data:
                mask = self.mask[indexed_frame_slices].squeeze()
            else:
                mask = self.mask[self.frame_slices].squeeze()
        else:
            mask = np.ones_like(intensity, dtype=np.int)
        return mask, intensity



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

            self._ismapped = True
            slow_axis_bounds = [0, self.slow_axis.shape[0]]
            fast_axis_bounds = [0, self.fast_axis.shape[-1]]

            set_slow_axis_bounds = self.p.positions.bounding_box.slow_axis_bounds
            set_fast_axis_bounds = self.p.positions.bounding_box.fast_axis_bounds

            if len(data_shape) == 4:
                self._scantype = "raster"

                if set_slow_axis_bounds is not None:
                    if isinstance(set_slow_axis_bounds, int):
                        slow_axis_bounds[0] = set_slow_axis_bounds
                    elif isinstance(slow_axis_bounds, (tuple, list)):
                        slow_axis_bounds = set_slow_axis_bounds
                if set_fast_axis_bounds is not None:
                    if isinstance(set_fast_axis_bounds, int):
                        fast_axis_bounds[0] = set_fast_axis_bounds
                    elif isinstance(fast_axis_bounds, (tuple, list)):
                        fast_axis_bounds = set_fast_axis_bounds

                indices = np.meshgrid(list(range(*fast_axis_bounds)), list(range(*slow_axis_bounds)))
                self.preview_indices = np.array([indices[1].flatten(), indices[0].flatten()], dtype=int)
                self.num_frames = len(self.preview_indices[0])
            else:
                if (set_slow_axis_bounds is not None) and (set_fast_axis_bounds is not None):
                    log(3, "Setting slow axis bounds for an arbitrary mapped scan doesn't make sense. "
                           "We will just use the fast axis bounds.")
                if set_fast_axis_bounds is not None:
                    if isinstance(set_fast_axis_bounds, int):
                        fast_axis_bounds[0] = set_fast_axis_bounds
                    elif isinstance(fast_axis_bounds, (tuple, list)):
                        fast_axis_bounds = set_fast_axis_bounds
                self._scantype = "arb"
                self.preview_indices = list(range(*fast_axis_bounds))
                self.num_frames = len(self.preview_indices)

        elif ((len(positions_fast_shape)>1) and (len(positions_slow_shape)>1)) and data_shape[0] == np.prod(positions_fast_shape) == np.prod(positions_slow_shape):
            '''
            cases covered:
            axis_data.shape (C, D) for data.shape (C*D, frame_size_m, frame_size_n) ,
            '''
            log(3, "Positions are raster, but data is a list of frames. Unpacking the data to match the positions...")
            slow_axis_bounds = [0, self.slow_axis.shape[1]]
            fast_axis_bounds = [0, self.fast_axis.shape[1]]

            set_slow_axis_bounds = self.p.positions.bounding_box.slow_axis_bounds
            set_fast_axis_bounds = self.p.positions.bounding_box.fast_axis_bounds
            if set_slow_axis_bounds is not None:
                if isinstance(set_slow_axis_bounds, int):
                    slow_axis_bounds[0] = set_slow_axis_bounds
                elif isinstance(slow_axis_bounds, (tuple, list)):
                    slow_axis_bounds = set_slow_axis_bounds
            if set_fast_axis_bounds is not None:
                if isinstance(set_fast_axis_bounds, int):
                    fast_axis_bounds[0] = set_fast_axis_bounds
                elif isinstance(fast_axis_bounds, (tuple, list)):
                    fast_axis_bounds = set_fast_axis_bounds

            indices = np.meshgrid(list(range(*fast_axis_bounds)), list(range(*slow_axis_bounds)))
            self.preview_indices = np.array([indices[1].flatten(), indices[0].flatten()])
            self.num_frames = len(self.preview_indices[0])
            self._ismapped = False
            self._scantype = 'raster'

        elif (len(positions_slow_shape) == 1) and (len(positions_fast_shape) == 1):
            if data_shape[:-2] == (positions_slow_shape[0], positions_fast_shape[0]):
                '''
                cases covered:
                axis_data.shape (C,) for data.shape (C, D, frame_size_m, frame_size_n) where D is the size of the other axis,
                '''
                log(3, "Assuming the axes are 1D and need to be meshed to match the raster style data")
                slow_axis_bounds = [0, self.slow_axis.shape[0]]
                fast_axis_bounds = [0, self.fast_axis.shape[0]]

                set_slow_axis_bounds = self.p.positions.bounding_box.slow_axis_bounds
                set_fast_axis_bounds = self.p.positions.bounding_box.fast_axis_bounds
                if set_slow_axis_bounds is not None:
                    if isinstance(set_slow_axis_bounds, int):
                        slow_axis_bounds[0] = set_slow_axis_bounds
                    elif isinstance(slow_axis_bounds, (tuple, list)):
                        slow_axis_bounds = set_slow_axis_bounds
                if set_fast_axis_bounds is not None:
                    if isinstance(set_fast_axis_bounds, int):
                        fast_axis_bounds[0] = set_fast_axis_bounds
                    elif isinstance(fast_axis_bounds, (tuple, list)):
                        fast_axis_bounds = set_fast_axis_bounds

                self.fast_axis, self.slow_axis = np.meshgrid(self.fast_axis[...], self.slow_axis[...])

                indices = np.meshgrid(list(range(*fast_axis_bounds)), list(range(*slow_axis_bounds)))
                self.preview_indices = np.array([indices[1].flatten(), indices[0].flatten()], dtype=int)
                self.num_frames = np.prod(indices[0].shape)

                self._ismapped = True
                self._scantype = 'raster'
            elif data_shape[0] == (positions_slow_shape[0] * positions_fast_shape[0]):
                '''
                cases covered:
                axis_data.shape (C,) for data.shape (C*D, frame_size_m, frame_size_n) where D is the size of the other axis.
                '''
                slow_axis_bounds = [0,self.slow_axis.shape[0]]
                fast_axis_bounds = [0, self.fast_axis.shape[0]]

                set_slow_axis_bounds = self.p.positions.bounding_box.slow_axis_bounds
                set_fast_axis_bounds = self.p.positions.bounding_box.fast_axis_bounds
                if set_slow_axis_bounds is not None:
                    if isinstance(set_slow_axis_bounds, int):
                        slow_axis_bounds[0] = set_slow_axis_bounds
                    elif isinstance(slow_axis_bounds, (tuple, list)):
                        slow_axis_bounds = set_slow_axis_bounds
                if set_fast_axis_bounds is not None:
                    if isinstance(set_fast_axis_bounds, int):
                        fast_axis_bounds[0] = set_fast_axis_bounds
                    elif isinstance(fast_axis_bounds, (tuple, list)):
                        fast_axis_bounds = set_fast_axis_bounds

                self.fast_axis, self.slow_axis = np.meshgrid(self.fast_axis[...], self.slow_axis[...])

                indices = np.meshgrid(list(range(*fast_axis_bounds)), list(range(*slow_axis_bounds)))

                self.preview_indices = np.array([indices[1].flatten(), indices[0].flatten()], dtype=int)
                self.num_frames = np.prod(indices[0].shape)


                self._ismapped = False
                self._scantype = 'raster'
            else:
                raise IOError("I don't know what to do with these positions/data shapes")
        else:
            raise IOError("I don't know what to do with these positions/data shapes")

