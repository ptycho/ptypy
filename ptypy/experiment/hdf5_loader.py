# -*- coding: utf-8 -*-
"""\
Scan loading recipe for the Diamond beamlines.

This file is part of the PTYPY package.

    :copyright: Copyright 2014 by the PTYPY team, see AUTHORS.
    :license: see LICENSE for details.
"""

import h5py as h5
import numpy as np

from ptypy import utils as u
from ptypy.core.data import PtyScan
from ptypy.experiment import register
from ptypy.utils import parallel
from ptypy.utils.verbose import log
from ptypy.utils.array_utils import _translate_to_pix

import os
from multiprocessing import Pool, RawArray

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

    [positions.skip]
    default = 1
    type = int
    help = Skip a given number of positions (in each direction)

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

    [mask.invert]
    default = False
    type = bool
    help = Inverting the mask

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

    [normalisation.sigma]
    default = 3
    type = int
    help = Sigma value applied for automatic detection of outliers in the normalisation dataset.

    [framefilter]
    default = 
    type = Param
    help = Parameters for the filtering of frames
    doc = The shape of loaded data is assumed to hvae the same dimensionality as data.shape[:-2]

    [framefilter.file]
    default = None
    type = str
    help = This is the path to the file containing the filter information. 

    [framefilter.key]
    default = None
    type = str
    help = This is the key to the frame filter entry in the hdf5 file.

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

    [recorded_energy.offset]
    default = 0.0
    type = float
    help = This is an optional offset for the recorded energy in keV.

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

    [recorded_distance.multiplier]
    default = 1.0
    type = float
    help = This is the multiplier for the recorded distance.

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

    [recorded_psize.multiplier]
    default = 1.0
    type = float
    help = This is the multiplier for the recorded detector psize.

    [shape]
    type = int, tuple
    default = None
    help = Shape of the region of interest cropped from the raw data.
    doc = Cropping dimension of the diffraction frame
      Can be None, (dimx, dimy), or dim. In the latter case shape will be (dim, dim).
    userlevel = 1

    [outer_index]
    type = int
    default = None
    help = Index for outer dimension (e.g. tomography, spectro scans), default is None.

    [padding]
    type = int, tuple, list
    default = None
    help = Option to pad the detector frames on all sides
    doc = A tuple of list with padding given as ( top, bottom, left, right)

    [electron_data]
    type = bool
    default = False
    help = Switch for loading data from electron ptychography experiments.
    doc = If True, the energy provided in keV will be considered as electron energy 
          and converted to electron wavelengths.    
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
        self.intensities_dtype = None
        self.slow_axis = None
        self.fast_axis = None
        self.data_shape = None
        self.positions_fast_shape = None
        self.positions_slow_shape = None
        self.darkfield = None
        self.flatfield = None
        self.mask = None
        self.normalisation = None
        self.normalisation_laid_out_like_positions = None
        self.darkfield_laid_out_like_data = None
        self.flatfield_field_laid_out_like_data = None
        self.mask_laid_out_like_data = None
        self.preview_indices = None
        self.framefilter = None
        self._is_spectro_scan = False

        self.fhandle_intensities = None
        self.fhandle_darkfield = None
        self.fhandle_flatfield = None
        self.fhandle_normalisation = None
        self.fhandle_mask = None

        self._params_check()
        log(4, u.verbose.report(self.info))
        self._spectro_scan_check()
        self._prepare_intensity_and_positions()
        self._prepare_framefilter()
        self.compute_scan_mapping_and_trajectory(self.data_shape, self.positions_fast_shape, self.positions_slow_shape)
        self._prepare_darkfield()
        self._prepare_flatfield()
        self._prepare_mask()
        self._prepare_normalisation()
        self._prepare_meta_info()
        self._prepare_center()

        # For electron data, convert energy
        if self.p.electron_data:
            self.meta.energy = u.m2keV(u.electron_wavelength(self.meta.energy))

        # it's much better to have this logic here than in load!
        if (self._ismapped and (self._scantype == 'arb')):
            log(3, "This scan looks to be a mapped arbitrary trajectory scan.")
            self.load = self.load_mapped_and_arbitrary_scan

        if (self._ismapped and (self._scantype == 'raster')):
            log(3, "This scan looks to be a mapped raster scan.")
            self.load = self.load_mapped_and_raster_scan

        if (self._scantype == 'raster') and not self._ismapped:
            log(3, "This scan looks to be an unmapped raster scan.")
            self.load = self.load_unmapped_raster_scan

    def _params_check(self):
        """
        raise error if essential parameters mising
        """
        if None in [self.p.intensities.file,
                    self.p.intensities.key,
                    self.p.positions.file,
                    self.p.positions.slow_key,
                    self.p.positions.fast_key]:
            raise RuntimeError("Missing some information about either the positions or the intensity mapping!")

        if True in [self.p.intensities.is_swmr,
                    self.p.positions.is_swmr,
                    self.p.normalisation.is_swmr]:
            raise NotImplementedError("Currently swmr functionality is not implemented! Coming soon...")

    def _spectro_scan_check(self):
        """
        make adjustments if dealing with a spectro scan
        """
        if None not in [self.p.recorded_energy.file, self.p.recorded_energy.key]:
            with h5.File(self.p.recorded_energy.file, 'r') as f:
                _energy_dset = f[self.p.recorded_energy.key]
                if len(_energy_dset.shape):
                    if _energy_dset.shape[0] > 1:
                        self._is_spectro_scan = True
        if self._is_spectro_scan and self.p.outer_index is None:
            self.p.outer_index = 0
        if self._is_spectro_scan:
            log(3, "This is appears to be a spectro scan, selecting index = {}".format(self.p.outer_index))


    def _prepare_intensity_and_positions(self):
        """
        Prep for loading intensity and position data
        """
        self.fhandle_intensities = h5.File(self.p.intensities.file, 'r')
        self.intensities = self.fhandle_intensities[self.p.intensities.key]
        self.intensities_dtype = self.intensities.dtype
        self.data_shape = self.intensities.shape
        if self._is_spectro_scan and self.p.outer_index is not None:
            self.data_shape = tuple(np.array(self.data_shape)[1:])

        with h5.File(self.p.positions.file, 'r') as f:
            fast_axis = f[self.p.positions.fast_key][...]
        if self._is_spectro_scan and self.p.outer_index is not None:
            fast_axis = fast_axis[self.p.outer_index]
        self.fast_axis = np.squeeze(fast_axis) if fast_axis.ndim > 2 else fast_axis
        self.positions_fast_shape = self.fast_axis.shape

        with h5.File(self.p.positions.file, 'r') as f:
            slow_axis = f[self.p.positions.slow_key][...]
        if self._is_spectro_scan and self.p.outer_index is not None:
            slow_axis = slow_axis[self.p.outer_index]
        self.slow_axis = np.squeeze(slow_axis) if slow_axis.ndim > 2 else slow_axis
        self.positions_slow_shape = self.slow_axis.shape

        log(3, "The shape of the \n\tdiffraction intensities is: {}\n\tslow axis data:{}\n\tfast axis data:{}".format(self.data_shape,
                                                                                                                      self.positions_slow_shape,
                                                                                                                      self.positions_fast_shape))
        if self.p.positions.skip > 1:
            log(3, "Skipping every {:d} positions".format(self.p.positions.skip))


    def _prepare_framefilter(self):
        """
        Prep for framefilter
        """
        if None not in [self.p.framefilter.file, self.p.framefilter.key]:
            with h5.File(self.p.framefilter.file, 'r') as f:
                self.framefilter = f[self.p.framefilter.key][()].squeeze() > 0 # turn into boolean
            if self._is_spectro_scan and self.p.outer_index is not None:
                self.framefilter = self.framefilter[self.p.outer_index]
            if (self.framefilter.shape == self.positions_fast_shape == self.positions_slow_shape):
                log(3, "The frame filter has the same dimensionality as the axis information.")
            elif self.framefilter.shape[:2] == self.positions_fast_shape == self.positions_slow_shape:
                log(3, "The frame filter matches the axis information, but will average the other dimensions.")
            else:
                raise RuntimeError("I have no idea what to do with this is shape of frame filter data.")
        else:
            log(3, "No frame filter will be applied.")

    def _prepare_darkfield(self):
        """
        Prep for darkfield
        """
        if None not in [self.p.darkfield.file, self.p.darkfield.key]:
            self.fhandle_darkfield =  h5.File(self.p.darkfield.file, 'r')
            self.darkfield = self.fhandle_darkfield[self.p.darkfield.key]
            log(3, "The darkfield has shape: {}".format(self.darkfield.shape))
            if self.darkfield.shape == self.data_shape:
                log(3, "The darkfield is laid out like the data.")
                self.darkfield_laid_out_like_data = True
            elif self.darkfield.shape == self.data_shape[-2:]:
                log(3, "The darkfield is not laid out like the data.")
                self.darkfield_laid_out_like_data = False
            elif (self.darkfield.shape[-2:] == self.data_shape[-2:]) and (self.darkfield.shape[0] == 1):
                log(3, "The darkfield is not laid out like the data.")
                self.darkfield_laid_out_like_data = False
                self.darkfield = self.darkfield[0]
            else:
                raise RuntimeError("I have no idea what to do with this shape of darkfield data.")
        else:
            log(3, "No darkfield will be applied.")

    def _prepare_flatfield(self):
        """
        Prep for flatfield
        """
        if None not in [self.p.flatfield.file, self.p.flatfield.key]:
            self.fhandle_flatfield = h5.File(self.p.flatfield.file, 'r')
            self.flatfield = self.fhandle_flatfield[self.p.flatfield.key]
            log(3, "The flatfield has shape: {}".format(self.flatfield.shape))
            if self.flatfield.shape == self.data_shape:
                log(3, "The flatfield is laid out like the data.")
                self.flatfield_laid_out_like_data = True
            elif self.flatfield.shape == self.data_shape[-2:]:
                log(3, "The flatfield is not laid out like the data.")
                self.flatfield_laid_out_like_data = False
            else:
                raise RuntimeError("I have no idea what to do with this shape of flatfield data.")
        else:
            log(3, "No flatfield will be applied.")

    def _prepare_mask(self):
        """
        Prep for mask
        """
        if None not in [self.p.mask.file, self.p.mask.key]:
            self.fhandle_mask = h5.File(self.p.mask.file, 'r')
            self.mask = self.fhandle_mask[self.p.mask.key]
            self.mask_dtype = self.mask.dtype
            log(3, "The mask has shape: {}".format(self.mask.shape))
            if self.mask.shape == self.data_shape:
                log(3, "The mask is laid out like the data.")
                self.mask_laid_out_like_data = True
            elif self.mask.shape == self.data_shape[-2:]:
                log(3, "The mask is not laid out like the data.")
                self.mask_laid_out_like_data = False
            else:
                raise RuntimeError("I have no idea what to do with this shape of mask data.")
        else:
            self.mask_dtype = np.int64
            log(3, "No mask will be applied.")


    def _prepare_normalisation(self):
        """
        Prep for normalisation
        """
        if None not in [self.p.normalisation.file, self.p.normalisation.key]:
            self.fhandle_normalisation = h5.File(self.p.normalisation.file, 'r')
            self.normalisation = self.fhandle_normalisation[self.p.normalisation.key]
            self.normalisation_mean = self.normalisation[:].mean()
            self.normalisation_std  = self.normalisation[:].std()
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

    def _prepare_meta_info(self):
        """
        Prep for meta info (energy, distance, psize)
        """
        if None not in [self.p.recorded_energy.file, self.p.recorded_energy.key]:
            with h5.File(self.p.recorded_energy.file, 'r') as f:
                if self._is_spectro_scan and self.p.outer_index is not None:
                    self.p.energy = float(f[self.p.recorded_energy.key][self.p.outer_index])
                else:
                    self.p.energy = float(f[self.p.recorded_energy.key][()])
            self.p.energy = self.p.energy * self.p.recorded_energy.multiplier + self.p.recorded_energy.offset
            self.meta.energy  = self.p.energy
            log(3, "loading energy={} from file".format(self.p.energy))

        if None not in [self.p.recorded_distance.file, self.p.recorded_distance.key]:
            with h5.File(self.p.recorded_distance.file, 'r') as f:
                self.p.distance = float(f[self.p.recorded_distance.key][()] * self.p.recorded_distance.multiplier)
            self.meta.distance = self.p.distance
            log(3, "loading distance={} from file".format(self.p.distance))
        
        if None not in [self.p.recorded_psize.file, self.p.recorded_psize.key]:
            with h5.File(self.p.recorded_psize.file, 'r') as f:
                self.p.psize = float(f[self.p.recorded_psize.key][()] * self.p.recorded_psize.multiplier)
            self.info.psize = self.p.psize
            log(3, "loading psize={} from file".format(self.p.psize))

        if self.p.padding is None:
            self.pad = np.array([0,0,0,0])
            log(3, "No padding will be applied.")
        else:
            self.pad = np.array(self.p.padding, dtype=int)
            assert self.pad.size == 4, "self.p.padding needs to of size 4"
            log(3, "Padding the detector frames by {}".format(self.p.padding))

    def _prepare_center(self):
        """
        define how data should be loaded (center, cropping)
        """
        # now lets figure out the cropping and centering roughly so we don't load the full data in.
        frame_shape = np.array(self.data_shape[-2:]) + self.pad.reshape(2,2).sum(1)
        center = frame_shape // 2 if self.p.center is None else u.expect2(self.p.center)
        center = np.array([_translate_to_pix(frame_shape[ix], center[ix]) for ix in range(len(frame_shape))])

        if self.p.shape is None:
            self.frame_slices = (slice(None, None, 1), slice(None, None, 1))
            self.frame_shape = self.data_shape[-2:]
            self.p.shape = frame_shape
            log(3, "Loading full shape frame.")
        elif self.p.shape is not None and not self.p.auto_center:
            pshape = u.expect2(self.p.shape)
            low_pix = center - pshape // 2
            high_pix = low_pix + pshape
            self.frame_slices = (slice(int(low_pix[0]), int(high_pix[0]), 1), slice(int(low_pix[1]), int(high_pix[1]), 1))
            self.frame_shape = self.p.shape
            self.p.center = pshape // 2 #the  new center going forward
            self.info.center = self.p.center
            self.p.shape = pshape
            log(3, "Loading in frame based on a center in:%i, %i" % tuple(center))
        else:
            self.frame_slices = (slice(None, None, 1), slice(None, None, 1))
            self.frame_shape = self.data_shape[-2:]
            self.info.center = None
            self.info.auto_center = self.p.auto_center
            log(3, "center is %s, auto_center: %s" % (self.info.center, self.info.auto_center))
            log(3, "The loader will not do any cropping.")

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

    def load_mapped_and_raster_scan(self, indices):
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
            weights[ii], intensities[ii] = self.get_corrected_intensities(jj)
            positions[ii] = np.array([self.slow_axis[jj] * self.p.positions.slow_multiplier,
                                      self.fast_axis[jj] * self.p.positions.fast_multiplier])

        log(3, 'Data loaded successfully.')
        return intensities, positions, weights

    def subtract_dark(self, raw, dark):
        """
        Subtract dark current from a raw frame
        and truncate negative values
        """
        corr = raw - dark
        corr[raw<dark] = 0
        return corr

    def get_corrected_intensities(self, index):
        '''
        Corrects the intensities for darkfield, flatfield and normalisations if they exist.
        There is a lot of logic here, I wonder if there is a better way to get rid of it.
        Limited a bit by the MPI, and thinking about extension to large data size.
        '''
        if not hasattr(index, '__iter__'):
            index = (index,)
        indexed_frame_slices = tuple([slice(ix, ix+1, 1) for ix in index])
        indexed_frame_slices += self.frame_slices
        if self._is_spectro_scan and self.p.outer_index is not None:
            indexed_frame_slices = (self.p.outer_index,) + indexed_frame_slices
        intensity = self.intensities[indexed_frame_slices].squeeze()

        # TODO: Remove these logic blocks into something a bit more sensible.
        if self.darkfield is not None:
            if self.darkfield_laid_out_like_data:
                df = self.darkfield[indexed_frame_slices].squeeze()
            else:
                df = self.darkfield[self.frame_slices].squeeze()
            intensity = self.subtract_dark(intensity, df)

        if self.flatfield is not None:
            if self.flatfield_laid_out_like_data:
                intensity[:] = intensity / self.flatfield[indexed_frame_slices].squeeze()
            else:
                intensity[:] = intensity / self.flatfield[self.frame_slices].squeeze()

        if self.normalisation is not None:
            if self.normalisation_laid_out_like_positions:
                scale =  self.normalisation[index]
            else:
                scale = np.squeeze(self.normalisation[indexed_frame_slices])
            if np.abs(scale - self.normalisation_mean) < (self.p.normalisation.sigma * self.normalisation_std):
                intensity[:] = intensity / scale * self.normalisation_mean

        if self.mask is not None:
            if self.mask_laid_out_like_data:
                mask = self.mask[indexed_frame_slices].squeeze()
            else:
                mask = self.mask[self.frame_slices].squeeze()
            if self.p.mask.invert:
                mask = 1 - mask
        else:
            mask = np.ones_like(intensity, dtype=int)

        if self.p.padding:
            intensity = np.pad(intensity, tuple(self.pad.reshape(2,2)), mode='constant')
            mask = np.pad(mask, tuple(self.pad.reshape(2,2)), mode='constant')

        return mask, intensity

    def compute_scan_mapping_and_trajectory(self, data_shape, positions_fast_shape, positions_slow_shape):
        '''
        This horrendous block of logic is all to do with making a semi-intelligent guess at what the data looks like.
        '''
        skip = self.p.positions.skip
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
                self.preview_indices = np.array([indices[1][::skip,::skip].flatten(), indices[0][::skip,::skip].flatten()], dtype=int)
                if self.framefilter is not None:
                    self.preview_indices = self.preview_indices[:,self.framefilter[indices[1][::skip,::skip], indices[0][::skip,::skip]].flatten()]
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
                indices = np.array(list(range(*fast_axis_bounds)))
                self.preview_indices = indices[::skip]
                if self.framefilter is not None:
                    self.preview_indices = self.preview_indices[self.framefilter[indices][::skip]]
                self.num_frames = len(self.preview_indices)

        elif ((len(positions_fast_shape)>1) and (len(positions_slow_shape)>1)) and data_shape[0] == np.prod(positions_fast_shape) == np.prod(positions_slow_shape):
            '''
            cases covered:
            axis_data.shape (C, D) for data.shape (C*D, frame_size_m, frame_size_n) ,
            '''
            log(3, "Positions are raster, but data is a list of frames. Unpacking the data to match the positions...")
            slow_axis_bounds = [0, self.slow_axis.shape[0]]
            fast_axis_bounds = [0, self.fast_axis.shape[-1]]

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
            self.preview_indices = np.array([indices[1][::skip,::skip].flatten(), indices[0][::skip,::skip].flatten()])
            if self.framefilter:
                log(3, "Framefilter not supported for this case")
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
                self.preview_indices = np.array([indices[1][::skip,::skip].flatten(), indices[0][::skip,::skip].flatten()], dtype=int)
                if self.framefilter:
                    log(3, "Framefilter not supported for this case")
                self.num_frames = len(self.preview_indices[0])
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
                self.preview_indices = np.array([indices[1][::skip,::skip].flatten(), indices[0][::skip,::skip].flatten()], dtype=int)
                if self.framefilter:
                    log(3, "Framefilter not supported for this case")
                self.num_frames = len(self.preview_indices[0])
                self._ismapped = False
                self._scantype = 'raster'

            else:
                raise IOError("I don't know what to do with these positions/data shapes")
        else:
            raise IOError("I don't know what to do with these positions/data shapes")

    def _finalize(self):
        """
        Close any open HDF5 files.
        """
        super()._finalize()
        for h in [self.fhandle_intensities,
                self.fhandle_darkfield,
                self.fhandle_flatfield,
                self.fhandle_normalisation,
                self.fhandle_mask]:
            try:
                h.close()
            except:
                pass

@register()
class Hdf5LoaderFast(Hdf5Loader):
    def __init__(self, pars=None, **kwargs):
        super().__init__(pars=pars, **kwargs)
        self.cpu_count_per_rank = max(os.cpu_count() // parallel.size,1)
        print("Rank %d has access to %d processes" %(parallel.rank, self.cpu_count_per_rank))
        self.intensities_array = None
        self.weights_array = None

    @staticmethod
    def subtract_dark(raw, dark):
        """
        Subtract dark current from a raw frame
        and truncate negative values
        """
        corr = raw - dark
        corr[raw<dark] = 0
        return corr

    @staticmethod
    def _init_worker(intensities_raw_array, weights_raw_array, 
                     intensities_handle,
                     weights_handle,
                     darkfield_handle,
                     flatfield_handle,
                     intensities_dtype, weights_dtype,
                     array_shape, 
                     mask_laid_out_like_data,
                     darkfield_laid_out_like_data,
                     flatfield_laid_out_like_data):
        Hdf5LoaderFast.worker_intensities_handle = intensities_handle
        Hdf5LoaderFast.worker_intensities_array = np.frombuffer(intensities_raw_array, intensities_dtype, -1).reshape(array_shape)
        Hdf5LoaderFast.worker_weights_handle = weights_handle
        Hdf5LoaderFast.worker_weights_array = np.frombuffer(weights_raw_array, weights_dtype, -1).reshape(array_shape) if weights_raw_array else None
        Hdf5LoaderFast.worker_mask_laid_out_like_data = mask_laid_out_like_data
        Hdf5LoaderFast.worker_darkfield_handle = darkfield_handle
        Hdf5LoaderFast.worker_darkfield_laid_out_like_data = darkfield_laid_out_like_data
        Hdf5LoaderFast.worker_flatfield_handle = flatfield_handle
        Hdf5LoaderFast.worker_flatfield_laid_out_like_data = flatfield_laid_out_like_data

    @staticmethod
    def _read_intensities_and_weights(slices):
        '''
        Copy intensities/weights into memory and correct for 
        darkfield/flatfield if they exist
        '''
        indexed_frame_slices, dest_slices = slices
        frame_slices = tuple(np.array(indexed_frame_slices)[-2:])

        # Handle / target array for intensities
        src_intensities  = Hdf5LoaderFast.worker_intensities_handle
        dest_intensities = Hdf5LoaderFast.worker_intensities_array

        # Handle / target array for mask/weights
        src_weights  = Hdf5LoaderFast.worker_weights_handle
        dest_weights = Hdf5LoaderFast.worker_weights_array
        mask_laid_out_like_data = Hdf5LoaderFast.worker_mask_laid_out_like_data

        # Handle for darkfield
        src_darkfield = Hdf5LoaderFast.worker_darkfield_handle
        darkfield_laid_out_like_data = Hdf5LoaderFast.worker_darkfield_laid_out_like_data

        # Handle for flatfield
        src_flatfield = Hdf5LoaderFast.worker_flatfield_handle
        flatfield_laid_out_like_data = Hdf5LoaderFast.worker_flatfield_laid_out_like_data

        # Copy intensities and weights
        src_intensities.read_direct(dest_intensities, indexed_frame_slices, dest_slices)
        if src_weights is not None:
            if mask_laid_out_like_data:
                src_weights.read_direct(dest_weights, indexed_frame_slices, dest_slices)
            else:
                src_weights.read_direct(dest_weights, frame_slices, dest_slices)

        # Correct darkfield
        if src_darkfield is not None:
            if darkfield_laid_out_like_data:
                df = src_darkfield[indexed_frame_slices].squeeze()
            else:
                df = src_darkfield[frame_slices].squeeze()
            dest_intensities[dest_slices] = Hdf5LoaderFast.subtract_dark(dest_intensities[dest_slices], df)

        # Correct flatfield
        if src_flatfield is not None:
            if flatfield_laid_out_like_data:
                dest_intensities[dest_slices] /= src_flatfield[indexed_frame_slices].squeeze()
            else:
                dest_intensities[dest_slices] /= src_flatfield[frame_slices].squeeze()

    def _setup_raw_intensity_buffer(self, dtype, sh):
        npixels = int(np.prod(sh))
        if (self.intensities_array is not None) and (self.intensities_array.size == npixels):
            return
        self._intensities_raw_array = RawArray(np.ctypeslib.as_ctypes_type(dtype), npixels)
        self.intensities_array = np.frombuffer(self._intensities_raw_array, self.intensities_dtype, -1).reshape(sh)
        
    def _setup_raw_weights_buffer(self, dtype, sh):
        npixels = int(np.prod(sh))
        if (self.weights_array is not None) and (self.weights_array.size == npixels):
            return
        if self.mask is not None:
            self._weights_raw_array = RawArray(np.ctypeslib.as_ctypes_type(dtype), npixels)
            self.weights_array = np.frombuffer(self._weights_raw_array, dtype, -1).reshape(sh)
        else:
            self._weights_raw_array = None
            self.weights_array = np.ones(sh, dtype=int)

    def load_multiprocessing(self, src_slices):
        sh = (len(src_slices),) + self.frame_shape
        self._setup_raw_intensity_buffer(self.intensities_dtype, sh)
        self._setup_raw_weights_buffer(self.mask_dtype, sh)
        dest_slices = [np.s_[i:i+1] for i in range(len(src_slices))]

        with Pool(self.cpu_count_per_rank, 
                  initializer=Hdf5LoaderFast._init_worker,
                  initargs=(self._intensities_raw_array, self._weights_raw_array,
                            self.intensities, self.mask, self.darkfield, self.flatfield,
                            self.intensities_dtype, self.mask_dtype,
                            sh, self.mask_laid_out_like_data,
                            self.darkfield_laid_out_like_data, 
                            self.flatfield_field_laid_out_like_data)) as p:
            p.map(self._read_intensities_and_weights, zip(src_slices, dest_slices))

    def load_unmapped_raster_scan(self, indices):

        slices = []
        for ii in indices:
            slow_idx, fast_idx = self.preview_indices[:, ii]
            jj = slow_idx * self.slow_axis.shape[1] + fast_idx
            indexed_frame_slices = (jj,)
            indexed_frame_slices += self.frame_slices
            if self._is_spectro_scan and self.p.outer_index is not None:
                indexed_frame_slices = (self.p.outer_index,) + indexed_frame_slices
            slices.append(indexed_frame_slices)

        self.load_multiprocessing(slices)

        intensities = {}
        positions = {}
        weights = {}
        for k,ii in enumerate(indices):
            slow_idx, fast_idx = self.preview_indices[:,ii]
            weights[ii], intensities[ii] = self.get_corrected_intensities(self.weights_array[k], self.intensities_array[k], ii, slices[k])
            positions[ii] = np.array([self.slow_axis[slow_idx, fast_idx] * self.p.positions.slow_multiplier,
                                      self.fast_axis[slow_idx, fast_idx] * self.p.positions.fast_multiplier])
        log(3, 'Data loaded successfully.')
        return intensities, positions, weights
    
    def load_mapped_and_raster_scan(self, indices):

        slices = []
        for ii in indices:
            index = self.preview_indices[:, ii]
            indexed_frame_slices = tuple(index)
            indexed_frame_slices += self.frame_slices
            if self._is_spectro_scan and self.p.outer_index is not None:
                indexed_frame_slices = (self.p.outer_index,) + indexed_frame_slices
            slices.append(indexed_frame_slices)
        
        self.load_multiprocessing(slices)

        intensities = {}
        positions = {}
        weights = {}
        for k,ii in enumerate(indices):
            slow_idx, fast_idx = self.preview_indices[:, ii]
            weights[ii], intensities[ii] = self.get_corrected_intensities(self.weights_array[k], self.intensities_array[k], ii, slices[k])
            positions[ii] = np.array([self.slow_axis[slow_idx, fast_idx] * self.p.positions.slow_multiplier,
                                      self.fast_axis[slow_idx, fast_idx] * self.p.positions.fast_multiplier])
        log(3, 'Data loaded successfully.')
        return intensities, positions, weights

    def load_mapped_and_arbitrary_scan(self, indices):

        slices = []
        for ii in indices:
            jj = self.preview_indices[ii]
            indexed_frame_slices = (jj,)
            indexed_frame_slices += self.frame_slices
            if self._is_spectro_scan and self.p.outer_index is not None:
                indexed_frame_slices = (self.p.outer_index,) + indexed_frame_slices
            slices.append(indexed_frame_slices)

        self.load_multiprocessing(slices)

        intensities = {}
        positions = {}
        weights = {}
        for k,ii in enumerate(indices):
            jj = self.preview_indices[ii]
            weights[ii], intensities[ii] = self.get_corrected_intensities(self.weights_array[k], self.intensities_array[k], ii, slices[k])
            positions[ii] = np.array([self.slow_axis[jj] * self.p.positions.slow_multiplier,
                                      self.fast_axis[jj] * self.p.positions.fast_multiplier])
        log(3, 'Data loaded successfully.')
        return intensities, positions, weights

    def get_corrected_intensities(self, weights, intensities, index, indexed_frame_slice):
        '''
        Corrects the intensities for normalisation and padding
        '''

        if self.normalisation is not None:
            if self.normalisation_laid_out_like_positions:
                scale =  self.normalisation[index]
            else:
                scale = np.squeeze(self.normalisation[indexed_frame_slice])
            if np.abs(scale - self.normalisation_mean) < (self.p.normalisation.sigma * self.normalisation_std):
                intensities *= 1 / (scale * self.normalisation_mean)

        if self.p.padding:
            intensities = np.pad(intensities, tuple(self.pad.reshape(2,2)), mode='constant')
            weights = np.pad(weights, tuple(self.pad.reshape(2,2)), mode='constant')

        if self.p.mask.invert:
            weights = 1 - weights

        return weights, intensities
