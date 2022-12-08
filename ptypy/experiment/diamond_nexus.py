# -*- coding: utf-8 -*-
"""\
Scan loading recipe for the I13 beamline, Diamond.

This file is part of the PTYPY package.

    :copyright: Copyright 2014 by the PTYPY team, see AUTHORS.
    :license: see LICENSE for details.
"""

import h5py as h5
import numpy as np

from ptypy import utils as u
from ptypy.core.data import PtyScan
from ptypy.experiment import register
from ptypy.utils.verbose import log
from ptypy.utils.array_utils import _translate_to_pix


@register()
class DiamondNexus(PtyScan):
    """
    Not quite NXptycho_cxi

    Defaults:

    [name]
    default = 'DiamondNexus'
    type = str
    help =

    [file]
    default = None
    type = str
    help = Path to the file containing the scan information.

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

    [shape]
    type = int, tuple
    default = None
    help = Shape of the region of interest cropped from the raw data.
    doc = Cropping dimension of the diffraction frame
      Can be None, (dimx, dimy), or dim. In the latter case shape will be (dim, dim).
    userlevel = 1

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

    [positions.fast_multiplier]
    default = 1.0
    type = float
    help = Multiplicative factor that converts motor positions to metres.

    [positions.slow_multiplier]
    default = 1.0
    type = float
    help = Multiplicative factor that converts motor positions to metres.

    [recorded_energy_multiplier]
    default = 1.0
    type = float
    help = This is the multiplier for the recorded energy.

    """

    def __init__(self, pars=None, **kwargs):
        """
        hdf5 data loader
        """
        self.p = self.DEFAULT.copy(99)
        self.p.update(pars, in_place_depth=99)

        super(DiamondNexus, self).__init__(self.p, **kwargs)

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

        INPUT_FILE = self.p.file
        f = h5.File(INPUT_FILE, 'r')
        INTENSITIES_KEY = 'entry_1/data/data'
        DARK_KEY = 'entry_1/instrument_1/detector_1/darkfield'
        FLAT_KEY = 'entry_1/instrument_1/detector_1/flatfield'
        POSITIONS_FAST_KEY = 'entry_1/data/x'
        POSITIONS_SLOW_KEY = 'entry_1/data/y'
        ENERGY_KEY = 'entry_1/instrument_1/beam_1/energy'
        DISTANCE_KEY = 'entry_1/instrument_1/detector_1/distance'
        MASK_FILE = self.p.mask.file
        MASK_KEY = self.p.mask.key
        PIXEL_SIZE_KEY = 'entry_1/instrument_1/detector_1/x_pixel_size'
        NORMALISATION_KEY = 'entry_1/instrument_1/monitor/data' if 'monitor' in f['entry_1/instrument_1'].keys() else None
        self.ENERGY_MULTIPLIER = self.p.recorded_energy_multiplier
        self.POSITIONS_FAST_MULTIPLIER = self.p.positions.fast_multiplier
        self.POSITIONS_SLOW_MULTIPLIER = self.p.positions.slow_multiplier


        self.intensities = h5.File(INPUT_FILE, 'r')[INTENSITIES_KEY]
        data_shape = self.intensities.shape

        fast_axis = h5.File(INPUT_FILE, 'r')[POSITIONS_FAST_KEY][...]
        self.fast_axis = np.squeeze(fast_axis) if fast_axis.ndim > 2 else fast_axis
        positions_fast_shape = self.fast_axis.shape


        slow_axis = h5.File(INPUT_FILE, 'r')[POSITIONS_SLOW_KEY][...]
        self.slow_axis = np.squeeze(slow_axis) if slow_axis.ndim > 2 else slow_axis
        positions_slow_shape = self.slow_axis.shape

        log(3, "The shape of the \n\tdiffraction intensities is: {}\n\tslow axis data:{}\n\tfast axis data:{}".format(data_shape,
                                                                                                                       positions_slow_shape,
                                                                                                                      positions_fast_shape))
        self.compute_scan_mapping_and_trajectory(data_shape, positions_fast_shape, positions_slow_shape)

        try:
            self.darkfield = h5.File(INPUT_FILE, 'r')[DARK_KEY]
            log(3, "The darkfield has shape: {}".format(self.darkfield.shape))
            if self.darkfield.shape == data_shape:
                log(3, "The darkfield is laid out like the data.")
                self.darkfield_laid_out_like_data = True
            elif self.darkfield.shape == data_shape[-2:]:
                log(3, "The darkfield is not laid out like the data.")
                self.darkfield_laid_out_like_data = False
            else:
                raise RuntimeError("I have no idea what to do with this shape of darkfield data.")
        except KeyError:
            log(3, "Could not find the darkfield in %s. No darkfield will be applied." % DARK_KEY)
        except:
            raise

        try:
            self.flatfield = h5.File(INPUT_FILE, 'r')[FLAT_KEY]
            log(3, "The flatfield has shape: {}".format(self.flatfield.shape))
            if self.flatfield.shape == data_shape:
                log(3, "The flatfield is laid out like the data.")
                self.flatfield_laid_out_like_data = True
            elif self.flatfield.shape == data_shape[-2:]:
                log(3, "The flatfield is not laid out like the data.")
                self.flatfield_laid_out_like_data = False
            else:
                raise RuntimeError("I have no idea what to do with this shape of flatfield data.")
        except KeyError:
           log(3, "Could not find the flatfield in %s. No flatfield will be applied." % FLAT_KEY)
        except:
            raise

        if None not in [MASK_FILE, MASK_KEY]:
            self.mask = h5.File(MASK_FILE, 'r')[MASK_KEY]
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

        try:
            self.normalisation = h5.File(INPUT_FILE, 'r')[NORMALISATION_KEY]
            if (self.normalisation.shape == self.fast_axis.shape == self.slow_axis.shape):
                log(3, "The normalisation is the same dimensionality as the axis information.")
                self.normalisation_laid_out_like_positions = True
            elif self.normalisation.shape[:2] == self.fast_axis.shape == self.slow_axis.shape:
                log(3, "The normalisation matches the axis information, but will average the other dimensions.")
                self.normalisation_laid_out_like_positions = False
            else:
                raise RuntimeError("I have no idea what to do with this is shape of normalisation data.")
        except KeyError:
            log(3, "Normalisation not found in: %s.No normalisation will be applied." % NORMALISATION_KEY)
        except:
            raise


        if None not in [INPUT_FILE, ENERGY_KEY]:
            self.p.energy = float(h5.File(INPUT_FILE, 'r')[ENERGY_KEY][()] * self.ENERGY_MULTIPLIER)
            self.meta.energy  = self.p.energy
            log(3, "loading energy={} from file".format(self.p.energy))


        if None not in [INPUT_FILE, DISTANCE_KEY]:
            self.p.distance = h5.File(INPUT_FILE, 'r')[DISTANCE_KEY][()]
            self.meta.distance = self.p.distance
            log(3, "loading distance={} from file".format(self.p.distance))
        
        if None not in [INPUT_FILE, PIXEL_SIZE_KEY]:
            self.p.psize = h5.File(INPUT_FILE, 'r')[PIXEL_SIZE_KEY][()]
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
        if (self._ismapped and (self._scantype == 'arb')):
            # easy peasy
            log(3, "This scan looks to be a mapped arbitrary trajectory scan.")
            self.load = self.load_mapped_and_arbitrary_scan

        if (self._ismapped and (self._scantype == 'raster')):
            log(3, "This scan looks to be a mapped raster scan.")
            self.load = self.loaded_mapped_and_raster_scan

        if (self._scantype == 'raster') and not self._ismapped:
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
            positions[ii] = np.array([self.slow_axis[slow_idx, fast_idx] * self.POSITIONS_SLOW_MULTIPLIER,
                                      self.fast_axis[slow_idx, fast_idx] * self.POSITIONS_FAST_MULTIPLIER])
        log(3, 'Data loaded successfully.')
        return intensities, positions, weights

    def loaded_mapped_and_raster_scan(self, indices):
        intensities = {}
        positions = {}
        weights = {}
        for jj in indices:
            slow_idx, fast_idx = self.preview_indices[:, jj]
            weights[jj], intensities[jj] = self.get_corrected_intensities((slow_idx, fast_idx))  # or the other way round???
            positions[jj] = np.array([self.slow_axis[slow_idx, fast_idx] * self.POSITIONS_SLOW_MULTIPLIER,
                                      self.fast_axis[slow_idx, fast_idx] * self.POSITIONS_FAST_MULTIPLIER])
        log(3, 'Data loaded successfully.')
        return intensities, positions, weights

    def load_mapped_and_arbitrary_scan(self, indices):
        intensities = {}
        positions = {}
        weights = {}
        for ii in indices:
            jj = self.preview_indices[ii]
            weights[ii], intensities[jj] = self.get_corrected_intensities(jj)
            positions[ii] = np.array([self.slow_axis[jj] * self.POSITIONS_SLOW_MULTIPLIER,
                                      self.fast_axis[jj] * self.POSITIONS_FAST_MULTIPLIER])

        log(3, 'Data loaded successfully.')

        return intensities, positions, weights

    def get_corrected_intensities(self, index):
        '''
        Corrects the intensities for darkfield, flatfield and normalisations if they exist.
        There is a lot of logic here, I wonder if there is a better way to get rid of it.
        Limited a bit by the MPI, adn thinking about extension to large data size.
        '''
        if not hasattr(index, '__iter__'):
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
                intensity[:] = intensity / self.flatfield[indexed_frame_slices].squeeze()
            else:
                intensity[:] = intensity / self.flatfield[self.frame_slices].squeeze()

        if self.normalisation is not None:
            if self.normalisation_laid_out_like_positions:
                intensity[:] = intensity / self.normalisation[index]
            else:
                intensity[:] = intensity / self.normalisation

        if self.mask is not None:
            if self.mask_laid_out_like_data:
                mask = self.mask[indexed_frame_slices].squeeze()
            else:
                mask = self.mask[self.frame_slices].squeeze()
        else:
            mask = np.ones_like(intensity, dtype=int)
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

