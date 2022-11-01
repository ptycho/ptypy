from ptypy.experiment import register

from ptypy.utils.array_utils import rebin_2d
from ptypy.utils.verbose import logger, log, headerline

import h5py as h5
import numpy as np

from mpi4py import MPI

from ptypy import utils as u
# from ptypy.core.data import PtyScan, EOS, CODES, WAIT
from ptypy.experiment import register
from ptypy.utils import parallel
from ptypy.utils.verbose import log
from ptypy.utils.array_utils import _translate_to_pix
from swmr_tools import KeyFollower
from ptypy.experiment.hdf5_loader import Hdf5Loader

import os


class MultiFileKeyFollower(KeyFollower):
    """Iterator for following key datasets in hdf5 files

    Parameters
    ----------
    args: alternating pairs h5py.File and list objects, where the File is the
    hdf5 file containing the required keys and the list is a list of string
    paths to the keys in that file.

    timeout: int (optional)
        The maximum time allowed for a dataset to update before the timeout
        termination condition is trigerred and iteration is halted. If a value
        is not set this will default to 10 seconds.

    finished_dataset: h5py.Dataset (optional)
        Scalar hdf5 dataset which is zero when the file is being
        written to and non-zero when the file is complete. Used to stop
        the iterator without waiting for the timeout



    Examples
    --------


    >>> # open hdf5 file using context manager with swmr mode activated
    >>> with h5py.File("/home/documents/work/data/example_1.h5", "r", swmr = True) as f1:
    >>>     with h5py.File("/home/documents/work/data/example_2.h5", "r", swmr = True) as f2:
    >>>     # create an instance of the Follower object to iterate through
    >>>         kf = KeyFollower(f1, ['path/to/key/one'],
    >>>                          f2, ['/path/to/key/two'],
    >>>                          timeout = 10,
    >>>                          finished_dataset = f2["/path/to/finished"])
    >>>         # iterate through the iterator as with a standard iterator/generator object
    >>>         for key in kf:
    >>>             print(key)


    """

    def __init__(self, *args, timeout=10, finished_dataset=None):
        # self.h5file = h5file
        assert len(args) % 2 == 0
        # pairs of h5files and lists of paths
        i = 0
        self.keys = []
        # self.num_keys = 0
        for arg in args:
            if i % 2 == 0: # even
                assert isinstance(arg, h5.File)
                file = arg
            else:
                assert isinstance(arg, list)
                self.keys.append([file, arg]) # could use dict instead of list, but
                # edge case where passed file objects are the same means dict keys are overwritten
            i+=1

        self.current_key = -1
        self.current_max = -1
        self.timeout = timeout
        self.start_time = None
        self.finished_dataset = finished_dataset # this should be actual dataset not just string for this class
        self._finish_tag = False
        self._check_successful = False
        self.scan_rank = -1
        self._prelim_finished_check = False
        self.maxshape = None

    def __iter__(self):
        return self


    def check_datasets(self):
        if self._check_successful:
            return

        rank = -1
        
        for h5file, paths in self.keys:
            for path in paths:
                # do some exception checking here
                tmp = h5file[path]
                r = self._get_rank(tmp.maxshape)

                if rank == -1:
                    rank = r

                if rank != -1 and rank != r:
                    raise RuntimeError("Key datasets must have the same rank!")

                if self.maxshape is None:
                    self.maxshape = tmp.maxshape[:rank]
                else:
                    if np.all(self.maxshape != tmp.maxshape[:rank]):
                        logger.warning("Max shape not consistent in keys")

        if self.finished_dataset is not None:
            # just check read here
            tmp = self.finished_dataset

        self.scan_rank = rank
        logger.debug("Dataset checks passed")

    def _get_key_list(self):
        raise NotImplementedError("Need not be called for MultiFileKeyFollower")

    def _get_keys(self):
        kds = []
        for h5file, paths in self.keys:
            for path in paths:
                dataset = h5file[path]
                dataset.refresh()
                d = dataset[...].flatten()
                kds.append(d)
        return kds

    def _check_finished_dataset(self):
        if self.finished_dataset is None:
            return False

        f = self.finished_dataset
        f.refresh()

        finished = f[0] == 1

        if self._prelim_finished_check and finished:
            return True

        # go through the timeout loop once more
        # in case finish is flushed slightly before
        # the last of the keys
        if finished:
            self._prelim_finished_check = True

        return False

    def refresh(self):
        """Force an update of the current maximum key"""
        return self._is_next()


@register()
class SwmrLoader(Hdf5Loader):
    """
    Defaults:
    [name]
    default = 'SwmrLoader'
    type = str
    help =

    [intensities]
    default =
    type = Param
    help = Parameters for the diffraction data.
    doc = Data shapes can be either (A, B, frame_size_m, frame_size_n) or (C, frame_size_m, frame_size_n).
          It is assumed in this latter case that the fast axis in the scan corresponds
          the fast axis on disc (i.e. C-ordered layout).

    [intensities.live_key]
    default = None
    type = str
    help = Key to live keys inside the intensities.file (used only if is_swmr is True)
    doc = Live_keys indicate where the data collection has progressed to. They are zero at the 
          scan start, but non-zero when the position is complete.

    [positions.live_key]
    default = None
    type = str
    help = Live_keys indicate where the data collection has progressed to. They are zero at the 
           scan start, but non-zero when the position is complete. If None whilst positions.is_swmr 
           is True, use "intensities.live_key".

    [positions.live_fast_key]
    default = None
    type = str
    help = 

    [positions.live_slow_key]
    default = None
    type = str
    help = 

    [darkfield.is_swmr]
    default = None
    type = bool
    help = 

    [mask.is_swmr]
    default = None
    type = bool
    help = 

    [flatfield.is_swmr]
    default = None
    type = bool
    help = 

    [framefilter.is_swmr]
    default = None
    type = bool
    help = 

    [recorded_energy.is_swmr]
    default = None
    type = bool
    help = 

    [recorded_distance.is_swmr]
    default = None
    type = bool
    help = 

    [recorded_psize.is_swmr]
    default = None
    type = bool
    help = 
    """
    def __init__(self, pars=None, **kwargs):
        """
        hdf5 data loader
        """
        self.p = self.DEFAULT.copy(99)
        self.p.update(pars, in_place_depth=99)

        # self.p = self.info

        super(Hdf5Loader, self).__init__(self.p, **kwargs)
        # calls Ptyscan.__init__

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
        self.framefilter = None
        self._is_spectro_scan = False
        _params = [self.p.intensities, self.p.positions, 
                      self.p.flatfield, self.p.darkfield,
                      self.p.normalisation, self.p.framefilter,
                      self.p.recorded_energy, self.p.recorded_distance,
                      self.p.recorded_psize]
        _params_with_files = [pm for pm in _params if pm.file != None]
        # if any of these files are swmr, ones which point to same file must be swmr
        swmr_files = [param.file for param in _params_with_files if getattr(param, 'is_swmr', False) == True]

        for param in _params_with_files: # for all file paths of all fields
            if not param.is_swmr:
                for swmr_file in swmr_files:
                    if os.path.samefile(param.file, swmr_file):
                        # if any of the other paths that point to the same file are swmr, make swmr
                        param.is_swmr = True
                        break

        # lets raise some exceptions here for the essentials
        if None in [self.p.intensities.file,
                    self.p.intensities.key,
                    self.p.positions.file,
                    self.p.positions.slow_key,
                    self.p.positions.fast_key]:
            raise RuntimeError("Missing some information about either the positions or the intensity mapping!")

        if self.p.intensities.live_key == None:
            raise RuntimeError("Missing information about the progress of"
                               " data collection for the intensities")

        if self.p.positions.live_key != None:
            self.live_slow_key = self.p.positions.live_key
            self.live_fast_key = self.p.positions.live_key
        elif None not in [self.p.positions.live_fast_key, self.p.positions.live_slow_key]:
            self.live_slow_key = self.p.positions.live_slow_key
            self.live_fast_key = self.p.positions.live_fast_key
        else:
            raise RuntimeError("Missing information about the progress"
                               "of data collection for the positions")

        log(4, u.verbose.report(self.info))

        # Check for spectro scans
        if None not in [self.p.recorded_energy.file, self.p.recorded_energy.key]:
            _energy_dset = self._make_h5_file(self.p.recorded_energy)[self.p.recorded_energy.key]

            if len(_energy_dset.shape):
                if _energy_dset.shape[0] > 1:
                    self._is_spectro_scan = True
        if self._is_spectro_scan and self.p.outer_index is None:
            self.p.outer_index = 0
        if self._is_spectro_scan:
            log(3, "This is appears to be a spectro scan, selecting index = {}".format(self.p.outer_index))

        self.intensities_file = self._make_h5_file(self.p.intensities)
        self.intensities = self.intensities_file[self.p.intensities.key]

        self.live_key = self.p.intensities.live_key

        if self.p.intensities.file == self.p.positions.file and 1==2:
            self.positions_file = self.intensities_file
            self.kf = KeyFollower(self.intensities_file, [self.live_key, self.live_fast_key, self.live_slow_key], timeout = 5)
        else:
            log(3, 'Checking keys across multiple files')
            self.positions_file = self._make_h5_file(self.p.positions)

            self.kf = MultiFileKeyFollower(self.intensities_file, [self.live_key],
                                           self.positions_file, [self.live_slow_key, self.live_fast_key])
        self.fast_axis = self.positions_file[self.p.positions.fast_key]
        self.slow_axis = self.positions_file[self.p.positions.slow_key]
        # see how it is squeezed at init in Hdf5Loader: not really possible with updating dataset
        
        data_shape = self.intensities.shape
        positions_fast_shape = self.fast_axis.shape
        positions_slow_shape = self.slow_axis.shape

        log(3, "The shape of the \n\tdiffraction intensities is: {}\n\tslow axis data:{}\n\tfast axis data:{}".format(data_shape,
                                                                                                                      positions_slow_shape,
                                                                                                                      positions_fast_shape))
        if self.p.positions.skip > 1:
            log(3, "Skipping every {:d} positions".format(self.p.positions.skip))

        self.num_frames_so_far = self.get_updated_num_frames()
        log(3, f"At initialisation, {self.num_frames_so_far} frames were found")

        if None not in [self.p.framefilter.file, self.p.framefilter.key]:
            self.framefilter = self._make_h5_file(self.p.framefilter)[self.p.framefilter.key][()].squeeze() > 0 # turn into boolean
            if self._is_spectro_scan and self.p.outer_index is not None:
                self.framefilter = self.framefilter[self.p.outer_index]
            if (self.framefilter.shape == self.fast_axis.shape == self.slow_axis.shape):
                log(3, "The frame filter has the same dimensionality as the axis information.")
            elif self.framefilter.shape[:2] == self.fast_axis.shape == self.slow_axis.shape:
                log(3, "The frame filter matches the axis information, but will average the other dimensions.")
            else:
                raise RuntimeError("I have no idea what to do with this is shape of frame filter data.")
        else:
            log(3, "No frame filter will be applied.")

        self.compute_scan_mapping_and_trajectory(data_shape, positions_fast_shape, positions_slow_shape)

        if None not in [self.p.darkfield.file, self.p.darkfield.key]:
            self.darkfield = self._make_h5_file(self.p.darkfield)[self.p.darkfield.key]
            log(3, "The darkfield has shape: {}".format(self.darkfield.shape))
            if self.darkfield.shape == data_shape:
                log(3, "The darkfield is laid out like the data.")
                self.darkfield_laid_out_like_data = True
            elif self.darkfield.shape == data_shape[-2:]:
                log(3, "The darkfield is not laid out like the data.")
                self.darkfield_laid_out_like_data = False
            elif np.array(self.darkfield).squeeze().shape == self.intensities.shape[-2:]:
                log(3, f"Taking squeezed axes of darkfield shape: {self.darkfield.shape}")
                self.darkfield = np.array(self.darkfield).squeeze()
                self.darkfield_laid_out_like_data = False
            else:
                raise RuntimeError("I have no idea what to do with this shape of darkfield data.")
        else:
            log(3, "No darkfield will be applied.")

        if None not in [self.p.flatfield.file, self.p.flatfield.key]:
            self.flatfield = self._make_h5_file(self.p.flatfield)[self.p.flatfield.key]
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
            self.mask = self._make_h5_file(self.p.mask)[self.p.mask.key]
            self.mask_dtype = self.mask.dtype
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
            self.mask_dtype = np.int64
            log(3, "No mask will be applied.")

        if None not in [self.p.normalisation.file, self.p.normalisation.key]:
            self.normalisation = self._make_h5_file(self.p.normalisation)[self.p.normalisation.key]
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

        if None not in [self.p.recorded_energy.file, self.p.recorded_energy.key]:
            if self._is_spectro_scan and self.p.outer_index is not None:
                self.p.energy = float(self._make_h5_file(self.p.recorded_energy)[self.p.recorded_energy.key][self.p.outer_index])
            else:
                self.p.energy = float(self._make_h5_file(self.p.recorded_energy)[self.p.recorded_energy.key][()])
            self.p.energy = self.p.energy * self.p.recorded_energy.multiplier + self.p.recorded_energy.offset
            self.meta.energy  = self.p.energy
            log(3, "loading energy={} from file".format(self.p.energy))

        if None not in [self.p.recorded_distance.file, self.p.recorded_distance.key]:
            self.p.distance = float(self._make_h5_file(self.p.recorded_distance)[self.p.recorded_distance.key][()] * self.p.recorded_distance.multiplier)
            self.meta.distance = self.p.distance
            log(3, "loading distance={} from file".format(self.p.distance))
        
        if None not in [self.p.recorded_psize.file, self.p.recorded_psize.key]:
            self.p.psize = float(self._make_h5_file(self.p.recorded_psize)[self.p.recorded_psize.key][()] * self.p.recorded_psize.multiplier)
            self.meta.psize = self.p.psize
            log(3, "loading psize={} from file".format(self.p.psize))

        if self.p.padding is None:
            self.pad = np.array([0,0,0,0])
            log(3, "No padding will be applied.")
        else:
            self.pad = np.array(self.p.padding, dtype=int)
            assert self.pad.size == 4, "self.p.padding needs to of size 4"
            log(3, "Padding the detector frames by {}".format(self.p.padding))

        # now lets figure out the cropping and centering roughly so we don't load the full data in.
        frame_shape = np.array(data_shape[-2:]) + self.pad.reshape(2,2).sum(1)
        center = frame_shape // 2 if self.p.center is None else u.expect2(self.p.center)
        center = np.array([_translate_to_pix(frame_shape[ix], center[ix]) for ix in range(len(frame_shape))])

        if self.p.shape is None:
            self.frame_slices = (slice(None, None, 1), slice(None, None, 1))
            self.frame_shape = data_shape[-2:]
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
            self.frame_shape = data_shape[-2:]
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
            self.load = self.load_mapped_and_raster_scan

        if (self._scantype == 'raster') and not self._ismapped:
            log(3, "This scan looks to be an unmapped raster scan.")
            self.load = self.load_unmapped_raster_scan

    def _make_h5_file(self, param, permissions = 'r'):
        '''
        Convenience function to make hdf5 File with is_swmr attribute 
        of the Param object
        '''
        return h5.File(param.file, permissions, swmr=param.is_swmr)

    def get_updated_num_frames(self):
        self.kf.refresh()
        return self.kf.get_current_max() + 1

    def refresh_datasets(self):
        self.intensities.refresh()
        self.slow_axis.refresh()
        self.fast_axis.refresh()

    def load_unmapped_raster_scan(self, indices):
        intensities = {}
        positions = {}
        weights = {}
        self.refresh_datasets()
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
        self.refresh_datasets()
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
        self.refresh_datasets()
        print(f"calling get_corrected_intensities for frames {indices}")
        for ii in indices:
            jj = self.preview_indices[ii]
            weights[ii], intensities[ii] = self.get_corrected_intensities(jj)
            positions[ii] = np.array([self.slow_axis[jj] * self.p.positions.slow_multiplier,
                                      self.fast_axis[jj] * self.p.positions.fast_multiplier])

        log(3, 'Data loaded successfully.')

        return intensities, positions, weights


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

        if self.darkfield is not None:
            if self.darkfield_laid_out_like_data:
                df = self.darkfield[indexed_frame_slices]
            else:
                df = self.darkfield[self.frame_slices].squeeze()
            pos_array = intensity > df
            intensity[pos_array] -= df[pos_array]
            intensity[~pos_array] = np.uint16(0)

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
        else:
            mask = np.ones_like(intensity, dtype=int)

        if self.p.padding:
            intensity = np.pad(intensity, tuple(self.pad.reshape(2,2)), mode='constant')
            mask = np.pad(mask, tuple(self.pad.reshape(2,2)), mode='constant')

        return mask, intensity

    def check(self, frames=None, start=None):

        if start is None:
            start = self.framestart

        if frames is None:
            frames = self.min_frames

        self.num_frames_so_far = self.get_updated_num_frames() # checks lowest
        print("num frames so far is ", self.num_frames_so_far)
        found_frames = self.num_frames_so_far - start # existing frames
        if found_frames < frames: # if less than say, 60
            if self.num_frames_so_far >= self.num_frames:
                frames_accessible = found_frames
                end_of_scan = 1
                print(f"{frames_accessible} frames will take us to the end of the scan")
            else:
                print(f"Not enough frames to process yet ({found_frames}/{min(frames, self.num_frames - start)})")
                end_of_scan = 0
                frames_accessible = 0
        else: # if there are enough frames to return, but it is not the end of the scan
            frames_accessible = frames
            end_of_scan = 0
            print(f"Returning {frames_accessible} frames, scan still ongoing")
        return frames_accessible, end_of_scan
