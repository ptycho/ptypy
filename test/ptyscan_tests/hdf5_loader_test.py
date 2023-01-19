import unittest
import tempfile
import os
import h5py as h5
import shutil
import numpy as np
import ptypy
from test.utils import PtyscanTestRunner
from ptypy.experiment.hdf5_loader import Hdf5Loader
from ptypy import utils as u


def create_file_and_dataset(path, key, data_type):
    with h5.File(path, 'w') as f:
        entry = f.create_group('entry')
        if isinstance(key, list):
            for ent in key:
                entry.create_dataset(ent, (10,), dtype=data_type)
        else:
            entry.create_dataset(key, (10,), dtype=data_type) #spoof it with a size just so we can do the linking later.


class Hdf5LoaderTestNoSWMR(unittest.TestCase):

    def setUp(self):
        '''
        This should create files for:
        intensity_file, positions, mask, dark, flat, normalisation. The entries for the data should be bound attributes so that we
        can overwrite the data with whatever we want.
        We should link the data into a top level file (which is supposed to be indicative of a nexus/ CXI with external
        links).
        '''
        self.outdir = tempfile.mkdtemp("Hdf5LoaderTestNoSWMR")

        self.intensity_file = os.path.join(self.outdir, 'intensity.h5')
        self.intensity_key = 'entry/intensity'
        create_file_and_dataset(path=self.intensity_file, key=self.intensity_key, data_type=float)

        self.positions_file = os.path.join(self.outdir, 'positions.h5')
        self.positions_slow_key = 'entry/positions_slow'
        self.positions_fast_key = 'entry/positions_fast'
        create_file_and_dataset(path=self.positions_file, key=[self.positions_slow_key, self.positions_fast_key],
                                data_type=float)

        self.mask_file = os.path.join(self.outdir, 'mask.h5')
        self.mask_key = 'entry/mask'
        create_file_and_dataset(path=self.mask_file, key=self.mask_key, data_type=int)

        self.dark_file = os.path.join(self.outdir, 'dark.h5')
        self.dark_key = 'entry/dark'
        create_file_and_dataset(path=self.dark_file, key=self.dark_key, data_type=float)

        self.flat_file = os.path.join(self.outdir, 'flat.h5')
        self.flat_key = 'entry/flat'
        create_file_and_dataset(path=self.flat_file, key=self.flat_key, data_type=float)

        self.normalisation_file = os.path.join(self.outdir, 'normalisation.h5')
        self.normalisation_key = 'entry/normalisation'
        create_file_and_dataset(path=self.normalisation_file, key=self.normalisation_key, data_type=float)

        self.framefilter_file = os.path.join(self.outdir, 'framefilter.h5')
        self.framefilter_key = 'entry/framefilter'
        create_file_and_dataset(path=self.framefilter_file, key=self.framefilter_key, data_type=bool)

        self.top_file = os.path.join(self.outdir, 'top_file.h5')
        self.recorded_energy_key = 'entry/energy'
        self.recorded_distance_key = 'entry/distance'
        self.recorded_psize_key = 'entry/psize'
        with h5.File(self.top_file, 'w') as f:
            f.create_group('entry')
            f[self.recorded_energy_key] = 9.1
            f[self.recorded_distance_key] = 1.5
            # now lets link some stuff in.
            f[self.intensity_key] = h5.ExternalLink(self.intensity_file, self.intensity_key)
            f[self.positions_slow_key] = h5.ExternalLink(self.positions_file, self.positions_slow_key)
            f[self.positions_fast_key] = h5.ExternalLink(self.positions_file, self.positions_fast_key)
            f[self.normalisation_key] = h5.ExternalLink(self.normalisation_file, self.normalisation_key)
            f[self.framefilter_key] = h5.ExternalLink(self.framefilter_file, self.framefilter_key)

    def tearDown(self):
        if os.path.exists(self.outdir):
            shutil.rmtree(self.outdir)

    def test_position_data_mapping_case_1(self):
        '''
        axis_data.shape (A, B) for data.shape (A, B, frame_size_m, frame_size_n),
        '''
        A = 106
        B = 101
        frame_size_m = 50
        frame_size_n = 50


        positions_slow = np.arange(A)
        positions_fast = np.arange(B)
        fast, slow = np.meshgrid(positions_fast, positions_slow) # just pretend it's a simple grid
        fast = fast[..., np.newaxis, np.newaxis]
        slow = slow[..., np.newaxis, np.newaxis]
        # now chuck them in the files
        with h5.File(self.positions_file, 'w') as f:
            f[self.positions_slow_key] = slow
            f[self.positions_fast_key] = fast

        # make up some data ...
        data = np.arange(A*B*frame_size_m*frame_size_n).reshape(A, B, frame_size_m, frame_size_n)
        with h5.File(self.intensity_file, 'w') as f:
            f[self.intensity_key] = data

        data_params = u.Param()
        data_params.auto_center = False
        data_params.intensities = u.Param()
        data_params.intensities.file = self.intensity_file
        data_params.intensities.key = self.intensity_key

        data_params.positions = u.Param()
        data_params.positions.file = self.positions_file
        data_params.positions.slow_key = self.positions_slow_key
        data_params.positions.fast_key = self.positions_fast_key
        output = PtyscanTestRunner(Hdf5Loader, data_params, auto_frames=A*B, cleanup=False)

    def test_flatfield_applied_case_1(self):
        '''
        Applies the flatfield and assumes it is shaped like the data
        '''
        k = 12
        frame_size_m = 50
        frame_size_n = 50

        positions_slow = np.arange(k)
        positions_fast = np.arange(k)

        # now chuck them in the files
        with h5.File(self.positions_file, 'w') as f:
            f[self.positions_slow_key] = positions_slow
            f[self.positions_fast_key] = positions_fast

        # make up some data ...
        data = np.arange(k*frame_size_m*frame_size_n).reshape(k, frame_size_m, frame_size_n)
        with h5.File(self.intensity_file, 'w') as f:
            f[self.intensity_key] = data

        ff= np.ones_like(data)
        with h5.File(self.flat_file, 'w') as f:
            f[self.flat_key] = ff

        data_params = u.Param()
        data_params.auto_center = False
        data_params.intensities = u.Param()
        data_params.intensities.file = self.intensity_file
        data_params.intensities.key = self.intensity_key

        data_params.flatfield = u.Param()
        data_params.flatfield.file = self.flat_file
        data_params.flatfield.key = self.flat_key

        data_params.positions = u.Param()
        data_params.positions.file = self.positions_file
        data_params.positions.slow_key = self.positions_slow_key
        data_params.positions.fast_key = self.positions_fast_key
        output = PtyscanTestRunner(Hdf5Loader, data_params, auto_frames=k, cleanup=False)

    def test_position_data_mapping_case_1_with_exclusion(self):
        '''
        axis_data.shape (A, B) for data.shape (A, B, frame_size_m, frame_size_n),
        masks out borders on the frame
        '''
        A = 6
        B = 5
        frame_size_m = 5
        frame_size_n = 5

        fast_axis_min, fast_axis_max = 1, 3
        slow_axis_min, slow_axis_max = 2, 5

        positions_slow = np.arange(A)
        positions_fast = np.arange(B)
        fast, slow = np.meshgrid(positions_fast, positions_slow) # just pretend it's a simple grid
        fast = fast[..., np.newaxis, np.newaxis]
        slow = slow[..., np.newaxis, np.newaxis]
        # now chuck them in the files
        with h5.File(self.positions_file, 'w') as f:
            f[self.positions_slow_key] = slow
            f[self.positions_fast_key] = fast

        # make up some data ...
        data = []
        for i in range(A*B):
            data.append([np.ones(frame_size_m*frame_size_n)*i])

        data = np.array(data).reshape(A, B, frame_size_m, frame_size_n)
        with h5.File(self.intensity_file, 'w') as f:
            f[self.intensity_key] = data

        data_params = u.Param()
        data_params.auto_center = False
        data_params.intensities = u.Param()
        data_params.intensities.file = self.intensity_file
        data_params.intensities.key = self.intensity_key

        data_params.positions = u.Param()
        data_params.positions.file = self.positions_file
        data_params.positions.slow_key = self.positions_slow_key
        data_params.positions.fast_key = self.positions_fast_key

        data_params.positions.bounding_box = u.Param()
        data_params.positions.bounding_box.fast_axis_bounds = fast_axis_min, fast_axis_max
        data_params.positions.bounding_box.slow_axis_bounds = slow_axis_min, slow_axis_max
        # data_params.positions.bounding_box.extra_axis_bounds = None

        output = PtyscanTestRunner(Hdf5Loader, data_params, auto_frames=A*B, cleanup=False)

        with h5.File(output['output_file'],'r') as f:
            out_data = f['chunks/0/data'][...].squeeze()
        with h5.File(output['output_file'],'r') as f:
            out_data_fast = f['chunks/0/positions'][:, 1]
        with h5.File(output['output_file'],'r') as f:
            out_data_slow = f['chunks/0/positions'][:, 0]

        ground_truth = data[slow_axis_min:slow_axis_max, fast_axis_min:fast_axis_max].reshape((-1, frame_size_m, frame_size_n))
        ground_truth_fast = fast[slow_axis_min:slow_axis_max, fast_axis_min:fast_axis_max].squeeze()
        ground_truth_fast = ground_truth_fast.reshape((np.prod(ground_truth_fast.shape)))
        ground_truth_slow = slow[slow_axis_min:slow_axis_max, fast_axis_min:fast_axis_max].squeeze()
        ground_truth_slow = ground_truth_slow.reshape((np.prod(ground_truth_slow.shape)))

        np.testing.assert_equal(ground_truth.shape, out_data.shape, err_msg="The shapes don't match for the bounding box for case1")
        np.testing.assert_array_equal(ground_truth, out_data, err_msg='There is something up with the bounding box for case1')
        np.testing.assert_array_equal(ground_truth_fast, out_data_fast,
                                      err_msg='There is something up with the bounding box for case1')
        np.testing.assert_array_equal(ground_truth_slow, out_data_slow,
                                      err_msg='There is something up with the bounding box for case1')

    def test_position_data_mapping_case_1_with_skipping(self):
        '''
        axis_data.shape (A, B) for data.shape (A, B, frame_size_m, frame_size_n),
        '''
        A = 106
        B = 101
        frame_size_m = 5
        frame_size_n = 5
        skip = 5

        positions_slow = np.arange(A)
        positions_fast = np.arange(B)
        fast, slow = np.meshgrid(positions_fast, positions_slow) # just pretend it's a simple grid
        fast = fast[..., np.newaxis, np.newaxis]
        slow = slow[..., np.newaxis, np.newaxis]
        # now chuck them in the files
        with h5.File(self.positions_file, 'w') as f:
            f[self.positions_slow_key] = slow
            f[self.positions_fast_key] = fast

        # make up some data ...
        data = np.arange(A*B*frame_size_m*frame_size_n).reshape(A, B, frame_size_m, frame_size_n)
        with h5.File(self.intensity_file, 'w') as f:
            f[self.intensity_key] = data

        data_params = u.Param()
        data_params.auto_center = False
        data_params.intensities = u.Param()
        data_params.intensities.file = self.intensity_file
        data_params.intensities.key = self.intensity_key

        data_params.positions = u.Param()
        data_params.positions.file = self.positions_file
        data_params.positions.slow_key = self.positions_slow_key
        data_params.positions.fast_key = self.positions_fast_key
        data_params.positions.skip = skip
        output = PtyscanTestRunner(Hdf5Loader, data_params, auto_frames=A*B, cleanup=False)

        with h5.File(output['output_file'],'r') as f:
            out_data = f['chunks/0/data'][...].squeeze()
        ground_truth = data[::skip,::skip].reshape((-1, frame_size_m, frame_size_n))
        np.testing.assert_equal(out_data.shape, ground_truth.shape, err_msg="The shapes don't match for the positions for case 1 with skipping")
        np.testing.assert_array_equal(out_data, ground_truth, err_msg='There is something up with the positions for case 1 with skipping')

    def test_position_data_mapping_case_1_with_framefilter(self):
        '''
        axis_data.shape (A, B) for data.shape (A, B, frame_size_m, frame_size_n),
        '''
        A = 106
        B = 101
        frame_size_m = 5
        frame_size_n = 5

        positions_slow = np.arange(A)
        positions_fast = np.arange(B)
        fast, slow = np.meshgrid(positions_fast, positions_slow) # just pretend it's a simple grid
        fast = fast[..., np.newaxis, np.newaxis]
        slow = slow[..., np.newaxis, np.newaxis]
        # now chuck them in the files
        with h5.File(self.positions_file, 'w') as f:
            f[self.positions_slow_key] = slow
            f[self.positions_fast_key] = fast

        # make up some data ...
        data = np.arange(A*B*frame_size_m*frame_size_n).reshape(A, B, frame_size_m, frame_size_n)
        with h5.File(self.intensity_file, 'w') as f:
            f[self.intensity_key] = data

        # create framefilter
        framefilter = data.sum(axis=(2,3)) > 0
        framefilter[50:60,40:50] = False 
        with h5.File(self.framefilter_file, 'w') as f:
            f[self.framefilter_key] = framefilter

        data_params = u.Param()
        data_params.auto_center = False
        data_params.intensities = u.Param()
        data_params.intensities.file = self.intensity_file
        data_params.intensities.key = self.intensity_key

        data_params.positions = u.Param()
        data_params.positions.file = self.positions_file
        data_params.positions.slow_key = self.positions_slow_key
        data_params.positions.fast_key = self.positions_fast_key

        data_params.framefilter = u.Param()
        data_params.framefilter.file = self.framefilter_file
        data_params.framefilter.key = self.framefilter_key
        output = PtyscanTestRunner(Hdf5Loader, data_params, auto_frames=A*B, cleanup=False)

        with h5.File(output['output_file'],'r') as f:
            out_data = f['chunks/0/data'][...].squeeze()
        ground_truth = data[framefilter].reshape((-1, frame_size_m, frame_size_n))
        np.testing.assert_equal(out_data.shape, ground_truth.shape, err_msg="The shapes don't match for the positions for case 1 with framefilter")
        np.testing.assert_array_equal(out_data, ground_truth, err_msg='There is something up with the positions for case 1 with framefilter')

    def test_darkfield_applied_case_1(self):
        '''
        Applies the darkfield and assumes it is shaped like the data
        '''
        k = 12
        frame_size_m = 50
        frame_size_n = 50

        positions_slow = np.arange(k)
        positions_fast = np.arange(k)

        # now chuck them in the files
        with h5.File(self.positions_file, 'w') as f:
            f[self.positions_slow_key] = positions_slow
            f[self.positions_fast_key] = positions_fast

        # make up some data ...
        data = np.arange(k*frame_size_m*frame_size_n).reshape(k, frame_size_m, frame_size_n)
        with h5.File(self.intensity_file, 'w') as f:
            f[self.intensity_key] = data

        darkfield = np.ones_like(data)
        with h5.File(self.dark_file, 'w') as f:
            f[self.dark_key] = darkfield

        data_params = u.Param()
        data_params.auto_center = False
        data_params.intensities = u.Param()
        data_params.intensities.file = self.intensity_file
        data_params.intensities.key = self.intensity_key

        data_params.darkfield = u.Param()
        data_params.darkfield.file = self.dark_file
        data_params.darkfield.key = self.dark_key

        data_params.positions = u.Param()
        data_params.positions.file = self.positions_file
        data_params.positions.slow_key = self.positions_slow_key
        data_params.positions.fast_key = self.positions_fast_key
        output = PtyscanTestRunner(Hdf5Loader, data_params, auto_frames=k, cleanup=False)

    def test_crop_load_works_case1(self):
        k = 12
        frame_size_m = 50
        frame_size_n = 50

        positions_slow = np.arange(k)
        positions_fast = np.arange(k)

        # now chuck them in the files
        with h5.File(self.positions_file, 'w') as f:
            f[self.positions_slow_key] = positions_slow
            f[self.positions_fast_key] = positions_fast
        distance = np.array([1.5])

        # make up some data ...
        data = np.arange(k*frame_size_m*frame_size_n).reshape((k, frame_size_m, frame_size_n))
        with h5.File(self.intensity_file, 'w') as f:
            f[self.intensity_key] = data

        mask = np.ones(data.shape[-2:], dtype=float)
        mask[::2] = 0
        mask[:, ::2] = 0
        with h5.File(self.mask_file, 'w') as f:
            f[self.mask_key] = mask

        data_params = u.Param()
        data_params.auto_center = False
        data_params.shape = (5, 5)
        data_params.center = (20, 30)
        data_params.intensities = u.Param()
        data_params.intensities.file = self.intensity_file
        data_params.intensities.key = self.intensity_key

        data_params.mask = u.Param()
        data_params.mask.file = self.mask_file
        data_params.mask.key = self.mask_key

        data_params.positions = u.Param()
        data_params.positions.file = self.positions_file
        data_params.positions.slow_key = self.positions_slow_key
        data_params.positions.fast_key = self.positions_fast_key
        output = PtyscanTestRunner(Hdf5Loader, data_params, auto_frames=k, cleanup=False)

    def test_position_data_mapping_case_2(self):
        '''
        axis_data.shape (k,) for data.shape (k, frame_size_m, frame_size_n)
        '''
        k = 12
        frame_size_m = 5
        frame_size_n = 5

        positions_slow = np.arange(k)
        positions_fast = np.arange(k)

        # now chuck them in the files
        with h5.File(self.positions_file, 'w') as f:
            f[self.positions_slow_key] = positions_slow
            f[self.positions_fast_key] = positions_fast

        # make up some data ...
        data = np.arange(k*frame_size_m*frame_size_n).reshape(k, frame_size_m, frame_size_n)
        with h5.File(self.intensity_file, 'w') as f:
            f[self.intensity_key] = data

        data_params = u.Param()
        data_params.auto_center = False
        data_params.intensities = u.Param()
        data_params.intensities.file = self.intensity_file
        data_params.intensities.key = self.intensity_key

        data_params.positions = u.Param()
        data_params.positions.file = self.positions_file
        data_params.positions.slow_key = self.positions_slow_key
        data_params.positions.fast_key = self.positions_fast_key
        output = PtyscanTestRunner(Hdf5Loader, data_params, auto_frames=k, cleanup=False)

    def test_position_data_mapping_case_2_with_exclusion(self):
        '''
        axis_data.shape (k,) for data.shape (k, frame_size_m, frame_size_n)
        '''
        k = 12
        fast_axis_min, fast_axis_max = 1, 3
        frame_size_m = 5
        frame_size_n = 5

        positions_slow = np.arange(k)
        positions_fast = np.arange(k)

        # now chuck them in the files
        with h5.File(self.positions_file, 'w') as f:
            f[self.positions_slow_key] = positions_slow
            f[self.positions_fast_key] = positions_fast

        # make up some data ...
        data = []
        for i in range(k):
            data.append(np.ones((frame_size_m, frame_size_n))*i)
        data = np.array(data)
        with h5.File(self.intensity_file, 'w') as f:
            f[self.intensity_key] = data
        data_params = u.Param()
        data_params.auto_center = False
        data_params.intensities = u.Param()
        data_params.intensities.file = self.intensity_file
        data_params.intensities.key = self.intensity_key

        data_params.positions = u.Param()
        data_params.positions.file = self.positions_file
        data_params.positions.slow_key = self.positions_slow_key
        data_params.positions.fast_key = self.positions_fast_key
        data_params.positions.bounding_box = u.Param()
        data_params.positions.bounding_box.fast_axis_bounds = fast_axis_min, fast_axis_max
        output = PtyscanTestRunner(Hdf5Loader, data_params, auto_frames=k, cleanup=False)

        with h5.File(output['output_file'], 'r') as f:
            out_data = f['chunks/0/data'][...].squeeze()
        with h5.File(output['output_file'], 'r') as f:
            out_data_fast = f['chunks/0/positions'][:, 1]
        with h5.File(output['output_file'], 'r') as f:
            out_data_slow = f['chunks/0/positions'][:, 0]

        ground_truth = data[fast_axis_min:fast_axis_max].reshape((-1, frame_size_m, frame_size_n))
        ground_truth_fast = positions_fast[fast_axis_min:fast_axis_max].squeeze()
        ground_truth_fast = ground_truth_fast.reshape((np.prod(ground_truth_fast.shape)))
        ground_truth_slow = positions_slow[fast_axis_min:fast_axis_max].squeeze()
        ground_truth_slow = ground_truth_slow.reshape((np.prod(ground_truth_slow.shape)))

        np.testing.assert_equal(ground_truth.shape, out_data.shape,
                                err_msg="The shapes don't match for the bounding box for case2")
        np.testing.assert_array_equal(ground_truth, out_data,
                                      err_msg='There is something up with the bounding box for case2')
        np.testing.assert_array_equal(ground_truth_fast, out_data_fast,
                                      err_msg='There is something up with the bounding box for case 2')
        np.testing.assert_array_equal(ground_truth_slow, out_data_slow,
                                      err_msg='There is something up with the bounding box for case 2')

    def test_position_data_mapping_case_2_with_skipping(self):
        '''
        axis_data.shape (k,) for data.shape (k, frame_size_m, frame_size_n)
        '''
        k = 12
        frame_size_m = 5
        frame_size_n = 5
        skip = 2

        positions_slow = np.arange(k)
        positions_fast = np.arange(k)

        # now chuck them in the files
        with h5.File(self.positions_file, 'w') as f:
            f[self.positions_slow_key] = positions_slow
            f[self.positions_fast_key] = positions_fast

        # make up some data ...
        data = np.arange(k*frame_size_m*frame_size_n).reshape(k, frame_size_m, frame_size_n)
        with h5.File(self.intensity_file, 'w') as f:
            f[self.intensity_key] = data

        data_params = u.Param()
        data_params.auto_center = False
        data_params.intensities = u.Param()
        data_params.intensities.file = self.intensity_file
        data_params.intensities.key = self.intensity_key

        data_params.positions = u.Param()
        data_params.positions.file = self.positions_file
        data_params.positions.slow_key = self.positions_slow_key
        data_params.positions.fast_key = self.positions_fast_key
        data_params.positions.skip = skip
        output = PtyscanTestRunner(Hdf5Loader, data_params, auto_frames=k, cleanup=False)

        with h5.File(output['output_file'], 'r') as f:
            out_data = f['chunks/0/data'][...].squeeze()
        ground_truth = data[::skip].reshape((-1, frame_size_m, frame_size_n))

        np.testing.assert_equal(ground_truth.shape, out_data.shape,
                                err_msg="The shapes don't match for the positions for case 2 with skipping")
        np.testing.assert_array_equal(ground_truth, out_data,
                                      err_msg='There is something up with the positions for case 2 with skipping')

    def test_position_data_mapping_case_2_with_framefilter(self):
        '''
        axis_data.shape (k,) for data.shape (k, frame_size_m, frame_size_n)
        '''
        k = 12
        frame_size_m = 5
        frame_size_n = 5

        positions_slow = np.arange(k)
        positions_fast = np.arange(k)

        # now chuck them in the files
        with h5.File(self.positions_file, 'w') as f:
            f[self.positions_slow_key] = positions_slow
            f[self.positions_fast_key] = positions_fast

        # make up some data ...
        data = np.arange(k*frame_size_m*frame_size_n).reshape(k, frame_size_m, frame_size_n)
        with h5.File(self.intensity_file, 'w') as f:
            f[self.intensity_key] = data

        # create framefilter
        framefilter = data.sum(axis=(1,2)) > 0
        framefilter[5:7] = False
        with h5.File(self.framefilter_file, 'w') as f:
            f[self.framefilter_key] = framefilter

        data_params = u.Param()
        data_params.auto_center = False
        data_params.intensities = u.Param()
        data_params.intensities.file = self.intensity_file
        data_params.intensities.key = self.intensity_key

        data_params.positions = u.Param()
        data_params.positions.file = self.positions_file
        data_params.positions.slow_key = self.positions_slow_key
        data_params.positions.fast_key = self.positions_fast_key

        data_params.framefilter = u.Param()
        data_params.framefilter.file = self.framefilter_file
        data_params.framefilter.key = self.framefilter_key

        output = PtyscanTestRunner(Hdf5Loader, data_params, auto_frames=k, cleanup=False)

        with h5.File(output['output_file'], 'r') as f:
            out_data = f['chunks/0/data'][...].squeeze()
        ground_truth = data[framefilter].reshape((-1, frame_size_m, frame_size_n))

        np.testing.assert_equal(ground_truth.shape, out_data.shape,
                                err_msg="The shapes don't match for the positions for case 2 with framefilter")
        np.testing.assert_array_equal(ground_truth, out_data,
                                      err_msg='There is something up with the positions for case 2 with framefilter')


    def test_flatfield_applied_case_2(self):
        '''
        Applies the flatfield and assumes it is shaped like a single frame
        '''
        k = 12
        frame_size_m = 50
        frame_size_n = 50

        positions_slow = np.arange(k)
        positions_fast = np.arange(k)

        # now chuck them in the files
        with h5.File(self.positions_file, 'w') as f:
            f[self.positions_slow_key] = positions_slow
            f[self.positions_fast_key] = positions_fast

        # make up some data ...
        data = np.arange(k*frame_size_m*frame_size_n).reshape(k, frame_size_m, frame_size_n)
        with h5.File(self.intensity_file, 'w') as f:
            f[self.intensity_key] = data

        flatfield = np.ones_like(data[0])
        with h5.File(self.flat_file, 'w') as f:
            f[self.flat_key] = flatfield

        data_params = u.Param()
        data_params.auto_center = False
        data_params.intensities = u.Param()
        data_params.intensities.file = self.intensity_file
        data_params.intensities.key = self.intensity_key

        data_params.flatfield = u.Param()
        data_params.flatfield.file = self.flat_file
        data_params.flatfield.key = self.flat_key

        data_params.positions = u.Param()
        data_params.positions.file = self.positions_file
        data_params.positions.slow_key = self.positions_slow_key
        data_params.positions.fast_key = self.positions_fast_key
        output = PtyscanTestRunner(Hdf5Loader, data_params, auto_frames=k, cleanup=False)

    def test_darkfield_applied_case_2(self):
        '''
        Applies the darkfield and assumes it is shaped like a single frame
        '''
        k = 12
        frame_size_m = 50
        frame_size_n = 50

        positions_slow = np.arange(k)
        positions_fast = np.arange(k)

        # now chuck them in the files
        with h5.File(self.positions_file, 'w') as f:
            f[self.positions_slow_key] = positions_slow
            f[self.positions_fast_key] = positions_fast

        # make up some data ...
        data = np.arange(k*frame_size_m*frame_size_n).reshape(k, frame_size_m, frame_size_n)
        with h5.File(self.intensity_file, 'w') as f:
            f[self.intensity_key] = data

        darkfield = np.ones_like(data[0])
        with h5.File(self.dark_file, 'w') as f:
            f[self.dark_key] = darkfield

        data_params = u.Param()
        data_params.auto_center = False
        data_params.intensities = u.Param()
        data_params.intensities.file = self.intensity_file
        data_params.intensities.key = self.intensity_key

        data_params.darkfield = u.Param()
        data_params.darkfield.file = self.dark_file
        data_params.darkfield.key = self.dark_key

        data_params.positions = u.Param()
        data_params.positions.file = self.positions_file
        data_params.positions.slow_key = self.positions_slow_key
        data_params.positions.fast_key = self.positions_fast_key
        output = PtyscanTestRunner(Hdf5Loader, data_params, auto_frames=k, cleanup=False)

    def test_crop_load_works_case2(self):
        C = 3
        D = 4
        frame_size_m = 50
        frame_size_n = 50


        slow = np.arange(C)
        fast = np.arange(D)
        # now chuck them in the files
        with h5.File(self.positions_file, 'w') as f:
            f[self.positions_slow_key] = slow
            f[self.positions_fast_key] = fast

        # make up some data ...
        data = np.arange(C*D*frame_size_m*frame_size_n).reshape(C*D, frame_size_m, frame_size_n)
        with h5.File(self.intensity_file, 'w') as f:
            f[self.intensity_key] = data

        data_params = u.Param()
        data_params.auto_center = False
        data_params.shape = (5, 5)
        data_params.center = (20, 30)
        data_params.intensities = u.Param()
        data_params.intensities.file = self.intensity_file
        data_params.intensities.key = self.intensity_key

        data_params.positions = u.Param()
        data_params.positions.file = self.positions_file
        data_params.positions.slow_key = self.positions_slow_key
        data_params.positions.fast_key = self.positions_fast_key
        output = PtyscanTestRunner(Hdf5Loader, data_params, auto_frames=C*D, cleanup=False)

    def test_position_data_mapping_case_3(self):
        '''
        axis_data.shape (C, D) for data.shape (C*D, frame_size_m, frame_size_n) ,
        '''
        C = 10
        D = 11
        frame_size_m = 5
        frame_size_n = 5


        positions_slow = np.arange(C)
        positions_fast = np.arange(D)
        fast, slow = np.meshgrid(positions_fast, positions_slow) # just pretend it's a simple grid
        # now chuck them in the files
        with h5.File(self.positions_file, 'w') as f:
            f[self.positions_slow_key] = slow
            f[self.positions_fast_key] = fast

        # make up some data ...
        data = np.arange(C*D*frame_size_m*frame_size_n).reshape(C*D, frame_size_m, frame_size_n)
        with h5.File(self.intensity_file, 'w') as f:
            f[self.intensity_key] = data

        data_params = u.Param()
        data_params.auto_center = False
        data_params.intensities = u.Param()
        data_params.intensities.file = self.intensity_file
        data_params.intensities.key = self.intensity_key

        data_params.positions = u.Param()
        data_params.positions.file = self.positions_file
        data_params.positions.slow_key = self.positions_slow_key
        data_params.positions.fast_key = self.positions_fast_key
        output = PtyscanTestRunner(Hdf5Loader, data_params, auto_frames=C*D, cleanup=False)

    def test_position_data_mapping_case_3_with_exclusion(self):
        '''
        axis_data.shape (C, D) for data.shape (C*D, frame_size_m, frame_size_n) ,
        '''
        C = 10
        D = 12
        frame_size_m = 5
        frame_size_n = 5
        fast_axis_min, fast_axis_max = 1, 3
        slow_axis_min, slow_axis_max = 5, 9

        positions_slow = np.arange(C)
        positions_fast = np.arange(D)
        fast, slow = np.meshgrid(positions_fast, positions_slow) # just pretend it's a simple grid
        # now chuck them in the files
        with h5.File(self.positions_file, 'w') as f:
            f[self.positions_slow_key] = slow
            f[self.positions_fast_key] = fast

        # make up some data ...
        data = []
        for i in range(C*D):
            data.append(np.ones((frame_size_m, frame_size_n))*i)
        data = np.array(data)
        with h5.File(self.intensity_file, 'w') as f:
            f[self.intensity_key] = data

        data = np.arange(C*D*frame_size_m*frame_size_n).reshape(C*D, frame_size_m, frame_size_n)
        with h5.File(self.intensity_file, 'w') as f:
            f[self.intensity_key] = data

        data_params = u.Param()
        data_params.auto_center = False
        data_params.intensities = u.Param()
        data_params.intensities.file = self.intensity_file
        data_params.intensities.key = self.intensity_key

        data_params.positions = u.Param()
        data_params.positions.file = self.positions_file
        data_params.positions.slow_key = self.positions_slow_key
        data_params.positions.fast_key = self.positions_fast_key
        data_params.positions.bounding_box = u.Param()
        data_params.positions.bounding_box.fast_axis_bounds = fast_axis_min, fast_axis_max
        data_params.positions.bounding_box.slow_axis_bounds = slow_axis_min, slow_axis_max

        output = PtyscanTestRunner(Hdf5Loader, data_params, auto_frames=C*D, cleanup=False)
        with h5.File(output['output_file'], 'r') as f:
            out_data = f['chunks/0/data'][...].squeeze()
        with h5.File(output['output_file'], 'r') as f:
            out_data_fast = f['chunks/0/positions'][:, 1]
        with h5.File(output['output_file'], 'r') as f:
            out_data_slow = f['chunks/0/positions'][:, 0]

        ground_truth = data.reshape((C, D, frame_size_m, frame_size_n))[slow_axis_min:slow_axis_max, fast_axis_min:fast_axis_max]
        ground_truth = ground_truth.reshape((-1, frame_size_m, frame_size_n))
        ground_truth_fast = fast[slow_axis_min:slow_axis_max, fast_axis_min:fast_axis_max].squeeze()
        ground_truth_fast = ground_truth_fast.reshape((np.prod(ground_truth_fast.shape)))
        ground_truth_slow = slow[slow_axis_min:slow_axis_max, fast_axis_min:fast_axis_max].squeeze()
        ground_truth_slow = ground_truth_slow.reshape((np.prod(ground_truth_slow.shape)))

        np.testing.assert_equal(ground_truth.shape, out_data.shape,
                                err_msg="The shapes don't match for the bounding box for case3")
        np.testing.assert_array_equal(ground_truth, out_data,
                                      err_msg='There is something up with the bounding box for case3')
        np.testing.assert_array_equal(ground_truth_fast, out_data_fast,
                                      err_msg='There is something up with the fast axis bounding box for case3')
        np.testing.assert_array_equal(ground_truth_slow, out_data_slow,
                                      err_msg='There is something up with the slow axis bounding box for case3')

    def test_position_data_mapping_case_3_with_skipping(self):
        '''
        axis_data.shape (C, D) for data.shape (C*D, frame_size_m, frame_size_n) ,
        '''
        C = 10
        D = 11
        frame_size_m = 5
        frame_size_n = 5
        skip = 2

        positions_slow = np.arange(C)
        positions_fast = np.arange(D)
        fast, slow = np.meshgrid(positions_fast, positions_slow) # just pretend it's a simple grid
        # now chuck them in the files
        with h5.File(self.positions_file, 'w') as f:
            f[self.positions_slow_key] = slow
            f[self.positions_fast_key] = fast

        # make up some data ...
        data = np.arange(C*D*frame_size_m*frame_size_n).reshape(C*D, frame_size_m, frame_size_n)
        with h5.File(self.intensity_file, 'w') as f:
            f[self.intensity_key] = data

        data_params = u.Param()
        data_params.auto_center = False
        data_params.intensities = u.Param()
        data_params.intensities.file = self.intensity_file
        data_params.intensities.key = self.intensity_key

        data_params.positions = u.Param()
        data_params.positions.file = self.positions_file
        data_params.positions.slow_key = self.positions_slow_key
        data_params.positions.fast_key = self.positions_fast_key
        data_params.positions.skip = skip
        output = PtyscanTestRunner(Hdf5Loader, data_params, auto_frames=C*D, cleanup=False)

        with h5.File(output['output_file'], 'r') as f:
            out_data = f['chunks/0/data'][...].squeeze()
        ground_truth = data.reshape((C, D, frame_size_m, frame_size_n))[::skip,::skip]
        ground_truth = ground_truth.reshape((-1, frame_size_m, frame_size_n))

        np.testing.assert_equal(out_data.shape, ground_truth.shape,
                                err_msg="The shapes don't match for the positions for case 4 with skipping")
        np.testing.assert_array_equal(out_data, ground_truth,
                                      err_msg='There is something up with the positions for case 4 with skipping')

    def test_position_data_mapping_case_4(self):
        '''
        axis_data.shape (C,) for data.shape (C, D, frame_size_m, frame_size_n) where D is the size of the other axis,
        '''
        C = 3
        D = 4
        frame_size_m = 50
        frame_size_n = 50


        slow = np.arange(C)
        fast = np.arange(D)
        # now chuck them in the files
        with h5.File(self.positions_file, 'w') as f:
            f[self.positions_slow_key] = slow
            f[self.positions_fast_key] = fast

        # make up some data ...
        data = np.arange(C*D*frame_size_m*frame_size_n).reshape(C, D, frame_size_m, frame_size_n)
        with h5.File(self.intensity_file, 'w') as f:
            f[self.intensity_key] = data

        data_params = u.Param()
        data_params.auto_center = False
        data_params.intensities = u.Param()
        data_params.intensities.file = self.intensity_file
        data_params.intensities.key = self.intensity_key

        data_params.positions = u.Param()
        data_params.positions.file = self.positions_file
        data_params.positions.slow_key = self.positions_slow_key
        data_params.positions.fast_key = self.positions_fast_key
        output = PtyscanTestRunner(Hdf5Loader, data_params, auto_frames=C*D, cleanup=False)

    def test_position_data_mapping_case_4_with_exclusion(self):
        '''
        axis_data.shape (C,) for data.shape (C, D, frame_size_m, frame_size_n) where D is the size of the other axis,
        '''
        C = 10
        D = 11
        frame_size_m = 5
        frame_size_n = 5
        fast_axis_min, fast_axis_max = 1, 3
        slow_axis_min, slow_axis_max = 5, 9

        slow = np.arange(C)
        fast = np.arange(D)
        # now chuck them in the files
        with h5.File(self.positions_file, 'w') as f:
            f[self.positions_slow_key] = slow
            f[self.positions_fast_key] = fast

        # make up some data ...
        data = np.arange(C*D*frame_size_m*frame_size_n).reshape(C, D, frame_size_m, frame_size_n)
        with h5.File(self.intensity_file, 'w') as f:
            f[self.intensity_key] = data

        data_params = u.Param()
        data_params.auto_center = False
        data_params.intensities = u.Param()
        data_params.intensities.file = self.intensity_file
        data_params.intensities.key = self.intensity_key

        data_params.positions = u.Param()
        data_params.positions.file = self.positions_file
        data_params.positions.slow_key = self.positions_slow_key
        data_params.positions.fast_key = self.positions_fast_key
        data_params.positions.bounding_box = u.Param()
        data_params.positions.bounding_box.fast_axis_bounds = fast_axis_min, fast_axis_max
        data_params.positions.bounding_box.slow_axis_bounds = slow_axis_min, slow_axis_max
        output = PtyscanTestRunner(Hdf5Loader, data_params, auto_frames=C*D, cleanup=False)
        with h5.File(output['output_file'], 'r') as f:
            out_data = f['chunks/0/data'][...].squeeze()
        with h5.File(output['output_file'], 'r') as f:
            out_data_fast = f['chunks/0/positions'][:, 1]
        with h5.File(output['output_file'], 'r') as f:
            out_data_slow = f['chunks/0/positions'][:, 0]


        ground_truth = data.reshape((C, D, frame_size_m, frame_size_n))[slow_axis_min:slow_axis_max, fast_axis_min:fast_axis_max]
        ground_truth = ground_truth.reshape((-1, frame_size_m, frame_size_n))


        expected_fast, expected_slow = np.meshgrid(slow, fast)
        expected_fast = expected_fast[slow_axis_min:slow_axis_max, fast_axis_min:fast_axis_max].squeeze()
        expected_slow = expected_slow[slow_axis_min:slow_axis_max, fast_axis_min:fast_axis_max].squeeze()
        expected_slow = expected_slow.flatten()
        expected_fast = expected_fast.flatten()

        np.testing.assert_equal(ground_truth.shape, out_data.shape,
                                err_msg="The shapes don't match for the bounding box for case 4 with exclusion")
        np.testing.assert_array_equal(ground_truth, out_data,
                                      err_msg='There is something up with the bounding box for case 4 with exclusion')
        np.testing.assert_array_equal(expected_fast, out_data_fast,
                                      err_msg='There is something up with the fast axis bounding box for case 4 with exclusion')
        np.testing.assert_array_equal(expected_slow, out_data_slow,
                                      err_msg='There is something up with the slow axis bounding box for case 4 with exclusion')

    def test_position_data_mapping_case_4_with_skipping(self):
        '''
        axis_data.shape (C,) for data.shape (C, D, frame_size_m, frame_size_n) where D is the size of the other axis,
        '''
        C = 4
        D = 8
        frame_size_m = 5
        frame_size_n = 5
        skip = 2

        slow = np.arange(C)
        fast = np.arange(D)
        # now chuck them in the files
        with h5.File(self.positions_file, 'w') as f:
            f[self.positions_slow_key] = slow
            f[self.positions_fast_key] = fast

        # make up some data ...
        data = np.arange(C*D*frame_size_m*frame_size_n).reshape(C, D, frame_size_m, frame_size_n)
        with h5.File(self.intensity_file, 'w') as f:
            f[self.intensity_key] = data

        data_params = u.Param()
        data_params.auto_center = False
        data_params.intensities = u.Param()
        data_params.intensities.file = self.intensity_file
        data_params.intensities.key = self.intensity_key

        data_params.positions = u.Param()
        data_params.positions.file = self.positions_file
        data_params.positions.slow_key = self.positions_slow_key
        data_params.positions.fast_key = self.positions_fast_key
        data_params.positions.skip = skip
        output = PtyscanTestRunner(Hdf5Loader, data_params, auto_frames=C*D, cleanup=False)

        with h5.File(output['output_file'], 'r') as f:
            out_data = f['chunks/0/data'][...].squeeze()
        ground_truth = data.reshape((C, D, frame_size_m, frame_size_n))[::skip,::skip]
        ground_truth = ground_truth.reshape((-1, frame_size_m, frame_size_n))

        np.testing.assert_equal(out_data.shape, ground_truth.shape,
                                err_msg="The shapes don't match for the positions for case 4 with skipping")
        np.testing.assert_array_equal(out_data, ground_truth,
                                      err_msg='There is something up with the positions for case 4 with skipping')

    def test_position_data_mapping_case_5(self):
        '''
        axis_data.shape (C,) for data.shape (C*D, frame_size_m, frame_size_n) where D is the size of the other axis.
        '''

        C = 3
        D = 4
        frame_size_m = 50
        frame_size_n = 50


        slow = np.arange(C)
        fast = np.arange(D)
        # now chuck them in the files
        with h5.File(self.positions_file, 'w') as f:
            f[self.positions_slow_key] = slow
            f[self.positions_fast_key] = fast

        # make up some data ...
        data = np.arange(C*D*frame_size_m*frame_size_n).reshape(C*D, frame_size_m, frame_size_n)
        with h5.File(self.intensity_file, 'w') as f:
            f[self.intensity_key] = data

        data_params = u.Param()
        data_params.auto_center = False
        data_params.intensities = u.Param()
        data_params.intensities.file = self.intensity_file
        data_params.intensities.key = self.intensity_key

        data_params.positions = u.Param()
        data_params.positions.file = self.positions_file
        data_params.positions.slow_key = self.positions_slow_key
        data_params.positions.fast_key = self.positions_fast_key
        output = PtyscanTestRunner(Hdf5Loader, data_params, auto_frames=C*D, cleanup=False)

    def test_position_data_mapping_case_5_with_exclusion(self):
        '''
        axis_data.shape (C,) for data.shape (C*D, frame_size_m, frame_size_n) where D is the size of the other axis.
        '''

        C = 10
        D = 11
        frame_size_m = 5
        frame_size_n = 5
        fast_axis_min, fast_axis_max = 1, 3
        slow_axis_min, slow_axis_max = 5, 9

        slow = np.arange(C)
        fast = np.arange(D)
        # now chuck them in the files
        with h5.File(self.positions_file, 'w') as f:
            f[self.positions_slow_key] = slow
            f[self.positions_fast_key] = fast

        # make up some data ...
        data = np.arange(C*D*frame_size_m*frame_size_n).reshape(C*D, frame_size_m, frame_size_n)
        with h5.File(self.intensity_file, 'w') as f:
            f[self.intensity_key] = data

        data_params = u.Param()
        data_params.auto_center = False
        data_params.intensities = u.Param()
        data_params.intensities.file = self.intensity_file
        data_params.intensities.key = self.intensity_key

        data_params.positions = u.Param()
        data_params.positions.file = self.positions_file
        data_params.positions.slow_key = self.positions_slow_key
        data_params.positions.fast_key = self.positions_fast_key
        data_params.positions.bounding_box = u.Param()
        data_params.positions.bounding_box.fast_axis_bounds = fast_axis_min, fast_axis_max
        data_params.positions.bounding_box.slow_axis_bounds = slow_axis_min, slow_axis_max
        output = PtyscanTestRunner(Hdf5Loader, data_params, auto_frames=C*D, cleanup=False)
        with h5.File(output['output_file'], 'r') as f:
            out_data = f['chunks/0/data'][...].squeeze()
        with h5.File(output['output_file'], 'r') as f:
            out_data_fast = f['chunks/0/positions'][:, 1]
        with h5.File(output['output_file'], 'r') as f:
            out_data_slow = f['chunks/0/positions'][:, 0]

        ground_truth = data.reshape((C, D, frame_size_m, frame_size_n))[slow_axis_min:slow_axis_max, fast_axis_min:fast_axis_max]
        ground_truth = ground_truth.reshape((-1, frame_size_m, frame_size_n))

        expected_fast, expected_slow = np.meshgrid(slow, fast)
        expected_fast = expected_fast[slow_axis_min:slow_axis_max, fast_axis_min:fast_axis_max].squeeze()
        expected_slow = expected_slow[slow_axis_min:slow_axis_max, fast_axis_min:fast_axis_max].squeeze()
        expected_slow = expected_slow.flatten()
        expected_fast = expected_fast.flatten()

        np.testing.assert_equal(ground_truth.shape, out_data.shape,
                                err_msg="The shapes don't match for the bounding box for case 4 with exclusion")
        np.testing.assert_array_equal(ground_truth, out_data,
                                      err_msg='There is something up with the bounding box for case 4 with exclusion')
        np.testing.assert_array_equal(expected_fast, out_data_fast,
                                      err_msg='There is something up with the fast axis bounding box for case 4 with exclusion')
        np.testing.assert_array_equal(expected_slow, out_data_slow,
                                      err_msg='There is something up with the slow axis bounding box for case 4 with exclusion')


    def test_position_data_mapping_case_5_with_skipping(self):
        '''
        axis_data.shape (C,) for data.shape (C*D, frame_size_m, frame_size_n) where D is the size of the other axis.
        '''

        C = 4
        D = 8
        frame_size_m = 5
        frame_size_n = 5
        skip = 2

        slow = np.arange(C)
        fast = np.arange(D)
        # now chuck them in the files
        with h5.File(self.positions_file, 'w') as f:
            f[self.positions_slow_key] = slow
            f[self.positions_fast_key] = fast

        # make up some data ...
        data = np.arange(C*D*frame_size_m*frame_size_n).reshape(C*D, frame_size_m, frame_size_n)
        with h5.File(self.intensity_file, 'w') as f:
            f[self.intensity_key] = data

        data_params = u.Param()
        data_params.auto_center = False
        data_params.intensities = u.Param()
        data_params.intensities.file = self.intensity_file
        data_params.intensities.key = self.intensity_key

        data_params.positions = u.Param()
        data_params.positions.file = self.positions_file
        data_params.positions.slow_key = self.positions_slow_key
        data_params.positions.fast_key = self.positions_fast_key
        data_params.positions.skip = skip
        output = PtyscanTestRunner(Hdf5Loader, data_params, auto_frames=C*D, cleanup=False)

        with h5.File(output['output_file'], 'r') as f:
            out_data = f['chunks/0/data'][...].squeeze()
        ground_truth = data.reshape((C, D, frame_size_m, frame_size_n))[::skip,::skip]
        ground_truth = ground_truth.reshape((-1, frame_size_m, frame_size_n))

        np.testing.assert_equal(out_data.shape, ground_truth.shape,
                                err_msg="The shapes don't match for the positions for case 5 with skipping")
        np.testing.assert_array_equal(out_data, ground_truth,
                                      err_msg='There is something up with the positions for case 5 with skipping')

    def test_normalisation_applied(self):
        k = 12
        frame_size_m = 50
        frame_size_n = 50

        positions_slow = np.arange(k)
        positions_fast = np.arange(k)

        # now chuck them in the files
        with h5.File(self.positions_file, 'w') as f:
            f[self.positions_slow_key] = positions_slow
            f[self.positions_fast_key] = positions_fast

        # make up some data ...
        data = np.arange(k*frame_size_m*frame_size_n).reshape(k, frame_size_m, frame_size_n)
        with h5.File(self.intensity_file, 'w') as f:
            f[self.intensity_key] = data

        normalisation = np.ones_like(positions_slow)
        with h5.File(self.normalisation_file, 'w') as f:
            f[self.normalisation_key] = normalisation

        data_params = u.Param()
        data_params.auto_center = False
        data_params.intensities = u.Param()
        data_params.intensities.file = self.intensity_file
        data_params.intensities.key = self.intensity_key

        data_params.normalisation = u.Param()
        data_params.normalisation.file = self.normalisation_file
        data_params.normalisation.key = self.normalisation_key

        data_params.positions = u.Param()
        data_params.positions.file = self.positions_file
        data_params.positions.slow_key = self.positions_slow_key
        data_params.positions.fast_key = self.positions_fast_key
        output = PtyscanTestRunner(Hdf5Loader, data_params, auto_frames=k, cleanup=False)

    def test_energy_loaded(self):
        k = 12
        frame_size_m = 50
        frame_size_n = 50

        positions_slow = np.arange(k)
        positions_fast = np.arange(k)

        # now chuck them in the files
        with h5.File(self.positions_file, 'w') as f:
            f[self.positions_slow_key] = positions_slow
            f[self.positions_fast_key] = positions_fast

        # make up some data ...
        data = np.arange(k*frame_size_m*frame_size_n).reshape(k, frame_size_m, frame_size_n)
        energy = np.array([9.1])
        with h5.File(self.intensity_file, 'w') as f:
            f[self.intensity_key] = data
            f[self.recorded_energy_key] = energy

        data_params = u.Param()
        data_params.auto_center = False
        data_params.intensities = u.Param()
        data_params.intensities.file = self.intensity_file
        data_params.intensities.key = self.intensity_key

        data_params.recorded_energy = u.Param()
        data_params.recorded_energy.file = self.intensity_file
        data_params.recorded_energy.key = self.recorded_energy_key
        data_params.recorded_energy.multiplier = 1.0

        data_params.positions = u.Param()
        data_params.positions.file = self.positions_file
        data_params.positions.slow_key = self.positions_slow_key
        data_params.positions.fast_key = self.positions_fast_key
        output = PtyscanTestRunner(Hdf5Loader, data_params, auto_frames=k, cleanup=False)

    def test_distance_loaded(self):
        k = 12
        frame_size_m = 50
        frame_size_n = 50

        positions_slow = np.arange(k)
        positions_fast = np.arange(k)

        # now chuck them in the files
        with h5.File(self.positions_file, 'w') as f:
            f[self.positions_slow_key] = positions_slow
            f[self.positions_fast_key] = positions_fast
        distance = np.array([1.5])

        # make up some data ...
        data = np.arange(k*frame_size_m*frame_size_n).reshape(k, frame_size_m, frame_size_n)
        with h5.File(self.intensity_file, 'w') as f:
            f[self.intensity_key] = data
            f[self.recorded_distance_key] = distance

        data_params = u.Param()
        data_params.auto_center = False
        data_params.intensities = u.Param()
        data_params.intensities.file = self.intensity_file
        data_params.intensities.key = self.intensity_key

        data_params.recorded_distance = u.Param()
        data_params.recorded_distance.file = self.intensity_file
        data_params.recorded_distance.key = self.recorded_distance_key
        data_params.recorded_distance.multiplier = 1.0

        data_params.positions = u.Param()
        data_params.positions.file = self.positions_file
        data_params.positions.slow_key = self.positions_slow_key
        data_params.positions.fast_key = self.positions_fast_key
        output = PtyscanTestRunner(Hdf5Loader, data_params, auto_frames=k, cleanup=False)

    def test_psize_loaded(self):
        k = 12
        frame_size_m = 50
        frame_size_n = 50

        positions_slow = np.arange(k)
        positions_fast = np.arange(k)

        # now chuck them in the files
        with h5.File(self.positions_file, 'w') as f:
            f[self.positions_slow_key] = positions_slow
            f[self.positions_fast_key] = positions_fast
        psize = np.array([55e-6])

        # make up some data ...
        data = np.arange(k*frame_size_m*frame_size_n).reshape(k, frame_size_m, frame_size_n)
        with h5.File(self.intensity_file, 'w') as f:
            f[self.intensity_key] = data
            f[self.recorded_psize_key] = psize

        data_params = u.Param()
        data_params.auto_center = False
        data_params.intensities = u.Param()
        data_params.intensities.file = self.intensity_file
        data_params.intensities.key = self.intensity_key

        data_params.recorded_psize = u.Param()
        data_params.recorded_psize.file = self.intensity_file
        data_params.recorded_psize.key = self.recorded_psize_key
        data_params.recorded_psize.multiplier = 1.0

        data_params.positions = u.Param()
        data_params.positions.file = self.positions_file
        data_params.positions.slow_key = self.positions_slow_key
        data_params.positions.fast_key = self.positions_fast_key
        output = PtyscanTestRunner(Hdf5Loader, data_params, auto_frames=k, cleanup=False)


    def test_mask_loaded(self):
        k = 12
        frame_size_m = 50
        frame_size_n = 50

        positions_slow = np.arange(k)
        positions_fast = np.arange(k)

        # now chuck them in the files
        with h5.File(self.positions_file, 'w') as f:
            f[self.positions_slow_key] = positions_slow
            f[self.positions_fast_key] = positions_fast
        distance = np.array([1.5])

        # make up some data ...
        data = np.arange(k*frame_size_m*frame_size_n).reshape((k, frame_size_m, frame_size_n))
        with h5.File(self.intensity_file, 'w') as f:
            f[self.intensity_key] = data

        mask = np.ones(data.shape[-2:], dtype=float)
        mask[::2] = 0
        mask[:, ::2] = 0
        with h5.File(self.mask_file, 'w') as f:
            f[self.mask_key] = mask

        data_params = u.Param()
        data_params.auto_center = False
        data_params.intensities = u.Param()
        data_params.intensities.file = self.intensity_file
        data_params.intensities.key = self.intensity_key

        data_params.mask = u.Param()
        data_params.mask.file = self.mask_file
        data_params.mask.key = self.mask_key

        data_params.positions = u.Param()
        data_params.positions.file = self.positions_file
        data_params.positions.slow_key = self.positions_slow_key
        data_params.positions.fast_key = self.positions_fast_key
        output = PtyscanTestRunner(Hdf5Loader, data_params, auto_frames=k, cleanup=False)



class Hdf5LoaderTestWithSWMR(unittest.TestCase):
    def test_something(self):
        pass

if __name__ == '__main__':
    unittest.main()
