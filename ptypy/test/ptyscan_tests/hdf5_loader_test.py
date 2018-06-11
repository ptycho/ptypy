import unittest
import tempfile
import os
import h5py as h5
import numpy as np

from ptypy.test.utils import PtyscanTestRunner
from ptypy.experiment import Hdf5Loader
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

    @classmethod
    def setUpClass(cls):
        '''
        This should create files for:
        intensity_file, positions, mask, dark, flat, normalisation. The entries for the data should be bound attributes so that we
        can overwrite the data with whatever we want.
        We should link the data into a top level file (which is supposed to be indicative of a nexus/ CXI with external
        links).
        '''
        cls.outdir = tempfile.mkdtemp("Hdf5LoaderTestNoSWMR")

        cls.intensity_file = os.path.join(cls.outdir, 'intensity.h5')
        cls.intensity_key = 'entry/intensity'
        create_file_and_dataset(path=cls.intensity_file, key=cls.intensity_key, data_type=np.float)

        cls.positions_file = os.path.join(cls.outdir, 'positions.h5')
        cls.positions_slow_key = 'entry/positions_slow'
        cls.positions_fast_key = 'entry/positions_fast'
        create_file_and_dataset(path=cls.positions_file, key=[cls.positions_slow_key, cls.positions_fast_key],
                                data_type=np.float)

        cls.mask_file = os.path.join(cls.outdir, 'mask.h5')
        cls.mask_key = 'entry/mask'
        create_file_and_dataset(path=cls.mask_file, key=cls.mask_key, data_type=np.int)

        cls.dark_file = os.path.join(cls.outdir, 'dark.h5')
        cls.dark_key = 'entry/dark'
        create_file_and_dataset(path=cls.dark_file, key=cls.dark_key, data_type=np.float)

        cls.flat_file = os.path.join(cls.outdir, 'flat.h5')
        cls.flat_key = 'entry/flat'
        create_file_and_dataset(path=cls.flat_file, key=cls.flat_key, data_type=np.float)

        cls.normalisation_file = os.path.join(cls.outdir, 'normalisation.h5')
        cls.normalisation_key = 'entry/normalisation'
        create_file_and_dataset(path=cls.normalisation_file, key=cls.normalisation_key, data_type=np.float)

        cls.top_file = os.path.join(cls.outdir, 'top_file.h5')
        cls.recorded_energy_key = 'entry/energy'
        cls.recorded_distance_key = 'entry/distance'
        with h5.File(cls.top_file, 'w') as f:
            f.create_group('entry')
            f[cls.recorded_energy_key] = 9.1
            f[cls.recorded_distance_key] = 1.5
            # now lets link some stuff in.
            f[cls.intensity_key] = h5.ExternalLink(cls.intensity_file, cls.intensity_key)
            f[cls.positions_slow_key] = h5.ExternalLink(cls.positions_file, cls.positions_slow_key)
            f[cls.positions_fast_key] = h5.ExternalLink(cls.positions_file, cls.positions_fast_key)
            f[cls.normalisation_key] = h5.ExternalLink(cls.normalisation_file, cls.normalisation_key)


    def test_position_data_mapping_case_1(self):
        '''
        axis_data.shape (A, B) for data.shape (A, B, frame_size_m, frame_size_n),
        '''
        A = 3
        B = 4
        frame_size_m = 50
        frame_size_n = 60


        positions_slow = np.arange(A)
        positions_fast = np.arange(B)
        fast, slow = np.meshgrid(positions_fast, positions_slow) # just pretend it's a simple grid
        # now chuck them in the files
        with h5.File(self.positions_file, 'w') as f:
            f[self.positions_slow_key] = slow
            f[self.positions_fast_key] = fast

        # make up some data ...
        data = np.arange(A*B*frame_size_m*frame_size_n).reshape(A, B, frame_size_m, frame_size_n)
        h5.File(self.intensity_file, 'w')[self.intensity_key] = data

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


    def test_position_data_mapping_case_2(self):
        '''
        axis_data.shape (k,) for data.shape (k, frame_size_m, frame_size_n)
        '''
        k = 12
        frame_size_m = 50
        frame_size_n = 60

        positions_slow = np.arange(k)
        positions_fast = np.arange(k)

        # now chuck them in the files
        with h5.File(self.positions_file, 'w') as f:
            f[self.positions_slow_key] = positions_slow
            f[self.positions_fast_key] = positions_fast

        # make up some data ...
        data = np.arange(k*frame_size_m*frame_size_n).reshape(k, frame_size_m, frame_size_n)
        h5.File(self.intensity_file, 'w')[self.intensity_key] = data

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

    def test_position_data_mapping_case_3(self):
        '''
        axis_data.shape (C, D) for data.shape (C*D, frame_size_m, frame_size_n) ,
        '''
        C = 3
        D = 4
        frame_size_m = 50
        frame_size_n = 60


        positions_slow = np.arange(C)
        positions_fast = np.arange(D)
        fast, slow = np.meshgrid(positions_fast, positions_slow) # just pretend it's a simple grid
        # now chuck them in the files
        with h5.File(self.positions_file, 'w') as f:
            f[self.positions_slow_key] = slow
            f[self.positions_fast_key] = fast

        # make up some data ...
        data = np.arange(C*D*frame_size_m*frame_size_n).reshape(C*D, frame_size_m, frame_size_n)
        h5.File(self.intensity_file, 'w')[self.intensity_key] = data

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


    def test_position_data_mapping_case_4(self):
        '''
        axis_data.shape (C,) for data.shape (C, D, frame_size_m, frame_size_n) where D is the size of the other axis,
        '''
        C = 3
        D = 4
        frame_size_m = 50
        frame_size_n = 60


        slow = np.arange(C)
        fast = np.arange(D)
        # now chuck them in the files
        with h5.File(self.positions_file, 'w') as f:
            f[self.positions_slow_key] = slow
            f[self.positions_fast_key] = fast

        # make up some data ...
        data = np.arange(C*D*frame_size_m*frame_size_n).reshape(C, D, frame_size_m, frame_size_n)
        h5.File(self.intensity_file, 'w')[self.intensity_key] = data

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

    def test_position_data_mapping_case_5(self):
        '''
        axis_data.shape (C,) for data.shape (C*D, frame_size_m, frame_size_n) where D is the size of the other axis.
        '''

        C = 3
        D = 4
        frame_size_m = 50
        frame_size_n = 60


        slow = np.arange(C)
        fast = np.arange(D)
        # now chuck them in the files
        with h5.File(self.positions_file, 'w') as f:
            f[self.positions_slow_key] = slow
            f[self.positions_fast_key] = fast

        # make up some data ...
        data = np.arange(C*D*frame_size_m*frame_size_n).reshape(C*D, frame_size_m, frame_size_n)
        h5.File(self.intensity_file, 'w')[self.intensity_key] = data

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


    def test_flatfield_applied_case_1(self):
        pass

    def test_flatfield_applied_case_2(self):
        pass

    def test_darkfield_applied_case_1(self):
        pass

    def test_darkfield_applied_case_2(self):
        pass

    def test_normalisation_applied_case(self):
        pass

    def test_energy_loaded(self):
        pass

    def test_distance_loaded(self):
        pass

    def test_mask_loaded(self):
        pass

    def test_crop_load_works(self):
        pass



class Hdf5LoaderTestWithSWMR(unittest.TestCase):
    def test_something(self):
        pass

if __name__ == '__main__':
    unittest.main()
