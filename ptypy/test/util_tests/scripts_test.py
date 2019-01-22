"""
Tests for functions in ptypy.utils.scripts
"""
import unittest
import numpy as np

from ptypy import defaults_tree
from ptypy.utils.scripts import diversify
from ptypy.utils import Param



class DiversifyTest(unittest.TestCase):

    def setUp(self):

        self.avail_arrays = {}

        # Not of much use but could help testing what happens with unexpected array sizes.
        self.avail_arrays['simple1d'] = np.linspace(0, 25, 30)

        # Random float array representing 100x100 image with 5 layers
        self.avail_arrays['array3dfloat5layers'] = np.resize(np.random.randint(0, 255, (100, 100)), (5, 100, 100)).astype('float64')

        # Random complex array representing 100x100 image with 5 layers
        self.avail_arrays['array3dcomplex'] = np.resize(np.random.randint(0, 255, (100, 100)), (5, 100, 100)).astype('complex64')

        # Random float array representing 100x100 image with 4 layers
        self.avail_arrays['array3dfloat4layers'] = np.resize(np.random.randint(0, 255, (100, 100)), (4, 100, 100)).astype('float64')


    # TODO test for shift argument type/value

    # TODO test that the np array is acted on in place

    def test_first_mode_unchanged(self):
        """
        Test that no changes are made to this first mode. This is complicated by the fact that when there are more
        than one mode there will always be some scaling to all the modes so produce the appropriate power
        """
        test_name = 'test_first_mode_unchanged'
        test_arrays = ['array3dfloat5layers', 'array3dcomplex']
        for array_name in test_arrays:
            print('Testing {} in {}'.format(array_name, test_name))
            array = self.avail_arrays[array_name]
            first_mode_initial = np.copy(array[0])
            first_mode_initial_pwr_correction = first_mode_initial / np.sqrt(array.shape[0])
            diversify(array,  shift=(10, 10), power=1.0)
            np.testing.assert_array_almost_equal(first_mode_initial_pwr_correction, array[0], decimal=5) # FIXME are we happy with this chosen precision


    def test_array_atributes(self):
        """
        Test that some of the attributes of the array are preserved
        """
        test_name = 'test_dtype'
        test_arrays = ['array3dfloat5layers', 'array3dcomplex', 'array3dfloat4layers']
        for array_name in test_arrays:
            print('Testing {} in {}'.format(array_name, test_name))
            array = self.avail_arrays[array_name]
            initial_dtype = array.dtype
            initial_shape = array.shape
            diversify(array)
            self.assertEqual(array.dtype, initial_dtype)
            self.assertEqual(array.shape, initial_shape)


if __name__ == "__main__":
    unittest.main()
