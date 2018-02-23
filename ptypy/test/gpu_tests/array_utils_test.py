'''
Tests for the array_utils module
'''


import unittest
from ptypy.array_based import array_utils as au
from ptypy.array_based import FLOAT_TYPE
from ptypy.gpu import array_utils as gau
from ptypy.gpu import FLOAT_TYPE as GPU_FLOAT_TYPE
import numpy as np


class ArrayUtilsTest(unittest.TestCase):

    @unittest.skip("This method is not implemented yet")
    def test_abs2_real_input_UNITY(self):
        x = np.ones((3,3))
        np.testing.assert_array_equal(au.abs2(x), gau.abs2(x))

    @unittest.skip("This method is not implemented yet")
    def test_abs2_complex_input_UNITY(self):
        x = np.ones((3,3)) + 1j*np.ones((3,3))
        np.testing.assert_array_equal(au.abs2(x), gau.abs2(x))

    @unittest.skip("This method is not implemented yet")
    def test_sum_to_buffer_UNITY(self):

        in1 = np.array([[1.0, 1.0, 1.0, 1.0],
                        [2.0, 2.0, 2.0, 2.0],
                        [3.0, 3.0, 3.0, 3.0],
                        [4.0, 4.0, 4.0, 4.0]])

        outshape = (2, 4)

        in1_addr = np.array([(0, 0, 0),
                            (1, 0, 0),
                            (2, 0, 0),
                            (3, 0, 0)])

        out1_addr = np.array([(0, 0, 0),
                              (1, 0, 0),
                              (0, 0, 0),
                              (1, 0, 0)])
        np.testing.assert_array_equal(au.sum_to_buffer(in1, outshape, in1_addr, out1_addr, dtype=FLOAT_TYPE),
                                      gau.sum_to_buffer(in1, outshape, in1_addr, out1_addr, dtype=GPU_FLOAT_TYPE))


if __name__=='__main__':
    unittest.main()