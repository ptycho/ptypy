import unittest
import numpy as np
from ptypy.utils.math_utils import delxf, delxb 

class DerivativesTest(unittest.TestCase):

    def test_delxf_1dim(self):
        inp = np.array([0, 1, 2, 4, 8, 0, 6], dtype=np.float32)

        outp = delxf(inp)

        exp = np.array([1, 1, 2, 4, -8, 6, 0], dtype=np.float32)
        np.testing.assert_array_equal(outp, exp)

    def test_delxf_1dim_inplace(self):
        inp = np.array([0, 1, 2, 4, 8, 0, 6], dtype=np.float32)
        outp = np.zeros_like(inp)

        delxf(inp, out=outp)

        exp = np.array([1, 1, 2, 4, -8, 6, 0], dtype=np.float32)
        np.testing.assert_array_equal(outp, exp)

    def test_delxf_2dim1(self):
        inp  = np.array([
            [0, 2, 6],
            [1, -4, 5]
        ], dtype=np.float32)

        outp = delxf(inp, axis=0)

        exp = np.array([
            [1, -6, -1],
            [0, 0, 0]
        ], dtype=np.float32)
        np.testing.assert_array_equal(outp, exp)

    def test_delxf_2dim2(self):
        inp  = np.array([
            [0, 2, 6],
            [1, -4, 5]
        ], dtype=np.float32)

        outp = delxf(inp, axis=1)

        exp = np.array([
            [2, 4, 0],
            [-5, 9, 0]
        ], dtype=np.float32)
        np.testing.assert_array_equal(outp, exp)


    def test_delxb_1dim(self):
        inp = np.array([0, 1, 2, 4, 8, 0, 6], dtype=np.float32)

        outp = delxb(inp)

        exp = np.array([0, 1, 1, 2, 4, -8, 6], dtype=np.float32)
        np.testing.assert_array_equal(outp, exp)

    def test_delxb_2dim1(self):
        inp  = np.array([
            [0, 2, 6],
            [1, -4, 5]
        ], dtype=np.float32)

        outp = delxb(inp, axis=0)

        exp = np.array([
            [0, 0, 0],
            [1, -6, -1],
        ], dtype=np.float32)
        np.testing.assert_array_equal(outp, exp)

    def test_delxb_2dim2(self):
        inp  = np.array([
            [0, 2, 6],
            [1, -4, 5]
        ], dtype=np.float32)

        outp = delxb(inp, axis=1)

        exp = np.array([
            [0, 2, 4],
            [0, -5, 9]
        ], dtype=np.float32)
        np.testing.assert_array_equal(outp, exp)

    