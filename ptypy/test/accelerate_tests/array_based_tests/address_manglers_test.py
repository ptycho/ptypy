import unittest
import sys
import numpy as np
from ptypy.accelerate.array_based.address_manglers import RandomIntMangle

COMPLEX_TYPE = np.complex64
FLOAT_TYPE = np.float32
INT_TYPE = np.int32

class AddressManglersTest(unittest.TestCase):

    def setUp(self):
        import sys
        np.set_printoptions(threshold=sys.maxsize, linewidth=np.inf)

    def tearDown(self):
        np.set_printoptions()


    def test_addr_original_set(self):

        max_bound = 10
        step_size = 3
        scan_pts = 2
        total_number_scan_positions = scan_pts ** 2
        num_modes = 3

        X, Y = np.meshgrid(range(scan_pts), range(scan_pts))
        X = X.reshape((total_number_scan_positions)) + max_bound  # max bound is added in the DM_serial engine.
        Y = Y.reshape((total_number_scan_positions)) + max_bound

        addr = np.zeros((total_number_scan_positions, num_modes, 5, 3), dtype=INT_TYPE)

        exit_idx = 0
        position_idx = 0
        for xpos, ypos in zip(X, Y):  #
            mode_idx = 0
            for pr_mode in range(num_modes):
                for ob_mode in range(1):
                    addr[position_idx, mode_idx] = np.array([[pr_mode, 0, 0],
                                                             [ob_mode, ypos, xpos],
                                                             [exit_idx, 0, 0],
                                                             [0, 0, 0],
                                                             [0, 0, 0]], dtype=INT_TYPE)
                    mode_idx += 1
                    exit_idx += 1
            position_idx += 1

        print(repr(addr))

        old_positions = np.zeros((total_number_scan_positions))

        differences_from_original = np.zeros((len(addr), 2))
        differences_from_original[::2] = 12  # so definitely more than the max_bound
        new_positions = addr[:, 0, 1, 1:] + differences_from_original

        mangler = RandomIntMangle(max_step_per_shift=step_size, max_bound=max_bound)

        # manually set the original_addr
        mangler.addr_original = addr

        mangler.apply_bounding_box(new_positions, old_positions)
        print(repr(new_positions))
        expected_new_positions = new_positions[:]
        expected_new_positions[::2] = 0

        print(repr(expected_new_positions))

        np.testing.assert_array_equal(expected_new_positions, new_positions)
        np.testing.assert_array_equal(expected_new_positions, new_positions)



