import unittest
import sys
import numpy as np
from ptypy.accelerate.base.address_manglers import BaseMangler, RandomIntMangler

COMPLEX_TYPE = np.complex64
FLOAT_TYPE = np.float32
INT_TYPE = np.int32

class AddressManglersTest(unittest.TestCase):

    def setUp(self):
        import sys
        np.set_printoptions(threshold=sys.maxsize, linewidth=np.inf)

    def tearDown(self):
        np.set_printoptions()

    def prepare_addresses(self, max_bound=10, scan_pts=2, num_modes=3):
        total_number_scan_positions = scan_pts ** 2
        X, Y = np.meshgrid(range(scan_pts), range(scan_pts))
        X = X.reshape((total_number_scan_positions)) + max_bound  # max bound is added in the DM_serial engine.
        Y = Y.reshape((total_number_scan_positions)) + max_bound

        addr_original = np.zeros((total_number_scan_positions, num_modes, 5, 3), dtype=INT_TYPE)

        exit_idx = 0
        position_idx = 0
        for xpos, ypos in zip(X, Y):  #
            mode_idx = 0
            for pr_mode in range(num_modes):
                for ob_mode in range(1):
                    addr_original[position_idx, mode_idx] = np.array([[pr_mode, 0, 0],
                                                             [ob_mode, ypos, xpos],
                                                             [exit_idx, 0, 0],
                                                             [0, 0, 0],
                                                             [0, 0, 0]], dtype=INT_TYPE)
                    mode_idx += 1
                    exit_idx += 1
            position_idx += 1
        
        return addr_original

    def test_apply_bounding_box(self):

        scan_pts=2
        max_bound=10
        addr = self.prepare_addresses(scan_pts=scan_pts, max_bound=max_bound)
        step_size = 3
        
        mangler = BaseMangler(step_size, 50, 100, nshifts=1, max_bound=max_bound, )
        min_oby = 1
        max_oby = 10
        min_obx = 2
        max_obx = 9
        mangler.apply_bounding_box(addr[:,:,1,1], min_oby, max_oby)
        mangler.apply_bounding_box(addr[:,:,1,2], min_obx, max_obx)
        
        np.testing.assert_array_less(addr[:,:,1,1], max_oby+1)
        np.testing.assert_array_less(addr[:,:,1,2], max_obx+1)
        np.testing.assert_array_less(min_oby-1, addr[:,:,1,1])
        np.testing.assert_array_less(min_obx-1, addr[:,:,1,2])


    def test_get_address(self):
        # the other manglers are using the BaseMangler's get_address function
        # so we set the deltas in a BaseMangler object and test get_address

        scan_pts=2
        addr_original = self.prepare_addresses(scan_pts=scan_pts)
        total_number_scan_positions = scan_pts ** 2
        addr1 = np.copy(addr_original)
        addr2 = np.copy(addr_original)
        nshifts=1
        step_size=2
        mglr = BaseMangler(step_size, 50, 100, nshifts, max_bound=2)
        # 2 shifts, with positive/negative shifting
        mglr.delta = np.array([
            [1, 2], 
            [-4, -2]
        ])
        mglr.get_address(0, addr_original, addr1, 10, 9)
        mglr.get_address(1, addr_original, addr2, 10, 9)

        exp1 = np.copy(addr_original)
        exp2 = np.copy(addr_original)
        # element-wise here to prepare reference
        for f in range(addr_original.shape[0]):
            for m in range(addr_original.shape[1]):
                exp1[f, m, 1, 1] = max(0, min(10, addr_original[f, m, 1, 1] + 1))
                exp1[f, m, 1, 2] = max(0, min(9, addr_original[f, m, 1, 2] + 2))
                exp2[f, m, 1, 1] = max(0, min(10, addr_original[f, m, 1, 1] - 4))
                exp2[f, m, 1, 2] = max(0, min(9, addr_original[f, m, 1, 2] - 2))

        np.testing.assert_array_equal(addr1, exp1)
        np.testing.assert_array_equal(addr2, exp2)
