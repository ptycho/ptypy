import unittest
import numpy as np
from . import perfrun, PyCudaTest, have_pycuda

if have_pycuda():
    from pycuda import gpuarray
    from ptypy.accelerate.base import address_manglers as am
    from ptypy.accelerate.cuda_pycuda import address_manglers as gam


COMPLEX_TYPE = np.complex64
FLOAT_TYPE = np.float32
INT_TYPE = np.int32

class AddressManglersTest(PyCudaTest):

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

    def test_get_address_REGRESSION(self):
        # the other manglers are using the BaseMangler's get_address function
        # so we set the deltas in a BaseMangler object and test get_address

        scan_pts=2
        addr_original = self.prepare_addresses(scan_pts=scan_pts)
        addr_original_dev = gpuarray.to_gpu(addr_original)
        nshifts=1
        step_size=2
        mglr = gam.BaseMangler(step_size, 50, 100, nshifts, max_bound=2)
        # 2 shifts, with positive/negative shifting
        mglr.delta = np.array([
            [1, 2], 
            [-4, -2]
        ], dtype=np.int32)
        mglr._setup_delta_gpu()
        
        addr1 = addr_original_dev.copy()
        mglr.get_address(0, addr_original_dev, addr1, 10, 9)
        
        addr2 = addr_original_dev.copy()
        mglr.get_address(1, addr_original_dev, addr2, 10, 9)

        exp1 = np.copy(addr_original)
        exp2 = np.copy(addr_original)
        # element-wise here to prepare reference
        for f in range(addr_original.shape[0]):
            for m in range(addr_original.shape[1]):
                exp1[f, m, 1, 1] = max(0, min(10, addr_original[f, m, 1, 1] + 1))
                exp1[f, m, 1, 2] = max(0, min(9, addr_original[f, m, 1, 2] + 2))
                exp2[f, m, 1, 1] = max(0, min(10, addr_original[f, m, 1, 1] - 4))
                exp2[f, m, 1, 2] = max(0, min(9, addr_original[f, m, 1, 2] - 2))

        np.testing.assert_array_equal(addr2.get(), exp2)
        np.testing.assert_array_equal(addr1.get(), exp1)
        
