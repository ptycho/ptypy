'''
tests for the object-probe interactions, including the specific DM, ePIE etc updates

'''

import unittest
import numpy as np
import utils as tu
from ptypy.gpu import data_utils as du
from ptypy.array_based import object_probe_interaction as opi
from ptypy.array_based import COMPLEX_TYPE, FLOAT_TYPE
from ptypy.gpu import object_probe_interaction as gopi
from copy import deepcopy
from utils import print_array_info

from ptypy.gpu.config import init_gpus, reset_function_cache
init_gpus(0)

class ObjectProbeInteractionTest(unittest.TestCase):

    def tearDown(self):
        # reset the cached GPU functions after each test
        reset_function_cache()

    def test_scan_and_multiply_UNITY(self):
        PtychoInstance = tu.get_ptycho_instance('pod_to_numpy_test')
        # now convert to arrays
        vectorised_scan = du.pod_to_arrays(PtychoInstance, 'S0000')
        addr = vectorised_scan['meta']['addr'] # probably want to extract these at a later date, but just to get stuff going...
        probe = vectorised_scan['probe']
        obj = vectorised_scan['obj']
        exit_wave = vectorised_scan['exit wave']
        addr_info = addr[:, 0]

        # add one, to avoid having a lot of zeros and hence disturbing the result
        probe = np.add(probe, 1)
        obj = np.add(obj,1)

        po = opi.scan_and_multiply(probe, obj, exit_wave.shape, addr_info)
        gpo = gopi.scan_and_multiply(probe, obj, exit_wave.shape, addr_info)
        
        np.testing.assert_array_equal(po, gpo)
        #np.testing.assert_allclose(po, gpo)
        
    def test_difference_map_realspace_constraint_UNITY(self):
        PtychoInstance = tu.get_ptycho_instance('pod_to_numpy_test')
        # now convert to arrays
        vectorised_scan = du.pod_to_arrays(PtychoInstance, 'S0000')
        addr = vectorised_scan['meta']['addr'] # probably want to extract these at a later date, but just to get stuff going...
        probe = vectorised_scan['probe']
        obj = vectorised_scan['obj']
        exit_wave = vectorised_scan['exit wave']
        
        view_dlayer = 0  # what is this?
        addr_info = addr[:, (view_dlayer)]  # addresses, object references
        probe_object = opi.scan_and_multiply(probe, obj, exit_wave.shape, addr_info)
        po = opi.difference_map_realspace_constraint(probe_object, exit_wave, alpha=1.0)
        gpo = gopi.difference_map_realspace_constraint(probe_object, exit_wave, alpha=1.0)
        np.testing.assert_array_equal(po, gpo)

    def test_extract_array_from_exit_wave_UNITY_case_a(self):
        # two cases for this a) the array to be updated is bigger than the extracted array (which is the same size as the exit wave)
        #                    b) the other way round
        B = 5
        C = 5

        D = 2
        E = B
        F = C

        npts_greater_than = 2
        G = 2
        H = B + npts_greater_than
        I = C + npts_greater_than

        scan_pts = 2
        A = scan_pts ** 2 * G * D  # this is a 16 point scan pattern (4x4 grid) over all the modes

        # shapes and types outlined here
        exit_wave = np.empty(shape=(A, B, C), dtype=COMPLEX_TYPE)
        for idx in range(A):
            exit_wave[idx] = np.ones((B, C)) * (idx + 1) + 1j * np.ones((B, C)) * (idx + 1)

        exit_addr = np.empty(shape=(A, 3), dtype=int)
        exit_addr[:, 0] = np.array(range(A))
        exit_addr[:, 1] = np.zeros((A,))
        exit_addr[:, 2] = np.zeros((A,))

        array_to_be_extracted = np.empty(shape=(D, E, F), dtype=COMPLEX_TYPE)
        for idx in range(D):
            array_to_be_extracted[idx] = np.ones((E, F)) * (idx + 1) + 1j * np.ones((E, F)) * (idx + 1)
        extract_addr = np.empty(shape=(A, 3), dtype=int)
        extract_addr[:, 0] = np.array(range(D)).repeat(A / D)
        extract_addr[:, 1] = np.zeros((A,))
        extract_addr[:, 2] = np.zeros((A,))

        array_to_be_updated = np.empty(shape=(G, H, I), dtype=COMPLEX_TYPE)
        for idx in range(G):
            array_to_be_updated[idx] = np.ones((H, I)) * (3 * idx + 1) + 1j * np.ones((H, I)) * (3 * idx + 1)
        update_addr = np.empty(shape=(A, 3), dtype=int)
        update_addr[:, 0] = np.array(range(G)).repeat(A / G)
        X, Y = np.meshgrid(range(scan_pts), range(scan_pts))
        X = X.reshape((scan_pts ** 2))
        Y = Y.reshape((scan_pts ** 2))
        for idx in range(G):
            for idy in range(D):
                index = idy + 2 * idx
                update_addr[index::scan_pts ** 2, 1] = X
                update_addr[index::scan_pts ** 2, 2] = Y

        weights = np.empty(shape=(G,), dtype=FLOAT_TYPE)
        weights[:] = np.linspace(-1, 1, G)
        cfact = np.empty_like(array_to_be_updated)
        for idx in range(G):
            cfact[idx] = np.ones((H, I)) * 10 * (idx + 1)

        garray_to_be_updated = deepcopy(array_to_be_updated)
        gcfact = deepcopy(cfact)

        opi.extract_array_from_exit_wave(exit_wave, exit_addr, array_to_be_extracted, extract_addr, array_to_be_updated,
                                         update_addr, cfact, weights)

        gopi.extract_array_from_exit_wave(exit_wave, exit_addr, array_to_be_extracted, extract_addr, garray_to_be_updated,
                                         update_addr, gcfact, weights)


        np.testing.assert_array_equal(array_to_be_updated,
                                      garray_to_be_updated,
                                      err_msg="The array has not been extracted properly from the exit wave.")

    def test_extract_array_from_exit_wave_UNITY_case_b(self):
        # two cases for this a) the array to be updated is bigger than the extracted array (which is the same size as the exit wave)
        #                    b) the other way round
        #
        npts_greater_than = 2
        B = 5
        C = 5

        D = 2
        E = C + npts_greater_than
        F = C + npts_greater_than

        G = 2
        H = B
        I = C

        scan_pts = 2
        A = scan_pts ** 2 * G * D  # this is a 16 point scan pattern (4x4 grid) over all the modes

        # shapes and types outlined here
        exit_wave = np.empty(shape=(A, B, C), dtype=COMPLEX_TYPE)
        for idx in range(A):
            exit_wave[idx] = np.ones((B, C)) * (idx + 1) + 1j * np.ones((B, C)) * (idx + 1)

        exit_addr = np.empty(shape=(A, 3), dtype=int)
        exit_addr[:, 0] = np.array(range(A))
        exit_addr[:, 1] = np.zeros((A,))
        exit_addr[:, 2] = np.zeros((A,))

        array_to_be_extracted = np.empty(shape=(D, E, F), dtype=COMPLEX_TYPE)
        for idx in range(D):
            array_to_be_extracted[idx] = np.ones((E, F)) * (idx + 1) + 1j * np.ones((E, F)) * (idx + 1)
        extract_addr = np.empty(shape=(A, 3), dtype=int)
        extract_addr[:, 0] = np.array(range(D)).repeat(A / D)
        update_addr = np.empty(shape=(A, 3), dtype=int)
        update_addr[:, 0] = np.array(range(G)).repeat(A / G)
        X, Y = np.meshgrid(range(scan_pts), range(scan_pts))
        X = X.reshape((scan_pts ** 2))
        Y = Y.reshape((scan_pts ** 2))
        for idx in range(G):
            for idy in range(D):
                index = idy + 2 * idx
                extract_addr[index::scan_pts ** 2, 1] = X
                extract_addr[index::scan_pts ** 2, 2] = Y

        array_to_be_updated = np.empty(shape=(G, H, I), dtype=COMPLEX_TYPE)
        for idx in range(G):
            array_to_be_updated[idx] = np.ones((H, I)) * (3 * idx + 1) + 1j * np.ones((H, I)) * (3 * idx + 1)
        update_addr = np.empty(shape=(A, 3), dtype=int)
        update_addr[:, 0] = np.array(range(G)).repeat(A / G)
        update_addr[:, 1] = np.zeros((A,))
        update_addr[:, 2] = np.zeros((A,))

        weights = np.empty(shape=(G,), dtype=FLOAT_TYPE)
        weights[:] = np.linspace(-1, 1, G)
        cfact = np.empty_like(array_to_be_updated)
        for idx in range(G):
            cfact[idx] = np.ones((H, I)) * 10 * (idx + 1)

        garray_to_be_updated = deepcopy(array_to_be_updated)
        gcfact = deepcopy(cfact)

        opi.extract_array_from_exit_wave(exit_wave, exit_addr, array_to_be_extracted, extract_addr,
                                         array_to_be_updated,
                                         update_addr, cfact, weights)
        gopi.extract_array_from_exit_wave(exit_wave, exit_addr, array_to_be_extracted, extract_addr,
                                         garray_to_be_updated,
                                         update_addr, gcfact, weights)


        np.testing.assert_array_equal(array_to_be_updated, garray_to_be_updated)

    @unittest.skip("This method is not implemented yet")
    def test_difference_map_update_probe_UNITY_with_support(self):
        '''
        This tests difference_map_update_probe, which wraps extract_array_from_exit_wave
        '''
        B = 5
        C = 5

        D = 2
        E = B
        F = C

        npts_greater_than = 2
        G = 2
        H = B + npts_greater_than
        I = C + npts_greater_than

        scan_pts = 2
        A = scan_pts ** 2 * G * D  # this is a 16 point scan pattern (4x4 grid) over all the modes

        # shapes and types outlined here
        exit_wave = np.empty(shape=(A, B, C), dtype=COMPLEX_TYPE)
        for idx in range(A):
            exit_wave[idx] = np.ones((B, C)) * (idx + 1) + 1j * np.ones((B, C)) * (idx + 1)

        exit_addr = np.empty(shape=(A, 3), dtype=int)
        exit_addr[:, 0] = np.array(range(A))
        exit_addr[:, 1] = np.zeros((A,))
        exit_addr[:, 2] = np.zeros((A,))

        array_to_be_extracted = np.empty(shape=(D, E, F), dtype=COMPLEX_TYPE)
        for idx in range(D):
            array_to_be_extracted[idx] = np.ones((E, F)) * (idx + 1) + 1j * np.ones((E, F)) * (idx + 1)
        extract_addr = np.empty(shape=(A, 3), dtype=int)
        extract_addr[:, 0] = np.array(range(D)).repeat(A / D)
        extract_addr[:, 1] = np.zeros((A,))
        extract_addr[:, 2] = np.zeros((A,))

        array_to_be_updated = np.empty(shape=(G, H, I), dtype=COMPLEX_TYPE)
        for idx in range(G):
            array_to_be_updated[idx] = np.ones((H, I)) * (3 * idx + 1) + 1j * np.ones((H, I)) * (3 * idx + 1)
        update_addr = np.empty(shape=(A, 3), dtype=int)
        update_addr[:, 0] = np.array(range(G)).repeat(A / G)
        X, Y = np.meshgrid(range(scan_pts), range(scan_pts))
        X = X.reshape((scan_pts ** 2))
        Y = Y.reshape((scan_pts ** 2))
        for idx in range(G):
            for idy in range(D):
                index = idy + 2 * idx
                update_addr[index::scan_pts ** 2, 1] = X
                update_addr[index::scan_pts ** 2, 2] = Y

        weights = np.empty(shape=(G,), dtype=FLOAT_TYPE)
        weights[:] = np.linspace(-1, 1, G)
        cfact = np.empty_like(array_to_be_updated)
        for idx in range(G):
            cfact[idx] = np.ones((H, I)) * 10 * (idx + 1)

        dummy_addr = np.zeros_like(extract_addr) #  these aren't used by the function, but are passed as a top level address book
        addr_info  = zip(update_addr, extract_addr, exit_addr, dummy_addr, dummy_addr)
        probe_support = np.ones_like(array_to_be_updated) * 100.0
        #(ob, probe_weights, probe, exit_wave, addr_info, cfact_probe, probe_support = None)

        garray_to_be_updated = deepcopy(array_to_be_updated)
        opi.difference_map_update_probe(array_to_be_extracted, weights, array_to_be_updated, exit_wave, addr_info, cfact, probe_support=probe_support)
        gopi.difference_map_update_probe(array_to_be_extracted, weights, array_to_be_updated, exit_wave, addr_info,
                                        cfact, probe_support=probe_support)

        np.testing.assert_array_equal(array_to_be_updated, garray_to_be_updated)

    @unittest.skip("This method is not implemented yet")
    def test_difference_map_update_probe_UNITY_without_support(self):
        '''
        This tests difference_map_update_probe, which wraps extract_array_from_exit_wave
        '''
        B = 5
        C = 5

        D = 2
        E = B
        F = C

        npts_greater_than = 2
        G = 2
        H = B + npts_greater_than
        I = C + npts_greater_than

        scan_pts = 2
        A = scan_pts ** 2 * G * D  # this is a 16 point scan pattern (4x4 grid) over all the modes

        # shapes and types outlined here
        exit_wave = np.empty(shape=(A, B, C), dtype=COMPLEX_TYPE)
        for idx in range(A):
            exit_wave[idx] = np.ones((B, C)) * (idx + 1) + 1j * np.ones((B, C)) * (idx + 1)

        exit_addr = np.empty(shape=(A, 3), dtype=int)
        exit_addr[:, 0] = np.array(range(A))
        exit_addr[:, 1] = np.zeros((A,))
        exit_addr[:, 2] = np.zeros((A,))

        array_to_be_extracted = np.empty(shape=(D, E, F), dtype=COMPLEX_TYPE)
        for idx in range(D):
            array_to_be_extracted[idx] = np.ones((E, F)) * (idx + 1) + 1j * np.ones((E, F)) * (idx + 1)
        extract_addr = np.empty(shape=(A, 3), dtype=int)
        extract_addr[:, 0] = np.array(range(D)).repeat(A / D)
        extract_addr[:, 1] = np.zeros((A,))
        extract_addr[:, 2] = np.zeros((A,))

        array_to_be_updated = np.empty(shape=(G, H, I), dtype=COMPLEX_TYPE)
        for idx in range(G):
            array_to_be_updated[idx] = np.ones((H, I)) * (3 * idx + 1) + 1j * np.ones((H, I)) * (3 * idx + 1)
        update_addr = np.empty(shape=(A, 3), dtype=int)
        update_addr[:, 0] = np.array(range(G)).repeat(A / G)
        X, Y = np.meshgrid(range(scan_pts), range(scan_pts))
        X = X.reshape((scan_pts ** 2))
        Y = Y.reshape((scan_pts ** 2))
        for idx in range(G):
            for idy in range(D):
                index = idy + 2 * idx
                update_addr[index::scan_pts ** 2, 1] = X
                update_addr[index::scan_pts ** 2, 2] = Y

        weights = np.empty(shape=(G,), dtype=FLOAT_TYPE)
        weights[:] = np.linspace(-1, 1, G)
        cfact = np.empty_like(array_to_be_updated)
        for idx in range(G):
            cfact[idx] = np.ones((H, I)) * 10 * (idx + 1)

        dummy_addr = np.zeros_like(extract_addr) #  these aren't used by the function, but are passed as a top level address book
        addr_info  = zip(update_addr, extract_addr, exit_addr, dummy_addr, dummy_addr)
        #(ob, probe_weights, probe, exit_wave, addr_info, cfact_probe, probe_support = None)

        garray_to_be_updated = deepcopy(array_to_be_updated)
        opi.difference_map_update_probe(array_to_be_extracted, weights, array_to_be_updated, exit_wave, addr_info, cfact, probe_support=None)
        gopi.difference_map_update_probe(array_to_be_extracted, weights, garray_to_be_updated, exit_wave, addr_info,
                                        cfact, probe_support=None)

        np.testing.assert_array_equal(array_to_be_updated, garray_to_be_updated)

    @unittest.skip("This method is not implemented yet")
    def test_difference_map_update_object_with_no_smooth_or_clip_UNITY(self):
        B = 5
        C = 5

        D = 2
        E = B
        F = C

        npts_greater_than = 2
        G = 2
        H = B + npts_greater_than
        I = C + npts_greater_than

        scan_pts = 2
        A = scan_pts ** 2 * G * D  # this is a 16 point scan pattern (4x4 grid) over all the modes

        # shapes and types outlined here
        exit_wave = np.empty(shape=(A, B, C), dtype=COMPLEX_TYPE)
        for idx in range(A):
            exit_wave[idx] = np.ones((B, C)) * (idx + 1) + 1j * np.ones((B, C)) * (idx + 1)

        exit_addr = np.empty(shape=(A, 3), dtype=int)
        exit_addr[:, 0] = np.array(range(A))
        exit_addr[:, 1] = np.zeros((A,))
        exit_addr[:, 2] = np.zeros((A,))

        array_to_be_extracted = np.empty(shape=(D, E, F), dtype=COMPLEX_TYPE)
        for idx in range(D):
            array_to_be_extracted[idx] = np.ones((E, F)) * (idx + 1) + 1j * np.ones((E, F)) * (idx + 1)
        extract_addr = np.empty(shape=(A, 3), dtype=int)
        extract_addr[:, 0] = np.array(range(D)).repeat(A / D)
        extract_addr[:, 1] = np.zeros((A,))
        extract_addr[:, 2] = np.zeros((A,))

        array_to_be_updated = np.empty(shape=(G, H, I), dtype=COMPLEX_TYPE)
        for idx in range(G):
            array_to_be_updated[idx] = np.ones((H, I)) * (3 * idx + 1) + 1j * np.ones((H, I)) * (3 * idx + 1)
        update_addr = np.empty(shape=(A, 3), dtype=int)
        update_addr[:, 0] = np.array(range(G)).repeat(A / G)
        X, Y = np.meshgrid(range(scan_pts), range(scan_pts))
        X = X.reshape((scan_pts ** 2))
        Y = Y.reshape((scan_pts ** 2))
        for idx in range(G):
            for idy in range(D):
                index = idy + 2 * idx
                update_addr[index::scan_pts ** 2, 1] = X
                update_addr[index::scan_pts ** 2, 2] = Y

        weights = np.empty(shape=(G,), dtype=FLOAT_TYPE)
        weights[:] = np.linspace(-1, 1, G)
        cfact = np.empty_like(array_to_be_updated)
        for idx in range(G):
            cfact[idx] = np.ones((H, I)) * 10 * (idx + 1)

        dummy_addr = np.zeros_like(extract_addr) #  these aren't used by the function, but are passed as a top level address book
        addr_info  = zip(extract_addr, update_addr , exit_addr, dummy_addr, dummy_addr)

        garray_to_be_updated = deepcopy(array_to_be_updated)
        opi.difference_map_update_object(array_to_be_updated, weights, array_to_be_extracted, exit_wave, addr_info, cfact, ob_smooth_std=None, clip_object=None)
        gopi.difference_map_update_object(garray_to_be_updated, weights, array_to_be_extracted, exit_wave, addr_info,
                                         cfact, ob_smooth_std=None, clip_object=None)

        np.testing.assert_array_equal(array_to_be_updated, garray_to_be_updated)

    @unittest.skip("This method is not implemented yet")
    def test_difference_map_update_object_with_smooth_but_no_clip_UNITY(self):
        B = 5
        C = 5

        D = 2
        E = B
        F = C

        npts_greater_than = 2
        G = 2
        H = B + npts_greater_than
        I = C + npts_greater_than

        scan_pts = 2
        A = scan_pts ** 2 * G * D  # this is a 16 point scan pattern (4x4 grid) over all the modes

        # shapes and types outlined here
        exit_wave = np.empty(shape=(A, B, C), dtype=COMPLEX_TYPE)
        for idx in range(A):
            exit_wave[idx] = np.ones((B, C)) * (idx + 1) + 1j * np.ones((B, C)) * (idx + 1)

        exit_addr = np.empty(shape=(A, 3), dtype=int)
        exit_addr[:, 0] = np.array(range(A))
        exit_addr[:, 1] = np.zeros((A,))
        exit_addr[:, 2] = np.zeros((A,))

        array_to_be_extracted = np.empty(shape=(D, E, F), dtype=COMPLEX_TYPE)
        for idx in range(D):
            array_to_be_extracted[idx] = np.ones((E, F)) * (idx + 1) + 1j * np.ones((E, F)) * (idx + 1)
        extract_addr = np.empty(shape=(A, 3), dtype=int)
        extract_addr[:, 0] = np.array(range(D)).repeat(A / D)
        extract_addr[:, 1] = np.zeros((A,))
        extract_addr[:, 2] = np.zeros((A,))

        array_to_be_updated = np.empty(shape=(G, H, I), dtype=COMPLEX_TYPE)
        for idx in range(G):
            array_to_be_updated[idx] = np.ones((H, I)) * (3 * idx + 1) + 1j * np.ones((H, I)) * (3 * idx + 1)
        update_addr = np.empty(shape=(A, 3), dtype=int)
        update_addr[:, 0] = np.array(range(G)).repeat(A / G)
        X, Y = np.meshgrid(range(scan_pts), range(scan_pts))
        X = X.reshape((scan_pts ** 2))
        Y = Y.reshape((scan_pts ** 2))
        for idx in range(G):
            for idy in range(D):
                index = idy + 2 * idx
                update_addr[index::scan_pts ** 2, 1] = X
                update_addr[index::scan_pts ** 2, 2] = Y

        weights = np.empty(shape=(G,), dtype=FLOAT_TYPE)
        weights[:] = np.linspace(-1, 1, G)
        cfact = np.empty_like(array_to_be_updated)
        for idx in range(G):
            cfact[idx] = np.ones((H, I)) * 10 * (idx + 1)

        dummy_addr = np.zeros_like(extract_addr) #  these aren't used by the function, but are passed as a top level address book
        addr_info  = zip(extract_addr, update_addr , exit_addr, dummy_addr, dummy_addr)
        obj_smooth_std = 2 # integer

        garray_to_be_updated = deepcopy(array_to_be_updated)
        opi.difference_map_update_object(array_to_be_updated, weights, array_to_be_extracted, exit_wave, addr_info, cfact, ob_smooth_std=obj_smooth_std, clip_object=None)
        gopi.difference_map_update_object(garray_to_be_updated, weights, array_to_be_extracted, exit_wave, addr_info,
                                         cfact, ob_smooth_std=obj_smooth_std, clip_object=None)

        np.testing.assert_array_equal(array_to_be_updated, garray_to_be_updated)

    @unittest.skip("This method is not implemented yet")
    def test_difference_map_update_object_with_no_smooth_but_clipping_UNITY(self):
        B = 5
        C = 5

        D = 2
        E = B
        F = C

        npts_greater_than = 2
        G = 2
        H = B + npts_greater_than
        I = C + npts_greater_than

        scan_pts = 2
        A = scan_pts ** 2 * G * D  # this is a 16 point scan pattern (4x4 grid) over all the modes

        # shapes and types outlined here
        exit_wave = np.empty(shape=(A, B, C), dtype=COMPLEX_TYPE)
        for idx in range(A):
            exit_wave[idx] = np.ones((B, C)) * (idx + 1) + 1j * np.ones((B, C)) * (idx + 1)

        exit_addr = np.empty(shape=(A, 3), dtype=int)
        exit_addr[:, 0] = np.array(range(A))
        exit_addr[:, 1] = np.zeros((A,))
        exit_addr[:, 2] = np.zeros((A,))

        array_to_be_extracted = np.empty(shape=(D, E, F), dtype=COMPLEX_TYPE)
        for idx in range(D):
            array_to_be_extracted[idx] = np.ones((E, F)) * (idx + 1) + 1j * np.ones((E, F)) * (idx + 1)
        extract_addr = np.empty(shape=(A, 3), dtype=int)
        extract_addr[:, 0] = np.array(range(D)).repeat(A / D)
        extract_addr[:, 1] = np.zeros((A,))
        extract_addr[:, 2] = np.zeros((A,))

        array_to_be_updated = np.empty(shape=(G, H, I), dtype=COMPLEX_TYPE)
        for idx in range(G):
            array_to_be_updated[idx] = np.ones((H, I)) * (3 * idx + 1) + 1j * np.ones((H, I)) * (3 * idx + 1)
        update_addr = np.empty(shape=(A, 3), dtype=int)
        update_addr[:, 0] = np.array(range(G)).repeat(A / G)
        X, Y = np.meshgrid(range(scan_pts), range(scan_pts))
        X = X.reshape((scan_pts ** 2))
        Y = Y.reshape((scan_pts ** 2))
        for idx in range(G):
            for idy in range(D):
                index = idy + 2 * idx
                update_addr[index::scan_pts ** 2, 1] = X
                update_addr[index::scan_pts ** 2, 2] = Y

        weights = np.empty(shape=(G,), dtype=FLOAT_TYPE)
        weights[:] = np.linspace(-1, 1, G)
        cfact = np.empty_like(array_to_be_updated)
        for idx in range(G):
            cfact[idx] = np.ones((H, I)) * 10 * (idx + 1)

        dummy_addr = np.zeros_like(extract_addr) #  these aren't used by the function, but are passed as a top level address book
        addr_info  = zip(extract_addr, update_addr , exit_addr, dummy_addr, dummy_addr)
        clip = (0.8, 1.0)

        garray_to_be_updated = deepcopy(array_to_be_updated)
        opi.difference_map_update_object(array_to_be_updated, weights, array_to_be_extracted, exit_wave, addr_info, cfact, ob_smooth_std=None, clip_object=clip)
        gopi.difference_map_update_object(garray_to_be_updated, weights, array_to_be_extracted, exit_wave, addr_info,
                                         cfact, ob_smooth_std=None, clip_object=clip)

        np.testing.assert_array_equal(array_to_be_updated, garray_to_be_updated)

    
    def test_center_probe_no_change_UNITY(self):
        npts = 64
        probe = np.zeros((1, npts, npts), dtype=COMPLEX_TYPE)
        rad = 10.0
        probe_vals = 2 + 3j
        x = np.array(range(npts)) - npts // 2
        X, Y = np.meshgrid(x, x)
        Xoff = 5.0
        Yoff = 2.0
        probe[0, (X-Xoff)**2 + (Y-Yoff)**2 < rad**2] = probe_vals
        center_tolerance = 10.0

        gprobe = deepcopy(probe)
        opi.center_probe(probe, center_tolerance)
        gopi.center_probe(gprobe, center_tolerance)

        np.testing.assert_array_equal(probe, gprobe)

    def test_center_probe_with_change_UNITY(self):
        npts = 64
        probe = np.zeros((1, npts, npts), dtype=COMPLEX_TYPE)
        rad = 10.0
        probe_vals = 2 + 3j
        x = np.array(range(npts)) - npts // 2
        X, Y = np.meshgrid(x, x)
        Xoff = 5.0
        Yoff = 2.0
        probe[0, (X-Xoff)**2 + (Y-Yoff)**2 < rad**2] = probe_vals
        center_tolerance = 1.0
        gprobe = deepcopy(probe)

        gopi.center_probe(gprobe, center_tolerance)
        opi.center_probe(probe, center_tolerance)
        np.testing.assert_array_almost_equal(probe, gprobe, decimal=8) # interpolation obviously won't make this exact!


if __name__ == "__main__":
    unittest.main()
