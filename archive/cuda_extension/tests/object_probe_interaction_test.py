'''
tests for the object-probe interactions, including the specific DM, ePIE etc updates

'''

import unittest
import numpy as np
from . import utils as tu
from ptypy.accelerate.array_based import data_utils as du
from ptypy.accelerate.array_based import object_probe_interaction as opi
from ptypy.accelerate.array_based import COMPLEX_TYPE, FLOAT_TYPE
from copy import deepcopy

from . import have_cuda, only_if_cuda_available
if have_cuda():
    from archive.cuda_extension.accelerate.cuda import object_probe_interaction as gopi
    from archive.cuda_extension.accelerate.cuda.config import init_gpus, reset_function_cache
    init_gpus(0)

@only_if_cuda_available
class ObjectProbeInteractionTest(unittest.TestCase):

    def tearDown(self):
        # reset the cached GPU functions after each test
        reset_function_cache()

    def test_scan_and_multiply_UNITY(self):
        PtychoInstance = tu.get_ptycho_instance('pod_to_numpy_test')
        # now convert to arrays
        vectorised_scan = du.pod_to_arrays(PtychoInstance, 'S0000')
        addr_info = vectorised_scan['meta']['addr'] # probably want to extract these at a later date, but just to get stuff going...
        probe = vectorised_scan['probe']
        obj = vectorised_scan['obj']
        exit_wave = vectorised_scan['exit wave']

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
        addr_info = vectorised_scan['meta']['addr'] # probably want to extract these at a later date, but just to get stuff going...
        probe = vectorised_scan['probe']
        obj = vectorised_scan['obj']
        exit_wave = vectorised_scan['exit wave']
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
        addr_info  = list(zip(update_addr, extract_addr, exit_addr, dummy_addr, dummy_addr))
        probe_support = np.ones_like(array_to_be_updated) * 100.0
        #(ob, probe_weights, probe, exit_wave, addr_info, cfact_probe, probe_support = None)

        garray_to_be_updated = deepcopy(array_to_be_updated)
        gcfact = deepcopy(cfact)
        err  = opi.difference_map_update_probe(array_to_be_extracted, weights, array_to_be_updated, exit_wave, addr_info, cfact, probe_support=probe_support)
                
        gerr = gopi.difference_map_update_probe(array_to_be_extracted, weights, garray_to_be_updated, exit_wave, addr_info,
                                        gcfact, probe_support=probe_support)

        self.assertAlmostEqual(err, gerr, 6)
        np.testing.assert_array_equal(array_to_be_updated, garray_to_be_updated)

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
        addr_info  = list(zip(update_addr, extract_addr, exit_addr, dummy_addr, dummy_addr))
        #(ob, probe_weights, probe, exit_wave, addr_info, cfact_probe, probe_support = None)

        garray_to_be_updated = deepcopy(array_to_be_updated)
        gcfact = deepcopy(cfact)
        opi.difference_map_update_probe(array_to_be_extracted, weights, array_to_be_updated, exit_wave, addr_info, cfact, probe_support=None)
        gopi.difference_map_update_probe(array_to_be_extracted, weights, garray_to_be_updated, exit_wave, addr_info,
                                        gcfact, probe_support=None)

        np.testing.assert_array_equal(array_to_be_updated, garray_to_be_updated)

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
        addr_info  = list(zip(extract_addr, update_addr , exit_addr, dummy_addr, dummy_addr))

        garray_to_be_updated = deepcopy(array_to_be_updated)
        gcfact = deepcopy(cfact)
        opi.difference_map_update_object(array_to_be_updated, weights, array_to_be_extracted, exit_wave, addr_info, cfact, ob_smooth_std=None, clip_object=None)
        gopi.difference_map_update_object(garray_to_be_updated, weights, array_to_be_extracted, exit_wave, addr_info,
                                         gcfact, ob_smooth_std=None, clip_object=None)

        np.testing.assert_array_equal(array_to_be_updated, garray_to_be_updated)

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
        addr_info  = list(zip(extract_addr, update_addr , exit_addr, dummy_addr, dummy_addr))
        obj_smooth_std = 2.0 # integer

        garray_to_be_updated = deepcopy(array_to_be_updated)
        gopi.difference_map_update_object(garray_to_be_updated, weights, array_to_be_extracted, exit_wave, addr_info,
                                         cfact, ob_smooth_std=obj_smooth_std, clip_object=None)
        opi.difference_map_update_object(array_to_be_updated, weights, array_to_be_extracted, exit_wave, addr_info, cfact, ob_smooth_std=obj_smooth_std, clip_object=None)

        print("Gpu={}".format(garray_to_be_updated))
        print("Cpu={}".format(array_to_be_updated))

        np.testing.assert_allclose(
            array_to_be_updated, 
            garray_to_be_updated,
            rtol=1e-6
            )

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
        addr_info  = list(zip(extract_addr, update_addr , exit_addr, dummy_addr, dummy_addr))
        clip = (0.8, 1.0)

        garray_to_be_updated = deepcopy(array_to_be_updated)
        gcfact = deepcopy(cfact)
        opi.difference_map_update_object(array_to_be_updated, weights, array_to_be_extracted, exit_wave, addr_info, cfact, ob_smooth_std=None, clip_object=clip)
        gopi.difference_map_update_object(garray_to_be_updated, weights, array_to_be_extracted, exit_wave, addr_info,
                                         gcfact, ob_smooth_std=None, clip_object=clip)

        np.testing.assert_allclose(array_to_be_updated, garray_to_be_updated)

    
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

    def test_difference_map_overlap_update_test_order_of_updates_a(self):
        '''
        This tests the order in which the object and probe are updated 
        '''
        smooth_std = None # anything else currently not supported
        max_iterations = 1
        update_object_first = True
        do_update_probe = False
        # this should mean that the object gets updated but the probe does not change
        ocf  = 1 # spam this for this test Not needed.

        # create some inputs - I should really make this a utility...
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

        probe = np.empty(shape=(D, E, F), dtype=COMPLEX_TYPE)
        for idx in range(D):
            probe[idx] = np.ones((E, F)) * (idx + 1) + 1j * np.ones((E, F)) * (idx + 1)
        probe_addr = np.empty(shape=(A, 3), dtype=int)
        probe_addr[:, 0] = np.array(range(D)).repeat(A / D)
        probe_addr[:, 1] = np.zeros((A,))
        probe_addr[:, 2] = np.zeros((A,))

        obj = np.empty(shape=(G, H, I), dtype=COMPLEX_TYPE)
        for idx in range(G):
            obj[idx] = np.ones((H, I)) * (3 * idx + 1) + 1j * np.ones((H, I)) * (3 * idx + 1)
        obj_addr = np.empty(shape=(A, 3), dtype=int)
        obj_addr[:, 0] = np.array(range(G)).repeat(A / G)
        X, Y = np.meshgrid(range(scan_pts), range(scan_pts))
        X = X.reshape((scan_pts ** 2))
        Y = Y.reshape((scan_pts ** 2))
        for idx in range(G):
            for idy in range(D):
                index = idy + 2 * idx
                obj_addr[index::scan_pts ** 2, 1] = X
                obj_addr[index::scan_pts ** 2, 2] = Y

        obj_weights = np.empty(shape=(G,), dtype=FLOAT_TYPE)
        obj_weights[:] = np.linspace(-1, 1, G)

        probe_weights = np.empty(shape=(D,), dtype=FLOAT_TYPE)
        probe_weights[:] = np.linspace(-1, 1, D)

        cfact_object = np.empty_like(obj)
        for idx in range(G):
            cfact_object[idx] = np.ones((H, I)) * 10 * (idx + 1)

        cfact_probe = np.empty_like(probe)
        for idx in range(G):
            cfact_probe[idx] = np.ones((E, F)) * 5 * (idx + 1)

        dummy_addr = np.zeros_like(probe_addr) #  these aren't used by the function, but are passed as a top level address book
        addr_info  = list(zip(probe_addr, obj_addr , exit_addr, dummy_addr, dummy_addr))

        gobj = deepcopy(obj)
        gprobe = deepcopy(probe)

        opi.difference_map_overlap_update(addr_info=addr_info,
                                          cfact_object=cfact_object,
                                          cfact_probe=cfact_probe,
                                          do_update_probe=do_update_probe,
                                          exit_wave=exit_wave,
                                          ob=obj,
                                          object_weights=obj_weights,
                                          probe=probe,
                                          probe_support=None,
                                          probe_weights=probe_weights,
                                          max_iterations=max_iterations,
                                          update_object_first=update_object_first,
                                          obj_smooth_std=smooth_std,
                                          overlap_converge_factor=ocf,
                                          probe_center_tol=None,
                                          clip_object=None)

        gopi.difference_map_overlap_update(addr_info=addr_info,
                                          cfact_object=cfact_object,
                                          cfact_probe=cfact_probe,
                                          do_update_probe=do_update_probe,
                                          exit_wave=exit_wave,
                                          ob=gobj,
                                          object_weights=obj_weights,
                                          probe=gprobe,
                                          probe_support=None,
                                          probe_weights=probe_weights,
                                          max_iterations=max_iterations,
                                          update_object_first=update_object_first,
                                          obj_smooth_std=smooth_std,
                                          overlap_converge_factor=ocf,
                                          probe_center_tol=None,
                                          clip_object=None)

        np.testing.assert_allclose(gprobe,
                                   probe,
                                   rtol=1e-6,
                                   err_msg="The cuda and numpy probes are different.")
        np.testing.assert_allclose(gobj,
                                      obj,
                                      rtol=1e-6,
                                      err_msg="The cuda and numpy object are different.")

    def test_difference_map_overlap_update_test_order_of_updates_b(self):
        '''
        This tests the order in which the object and probe are updated
        '''

        smooth_std = None # anything else currently not supported
        max_iterations = 1
        update_object_first = False
        do_update_probe = True
        # This should mean that the probe is updated, but not the object since max_iterations=1
        ocf  = 1 # spam this for this test Not needed.

        # create some inputs - I should really make this a utility...
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

        probe = np.empty(shape=(D, E, F), dtype=COMPLEX_TYPE)
        for idx in range(D):
            probe[idx] = np.ones((E, F)) * (idx + 1) + 1j * np.ones((E, F)) * (idx + 1)
        probe_addr = np.empty(shape=(A, 3), dtype=int)
        probe_addr[:, 0] = np.array(range(D)).repeat(A / D)
        probe_addr[:, 1] = np.zeros((A,))
        probe_addr[:, 2] = np.zeros((A,))

        obj = np.empty(shape=(G, H, I), dtype=COMPLEX_TYPE)
        for idx in range(G):
            obj[idx] = np.ones((H, I)) * (3 * idx + 1) + 1j * np.ones((H, I)) * (3 * idx + 1)
        obj_addr = np.empty(shape=(A, 3), dtype=int)
        obj_addr[:, 0] = np.array(range(G)).repeat(A / G)
        X, Y = np.meshgrid(range(scan_pts), range(scan_pts))
        X = X.reshape((scan_pts ** 2))
        Y = Y.reshape((scan_pts ** 2))
        for idx in range(G):
            for idy in range(D):
                index = idy + 2 * idx
                obj_addr[index::scan_pts ** 2, 1] = X
                obj_addr[index::scan_pts ** 2, 2] = Y

        obj_weights = np.empty(shape=(G,), dtype=FLOAT_TYPE)
        obj_weights[:] = np.linspace(-1, 1, G)

        probe_weights = np.empty(shape=(D,), dtype=FLOAT_TYPE)
        probe_weights[:] = np.linspace(-1, 1, D)

        cfact_object = np.empty_like(obj)
        for idx in range(G):
            cfact_object[idx] = np.ones((H, I)) * 10 * (idx + 1)

        cfact_probe = np.empty_like(probe)
        for idx in range(G):
            cfact_probe[idx] = np.ones((E, F)) * 5 * (idx + 1)

        dummy_addr = np.zeros_like(probe_addr) #  these aren't used by the function, but are passed as a top level address book
        addr_info  = list(zip(probe_addr, obj_addr , exit_addr, dummy_addr, dummy_addr))

        gobj = deepcopy(obj)
        gprobe = deepcopy(probe)

        opi.difference_map_overlap_update(addr_info=addr_info,
                                          cfact_object=cfact_object,
                                          cfact_probe=cfact_probe,
                                          do_update_probe=do_update_probe,
                                          exit_wave=exit_wave,
                                          ob=obj,
                                          object_weights=obj_weights,
                                          probe=probe,
                                          probe_support=None,
                                          probe_weights=probe_weights,
                                          max_iterations=max_iterations,
                                          update_object_first=update_object_first,
                                          obj_smooth_std=smooth_std,
                                          overlap_converge_factor=ocf,
                                          probe_center_tol=None,
                                          clip_object=None)

        gopi.difference_map_overlap_update(addr_info=addr_info,
                                          cfact_object=cfact_object,
                                          cfact_probe=cfact_probe,
                                          do_update_probe=do_update_probe,
                                          exit_wave=exit_wave,
                                          ob=gobj,
                                          object_weights=obj_weights,
                                          probe=gprobe,
                                          probe_support=None,
                                          probe_weights=probe_weights,
                                          max_iterations=max_iterations,
                                          update_object_first=update_object_first,
                                          obj_smooth_std=smooth_std,
                                          overlap_converge_factor=ocf,
                                          probe_center_tol=None,
                                          clip_object=None)

        np.testing.assert_allclose(gprobe,
                                      probe,
                                      rtol=1e-6,
                                      err_msg="The cuda and numpy probes are different.")
        np.testing.assert_allclose(gobj,
                                      obj,
                                      rtol=1e-6,
                                      err_msg="The cuda and numpy object are different.")

    def test_difference_map_overlap_update_test_order_of_updates_c(self):
        '''
        This tests the order in which the object and probe are updated
        '''

        smooth_std = None # anything else currently not supported
        max_iterations = 1
        update_object_first = False
        do_update_probe = False
        # neither the probe or the object are updated
        ocf  = 1 # spam this for this test Not needed.

        # create some inputs - I should really make this a utility...
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

        probe = np.empty(shape=(D, E, F), dtype=COMPLEX_TYPE)
        for idx in range(D):
            probe[idx] = np.ones((E, F)) * (idx + 1) + 1j * np.ones((E, F)) * (idx + 1)
        probe_addr = np.empty(shape=(A, 3), dtype=int)
        probe_addr[:, 0] = np.array(range(D)).repeat(A / D)
        probe_addr[:, 1] = np.zeros((A,))
        probe_addr[:, 2] = np.zeros((A,))

        obj = np.empty(shape=(G, H, I), dtype=COMPLEX_TYPE)
        for idx in range(G):
            obj[idx] = np.ones((H, I)) * (3 * idx + 1) + 1j * np.ones((H, I)) * (3 * idx + 1)
        obj_addr = np.empty(shape=(A, 3), dtype=int)
        obj_addr[:, 0] = np.array(range(G)).repeat(A / G)
        X, Y = np.meshgrid(range(scan_pts), range(scan_pts))
        X = X.reshape((scan_pts ** 2))
        Y = Y.reshape((scan_pts ** 2))
        for idx in range(G):
            for idy in range(D):
                index = idy + 2 * idx
                obj_addr[index::scan_pts ** 2, 1] = X
                obj_addr[index::scan_pts ** 2, 2] = Y

        obj_weights = np.empty(shape=(G,), dtype=FLOAT_TYPE)
        obj_weights[:] = np.linspace(-1, 1, G)

        probe_weights = np.empty(shape=(D,), dtype=FLOAT_TYPE)
        probe_weights[:] = np.linspace(-1, 1, D)

        cfact_object = np.empty_like(obj)
        for idx in range(G):
            cfact_object[idx] = np.ones((H, I)) * 10 * (idx + 1)

        cfact_probe = np.empty_like(probe)
        for idx in range(G):
            cfact_probe[idx] = np.ones((E, F)) * 5 * (idx + 1)

        dummy_addr = np.zeros_like(probe_addr) #  these aren't used by the function, but are passed as a top level address book
        addr_info  = list(zip(probe_addr, obj_addr , exit_addr, dummy_addr, dummy_addr))


        gobj = deepcopy(obj)
        gprobe = deepcopy(probe)

        opi.difference_map_overlap_update(addr_info=addr_info,
                                          cfact_object=cfact_object,
                                          cfact_probe=cfact_probe,
                                          do_update_probe=do_update_probe,
                                          exit_wave=exit_wave,
                                          ob=obj,
                                          object_weights=obj_weights,
                                          probe=probe,
                                          probe_support=None,
                                          probe_weights=probe_weights,
                                          max_iterations=max_iterations,
                                          update_object_first=update_object_first,
                                          obj_smooth_std=smooth_std,
                                          overlap_converge_factor=ocf,
                                          probe_center_tol=None,
                                          clip_object=None)

        gopi.difference_map_overlap_update(addr_info=addr_info,
                                          cfact_object=cfact_object,
                                          cfact_probe=cfact_probe,
                                          do_update_probe=do_update_probe,
                                          exit_wave=exit_wave,
                                          ob=gobj,
                                          object_weights=obj_weights,
                                          probe=gprobe,
                                          probe_support=None,
                                          probe_weights=probe_weights,
                                          max_iterations=max_iterations,
                                          update_object_first=update_object_first,
                                          obj_smooth_std=smooth_std,
                                          overlap_converge_factor=ocf,
                                          probe_center_tol=None,
                                          clip_object=None)

        np.testing.assert_allclose(gprobe,
                                      probe,
                                      rtol=1e-6,
                                      err_msg="The cuda and numpy probes are different.")
        np.testing.assert_allclose(gobj,
                                      obj,
                                      rtol=1e-6,
                                      err_msg="The cuda and numpy object are different.")

    def test_difference_map_overlap_update_test_order_of_updates_d(self):
        '''
        This tests the order in which the object and probe are updated
        '''

        smooth_std = None # anything else currently not supported
        max_iterations = 1
        update_object_first = True
        do_update_probe = True
        # both the object and the probe are updated
        ocf  = 1 # spam this for this test Not needed.

        # create some inputs - I should really make this a utility...
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

        probe = np.empty(shape=(D, E, F), dtype=COMPLEX_TYPE)
        for idx in range(D):
            probe[idx] = np.ones((E, F)) * (idx + 1) + 1j * np.ones((E, F)) * (idx + 1)
        probe_addr = np.empty(shape=(A, 3), dtype=int)
        probe_addr[:, 0] = np.array(range(D)).repeat(A / D)
        probe_addr[:, 1] = np.zeros((A,))
        probe_addr[:, 2] = np.zeros((A,))

        obj = np.empty(shape=(G, H, I), dtype=COMPLEX_TYPE)
        for idx in range(G):
            obj[idx] = np.ones((H, I)) * (3 * idx + 1) + 1j * np.ones((H, I)) * (3 * idx + 1)
        obj_addr = np.empty(shape=(A, 3), dtype=int)
        obj_addr[:, 0] = np.array(range(G)).repeat(A / G)
        X, Y = np.meshgrid(range(scan_pts), range(scan_pts))
        X = X.reshape((scan_pts ** 2))
        Y = Y.reshape((scan_pts ** 2))
        for idx in range(G):
            for idy in range(D):
                index = idy + 2 * idx
                obj_addr[index::scan_pts ** 2, 1] = X
                obj_addr[index::scan_pts ** 2, 2] = Y

        obj_weights = np.empty(shape=(G,), dtype=FLOAT_TYPE)
        obj_weights[:] = np.linspace(-1, 1, G)

        probe_weights = np.empty(shape=(D,), dtype=FLOAT_TYPE)
        probe_weights[:] = np.linspace(-1, 1, D)

        cfact_object = np.empty_like(obj)
        for idx in range(G):
            cfact_object[idx] = np.ones((H, I)) * 10 * (idx + 1)

        cfact_probe = np.empty_like(probe)
        for idx in range(G):
            cfact_probe[idx] = np.ones((E, F)) * 5 * (idx + 1)

        dummy_addr = np.zeros_like(
            probe_addr)  # these aren't used by the function, but are passed as a top level address book
        addr_info = list(zip(probe_addr, obj_addr, exit_addr, dummy_addr, dummy_addr))

        gobj = deepcopy(obj)
        gprobe = deepcopy(probe)

        opi.difference_map_overlap_update(addr_info=addr_info,
                                          cfact_object=cfact_object,
                                          cfact_probe=cfact_probe,
                                          do_update_probe=do_update_probe,
                                          exit_wave=exit_wave,
                                          ob=obj,
                                          object_weights=obj_weights,
                                          probe=probe,
                                          probe_support=None,
                                          probe_weights=probe_weights,
                                          max_iterations=max_iterations,
                                          update_object_first=update_object_first,
                                          obj_smooth_std=smooth_std,
                                          overlap_converge_factor=ocf,
                                          probe_center_tol=None,
                                          clip_object=None)

        gopi.difference_map_overlap_update(addr_info=addr_info,
                                          cfact_object=cfact_object,
                                          cfact_probe=cfact_probe,
                                          do_update_probe=do_update_probe,
                                          exit_wave=exit_wave,
                                          ob=gobj,
                                          object_weights=obj_weights,
                                          probe=gprobe,
                                          probe_support=None,
                                          probe_weights=probe_weights,
                                          max_iterations=max_iterations,
                                          update_object_first=update_object_first,
                                          obj_smooth_std=smooth_std,
                                          overlap_converge_factor=ocf,
                                          probe_center_tol=None,
                                          clip_object=None)

        np.testing.assert_allclose(gprobe,
                                      probe,
                                      rtol=1e-6,
                                      err_msg="The cuda and numpy probes are different.")
        np.testing.assert_allclose(gobj,
                                      obj,
                                      rtol=1e-6,
                                      err_msg="The cuda and numpy object are different.")




    def test_difference_map_overlap_update_break_when_in_tolerance(self):
        '''
        This tests if the loop breaks according to the convergence criterion.
        '''

        '''
        This tests the order in which the object and probe are updated
        '''

        smooth_std = None # anything else currently not supported
        max_iterations = 100
        update_object_first = False
        do_update_probe = True
        # both the object and the probe are updated
        ocf  = 4.2e-2 # chosen so that this should terminate on teh 6th iteration

        # create some inputs - I should really make this a utility...
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

        probe = np.empty(shape=(D, E, F), dtype=COMPLEX_TYPE)
        for idx in range(D):
            probe[idx] = np.ones((E, F)) * (idx + 1) + 1j * np.ones((E, F)) * (idx + 1)
        probe_addr = np.empty(shape=(A, 3), dtype=int)
        probe_addr[:, 0] = np.array(range(D)).repeat(A / D)
        probe_addr[:, 1] = np.zeros((A,))
        probe_addr[:, 2] = np.zeros((A,))

        obj = np.empty(shape=(G, H, I), dtype=COMPLEX_TYPE)
        for idx in range(G):
            obj[idx] = np.ones((H, I)) * (3 * idx + 1) + 1j * np.ones((H, I)) * (3 * idx + 1)
        obj_addr = np.empty(shape=(A, 3), dtype=int)
        obj_addr[:, 0] = np.array(range(G)).repeat(A / G)
        X, Y = np.meshgrid(range(scan_pts), range(scan_pts))
        X = X.reshape((scan_pts ** 2))
        Y = Y.reshape((scan_pts ** 2))
        for idx in range(G):
            for idy in range(D):
                index = idy + 2 * idx
                obj_addr[index::scan_pts ** 2, 1] = X
                obj_addr[index::scan_pts ** 2, 2] = Y

        obj_weights = np.empty(shape=(G,), dtype=FLOAT_TYPE)
        obj_weights[:] = np.linspace(-1, 1, G)

        probe_weights = np.empty(shape=(D,), dtype=FLOAT_TYPE)
        probe_weights[:] = np.linspace(-1, 1, D)

        cfact_object = np.empty_like(obj)
        for idx in range(G):
            cfact_object[idx] = np.ones((H, I)) * 10 * (idx + 1)

        cfact_probe = np.empty_like(probe)
        for idx in range(G):
            cfact_probe[idx] = np.ones((E, F)) * 5 * (idx + 1)

        dummy_addr = np.zeros_like(probe_addr)  # these aren't used by the function, but are passed as a top level address book
        addr_info = list(zip(probe_addr, obj_addr, exit_addr, dummy_addr, dummy_addr))

        expected_probe = np.array([[[ 9.09412193-0.70329767j,  9.09412193-0.70329767j,  9.09412193-0.70329767j,
                                      9.09412193-0.70329767j,  9.09412193-0.70329767j],
                                    [ 6.54680109-0.50316954j,  6.54680109-0.50316954j,  6.54680109-0.50316954j,
                                      6.54680109-0.50316954j,  6.54680109-0.50316954j],
                                    [ 6.14579964-0.47170654j,  6.14579964-0.47170654j,  6.14579964-0.47170654j,
                                      6.14579964-0.47170654j,  6.14579964-0.47170654j],
                                    [ 5.90408182-0.4541125j ,  5.90408182-0.4541125j,   5.90408182-0.4541125j,
                                      5.90408182-0.4541125j ,  5.90408182-0.4541125j ],
                                    [ 4.61261368-0.35782164j,  4.61261368-0.35782164j,  4.61261368-0.35782164j,
                                      4.61261368-0.35782164j,  4.61261368-0.35782164j]],

                                   [[ 3.49120140+0.0705148j,   3.49120140+0.0705148j,   3.49120140+0.0705148j,
                                      3.49120140+0.0705148j,   3.49120140+0.0705148j ],
                                    [ 3.14379764+0.06552192j,  3.14379764+0.06552192j,  3.14379764+0.06552192j,
                                      3.14379764+0.06552192j,  3.14379764+0.06552192j],
                                    [ 3.08963704+0.06493596j,  3.08963704+0.06493596j,  3.08963704+0.06493596j,
                                      3.08963704+0.06493596j,  3.08963704+0.06493596j],
                                    [ 3.04668784+0.06343807j,  3.04668784+0.06343807j,  3.04668784+0.06343807j,
                                      3.04668784+0.06343807j,  3.04668784+0.06343807j],
                                    [ 2.78638887+0.05607619j,  2.78638887+0.05607619j,  2.78638887+0.05607619j,
                                      2.78638887+0.05607619j,  2.78638887+0.05607619j]]], dtype=COMPLEX_TYPE)

        expected_object=np.array([[[0.27179495+0.31753245j,0.27179495+0.31753245j,0.27179495+0.31753245j,
                                    0.27179495+0.31753245j,0.27179495+0.31753245j,1.00000000+1.j,
                                    1.00000000+1.j],
                                   [0.58112150+0.67839354j,0.58112150+0.67839354j,0.58112150+0.67839354j,
                                    0.58112150+0.67839354j,0.58112150+0.67839354j,1.00000000+1.j,
                                    1.00000000+1.j],
                                   [0.66542435+0.77613211j,0.66542435+0.77613211j,0.66542435+0.77613211j,
                                    0.66542435+0.77613211j,0.66542435+0.77613211j,1.00000000+1.j,
                                    1.00000000+1.j],
                                   [0.68367511+0.7973848j,0.68367511+0.7973848j,0.68367511+0.7973848j,
                                    0.68367511+0.7973848j,0.68367511+0.7973848j,1.00000000+1.j,
                                    1.00000000+1.j],
                                   [0.77851987+0.90848058j,0.77851987+0.90848058j,0.77851987+0.90848058j,
                                    0.77851987+0.90848058j,0.77851987+0.90848058j,1.00000000+1.j,
                                    1.00000000+1.j],
                                   [1.20245206+1.40492487j,1.20245206+1.40492487j,1.20245206+1.40492487j,
                                    1.20245206+1.40492487j,1.20245206+1.40492487j,1.00000000+1.j,
                                    1.00000000+1.j],
                                   [1.00000000+1.j,1.00000000+1.j,1.00000000+1.j,
                                    1.00000000+1.j,1.00000000+1.j,1.00000000+1.j,
                                    1.00000000+1.j]],

                                  [[4.00000000+4.j,3.17660427+3.05242205j,3.17660427+3.05242205j,
                                    3.17660427+3.05242205j,3.17660427+3.05242205j,3.17660427+3.05242205j,
                                    4.00000000+4.j],
                                   [4.00000000+4.j,3.94026518+3.78214717j,3.94026518+3.78214717j,
                                    3.94026518+3.78214717j,3.94026518+3.78214717j,3.94026518+3.78214717j,
                                    4.00000000+4.j,],
                                   [4.00000000+4.j,4.11254692+3.94367075j,4.11254692+3.94367075j,
                                    4.11254692+3.94367075j,4.11254692+3.94367075j,4.11254692+3.94367075j,
                                    4.00000000+4.j],
                                   [4.00000000+4.j,4.14782619+3.9773953j,4.14782619+3.9773953j,
                                    4.14782619+3.9773953j,4.14782619+3.9773953j,4.14782619+3.9773953j,
                                    4.00000000+4.j,],
                                   [4.00000000+4.j,4.31528330+4.14124584j,4.31528330+4.14124584j,
                                    4.31528330+4.14124584j,4.31528330+4.14124584j,4.31528330+4.14124584j,
                                    4.00000000+4.j],
                                   [4.00000000+4.j,5.13210011+4.93122625j,5.13210011+4.93122625j,
                                    5.13210011+4.93122625j, 5.13210011+4.93122625j,5.13210011+4.93122625j,
                                    4.00000000+4.j,],
                                   [4.00000000+4.j,4.00000000+4.j,4.00000000+4.j,
                                    4.00000000+4.j,4.00000000+4.j,4.00000000+4.j,
                                    4.00000000+4.j]]],dtype=COMPLEX_TYPE)

        gobj = deepcopy(obj)
        gprobe = deepcopy(probe)

        opi.difference_map_overlap_update(addr_info=addr_info,
                                          cfact_object=cfact_object,
                                          cfact_probe=cfact_probe,
                                          do_update_probe=do_update_probe,
                                          exit_wave=exit_wave,
                                          ob=obj,
                                          object_weights=obj_weights,
                                          probe=probe,
                                          probe_support=None,
                                          probe_weights=probe_weights,
                                          max_iterations=max_iterations,
                                          update_object_first=update_object_first,
                                          obj_smooth_std=smooth_std,
                                          overlap_converge_factor=ocf,
                                          probe_center_tol=None,
                                          clip_object=None)

        gopi.difference_map_overlap_update(addr_info=addr_info,
                                          cfact_object=cfact_object,
                                          cfact_probe=cfact_probe,
                                          do_update_probe=do_update_probe,
                                          exit_wave=exit_wave,
                                          ob=gobj,
                                          object_weights=obj_weights,
                                          probe=gprobe,
                                          probe_support=None,
                                          probe_weights=probe_weights,
                                          max_iterations=max_iterations,
                                          update_object_first=update_object_first,
                                          obj_smooth_std=smooth_std,
                                          overlap_converge_factor=ocf,
                                          probe_center_tol=None,
                                          clip_object=None)

        np.testing.assert_allclose(gprobe,
                                      probe,
                                      rtol=5e-5,
                                      err_msg="The cuda and numpy probes are different.")
        np.testing.assert_allclose(gobj,
                                      obj,
                                      rtol=5e-5,
                                      err_msg="The cuda and numpy object are different.")


if __name__ == "__main__":
    unittest.main()
