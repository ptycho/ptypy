'''
tests for the object-probe interactions, including the specific DM, ePIE etc updates

'''

import unittest
import numpy as np
from test.archive_tests.array_based_tests import utils as tu
from copy import deepcopy
from ptypy.accelerate.array_based import COMPLEX_TYPE, FLOAT_TYPE
from ptypy.accelerate.array_based import data_utils as du
from ptypy.accelerate.array_based import object_probe_interaction as opi



class ObjectProbeInteractionRegressionTest(unittest.TestCase):
    def test_scan_and_multiply(self):
        PtychoInstance = tu.get_ptycho_instance('pod_to_numpy_test')
        # now convert to arrays
        vectorised_scan = du.pod_to_arrays(PtychoInstance, 'S0000')
        addr_info = vectorised_scan['meta']['addr']  # probably want to extract these at a later date, but just to get stuff going...
        probe = vectorised_scan['probe']
        obj = vectorised_scan['obj']
        exit_wave = vectorised_scan['exit wave']
        blank = np.ones_like(probe)
        po = opi.scan_and_multiply(blank, obj, exit_wave.shape, addr_info)

        for idx, p in enumerate(iter(PtychoInstance.pods.values())):
            np.testing.assert_array_equal(po[idx], p.object)

    def test_exit_wave_calculation(self):
        PtychoInstance = tu.get_ptycho_instance('pod_to_numpy_test')
        # now convert to arrays
        vectorised_scan = du.pod_to_arrays(PtychoInstance, 'S0000')
        addr_info = vectorised_scan['meta']['addr']  # probably want to extract these at a later date, but just to get stuff going...
        probe = vectorised_scan['probe']
        obj = vectorised_scan['obj']
        exit_wave = vectorised_scan['exit wave']

        po = opi.scan_and_multiply(probe, obj, exit_wave.shape, addr_info)
        for idx, p in enumerate(iter(PtychoInstance.pods.values())):
            np.testing.assert_array_equal(po[idx], p.object * p.probe)

    def test_difference_map_realspace_constraint(self):
        PtychoInstance = tu.get_ptycho_instance('pod_to_numpy_test')
        # now convert to arrays
        vectorised_scan = du.pod_to_arrays(PtychoInstance, 'S0000')
        addr_info = vectorised_scan['meta']['addr']  # probably want to extract these at a later date, but just to get stuff going...
        probe = vectorised_scan['probe']
        obj = vectorised_scan['obj']
        exit_wave = vectorised_scan['exit wave']
        probe_and_object = opi.scan_and_multiply(probe, obj, exit_wave.shape, addr_info)

        opi.difference_map_realspace_constraint(probe_and_object,
                                                exit_wave,
                                                alpha=1.0)

    def test_extract_array_from_exit_wave_regression_case_a(self):
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

        opi.extract_array_from_exit_wave(exit_wave, exit_addr, array_to_be_extracted, extract_addr, array_to_be_updated,
                                         update_addr, cfact, weights)

        expected = np.array([[-9.50000000 + 0.5j, 0.20000000 + 0.2j],
                    [-9.50000000 + 0.5j, 4.80952406 + 0.04761905j],
                    [-9.50000000 + 0.5j, 4.80952406 + 0.04761905j],
                    [-9.50000000 + 0.5j, 4.80952406 + 0.04761905j],
                    [-9.50000000 + 0.5j, 4.80952406 + 0.04761905j],
                    [0.10000000 + 0.1j,  4.80952406 + 0.04761905j],
                    [0.10000000 + 0.1j, 0.20000000 + 0.2j]], dtype=COMPLEX_TYPE)

        np.testing.assert_array_equal(np.diagonal(array_to_be_updated),
                                      expected,
                                      err_msg="The array has not been extracted properly from the exit wave.")

    def test_extract_array_from_exit_wave_regression_case_b(self):
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

        opi.extract_array_from_exit_wave(exit_wave, exit_addr, array_to_be_extracted, extract_addr,
                                         array_to_be_updated,
                                         update_addr, cfact, weights)

        expected = np.array([[11.83333397 - 0.16666667j, 4.80952406 + 0.04761905j],
                             [11.83333397 - 0.16666667j, 4.80952406 + 0.04761905j],
                             [11.83333397 - 0.16666667j, 4.80952406 + 0.04761905j],
                             [11.83333397 - 0.16666667j, 4.80952406 + 0.04761905j],
                             [11.83333397 - 0.16666667j, 4.80952406 + 0.04761905j]],
                            dtype=COMPLEX_TYPE)
        np.testing.assert_array_equal(expected, np.diagonal(array_to_be_updated))

    def test_difference_map_update_probe_regression_with_support(self):
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
        opi.difference_map_update_probe(array_to_be_extracted, weights, array_to_be_updated, exit_wave, addr_info, cfact, probe_support=probe_support)
        expected_output = np.array([[-500.00000000 + 500.j, 400.00000000 + 400.j],
                                    [-500.00000000 + 500.j, 571.42858887 + 95.23809814j],
                                    [-500.00000000 + 500.j, 571.42858887 + 95.23809814j],
                                    [-500.00000000 + 500.j, 571.42858887 + 95.23809814j],
                                    [-500.00000000 + 500.j, 571.42858887 + 95.23809814j],
                                    [100.00000000 + 100.j, 571.42858887 + 95.23809814j],
                                    [100.00000000 + 100.j, 400.00000000 + 400.j]], dtype=COMPLEX_TYPE)
        np.testing.assert_array_equal(np.diagonal(array_to_be_updated), expected_output)

    def test_difference_map_update_probe_regression_without_support(self):
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
        opi.difference_map_update_probe(array_to_be_extracted, weights, array_to_be_updated, exit_wave, addr_info, cfact, probe_support=None)

        expected_output = np.array([[-5.00000000+5.j, 4.00000000+4.j],
                                    [-5.00000000+5.j, 5.71428585+0.95238096j],
                                    [-5.00000000+5.j, 5.71428585+0.95238096j],
                                    [-5.00000000+5.j, 5.71428585+0.95238096j],
                                    [-5.00000000+5.j, 5.71428585+0.95238096j],
                                    [ 1.00000000+1.j, 5.71428585+0.95238096j],
                                    [ 1.00000000+1.j, 4.00000000+4.j]], dtype=COMPLEX_TYPE)

        np.testing.assert_array_equal(np.diagonal(array_to_be_updated), expected_output)


    def test_difference_map_update_object_with_no_smooth_or_clip_regression(self):
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

        opi.difference_map_update_object(array_to_be_updated, weights, array_to_be_extracted, exit_wave, addr_info, cfact, ob_smooth_std=None, clip_object=None)
        expected_output = np.array([[-5.00000000 + 5.j, 4.00000000 + 4.j],
                                    [-5.00000000 + 5.j, 5.71428585 + 0.95238096j],
                                    [-5.00000000 + 5.j, 5.71428585 + 0.95238096j],
                                    [-5.00000000 + 5.j, 5.71428585 + 0.95238096j],
                                    [-5.00000000 + 5.j, 5.71428585 + 0.95238096j],
                                    [1.00000000 + 1.j, 5.71428585 + 0.95238096j],
                                    [1.00000000 + 1.j, 4.00000000 + 4.j]], dtype=COMPLEX_TYPE)

        np.testing.assert_array_equal(np.diagonal(array_to_be_updated), expected_output)


    def test_difference_map_update_object_with_smooth_but_no_clip_regression(self):
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
        obj_smooth_std = 2 # integer
        opi.difference_map_update_object(array_to_be_updated, weights, array_to_be_extracted, exit_wave, addr_info, cfact, ob_smooth_std=obj_smooth_std, clip_object=None)
        expected_output = np.array([[-5.00000000+5.j, 4.00000000+4.j],
                                    [-5.00000000+5.j, 5.71428585+0.95238096j],
                                    [-5.00000000+5.j, 5.71428585+0.95238096j],
                                    [-5.00000000+5.j, 5.71428585+0.95238096j],
                                    [-5.00000000+5.j, 5.71428585+0.95238096j],
                                    [ 1.00000000+1.j, 5.71428585+0.95238096j],
                                    [ 1.00000000+1.j, 4.00000000+4.j]], dtype=COMPLEX_TYPE)

        np.testing.assert_array_equal(np.diagonal(array_to_be_updated), expected_output)

    @unittest.skip("Not used at the moment.")
    def test_difference_map_update_object_with_no_smooth_but_clipping_regression(self):
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
        opi.difference_map_update_object(array_to_be_updated, weights, array_to_be_extracted, exit_wave, addr_info, cfact, ob_smooth_std=None, clip_object=clip)
        expected_output = np.array([[-0.70710677+0.70710677j, 0.70710677+0.70710683j],
                                    [-0.70710677+0.70710677j, 0.98639393+0.16439897j],
                                    [-0.70710677+0.70710677j, 0.98639393+0.16439897j],
                                    [-0.70710677+0.70710677j, 0.98639393+0.16439897j],
                                    [-0.70710677+0.70710677j, 0.98639393+0.16439897j],
                                    [ 0.70710677+0.70710683j, 0.98639393+0.16439897j],
                                    [ 0.70710677+0.70710683j, 0.70710677+0.70710683j]], dtype=COMPLEX_TYPE)

        np.testing.assert_array_equal(np.diagonal(array_to_be_updated), expected_output)

    def test_center_probe_no_change_regression(self):
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
        original_probe = np.copy(probe)
        opi.center_probe(probe, center_tolerance)

        np.testing.assert_array_equal(probe, original_probe)

    def test_center_probe_with_change_regression(self):
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

        not_shifted_probe = np.zeros((1, npts, npts), dtype=COMPLEX_TYPE)
        not_shifted_probe[0, (X)**2 + (Y)**2 < rad**2] = probe_vals
        opi.center_probe(probe, center_tolerance)
        np.testing.assert_array_almost_equal(probe, not_shifted_probe, decimal=8) # interpolation obviously won't make this exact!

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

        original_probe = deepcopy(probe)
        expected_object=np.array([[[-5.00000000+5.j,-5.00000000+5.j,-5.00000000+5.j,
                                    -5.00000000+5.j,-5.00000000+5.j,1.00000000+1.j,1.00000000+1.j],
                                   [10.33333397-1.66666675j,10.33333397-1.66666675j,
                                    10.33333397-1.66666675j,10.33333397-1.66666675j,
                                    10.33333397-1.66666675j,1.00000000+1.j,1.00000000+1.j],
                                   [10.33333397-1.66666675j,10.33333397-1.66666675j,
                                    10.33333397-1.66666675j,10.33333397-1.66666675j,
                                    10.33333397-1.66666675j,1.00000000+1.j,1.00000000+1.j],
                                   [10.33333397-1.66666675j,10.33333397-1.66666675j,
                                    10.33333397-1.66666675j,10.33333397-1.66666675j,
                                    10.33333397-1.66666675j,1.00000000+1.j,1.00000000+1.j],
                                   [10.33333397-1.66666675j,10.33333397-1.66666675j,
                                    10.33333397-1.66666675j,10.33333397-1.66666675j,
                                    10.33333397-1.66666675j,1.00000000+1.j,1.00000000+1.j],
                                   [-21.00000000+5.j,-21.00000000+5.j,-21.00000000+5.j,
                                    -21.00000000+5.j,-21.00000000+5.j,1.00000000+1.j,
                                    1.00000000+1.j],
                                   [1.00000000+1.j,1.00000000+1.j,1.00000000+1.j,
                                    1.00000000+1.j,1.00000000+1.j,1.00000000+1.j,
                                    1.00000000+1.j]],

                                  [[4.00000000+4.j,4.76923084+1.53846157j,
                                    4.76923084+1.53846157j,4.76923084+1.53846157j,
                                    4.76923084+1.53846157j,4.76923084+1.53846157j,4.00000000+4.j],
                                   [4.00000000+4.j,5.71428585+0.95238096j,
                                    5.71428585+0.95238096j,5.71428585+0.95238096j,
                                    5.71428585+0.95238096j,5.71428585+0.95238096j,4.00000000+4.j],
                                   [4.00000000+4.j,5.71428585+0.95238096j,
                                    5.71428585+0.95238096j,5.71428585+0.95238096j,
                                    5.71428585+0.95238096j,5.71428585+0.95238096j,4.00000000+4.j],
                                   [4.00000000+4.j,5.71428585+0.95238096j,
                                    5.71428585+0.95238096j,5.71428585+0.95238096j,
                                    5.71428585+0.95238096j,5.71428585+0.95238096j,4.00000000+4.j],
                                   [4.00000000+4.j,5.71428585+0.95238096j,
                                    5.71428585+0.95238096j,5.71428585+0.95238096j,
                                    5.71428585+0.95238096j,5.71428585+0.95238096j,4.00000000+4.j],
                                   [4.00000000+4.j,6.00000000+1.53846157j,
                                    6.00000000+1.53846157j,6.00000000+1.53846157j,
                                    6.00000000+1.53846157j,6.00000000+1.53846157j,4.00000000+4.j],
                                   [4.00000000+4.j,4.00000000+4.j,4.00000000+4.j,
                                    4.00000000+4.j,4.00000000+4.j,4.00000000+4.j,
                                    4.00000000+4.j]]], dtype=COMPLEX_TYPE)

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

        np.testing.assert_array_equal(original_probe,
                                      probe,
                                      err_msg="The probe has been updated when it shouldn't have been.")
        np.testing.assert_array_equal(expected_object,
                                      obj,
                                      err_msg="The object has not been updated correctly.")


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

        original_obj = deepcopy(obj)

        expected_probe = np.array([[[ 6.09090948-0.45454547j, 6.09090948-0.45454547j, 6.09090948-0.45454547j,
                                      6.09090948-0.45454547j, 6.09090948-0.45454547j],
                                    [ 6.09090948-0.45454547j, 6.09090948-0.45454547j, 6.09090948-0.45454547j,
                                      6.09090948-0.45454547j, 6.09090948-0.45454547j],
                                    [ 6.09090948-0.45454547j, 6.09090948-0.45454547j, 6.09090948-0.45454547j,
                                      6.09090948-0.45454547j, 6.09090948-0.45454547j],
                                    [ 6.09090948-0.45454547j, 6.09090948-0.45454547j, 6.09090948-0.45454547j,
                                      6.09090948-0.45454547j, 6.09090948-0.45454547j],
                                    [ 6.09090948-0.45454547j, 6.09090948-0.45454547j, 6.09090948-0.45454547j,
                                      6.09090948-0.45454547j, 6.09090948-0.45454547j]],
                                   [[ 3.08270693+0.07518797j, 3.08270693+0.07518797j, 3.08270693+0.07518797j,
                                      3.08270693+0.07518797j, 3.08270693+0.07518797j],
                                    [ 3.08270693+0.07518797j, 3.08270693+0.07518797j, 3.08270693+0.07518797j,
                                      3.08270693+0.07518797j, 3.08270693+0.07518797j],
                                    [ 3.08270693+0.07518797j, 3.08270693+0.07518797j, 3.08270693+0.07518797j,
                                      3.08270693+0.07518797j, 3.08270693+0.07518797j],
                                    [ 3.08270693+0.07518797j, 3.08270693+0.07518797j, 3.08270693+0.07518797j,
                                      3.08270693+0.07518797j, 3.08270693+0.07518797j],
                                    [ 3.08270693+0.07518797j, 3.08270693+0.07518797j, 3.08270693+0.07518797j,
                                      3.08270693+0.07518797j, 3.08270693+0.07518797j]]], dtype=COMPLEX_TYPE)

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

        np.testing.assert_array_equal(original_obj,
                                      obj,
                                      err_msg="The object has been updated when it shouldn't have been.")
        np.testing.assert_array_equal(expected_probe,
                                      probe,
                                      err_msg="The probe has not been updated correctly.")

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

        original_obj = deepcopy(obj)
        original_probe = deepcopy(probe)


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

        np.testing.assert_array_equal(original_obj,
                                      obj,
                                      err_msg="The object has been updated when it shouldn't have been.")
        np.testing.assert_array_equal(original_probe,
                                      probe,
                                      err_msg="The probe has been updated when it shouldn't have been.")

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

        expected_probe = np.array([[[ 0.34795576+0.32689944j, 0.34795576+0.32689944j,  0.34795576+0.32689944j,
                                      0.34795576+0.32689944j, 0.34795576+0.32689944j],
                                    [ 0.35228869+0.48999104j,  0.35228869+0.48999104j,  0.35228869+0.48999104j,
                                      0.35228869+0.48999104j,  0.35228869+0.48999104j],
                                    [ 0.35228869+0.48999104j,  0.35228869+0.48999104j,  0.35228869+0.48999104j,
                                      0.35228869+0.48999104j,  0.35228869+0.48999104j],
                                    [ 0.35228869+0.48999104j,  0.35228869+0.48999104j,  0.35228869+0.48999104j,
                                      0.35228869+0.48999104j,  0.35228869+0.48999104j],
                                    [-0.14553808-0.24420798j, -0.14553808-0.24420798j, -0.14553808-0.24420798j,
                                     -0.14553808-0.24420798j, -0.14553808-0.24420798j]],

                                   [[ 2.74465489+1.76501989j,  2.74465489+1.76501989j,  2.74465489+1.76501989j,
                                      2.74465489+1.76501989j,  2.74465489+1.76501989j],
                                    [ 2.46576023+1.78177691j,  2.46576023+1.78177691j,  2.46576023+1.78177691j,
                                      2.46576023+1.78177691j,  2.46576023+1.78177691j],
                                    [ 2.46576023+1.78177691j,  2.46576023+1.78177691j,  2.46576023+1.78177691j,
                                      2.46576023+1.78177691j,  2.46576023+1.78177691j],
                                    [ 2.46576023+1.78177691j,  2.46576023+1.78177691j,  2.46576023+1.78177691j,
                                      2.46576023+1.78177691j,  2.46576023+1.78177691j],
                                    [ 2.47635674+1.60818517j,  2.47635674+1.60818517j,  2.47635674+1.60818517j,
                                      2.47635674+1.60818517j,  2.47635674+1.60818517j]]], dtype=COMPLEX_TYPE)

        expected_object = np.array([[[-5.00000000 + 5.j, -5.00000000 + 5.j, -5.00000000 + 5.j,
                                      -5.00000000 + 5.j, -5.00000000 + 5.j, 1.00000000 + 1.j, 1.00000000 + 1.j],
                                     [10.33333397 - 1.66666675j, 10.33333397 - 1.66666675j,
                                      10.33333397 - 1.66666675j, 10.33333397 - 1.66666675j,
                                      10.33333397 - 1.66666675j, 1.00000000 + 1.j, 1.00000000 + 1.j],
                                     [10.33333397 - 1.66666675j, 10.33333397 - 1.66666675j,
                                      10.33333397 - 1.66666675j, 10.33333397 - 1.66666675j,
                                      10.33333397 - 1.66666675j, 1.00000000 + 1.j, 1.00000000 + 1.j],
                                     [10.33333397 - 1.66666675j, 10.33333397 - 1.66666675j,
                                      10.33333397 - 1.66666675j, 10.33333397 - 1.66666675j,
                                      10.33333397 - 1.66666675j, 1.00000000 + 1.j, 1.00000000 + 1.j],
                                     [10.33333397 - 1.66666675j, 10.33333397 - 1.66666675j,
                                      10.33333397 - 1.66666675j, 10.33333397 - 1.66666675j,
                                      10.33333397 - 1.66666675j, 1.00000000 + 1.j, 1.00000000 + 1.j],
                                     [-21.00000000 + 5.j, -21.00000000 + 5.j, -21.00000000 + 5.j,
                                      -21.00000000 + 5.j, -21.00000000 + 5.j, 1.00000000 + 1.j,
                                      1.00000000 + 1.j],
                                     [1.00000000 + 1.j, 1.00000000 + 1.j, 1.00000000 + 1.j,
                                      1.00000000 + 1.j, 1.00000000 + 1.j, 1.00000000 + 1.j,
                                      1.00000000 + 1.j]],

                                    [[4.00000000 + 4.j, 4.76923084 + 1.53846157j,
                                      4.76923084 + 1.53846157j, 4.76923084 + 1.53846157j,
                                      4.76923084 + 1.53846157j, 4.76923084 + 1.53846157j, 4.00000000 + 4.j],
                                     [4.00000000 + 4.j, 5.71428585 + 0.95238096j,
                                      5.71428585 + 0.95238096j, 5.71428585 + 0.95238096j,
                                      5.71428585 + 0.95238096j, 5.71428585 + 0.95238096j, 4.00000000 + 4.j],
                                     [4.00000000 + 4.j, 5.71428585 + 0.95238096j,
                                      5.71428585 + 0.95238096j, 5.71428585 + 0.95238096j,
                                      5.71428585 + 0.95238096j, 5.71428585 + 0.95238096j, 4.00000000 + 4.j],
                                     [4.00000000 + 4.j, 5.71428585 + 0.95238096j,
                                      5.71428585 + 0.95238096j, 5.71428585 + 0.95238096j,
                                      5.71428585 + 0.95238096j, 5.71428585 + 0.95238096j, 4.00000000 + 4.j],
                                     [4.00000000 + 4.j, 5.71428585 + 0.95238096j,
                                      5.71428585 + 0.95238096j, 5.71428585 + 0.95238096j,
                                      5.71428585 + 0.95238096j, 5.71428585 + 0.95238096j, 4.00000000 + 4.j],
                                     [4.00000000 + 4.j, 6.00000000 + 1.53846157j,
                                      6.00000000 + 1.53846157j, 6.00000000 + 1.53846157j,
                                      6.00000000 + 1.53846157j, 6.00000000 + 1.53846157j, 4.00000000 + 4.j],
                                     [4.00000000 + 4.j, 4.00000000 + 4.j, 4.00000000 + 4.j,
                                      4.00000000 + 4.j, 4.00000000 + 4.j, 4.00000000 + 4.j,
                                      4.00000000 + 4.j]]], dtype=COMPLEX_TYPE)

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

        np.testing.assert_array_equal(expected_probe,
                                      probe,
                                      err_msg="The probe has been updated when it shouldn't have been.")
        np.testing.assert_array_equal(expected_object,
                                      obj,
                                      err_msg="The object has not been updated correctly.")




    def test_difference_map_overlap_update_break_when_in_tolerance(self):
        '''
        This tests if the loop breaks according to the convergence criterion.
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

        expected_probe = np.array([[[ 45.64985275-4.16102743j,  45.64985275-4.16102743j,
                                      45.64985275-4.16102743j,  45.64985275-4.16102743j,
                                      45.64985275-4.16102743j],
                                    [ -9.70029163+0.79768848j,  -9.70029163+0.79768848j,
                                      -9.70029163+0.79768848j,  -9.70029163+0.79768848j,
                                      -9.70029163+0.79768848j],
                                    [  4.76249838-0.31339467j,   4.76249838-0.31339467j,
                                       4.76249838-0.31339467j,   4.76249838-0.31339467j,
                                       4.76249838-0.31339467j],
                                    [  3.71407413-0.28579614j,   3.71407413-0.28579614j,
                                       3.71407413-0.28579614j,   3.71407413-0.28579614j,
                                       3.71407413-0.28579614j],
                                    [  2.35006571-0.19345209j,   2.35006571-0.19345209j,
                                       2.35006571-0.19345209j,   2.35006571-0.19345209j,
                                       2.35006571-0.19345209j]],

                                   [[  3.99001932+0.07404561j,   3.99001932+0.07404561j,
                                       3.99001932+0.07404561j,   3.99001932+0.07404561j,
                                       3.99001932+0.07404561j],
                                    [  3.36987257+0.06362584j,   3.36987257+0.06362584j,
                                       3.36987257+0.06362584j,   3.36987257+0.06362584j,
                                       3.36987257+0.06362584j],
                                    [  3.10962296+0.05934116j,   3.10962296+0.05934116j,
                                       3.10962296+0.05934116j,   3.10962296+0.05934116j,
                                       3.10962296+0.05934116j],
                                    [  2.90126610+0.05487255j,   2.90126610+0.05487255j,
                                       2.90126610+0.05487255j,   2.90126610+0.05487255j,
                                       2.90126610+0.05487255j],
                                    [  2.51116419+0.04660653j,   2.51116419+0.04660653j,
                                       2.51116419+0.04660653j,   2.51116419+0.04660653j,
                                       2.51116419+0.04660653j]]], dtype=COMPLEX_TYPE)

        expected_object=np.array([[[ 0.04918606+0.05905421j,  0.04918606+0.05905421j,  0.04918606+0.05905421j,
                                     0.04918606+0.05905421j,  0.04918606+0.05905421j,  1.00000000+1.j, 1.00000000+1.j],
                                   [ 0.11200862+0.13464974j,  0.11200862+0.13464974j,  0.11200862+0.13464974j,
                                     0.11200862+0.13464974j,  0.11200862+0.13464974j,  1.00000000+1.j,1.00000000+1.j],
                                   [-0.51355976-0.61082107j, -0.51355976-0.61082107j, -0.51355976-0.61082107j,
                                    -0.51355976-0.61082107j, -0.51355976-0.61082107j,  1.00000000+1.j, 1.00000000+1.j],
                                   [ 0.99391210+1.14208853j,  0.99391210+1.14208853j,  0.99391210+1.14208853j,
                                     0.99391210+1.14208853j,  0.99391210+1.14208853j,  1.00000000+1.j, 1.00000000+1.j],
                                   [ 1.22169828+1.43459976j,  1.22169828+1.43459976j,  1.22169828+1.43459976j,
                                     1.22169828+1.43459976j,  1.22169828+1.43459976j,  1.00000000+1.j, 1.00000000+1.j],
                                   [ 2.36522031+2.79235673j,  2.36522031+2.79235673j,  2.36522031+2.79235673j,
                                     2.36522031+2.79235673j,  2.36522031+2.79235673j,  1.00000000+1.j,1.00000000+1.j],
                                   [ 1.00000000+1.j,          1.00000000+1.j,          1.00000000+1.j,
                                     1.00000000+1.j,          1.00000000+1.j,          1.00000000+1.j, 1.00000000+1.j  ]],
                                  [[ 4.00000000+4.j,          2.80242014+2.70098448j,  2.80242014+2.70098448j,
                                     2.80242014+2.70098448j,  2.80242014+2.70098448j,  2.80242014+2.70098448j,
                                     4.00000000+4.j],
                                   [ 4.00000000+4.j,          3.59294462+3.46147537j,  3.59294462+3.46147537j,
                                     3.59294462+3.46147537j,  3.59294462+3.46147537j,  3.59294462+3.46147537j,
                                     4.00000000+4.j],
                                   [ 4.00000000+4.j,          3.99926472+3.8498373j,   3.99926472+3.8498373j,
                                     3.99926472+3.8498373j,   3.99926472+3.8498373j,   3.99926472+3.8498373j,
                                     4.00000000+4.j],
                                   [ 4.00000000+4.j,          4.21914482+4.06092405j,  4.21914482+4.06092405j,
                                     4.21914482+4.06092405j,  4.21914482+4.06092405j,  4.21914482+4.06092405j,
                                     4.00000000+4.j],
                                   [ 4.00000000+4.j,          4.60679293+4.43693876j,  4.60679293+4.43693876j,
                                     4.60679293+4.43693876j,  4.60679293+4.43693876j,  4.60679293+4.43693876j,
                                     4.00000000+4.j],
                                   [ 4.00000000+4.j,          5.59837151+5.39574909j,  5.59837151+5.39574909j,
                                     5.59837151+5.39574909j,  5.59837151+5.39574909j,  5.59837151+5.39574909j,
                                     4.00000000+4.j],
                                   [ 4.00000000+4.j,          4.00000000+4.j,          4.00000000+4.j,
                                     4.00000000+4.j,          4.00000000+4.j,          4.00000000+4.j,
                                     4.00000000+4.j]]] ,dtype=COMPLEX_TYPE)

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



        np.testing.assert_allclose(expected_probe,
                                      probe,
                                      err_msg="The probe has not been updated correctly")

        print(obj)
        np.testing.assert_allclose(expected_object,
                                      obj,
                                      err_msg="The object has not been updated correctly.")


    if __name__ == "__main__":
        unittest.main()
