'''
tests for the object-probe interactions, including the specific DM, ePIE etc updates

'''

import unittest
import numpy as np
import utils as tu
from ptypy.array_based import COMPLEX_TYPE, FLOAT_TYPE
from ptypy.array_based import data_utils as du
from ptypy.array_based import object_probe_interaction as opi
from collections import OrderedDict


class ObjectProbeInteractionTest(unittest.TestCase):
    def test_scan_and_multiply(self):
        PtychoInstance = tu.get_ptycho_instance('pod_to_numpy_test')
        # now convert to arrays
        vectorised_scan = du.pod_to_arrays(PtychoInstance, 'S0000')
        addr = vectorised_scan['meta'][
            'addr']  # probably want to extract these at a later date, but just to get stuff going...
        probe = vectorised_scan['probe']
        obj = vectorised_scan['obj']
        exit_wave = vectorised_scan['exit wave']
        blank = np.ones_like(probe)
        addr_info = addr[:, 0]

        po = opi.scan_and_multiply(blank, obj, exit_wave.shape, addr_info)

        for idx, p in enumerate(PtychoInstance.pods.itervalues()):
            np.testing.assert_array_equal(po[idx], p.object)

    def test_exit_wave_calculation(self):
        PtychoInstance = tu.get_ptycho_instance('pod_to_numpy_test')
        # now convert to arrays
        vectorised_scan = du.pod_to_arrays(PtychoInstance, 'S0000')
        addr = vectorised_scan['meta'][
            'addr']  # probably want to extract these at a later date, but just to get stuff going...
        probe = vectorised_scan['probe']
        obj = vectorised_scan['obj']
        exit_wave = vectorised_scan['exit wave']
        addr_info = addr[:, 0]

        po = opi.scan_and_multiply(probe, obj, exit_wave.shape, addr_info)
        for idx, p in enumerate(PtychoInstance.pods.itervalues()):
            np.testing.assert_array_equal(po[idx], p.object * p.probe)

    def test_difference_map_realspace_constraint(self):
        PtychoInstance = tu.get_ptycho_instance('pod_to_numpy_test')
        # now convert to arrays
        vectorised_scan = du.pod_to_arrays(PtychoInstance, 'S0000')
        addr = vectorised_scan['meta'][
            'addr']  # probably want to extract these at a later date, but just to get stuff going...
        probe = vectorised_scan['probe']
        obj = vectorised_scan['obj']
        exit_wave = vectorised_scan['exit wave']

        view_dlayer = 0  # what is this?
        addr_info = addr[:, (view_dlayer)]  # addresses, object references
        probe_and_object = opi.scan_and_multiply(probe, obj, exit_wave.shape, addr_info)

        opi.difference_map_realspace_constraint(probe_and_object,
                                                exit_wave,
                                                alpha=1.0)

    def test_difference_map_realspace_constraint_UNITY(self):
        PtychoInstance = tu.get_ptycho_instance('pod_to_numpy_test')
        a_ptycho_instance = tu.get_ptycho_instance('pod_to_numpy_test')
        # now convert to arrays
        vectorised_scan = du.pod_to_arrays(PtychoInstance, 'S0000')
        addr = vectorised_scan['meta'][
            'addr']  # probably want to extract these at a later date, but just to get stuff going...
        probe = vectorised_scan['probe']
        obj = vectorised_scan['obj']
        exit_wave = vectorised_scan['exit wave']

        view_dlayer = 0  # what is this?
        addr_info = addr[:, (view_dlayer)]  # addresses, object references
        probe_and_object = opi.scan_and_multiply(probe, obj, exit_wave.shape, addr_info)

        ptypy_dm_constraint = self.ptypy_apply_difference_map(a_ptycho_instance)
        numpy_dm_constraint = opi.difference_map_realspace_constraint(probe_and_object,
                                                                      exit_wave,
                                                                      alpha=1.0)
        for idx, key in enumerate(ptypy_dm_constraint):
            np.testing.assert_allclose(ptypy_dm_constraint[key], numpy_dm_constraint[idx])

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

        def test_difference_map_update_object(self):
            pass

        def test_difference_map_update_object_regression(self):
            pass

        def test_difference_map_update_probe(self):
            pass

        def test_difference_map_update_probe_regression(self):
            pass

        def test_center_probe(self):
            pass

        def test_center_probe_regression(self):
            pass

        def ptypy_apply_difference_map(self, a_ptycho_instance):
            f = OrderedDict()
            alpha = 1.0
            for dname, diff_view in a_ptycho_instance.diff.views.iteritems():
                for name, pod in diff_view.pods.iteritems():
                    if not pod.active:
                        continue
                    f[name] = (1 + alpha) * pod.probe * pod.object - alpha * pod.exit
            return f

    if __name__ == "__main__":
        unittest.main()
