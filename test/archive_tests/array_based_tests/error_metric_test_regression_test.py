'''
A test for the module of the relevant error metrics
'''

import unittest
import numpy as np
from test.archive_tests.array_based_tests import utils as tu
from ptypy.accelerate.array_based import data_utils as du
from ptypy.accelerate.array_based import COMPLEX_TYPE, FLOAT_TYPE
import ptypy.utils as u
from collections import OrderedDict
from archive.array_based.error_metrics import log_likelihood, far_field_error, realspace_error
from ptypy.accelerate.array_based.object_probe_interaction import scan_and_multiply


class ErrorMetricRegressionTest(unittest.TestCase):

    def test_loglikelihood_regression(self):
        '''
        Test that it runs
        '''
        # should be able to completely remove this
        PtychoInstance = tu.get_ptycho_instance('log_likelihood_test')
        vectorised_scan = du.pod_to_arrays(PtychoInstance, 'S0000')
        first_view_id = vectorised_scan['meta']['view_IDs'][0]
        propagator = PtychoInstance.diff.V[first_view_id].pod.geometry.propagator
        addr_info = vectorised_scan['meta']['addr']  # probably want to extract these at a later date, but just to get stuff going...
        probe = vectorised_scan['probe']
        obj = vectorised_scan['obj']
        mask = vectorised_scan['mask']
        exit_wave = vectorised_scan['exit wave']
        diffraction = vectorised_scan['diffraction']

        probe_object = scan_and_multiply(probe, obj, exit_wave.shape, addr_info)
        log_likelihood(probe_object, mask, diffraction, propagator.pre_fft, propagator.post_fft, addr_info)


    def test_far_field_error_regression(self):
        PtychoInstance = tu.get_ptycho_instance('log_likelihood_test')
        af, fmag, mask = self.get_current_and_measured_solution(PtychoInstance)
        far_field_error(af, fmag, mask)


    def test_realspace_error_regression_a(self):
        # the case when there is only one mode
        I = 5
        M = 20
        N = 30
        out_length = I
        ea_first_column = range(I)
        da_first_column = range(I)

        difference = np.empty(shape=(I, M, N), dtype=COMPLEX_TYPE)
        for idx in range(I):
            difference[idx] = np.ones((M, N)) *idx + 1j * np.ones((M, N)) *idx

        error = realspace_error(difference, ea_first_column, da_first_column, out_length)

        expected_error = np.array([ 0.0, 2.0, 8.0, 18.0, 32.0], dtype=FLOAT_TYPE)
        np.testing.assert_array_equal(error, expected_error)

    def test_realspace_error_regression_b(self):
        # multiple modes
        I = 10
        M = 20
        N = 30
        out_length = 5
        ea_first_column = range(I)
        da_first_column = list(range(int(I/2))) + list(range(int(I/2)))

        difference = np.empty(shape=(I, M, N), dtype=COMPLEX_TYPE)
        for idx in range(I):
            difference[idx] = np.ones((M, N)) * idx + 1j * np.ones((M, N)) * idx

        error = realspace_error(difference, ea_first_column, da_first_column, out_length)

        expected_error = np.array([50., 74., 106., 146., 194.], dtype=FLOAT_TYPE)
        np.testing.assert_array_equal(error, expected_error)


    def get_current_and_measured_solution(self, a_ptycho_instance):
        alpha = 1.0
        fmag = []
        af = []
        mask = []
        for dname, diff_view in a_ptycho_instance.diff.views.items():
            fmag.append(np.sqrt(np.abs(diff_view.data)))
            af2 = np.zeros_like(diff_view.data)
            f = OrderedDict()
            for name, pod in diff_view.pods.items():
                if not pod.active:
                    continue
                f[name] = pod.fw((1 + alpha) * pod.probe * pod.object
                                 - alpha * pod.exit)
                af2 += u.abs2(f[name])
            mask.append(diff_view.pod.mask)
            af.append(np.sqrt(af2))
        return np.array(af), np.array(fmag), np.array(mask)



if __name__ == '__main__':
    unittest.main()
