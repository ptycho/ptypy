'''
A test for the module of the relevant error metrics
'''

import unittest
import numpy as np
import utils as tu
from ptypy.array_based import data_utils as du
from ptypy.array_based.constraints import difference_map_realspace_constraint, scan_and_multiply
from ptypy.array_based.propagation import farfield_propagator
import ptypy.array_based.array_utils  as au
from ptypy.array_based import FLOAT_TYPE
from ptypy.gpu.error_metrics import log_likelihood as glog_likelihood
from ptypy.gpu.error_metrics import far_field_error as gfar_field_error
from ptypy.array_based.error_metrics import log_likelihood, far_field_error


class ErrorMetricTest(unittest.TestCase):

    @unittest.skip("This method is not implemented yet")
    def test_loglikelihood_numpy_UNITY(self):
        PtychoInstance = tu.get_ptycho_instance('pod_to_numpy_test')
        vectorised_scan = du.pod_to_arrays(PtychoInstance, 'S0000')
        first_view_id = vectorised_scan['meta']['view_IDs'][0]
        propagator = PtychoInstance.di.V[first_view_id].pod.geometry.propagator
        addr = vectorised_scan['meta']['addr'] # probably want to extract these at a later date, but just to get stuff going...
        probe = vectorised_scan['probe']
        obj = vectorised_scan['obj']
        diffraction=vectorised_scan['diffraction']
        mask = vectorised_scan['mask']
        exit_wave = vectorised_scan['exit wave']
        
        view_dlayer = 0  # what is this?
        addr_info = addr[:, (view_dlayer)]  # addresses, object references
        probe_object = scan_and_multiply(probe, obj, exit_wave.shape, addr_info)
        
        ll = log_likelihood(probe_object, mask, exit_wave, diffraction, propagator.pre_fft, propagator.post_fft, addr)
        gll = glog_likelihood(probe_object, mask, exit_wave, diffraction, propagator.pre_fft, propagator.post_fft, addr)
        np.testing.assert_array_equal(ll, gll)



    @unittest.skip("This method is not implemented yet")
    def test_far_field_error_UNITY(self):
        PtychoInstance = tu.get_ptycho_instance('pod_to_numpy_test')
        vectorised_scan = du.pod_to_arrays(PtychoInstance, 'S0000')
        first_view_id = vectorised_scan['meta']['view_IDs'][0]
        propagator = PtychoInstance.di.V[first_view_id].pod.geometry.propagator
        addr = vectorised_scan['meta']['addr'] # probably want to extract these at a later date, but just to get stuff going...
        addr_info = addr[:, 0]
        probe = vectorised_scan['probe']
        obj = vectorised_scan['obj']
        diffraction=vectorised_scan['diffraction']
        mask = vectorised_scan['mask']
        exit_wave = vectorised_scan['exit wave']

        view_dlayer = 0  # what is this?
        addr_info = addr[:, (view_dlayer)]  # addresses, object references
        probe_object = scan_and_multiply(probe, obj, exit_wave.shape, addr_info)
        constrained = difference_map_realspace_constraint(probe_object, exit_wave, alpha=1.0)
        f = farfield_propagator(constrained, propagator.pre_fft, propagator.post_fft, direction='forward')
        pa, oa, ea, da, ma = zip(*addr_info)
        af2 = au.sum_to_buffer(au.abs2(f), diffraction.shape, ea, da, dtype=FLOAT_TYPE)

        fmag = np.sqrt(np.abs(diffraction))
        af = np.sqrt(af2)

        ff_error = far_field_error(af, fmag, mask)
        gff_error = gfar_field_error(af, fmag, mask)
        np.testing.assert_array_equal(ff_error, gff_error)


if __name__ == '__main__':
    unittest.main()
