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
from ptypy.gpu.error_metrics import realspace_error as grealspace_error
from ptypy.array_based.error_metrics import log_likelihood, far_field_error, realspace_error
from ptypy.array_based import COMPLEX_TYPE, FLOAT_TYPE

class ErrorMetricTest(unittest.TestCase):

    #@unittest.skip("This method is not implemented yet")
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
        
        ll = log_likelihood(probe_object, mask, diffraction, propagator.pre_fft, propagator.post_fft, addr)
        gll = glog_likelihood(probe_object, mask, exit_wave, diffraction, propagator.pre_fft, propagator.post_fft, addr)
        np.testing.assert_allclose(ll, gll, rtol=1e-6, atol=5e-4)



    #@unittest.skip("This method is not implemented yet")
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

        #print("af  : {}, {}".format(af.shape, af.dtype))
        #print("fmag: {}, {}".format(fmag.shape, fmag.dtype))
        #print("mask: {}, {}".format(mask.shape, mask.dtype))
        ff_error = far_field_error(af, fmag, mask).astype(np.float32)
        #print("out : {}, {}".format(ff_error.shape, ff_error.dtype))
        
        gff_error = gfar_field_error(af, fmag, mask)
        np.testing.assert_allclose(ff_error, gff_error, rtol=1e-6)
        # Detailed errors
        #max_relerr = 0.0
        #max_abserr = 0.0
        #for i in xrange(ff_error.shape[0]):
        #    relerr = (ff_error[i] - gff_error[i]) / ff_error[i]
        #    abserr = (ff_error[i] - gff_error[i])
        #    if (abs(relerr) > max_relerr):
        #        max_relerr = abs(relerr)
        #    if (abs(abserr) > max_abserr):
        #        max_abserr = abs(abserr)
        #    print("{} vs {} (rel={}, abs={})".format(ff_error[i], gff_error[i], relerr, abserr))
        #print("Max rel={}, abs={}".format(max_relerr, max_abserr))

    def test_realspace_error_regression_UNITY(self):
        I = 5
        M = 20
        N = 30

        difference = np.empty(shape=(I, M, N), dtype=COMPLEX_TYPE)
        for idx in range(I):
            difference[idx] = np.ones((M, N)) *idx + 1j * np.ones((M, N)) *idx

        #print("difference: {}, {}".format(difference.shape, difference.dtype))
        error = realspace_error(difference)
        #print("error: {}, {}".format(error.shape, error.dtype))
        gerror = grealspace_error(difference)
        
        np.testing.assert_allclose(error, gerror, rtol=1e-6)
        # detailed errors
        #max_relerr = 0.0
        #max_abserr = 0.0
        #for i in xrange(error.shape[0]):
        #    relerr = (error[i] - gerror[i]) / error[i]
        #    abserr = (error[i] - gerror[i])
        #    if (abs(relerr) > max_relerr):
        #        max_relerr = abs(relerr)
        #    if (abs(abserr) > max_abserr):
        #        max_abserr = abs(abserr)
        #    print("{} vs {} (rel={}, abs={})".format(error[i], gerror[i], relerr, abserr))
        #print("Max rel={}, abs={}".format(max_relerr, max_abserr))


if __name__ == '__main__':
    unittest.main()
        