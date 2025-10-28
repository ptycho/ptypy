'''
A test for the module of the relevant error metrics
'''

import unittest
import numpy as np
from . import utils as tu
from ptypy.accelerate.array_based import data_utils as du
from ptypy.accelerate.array_based.constraints import difference_map_realspace_constraint, scan_and_multiply
from archive.array_based.propagation import farfield_propagator
import ptypy.accelerate.array_based.array_utils  as au
from archive.array_based.error_metrics import log_likelihood, far_field_error, realspace_error
from ptypy.accelerate.array_based import COMPLEX_TYPE, FLOAT_TYPE

from . import have_cuda, only_if_cuda_available
if have_cuda():
    from archive.cuda_extension.accelerate.cuda.error_metrics import log_likelihood as glog_likelihood
    from archive.cuda_extension.accelerate.cuda.error_metrics import far_field_error as gfar_field_error
    from archive.cuda_extension.accelerate.cuda.error_metrics import realspace_error as grealspace_error
    from archive.cuda_extension.accelerate.cuda.config import init_gpus, reset_function_cache
    init_gpus(0)

@only_if_cuda_available
class ErrorMetricTest(unittest.TestCase):

    def tearDown(self):
        # reset the cached GPU functions after each test
        reset_function_cache()

    def test_loglikelihood_numpy_UNITY(self):
        PtychoInstance = tu.get_ptycho_instance('pod_to_numpy_test')
        vectorised_scan = du.pod_to_arrays(PtychoInstance, 'S0000')
        first_view_id = vectorised_scan['meta']['view_IDs'][0]
        propagator = PtychoInstance.di.V[first_view_id].pod.geometry.propagator
        addr_info = vectorised_scan['meta']['addr'] # probably want to extract these at a later date, but just to get stuff going...
        probe = vectorised_scan['probe']
        obj = vectorised_scan['obj']
        diffraction=vectorised_scan['diffraction']
        mask = vectorised_scan['mask']
        exit_wave = vectorised_scan['exit wave']

        probe_object = scan_and_multiply(probe, obj, exit_wave.shape, addr_info)
        
        ll = log_likelihood(probe_object, mask, diffraction, propagator.pre_fft, propagator.post_fft, addr_info)
        gll = glog_likelihood(probe_object, mask, diffraction, propagator.pre_fft, propagator.post_fft, addr_info)
        np.testing.assert_allclose(ll, gll, rtol=1e-6, atol=5e-4)

    def test_far_field_error_UNITY(self):
        PtychoInstance = tu.get_ptycho_instance('pod_to_numpy_test')
        vectorised_scan = du.pod_to_arrays(PtychoInstance, 'S0000')
        first_view_id = vectorised_scan['meta']['view_IDs'][0]
        propagator = PtychoInstance.di.V[first_view_id].pod.geometry.propagator
        addr_info = vectorised_scan['meta']['addr'] # probably want to extract these at a later date, but just to get stuff going...

        probe = vectorised_scan['probe']
        obj = vectorised_scan['obj']
        diffraction=vectorised_scan['diffraction']
        mask = vectorised_scan['mask']
        exit_wave = vectorised_scan['exit wave']

        probe_object = scan_and_multiply(probe, obj, exit_wave.shape, addr_info)
        constrained = difference_map_realspace_constraint(probe_object, exit_wave, alpha=1.0)
        f = farfield_propagator(constrained, propagator.pre_fft, propagator.post_fft, direction='forward')
        pa, oa, ea, da, ma = zip(*addr_info)
        af2 = au.sum_to_buffer(au.abs2(f), diffraction.shape, ea, da, dtype=FLOAT_TYPE)

        fmag = np.sqrt(np.abs(diffraction))
        af = np.sqrt(af2)

        ff_error = far_field_error(af, fmag, mask).astype(np.float32)
        
        gff_error = gfar_field_error(af, fmag, mask)
        np.testing.assert_allclose(ff_error, gff_error, rtol=1e-6)

    def test_realspace_error_regression1_UNITY(self):
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
        gerror = grealspace_error(difference, ea_first_column, da_first_column, out_length)
        
        np.testing.assert_allclose(error, gerror, rtol=1e-6)

    def test_realspace_error_regression2_UNITY(self):
        I = 5
        M = 20
        N = 30
        out_length = 5
        ea_first_column = range(I)
        da_first_column = list(range(int(I/2))) + list(range(int(I/2)))

        difference = np.empty(shape=(I, M, N), dtype=COMPLEX_TYPE)
        for idx in range(I):
            difference[idx] = np.ones((M, N)) * idx + 1j * np.ones((M, N)) * idx

        error = realspace_error(difference, ea_first_column, da_first_column, out_length)
        gerror = grealspace_error(difference, ea_first_column, da_first_column, out_length)
        
        np.testing.assert_allclose(error, gerror, rtol=1e-6)


if __name__ == '__main__':
    unittest.main()
        
