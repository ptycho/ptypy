'''
Test for the propagation in numpy
'''

import unittest
import numpy as np
import utils as tu
from ptypy.array_based import data_utils as du
from ptypy.array_based import object_probe_interaction as opi
from ptypy.gpu import propagation as gprop
from ptypy.array_based import propagation as prop



class FarfieldPropagatorTest(unittest.TestCase):

    @unittest.skip("This method is not implemented yet")
    def test_fourier_transform_farfield_nofilter_UNITY(self):
        PtychoInstance = tu.get_ptycho_instance('pod_to_numpy_test')
        vectorised_scan = du.pod_to_arrays(PtychoInstance, 'S0000')
        exit_wave = opi.scan_and_multiply(vectorised_scan['probe'],
                                        vectorised_scan['obj'],
                                        vectorised_scan['exit wave'].shape,
                                        vectorised_scan['meta']['addr'][:, 0])

        array_propagated = prop.farfield_propagator(exit_wave, prefilter=None, postfilter=None)
        gpu_propagated = gprop.farfield_propagator(exit_wave, prefilter=None, postfilter=None)

        np.testing.assert_array_equal(array_propagated,
                                      gpu_propagated)




    @unittest.skip("This method is not implemented yet")
    def test_fourier_transform_farfield_with_prefilter_UNITY(self):
        PtychoInstance = tu.get_ptycho_instance('pod_to_numpy_test')
        vectorised_scan = du.pod_to_arrays(PtychoInstance, 'S0000')
        first_view_id = vectorised_scan['meta']['view_IDs'][0]
        propagator = PtychoInstance.di.V[first_view_id].pod.geometry.propagator
        exit_wave = opi.scan_and_multiply(vectorised_scan['probe'],
                                        vectorised_scan['obj'],
                                        vectorised_scan['exit wave'].shape,
                                        vectorised_scan['meta']['addr'][:, 0])

        array_propagated = prop.farfield_propagator(exit_wave, prefilter=propagator.pre_fft, postfilter=None)
        gpu_propagated = gprop.farfield_propagator(exit_wave, prefilter=propagator.pre_fft, postfilter=None)

        np.testing.assert_array_equal(array_propagated,
                                      gpu_propagated)

    @unittest.skip("This method is not implemented yet")
    def test_fourier_transform_farfield_with_postfilter_UNITY(self):
        PtychoInstance = tu.get_ptycho_instance('pod_to_numpy_test')
        vectorised_scan = du.pod_to_arrays(PtychoInstance, 'S0000')
        first_view_id = vectorised_scan['meta']['view_IDs'][0]
        propagator = PtychoInstance.di.V[first_view_id].pod.geometry.propagator
        exit_wave = opi.scan_and_multiply(vectorised_scan['probe'],
                                        vectorised_scan['obj'],
                                        vectorised_scan['exit wave'].shape,
                                        vectorised_scan['meta']['addr'][:, 0])

        array_propagated = prop.farfield_propagator(exit_wave, prefilter=None, postfilter=propagator.post_fft)
        gpu_propagated = gprop.farfield_propagator(exit_wave, prefilter=None, postfilter=propagator.post_fft)

        np.testing.assert_array_equal(array_propagated,
                                      gpu_propagated)

    @unittest.skip("This method is not implemented yet")
    def test_fourier_transform_farfield_with_pre_and_post_filter_UNITY(self):
        PtychoInstance = tu.get_ptycho_instance('pod_to_numpy_test')
        vectorised_scan = du.pod_to_arrays(PtychoInstance, 'S0000')
        first_view_id = vectorised_scan['meta']['view_IDs'][0]
        propagator = PtychoInstance.di.V[first_view_id].pod.geometry.propagator
        exit_wave = opi.scan_and_multiply(vectorised_scan['probe'],
                                        vectorised_scan['obj'],
                                        vectorised_scan['exit wave'].shape,
                                        vectorised_scan['meta']['addr'][:, 0])

        array_propagated = prop.farfield_propagator(exit_wave, prefilter=propagator.pre_fft, postfilter=propagator.post_fft)
        gpu_propagated = gprop.farfield_propagator(exit_wave, prefilter=propagator.pre_fft, postfilter=propagator.post_fft)

        np.testing.assert_array_equal(array_propagated,
                                      gpu_propagated)

    @unittest.skip("This method is not implemented yet")
    def test_inverse_fourier_transform_farfield_nofilter_UNITY(self):
        PtychoInstance = tu.get_ptycho_instance('pod_to_numpy_test')
        vectorised_scan = du.pod_to_arrays(PtychoInstance, 'S0000',)
        exit_wave = opi.scan_and_multiply(vectorised_scan['probe'],
                                        vectorised_scan['obj'],
                                        vectorised_scan['exit wave'].shape,
                                        vectorised_scan['meta']['addr'][:, 0])

        array_propagated = prop.farfield_propagator(exit_wave, prefilter=None, postfilter=None, direction='backward')
        gpu_propagated = gprop.farfield_propagator(exit_wave, prefilter=None, postfilter=None, direction='backward')

        np.testing.assert_array_equal(array_propagated,
                                      gpu_propagated)

    @unittest.skip("This method is not implemented yet")
    def test_inverse_fourier_transform_farfield_with_prefilter_UNITY(self):
        PtychoInstance = tu.get_ptycho_instance('pod_to_numpy_test')
        vectorised_scan = du.pod_to_arrays(PtychoInstance, 'S0000')
        first_view_id = vectorised_scan['meta']['view_IDs'][0]
        propagator = PtychoInstance.di.V[first_view_id].pod.geometry.propagator
        exit_wave = opi.scan_and_multiply(vectorised_scan['probe'],
                                        vectorised_scan['obj'],
                                        vectorised_scan['exit wave'].shape,
                                        vectorised_scan['meta']['addr'][:, 0])

        array_propagated = prop.farfield_propagator(exit_wave, prefilter=propagator.pre_fft, postfilter=None, direction='backward')
        gpu_propagated = gprop.farfield_propagator(exit_wave, prefilter=propagator.pre_fft, postfilter=None, direction='backward')

        np.testing.assert_array_equal(array_propagated,
                                      gpu_propagated)

    @unittest.skip("This method is not implemented yet")
    def test_inverse_fourier_transform_farfield_with_postfilter_UNITY(self):
        PtychoInstance = tu.get_ptycho_instance('pod_to_numpy_test')
        vectorised_scan = du.pod_to_arrays(PtychoInstance, 'S0000')
        first_view_id = vectorised_scan['meta']['view_IDs'][0]
        propagator = PtychoInstance.di.V[first_view_id].pod.geometry.propagator
        exit_wave = opi.scan_and_multiply(vectorised_scan['probe'],
                                        vectorised_scan['obj'],
                                        vectorised_scan['exit wave'].shape,
                                        vectorised_scan['meta']['addr'][:, 0])

        array_propagated = prop.farfield_propagator(exit_wave, prefilter=None, postfilter=propagator.post_fft, direction='backward')
        gpu_propagated = gprop.farfield_propagator(exit_wave, prefilter=None, postfilter=propagator.post_fft, direction='backward')

        np.testing.assert_array_equal(array_propagated,
                                      gpu_propagated)

    @unittest.skip("This method is not implemented yet")
    def test_inverse_fourier_transform_farfield_with_pre_and_post_filter_UNITY(self):
        PtychoInstance = tu.get_ptycho_instance('pod_to_numpy_test')
        vectorised_scan = du.pod_to_arrays(PtychoInstance, 'S0000')
        first_view_id = vectorised_scan['meta']['view_IDs'][0]
        propagator = PtychoInstance.di.V[first_view_id].pod.geometry.propagator
        exit_wave = opi.scan_and_multiply(vectorised_scan['probe'],
                                        vectorised_scan['obj'],
                                        vectorised_scan['exit wave'].shape,
                                        vectorised_scan['meta']['addr'][:, 0])

        array_propagated = prop.farfield_propagator(exit_wave, prefilter=propagator.pre_fft, postfilter=propagator.post_fft, direction='backward')
        gpu_propagated = gprop.farfield_propagator(exit_wave, prefilter=propagator.pre_fft, postfilter=propagator.post_fft, direction='backward')

        np.testing.assert_array_equal(array_propagated,
                                      gpu_propagated)



#

if __name__ == "__main__":
    unittest.main()
