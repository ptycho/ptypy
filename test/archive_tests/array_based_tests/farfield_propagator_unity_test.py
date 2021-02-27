'''
Test for the propagation in numpy
SHOULD REFACTOR HERE to be less dependent on the main framework. We just want to test the propagator works with 3x3 data.
'''

import unittest
import numpy as np
from test.archive_tests.array_based_tests import utils as tu
from ptypy.accelerate.array_based import data_utils as du
from ptypy.accelerate.array_based import object_probe_interaction as opi
from archive.array_based import propagation as prop
from copy import deepcopy as copy
TOLERANCE=4


class FarfieldPropagatorUnityTest(unittest.TestCase):
    def setUp(self):
        self.PtychoInstance = tu.get_ptycho_instance('pod_to_numpy_test')
        self.GeoPtychoInstance = tu.get_ptycho_instance('pod_to_numpy_test')
        self.vectorised_scan = du.pod_to_arrays(self.PtychoInstance, 'S0000')
        self.pod_vectorised_scan = du.pod_to_arrays(self.GeoPtychoInstance, 'S0000')
        self.first_view_id = self.pod_vectorised_scan['meta']['view_IDs'][0]

    def test_fourier_transform_farfield_nofilter_UNITY(self):
        vec_ew = self.get_exit_wave(self.vectorised_scan)
        pod_ew = self.get_exit_wave(self.pod_vectorised_scan)
        geo_propagator = copy(self.GeoPtychoInstance.di.V[self.first_view_id].pod.geometry.propagator)
        propagator = copy(self.PtychoInstance.di.V[self.first_view_id].pod.geometry.propagator)

        pre_fft = 1.0
        post_fft = 1.0

        geo_propagator.pre_fft = pre_fft
        geo_propagator.post_fft = post_fft

        result_array_npy = prop.farfield_propagator(vec_ew, prefilter=None, postfilter=None)
        result_array_geo = self.diffraction_transform_with_geo(geo_propagator, pod_ew)
        np.testing.assert_array_almost_equal(result_array_npy, result_array_geo, decimal=TOLERANCE)

    def test_fourier_transform_farfield_with_prefilter_UNITY(self):
        vec_ew = self.get_exit_wave(self.vectorised_scan)
        pod_ew = self.get_exit_wave(self.pod_vectorised_scan)
        geo_propagator = copy(self.GeoPtychoInstance.di.V[self.first_view_id].pod.geometry.propagator)
        propagator = copy(self.PtychoInstance.di.V[self.first_view_id].pod.geometry.propagator)

        post_fft = 1.0

        geo_propagator.post_fft = post_fft

        result_array_npy = prop.farfield_propagator(vec_ew, prefilter=propagator.pre_fft, postfilter=None)
        result_array_geo = self.diffraction_transform_with_geo(geo_propagator, pod_ew)
        np.testing.assert_array_almost_equal(result_array_npy, result_array_geo, decimal=TOLERANCE)

    def test_fourier_transform_farfield_with_postfilter_UNITY(self):
        vec_ew = self.get_exit_wave(self.vectorised_scan)
        pod_ew = self.get_exit_wave(self.pod_vectorised_scan)
        geo_propagator = copy(self.GeoPtychoInstance.di.V[self.first_view_id].pod.geometry.propagator)
        propagator = copy(self.PtychoInstance.di.V[self.first_view_id].pod.geometry.propagator)

        pre_fft = 1.0

        geo_propagator.pre_fft = pre_fft

        result_array_npy = prop.farfield_propagator(vec_ew, prefilter=None, postfilter=propagator.post_fft)
        result_array_geo = self.diffraction_transform_with_geo(geo_propagator, pod_ew)
        np.testing.assert_array_almost_equal(result_array_npy, result_array_geo, decimal=TOLERANCE)

    def test_fourier_transform_farfield_with_pre_and_post_filter_UNITY(self):
        vec_ew = self.get_exit_wave(self.vectorised_scan)
        pod_ew = self.get_exit_wave(self.pod_vectorised_scan)
        geo_propagator = copy(self.GeoPtychoInstance.di.V[self.first_view_id].pod.geometry.propagator)
        propagator = copy(self.PtychoInstance.di.V[self.first_view_id].pod.geometry.propagator)


        result_array_npy = prop.farfield_propagator(vec_ew, prefilter=propagator.pre_fft, postfilter=propagator.post_fft)
        result_array_geo = self.diffraction_transform_with_geo(geo_propagator, pod_ew)
        np.testing.assert_array_almost_equal(result_array_npy, result_array_geo)

    def test_inverse_fourier_transform_farfield_nofilter_UNITY(self):
        vec_ew = self.get_exit_wave(self.vectorised_scan)
        pod_ew = self.get_exit_wave(self.pod_vectorised_scan)
        geo_propagator = copy(self.GeoPtychoInstance.di.V[self.first_view_id].pod.geometry.propagator)
        propagator = copy(self.PtychoInstance.di.V[self.first_view_id].pod.geometry.propagator)

        pre_ifft = 1.0
        post_ifft = 1.0

        geo_propagator.pre_ifft = pre_ifft
        geo_propagator.post_ifft = post_ifft

        result_array_npy = prop.farfield_propagator(vec_ew, prefilter=None, postfilter=None, direction='backward')
        result_array_geo = self.diffraction_transform_with_geo(geo_propagator, pod_ew, direction='backward')
        np.testing.assert_array_almost_equal(result_array_npy, result_array_geo, decimal=TOLERANCE)

    def test_inverse_fourier_transform_farfield_with_prefilter_UNITY(self):
        vec_ew = self.get_exit_wave(self.vectorised_scan)
        pod_ew = self.get_exit_wave(self.pod_vectorised_scan)
        geo_propagator = copy(self.GeoPtychoInstance.di.V[self.first_view_id].pod.geometry.propagator)
        propagator = copy(self.PtychoInstance.di.V[self.first_view_id].pod.geometry.propagator)

        post_ifft = 1.0

        geo_propagator.post_ifft = post_ifft

        result_array_npy = prop.farfield_propagator(vec_ew, prefilter=propagator.pre_ifft, postfilter=None, direction='backward')
        result_array_geo = self.diffraction_transform_with_geo(geo_propagator, pod_ew, direction='backward')
        np.testing.assert_array_almost_equal(result_array_npy, result_array_geo, decimal=TOLERANCE)

    def test_inverse_fourier_transform_farfield_with_postfilter_UNITY(self):
        vec_ew = self.get_exit_wave(self.vectorised_scan)
        pod_ew = self.get_exit_wave(self.pod_vectorised_scan)
        geo_propagator = copy(self.GeoPtychoInstance.di.V[self.first_view_id].pod.geometry.propagator)
        propagator = copy(self.PtychoInstance.di.V[self.first_view_id].pod.geometry.propagator)

        pre_ifft = 1.0

        geo_propagator.pre_ifft = pre_ifft

        result_array_npy = prop.farfield_propagator(vec_ew, prefilter=None, postfilter=propagator.post_ifft, direction='backward')
        result_array_geo = self.diffraction_transform_with_geo(geo_propagator, pod_ew, direction='backward')
        np.testing.assert_array_almost_equal(result_array_npy, result_array_geo, decimal=TOLERANCE)

    def test_inverse_fourier_transform_farfield_with_pre_and_post_filter_UNITY(self):
        vec_ew = self.get_exit_wave(self.vectorised_scan)
        pod_ew = self.get_exit_wave(self.pod_vectorised_scan)
        geo_propagator = copy(self.GeoPtychoInstance.di.V[self.first_view_id].pod.geometry.propagator)
        propagator = copy(self.PtychoInstance.di.V[self.first_view_id].pod.geometry.propagator)


        result_array_npy = prop.farfield_propagator(vec_ew, prefilter=propagator.pre_ifft, postfilter=propagator.post_ifft, direction='backward')
        result_array_geo = self.diffraction_transform_with_geo(geo_propagator, pod_ew, direction='backward')
        np.testing.assert_array_almost_equal(result_array_npy, result_array_geo, decimal=TOLERANCE)


    def get_exit_wave(self, a_vectorised_scan):
        '''
        a pretested method
        :param a_vectorised_scan: A scan that has been vectorised. 
        :return: the exit wave
        '''
        vec_addr_info = a_vectorised_scan['meta']['addr']
        vec_probe = a_vectorised_scan['probe']
        vec_obj = a_vectorised_scan['obj']
        vec_ew = a_vectorised_scan['exit wave']
        return opi.scan_and_multiply(vec_probe,
                                     vec_obj,
                                     vec_ew.shape,
                                     vec_addr_info)


    def diffraction_transform_with_geo(self, propagator, ew, direction='forward'):
        result_array_geo = np.zeros_like(ew)
        meta = self.pod_vectorised_scan['meta'] #  probably want to extract these at a later date, but just to get stuff going...
        addr_info = meta['addr'] # addresses, object references
        for _pa, _oa, ea,  _da, _ma in addr_info:
            if direction=='forward':
                result_array_geo[ea[0]] = propagator.fw(ew[ea[0]])
            else:
                result_array_geo[ea[0]] = propagator.bw(ew[ea[0]])
        return result_array_geo


#

if __name__ == "__main__":
    unittest.main()
