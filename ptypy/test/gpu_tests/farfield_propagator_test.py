'''
Test for the propagation in numpy
'''

import unittest
import numpy as np
import utils as tu
from ptypy.gpu import data_utils as du
from ptypy.gpu import object_probe_interaction as opi
from ptypy.gpu import propagation as prop
from copy import deepcopy as copy

class FarfieldPropagatorTest(unittest.TestCase):
    def setUp(self):
        self.PtychoInstance = tu.get_ptycho_instance('pod_to_numpy_test')
        self.vectorised_scan = du.pod_to_arrays(self.PtychoInstance, 'S0000')
        self.addr = self.vectorised_scan['meta']['addr'] # probably want to extract these at a later date, but just to get stuff going...
        self.probe = self.vectorised_scan['probe']
        self.obj = self.vectorised_scan['obj']
        self.exit_wave = self.vectorised_scan['exit wave']
        first_view_id = self.vectorised_scan['meta']['view_IDs'][0]
        master_pod = self.PtychoInstance.diff.V[first_view_id].pod
        self.propagator = master_pod.geometry.propagator
        addr_info = self.addr[:, 0]
        self.exit_wave = opi.scan_and_multiply(self.probe, self.obj, self.exit_wave.shape, addr_info) # already tested

    def test_fourier_transform_farfield_nofilter(self):
        prop.farfield_propagator(self.exit_wave)

    def test_fourier_transform_farfield_nofilter_UNITY(self):
        pre_fft = 1.0
        post_fft = 1.0
        propagator = copy(self.propagator)
        propagator.pre_fft = pre_fft
        propagator.post_fft = post_fft
        result_array_npy = prop.farfield_propagator(self.exit_wave)
        result_array_geo = self.diffraction_transform_with_geo(propagator)
        np.testing.assert_allclose(result_array_npy, result_array_geo)

    def test_fourier_transform_farfield_with_prefilter(self):
        prop.farfield_propagator(self.exit_wave, prefilter=self.propagator.pre_fft)

    def test_fourier_transform_farfield_with_prefilter_UNITY(self):
        pre_fft = self.propagator.pre_fft
        post_fft = 1.0
        propagator = copy(self.propagator)
        propagator.pre_fft = pre_fft
        propagator.post_fft = post_fft
        result_array_npy = prop.farfield_propagator(self.exit_wave, prefilter=pre_fft)
        result_array_geo = self.diffraction_transform_with_geo(propagator)
        np.testing.assert_allclose(result_array_npy, result_array_geo)

    def test_fourier_transform_farfield_with_postfilter(self):
        prop.farfield_propagator(self.exit_wave,
                                 postfilter=self.propagator.post_fft)

    def test_fourier_transform_farfield_with_postfilter_UNITY(self):
        pre_fft = 1.0
        post_fft = self.propagator.post_fft
        propagator = copy(self.propagator)
        propagator.pre_fft = pre_fft
        propagator.post_fft = post_fft
        result_array_npy= prop.farfield_propagator(self.exit_wave,
                                                   postfilter=post_fft)
        result_array_geo = self.diffraction_transform_with_geo(propagator)

        np.testing.assert_allclose(result_array_npy, result_array_geo)

    def test_fourier_transform_farfield_with_pre_and_post_filter(self):
        prop.farfield_propagator(self.exit_wave,
                                prefilter=self.propagator.pre_fft,
                                postfilter=self.propagator.post_fft)

    def test_fourier_transform_farfield_with_pre_and_post_filter_UNITY(self):
        pre_fft = self.propagator.pre_fft
        post_fft = self.propagator.post_fft
        propagator = copy(self.propagator)
        propagator.pre_fft = pre_fft
        propagator.post_fft = post_fft
        result_array_npy = prop.farfield_propagator(self.exit_wave,
                                                    prefilter=pre_fft,
                                                    postfilter=post_fft)
        result_array_geo = self.diffraction_transform_with_geo(propagator)

        np.testing.assert_allclose(result_array_npy, result_array_geo)


    def test_inverse_fourier_transform_farfield_nofilter(self):
        prop.farfield_propagator(self.exit_wave, direction='backward')

    def test_inverse_fourier_transform_farfield_nofilter_UNITY(self):
        pre_ifft = 1.0
        post_ifft = 1.0
        propagator = copy(self.propagator)
        propagator.pre_ifft = pre_ifft
        propagator.post_ifft = post_ifft
        result_array_npy = prop.farfield_propagator(self.exit_wave,
                                                    direction='backward')
        result_array_geo = self.inverse_diffraction_transform_with_geo(propagator)

        np.testing.assert_allclose(result_array_npy, result_array_geo)

    def test_inverse_fourier_transform_farfield_with_prefilter(self):
        prop.farfield_propagator(self.exit_wave,
                                 prefilter=self.propagator.pre_ifft)

    def test_inverse_fourier_transform_farfield_with_prefilter_UNITY(self):
        pre_ifft = self.propagator.pre_ifft
        post_ifft = 1.0
        propagator = copy(self.propagator)
        propagator.pre_ifft = pre_ifft
        propagator.post_ifft = post_ifft
        result_array_npy = prop.farfield_propagator(self.exit_wave,
                                                    prefilter=pre_ifft,
                                                    direction='backward')
        result_array_geo = self.inverse_diffraction_transform_with_geo(propagator)

        np.testing.assert_allclose(result_array_npy, result_array_geo)

    def test_inverse_fourier_transform_farfield_with_postfilter(self):
        prop.farfield_propagator(self.exit_wave,
                                 postfilter=self.propagator.post_ifft,
                                 direction='backward')

    def test_inverse_fourier_transform_farfield_with_postfilter_UNITY(self):
        pre_ifft = 1.0
        post_ifft = self.propagator.post_ifft
        propagator = copy(self.propagator)
        propagator.pre_ifft = pre_ifft
        propagator.post_ifft = post_ifft
        result_array_npy = prop.farfield_propagator(self.exit_wave,
                                                    postfilter=post_ifft,
                                                    direction='backward')
        result_array_geo = self.inverse_diffraction_transform_with_geo(propagator)

        np.testing.assert_allclose(result_array_npy, result_array_geo)

    def test_inverse_fourier_transform_farfield_with_pre_and_post_filter(self):
        prop.farfield_propagator(self.exit_wave,
                               prefilter=self.propagator.pre_ifft,
                               postfilter=self.propagator.post_ifft,
                                direction='backward')

    def test_inverse_fourier_transform_farfield_with_pre_and_post_filter_UNITY(self):
        pre_ifft = self.propagator.pre_ifft
        post_ifft = self.propagator.post_ifft
        propagator = copy(self.propagator)
        propagator.pre_ifft = pre_ifft
        propagator.post_ifft = post_ifft
        result_array_npy = prop.farfield_propagator(self.exit_wave,
                                                    prefilter=pre_ifft,
                                                    postfilter=post_ifft,
                                                    direction='backward')
        result_array_geo = self.inverse_diffraction_transform_with_geo(propagator)

        np.testing.assert_allclose(result_array_npy, result_array_geo)


    def inverse_diffraction_transform_with_geo(self, propagator):
        result_array_geo = np.zeros_like(self.exit_wave)
        meta = self.vectorised_scan['meta'] #  probably want to extract these at a later date, but just to get stuff going...
        view_dlayer = 0 # what is this?
        addr_info = meta['addr'][:,view_dlayer] # addresses, object references
        for _pa, _oa, ea,  _da, _ma in addr_info:
            result_array_geo[ea[0]] = propagator.bw(self.exit_wave[ea[0]])
        return result_array_geo


    def diffraction_transform_with_geo(self, propagator):
        result_array_geo = np.zeros_like(self.exit_wave)
        meta = self.vectorised_scan['meta'] #  probably want to extract these at a later date, but just to get stuff going...
        view_dlayer = 0 # what is this?
        addr_info = meta['addr'][:,view_dlayer] # addresses, object references
        for _pa, _oa, ea,  _da, _ma in addr_info:
            result_array_geo[ea[0]] = propagator.fw(self.exit_wave[ea[0]])
        return result_array_geo
#
if __name__ == "__main__":
    unittest.main()
