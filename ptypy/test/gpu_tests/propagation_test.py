'''
Test for the propagation in numpy
'''

import unittest
import numpy as np
import utils as tu
from ptypy.gpu import data_utils as du
from ptypy.gpu import object_probe_interaction as opi
from ptypy.gpu import propagation as prop


class PropagationTest(unittest.TestCase):
    def setUp(self):
        self.PtychoInstance = tu.get_ptycho_instance('pod_to_numpy_test')
        # now convert to arrays
        self.serialized_scan = du.pod_to_arrays(self.PtychoInstance, 'S0000')
        self.serialized_scan = opi.difference_map_realspace_constraint(self.serialized_scan, alpha=1.0)
        first_view_id = self.serialized_scan['meta']['view_IDs'][0]
        self.exit_wave = self.serialized_scan['exit wave']
        self.master_pod = self.PtychoInstance.diff.V[first_view_id].pod

    def test_fourier_transform_farfield_nofilter(self):
        prop.farfield_propagator(self.exit_wave)
#  
#     def test_fourier_transform_farfield_nofilter_UNITY(self):
#         propagator = self.master_pod.geometry.propagator
#         propagator.pre_fft = np.ones(self.exit_wave['exit wave'].shape[-2:])
#         propagator.post_fft = np.ones(self.exit_wave['exit wave'].shape[-2:])
#         result_array_npy = prop.farfield_propagator(self.exit_wave, mode="farfield")
#         result_array_geo = self.diffraction_transform_with_geo(propagator)
#         self.assertTrue(np.allclose(result_array_npy, result_array_geo), atol=1e-5)

    def test_fourier_transform_farfield_with_prefilter(self):
        propagator = self.master_pod.geometry.propagator
        prop.farfield_propagator(self.exit_wave, prefilter=propagator.pre_fft)
   
#     def test_fourier_transform_farfield_with_prefilter_UNITY(self):
#         propagator = self.master_pod.geometry.propagator
#         result_array_npy = prop.farfield_propagator(self.exit_wave, mode="farfield", prefilter=propagator.pre_fft)
#         propagator.post_fft = np.ones(self.exit_wave['exit wave'].shape[-2:])
#         result_array_geo = self.diffraction_transform_with_geo(propagator)
#         dnp.plot.image(np.log10(np.abs(result_array_geo[0])), name='result geo')
#         dnp.plot.image(np.log10(np.abs(result_array_npy[0])), name='result np')
#         self.assertTrue(np.allclose(result_array_npy, result_array_geo))
# #   
    def test_fourier_transform_farfield_with_postfilter(self):
        propagator = self.master_pod.geometry.propagator
        prop.farfield_propagator(self.exit_wave, mode="farfield", postfilter=propagator.post_fft)
#   
#     def test_fourier_transform_farfield_with_postfilter_UNITY(self):
#         geo = copy(self.master_geo)
#         shape = self.exit_wave['probe'].shape[-2:]
#         filters = prop.generate_far_field_fft_filters(geo.lam * geo.distance, shape, geo.resolution, geo.psize)
#         result_array_npy = prop.farfield_propagator(self.exit_wave, mode="farfield", postfilter=filters[1])
#         geo.pre_fft = np.ones_like(self.exit_wave['exit wave'].shape[-2:])
#         result_array_geo = self.diffraction_transform_with_geo(geo)
#         self.assertTrue(np.allclose(result_array_npy, result_array_geo))
#   
    def test_fourier_transform_farfield_with_pre_and_post_filter(self):
        propagator = self.master_pod.geometry.propagator
        prop.farfield_propagator(self.exit_wave, mode="farfield",
                               prefilter=propagator.pre_fft,
                               postfilter=propagator.post_fft)
#   
#     def test_fourier_transform_farfield_with_pre_and_post_filter_UNITY(self):
#         geo = copy(self.master_geo)
#         shape = self.exit_wave['probe'].shape[-2:]
#         filters = prop.generate_far_field_fft_filters(geo.lam * geo.distance, shape, geo.resolution, geo.psize)
#         result_array_npy = prop.farfield_propagator(self.exit_wave, mode="farfield", prefilter=filters[0], postfilter=filters[1])
#         result_array_geo = self.diffraction_transform_with_geo(geo)
#         self.assertTrue(np.allclose(result_array_npy, result_array_geo))
#   
    def test_inverse_fourier_transform_farfield_nofilter(self):
        propagator = self.master_pod.geometry.propagator
        farfield_stack = prop.farfield_propagator(self.exit_wave,
                                                mode="farfield")
        prop.farfield_propagator(farfield_stack,
                               mode="farfield",
                               direction='backward')
#         
   
    def test_inverse_fourier_transform_farfield_with_prefilter(self):
        propagator = self.master_pod.geometry.propagator
        farfield_stack = prop.farfield_propagator(self.exit_wave,
                                                mode="farfield",
                                                prefilter=propagator.pre_fft)
        prop.farfield_propagator(farfield_stack,
                               mode="farfield",
                               prefilter=propagator.pre_ifft,
                               direction='backward')

    def test_inverse_fourier_transform_farfield_with_postfilter(self):
        propagator = self.master_pod.geometry.propagator
        farfield_stack = prop.farfield_propagator(self.exit_wave,
                                                mode="farfield",
                                                postfilter=propagator.post_fft)
        prop.farfield_propagator(farfield_stack,
                               mode="farfield",
                               postfilter=propagator.post_ifft,
                               direction='backward')
   
    def test_inverse_fourier_transform_farfield_with_pre_and_post_filter(self):
        propagator = self.master_pod.geometry.propagator
        farfield_stack = prop.farfield_propagator(self.exit_wave,
                                                mode="farfield",
                                                prefilter=propagator.pre_fft,
                                                postfilter=propagator.post_fft)

        result = prop.farfield_propagator(farfield_stack,
                                        mode="farfield", 
                                        prefilter=propagator.pre_ifft,
                                        postfilter=propagator.post_ifft,
                                        direction='backward')
       
    def test_inverse_fourier_transform_farfield_with_pre_and_post_filter_self_consistency(self):
        propagator = self.master_pod.geometry.propagator
        farfield_stack = prop.farfield_propagator(self.exit_wave,
                                                mode="farfield",
                                                prefilter=propagator.pre_fft,
                                                postfilter=propagator.post_fft)
        result = prop.farfield_propagator(farfield_stack,
                                        mode="farfield", 
                                        prefilter=propagator.pre_ifft,
                                        postfilter=propagator.post_ifft)
        self.assertTrue(np.allclose(result, self.exit_wave,atol=1e-5))

    def test_inverse_fourier_transform_farfield_self_consistency(self):
        propagator = self.master_pod.geometry.propagator
        farfield_stack = prop.farfield_propagator(self.exit_wave,
                                                mode="farfield")
        result = prop.farfield_propagator(farfield_stack,
                                        mode="farfield",
                                        direction='backward')

        self.assertTrue(np.allclose(result, self.exit_wave, atol=1e-5)) # this tolerance is equivalent to the current scipy implementation. It comes from the filters
        
    def diffraction_transform_with_geo(self, propagator):
        result_array_geo = np.zeros_like(self.exit_wave)
        meta = self.serialized_scan['meta'] # probably want to extract these at a later date, but just to get stuff going...
        view_dlayer = 0 # what is this?
        addr_info = meta['addr'][:,(view_dlayer)] # addresses, object references
        for _pa, _oa, ea,  _da, _ma in addr_info:
            result_array_geo[ea[0]] = propagator.bw(propagator.fw(self.exit_wave[ea[0]]))
        return result_array_geo
  
if __name__ == "__main__":
    unittest.main()
