'''
Test for the propagation in numpy
'''

import unittest
import numpy as np
import utils as tu
import scisoftpy as dnp
from copy import deepcopy as copy
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
        self.master_pod = self.PtychoInstance.diff.V[first_view_id].pod

    def test_generate_far_field_fft_filters(self):
        '''
        We might just be able to use the ones from the geo class and just copy them to the gpu. 
        This function essentially replicates this code, but for the gpu.
        '''
        geo = self.master_pod.geometry
        shape = self.serialized_scan['probe'].shape[-2:]
        prop.generate_far_field_fft_filters(geo.lam * geo.distance, shape, geo.resolution, geo.psize)

    def test_generate_far_field_fft_filters_UNITY(self):
        '''
        This test checks to see if the numpy version reproduces the same filters as the geo class.
        '''
        
        geo = self.master_pod.geometry
        shape = self.serialized_scan['probe'].shape[-2:]
        filters = prop.generate_far_field_fft_filters(geo.lam * geo.distance, shape, geo.resolution, geo.psize)
        dnp.plot.image(filters[0], name='prefilter')
        dnp.plot.image(geo.propagator.pre_fft, name='geo prefilter')
        self.assertTrue(np.allclose(filters[0], geo.propagator.pre_fft))
        self.assertTrue(np.allclose(filters[1], geo.propagator.post_fft))
        self.assertTrue(np.allclose(filters[2], geo.propagator.pre_fft))
        self.assertTrue(np.allclose(filters[3], geo.propagator.post_fft))
# 
#     def test_fourier_transform_farfield_nofilter(self):
#         prop.forward_transform(self.serialized_scan, mode="farfield")
# 
#     def test_fourier_transform_farfield_nofilter_UNITY(self):
#         geo = self.master_pod.geometry
#         prefft = geo.pre_fft
#         postfft = geo.post_fft
#         geo.pre_fft = np.ones_like(self.serialized_scan['exit wave'].shape[-2:])
#         geo.post_fft = np.ones_like(self.serialized_scan['exit wave'].shape[-2:])
#         result_array_npy = prop.forward_transform(self.serialized_scan, mode="farfield")
#         result_array_geo = self.forward_transform_with_geo(geo)
#         self.assertTrue(np.allclose(result_array_npy, result_array_geo))
#         geo.pre_fft = prefft
#         geo.post_fft = postfft
# 
#     def test_fourier_transform_farfield_with_prefilter(self):
#         geo = self.master_geo
#         shape = self.serialized_scan['probe'].shape[-2:]
#         filters = prop.generate_far_field_fft_filters(geo.lam * geo.distance, shape, geo.resolution, geo.psize)
#         prop.forward_transform(self.serialized_scan, mode="farfield", prefilter=filters[0])
# 
#     def test_fourier_transform_farfield_with_prefilter_UNITY(self):
#         geo = copy(self.master_geo)
#         shape = self.serialized_scan['probe'].shape[-2:]
#         filters = prop.generate_far_field_fft_filters(geo.lam * geo.distance, shape, geo.resolution, geo.psize)
#         result_array_npy = prop.forward_transform(self.serialized_scan, mode="farfield", prefilter=filters[0])
#         geo.post_fft = np.ones_like(self.serialized_scan['exit wave'].shape[-2:])
#         result_array_geo = self.forward_transform_with_geo(geo)
#         self.assertTrue(np.allclose(result_array_npy, result_array_geo))
# 
#     def test_fourier_transform_farfield_with_postfilter(self):
#         geo = self.master_geo
#         shape = self.serialized_scan['probe'].shape[-2:]
#         filters = prop.generate_far_field_fft_filters(geo.lam * geo.distance, shape, geo.resolution, geo.psize)
#         prop.forward_transform(self.serialized_scan, mode="farfield", postfilter=filters[1])
# 
#     def test_fourier_transform_farfield_with_postfilter_UNITY(self):
#         geo = copy(self.master_geo)
#         shape = self.serialized_scan['probe'].shape[-2:]
#         filters = prop.generate_far_field_fft_filters(geo.lam * geo.distance, shape, geo.resolution, geo.psize)
#         result_array_npy = prop.forward_transform(self.serialized_scan, mode="farfield", postfilter=filters[1])
#         geo.pre_fft = np.ones_like(self.serialized_scan['exit wave'].shape[-2:])
#         result_array_geo = self.forward_transform_with_geo(geo)
#         self.assertTrue(np.allclose(result_array_npy, result_array_geo))
# 
#     def test_fourier_transform_farfield_with_pre_and_post_filter(self):
#         geo = self.master_geo
#         shape = self.serialized_scan['probe'].shape[-2:]
#         filters = prop.generate_far_field_fft_filters(geo.lam * geo.distance, shape, geo.resolution, geo.psize)
#         prop.forward_transform(self.serialized_scan, mode="farfield", prefilter=filters[0], postfilter=filters[1])
# 
#     def test_fourier_transform_farfield_with_pre_and_post_filter_UNITY(self):
#         geo = copy(self.master_geo)
#         shape = self.serialized_scan['probe'].shape[-2:]
#         filters = prop.generate_far_field_fft_filters(geo.lam * geo.distance, shape, geo.resolution, geo.psize)
#         result_array_npy = prop.forward_transform(self.serialized_scan, mode="farfield", prefilter=filters[0], postfilter=filters[1])
#         result_array_geo = self.forward_transform_with_geo(geo)
#         self.assertTrue(np.allclose(result_array_npy, result_array_geo))
# 
#     def test_inverse_fourier_transform_farfield_nofilter(self):
#         prop.forward_transform(self.serialized_scan, mode="farfield")
# 
#     def test_inverse_fourier_transform_farfield_with_prefilter(self):
#         pass
# 
#     def test_inverse_fourier_transform_farfield_with_postfilter(self):
#         pass
# 
#     def test_inverse_fourier_transform_farfield_with_pre_and_post_filter(self):
#         pass
#     
#     def forward_transform_with_geo(self, propagator):
#         exit_wave = self.serialized_scan['exit wave']
#         result_array_geo = np.zeros_like(exit_wave)
#         meta = self.serialized_scan['meta'] # probably want to extract these at a later date, but just to get stuff going...
#         view_dlayer = 0 # what is this?
#         addr_info = meta['addr'][:,(view_dlayer)] # addresses, object references
#         for _pa, _oa, _ea,  da, _ma in addr_info:
#             result_array_geo[da[0]] = propagator.fw(exit_wave[da[0]])
#         return result_array_geo

if __name__ == "__main__":
    unittest.main()
