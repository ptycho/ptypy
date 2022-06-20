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

class FarfieldPropagatorRegressionTest(unittest.TestCase):
    def setUp(self):
        self.PtychoInstance = tu.get_ptycho_instance('pod_to_numpy_test')
        self.GeoPtychoInstance = tu.get_ptycho_instance('pod_to_numpy_test')
        self.vectorised_scan = du.pod_to_arrays(self.PtychoInstance, 'S0000')
        self.pod_vectorised_scan = du.pod_to_arrays(self.GeoPtychoInstance, 'S0000')
        self.first_view_id = self.pod_vectorised_scan['meta']['view_IDs'][0]

    def test_fourier_transform_farfield_nofilter(self):
        vec_ew = self.get_exit_wave(self.vectorised_scan)
        prop.farfield_propagator(vec_ew)

    def test_fourier_transform_farfield_with_prefilter(self):
        vec_ew = self.get_exit_wave(self.vectorised_scan)
        propagator = copy(self.PtychoInstance.di.V[self.first_view_id].pod.geometry.propagator)
        prop.farfield_propagator(vec_ew, prefilter=propagator.pre_fft)


    def test_fourier_transform_farfield_with_postfilter(self):
        vec_ew = self.get_exit_wave(self.vectorised_scan)
        propagator = copy(self.PtychoInstance.di.V[self.first_view_id].pod.geometry.propagator)
        prop.farfield_propagator(vec_ew, prefilter=None, postfilter=propagator.post_fft)

    def test_fourier_transform_farfield_with_pre_and_post_filter(self):
        vec_ew = self.get_exit_wave(self.vectorised_scan)
        propagator = copy(self.PtychoInstance.di.V[self.first_view_id].pod.geometry.propagator)
        prop.farfield_propagator(vec_ew, prefilter=propagator.pre_fft, postfilter=propagator.post_fft)

    def test_inverse_fourier_transform_farfield_nofilter(self):
        vec_ew = self.get_exit_wave(self.vectorised_scan)
        prop.farfield_propagator(vec_ew, direction='backward')

    def test_inverse_fourier_transform_farfield_with_prefilter(self):
        vec_ew = self.get_exit_wave(self.vectorised_scan)
        propagator = copy(self.PtychoInstance.di.V[self.first_view_id].pod.geometry.propagator)
        prop.farfield_propagator(vec_ew, prefilter=propagator.pre_ifft, direction='backward')

    def test_inverse_fourier_transform_farfield_with_postfilter(self):
        vec_ew = self.get_exit_wave(self.vectorised_scan)
        propagator = copy(self.PtychoInstance.di.V[self.first_view_id].pod.geometry.propagator)
        prop.farfield_propagator(vec_ew, prefilter=None, postfilter=propagator.post_ifft, direction='backward')

    def test_inverse_fourier_transform_farfield_with_pre_and_post_filter(self):
        vec_ew = self.get_exit_wave(self.vectorised_scan)
        propagator = copy(self.PtychoInstance.di.V[self.first_view_id].pod.geometry.propagator)
        prop.farfield_propagator(vec_ew, prefilter=propagator.pre_ifft, postfilter=propagator.post_ifft, direction='backward')

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
        view_dlayer = 0 # what is this?
        addr_info = meta['addr'][:,view_dlayer] # addresses, object references
        for _pa, _oa, ea,  _da, _ma in addr_info:
            if direction=='forward':
                result_array_geo[ea[0]] = propagator.fw(ew[ea[0]])
            else:
                result_array_geo[ea[0]] = propagator.bw(ew[ea[0]])
        return result_array_geo


if __name__ == "__main__":
    unittest.main()
