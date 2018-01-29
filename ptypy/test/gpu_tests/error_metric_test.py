'''
A test for the module of the relevant error metrics
'''

import unittest
import numpy as np
import utils as tu
from ptypy.gpu import data_utils as du
import ptypy.utils as u
from collections import OrderedDict
from ptypy.gpu.error_metrics import log_likelihood
from ptypy.gpu import object_probe_interaction as opi

class ErrorMetricTest(unittest.TestCase):
    def setUp(self):
        self.PtychoInstance = tu.get_ptycho_instance('pod_to_numpy_test')
        # now convert to arrays
        self.serialized_scan = du.pod_to_arrays(self.PtychoInstance, 'S0000')
        self.addr = self.serialized_scan['meta']['addr'] # probably want to extract these at a later date, but just to get stuff going...
        self.probe = self.serialized_scan['probe']
        self.obj = self.serialized_scan['obj']
        self.mask = self.serialized_scan['mask']
        self.exit_wave = self.serialized_scan['exit wave']
        self.diffraction = self.serialized_scan['diffraction']
        view_names = self.PtychoInstance.diff.views.keys()
        self.error_metric = OrderedDict.fromkeys(view_names)
        first_view_id = self.serialized_scan['meta']['view_IDs'][0]
        master_pod = self.PtychoInstance.diff.V[first_view_id].pod
        self.propagator = master_pod.geometry.propagator

    def test_loglikelihood_numpy(self):
        '''
        Test that it runs
        '''
        log_likelihood(self.probe, self.obj, self.mask, self.exit_wave, self.diffraction, self.propagator.pre_fft, self.propagator.post_fft, self.addr)


    def test_loglikelihood_numpy_UNITY(self):
        '''
        Check that it gives the same result as the ptypy original

        '''
        ptypy_error_metric =self.get_ptypy_loglikelihood()
        vals =log_likelihood(self.probe, self.obj, self.mask, self.exit_wave, self.diffraction, self.propagator.pre_fft, self.propagator.post_fft, self.addr)
        for idx, key in enumerate(self.error_metric.keys()):
            self.error_metric[key] = vals[idx]


        for key in self.error_metric.keys():
            ptypy_error = ptypy_error_metric[key]
            numpy_error = self.error_metric[key]
            np.testing.assert_allclose(ptypy_error, numpy_error, rtol=1e-3)


    def get_ptypy_loglikelihood(self):
        error_dct = {}
        for dname, diff_view in self.PtychoInstance.diff.views.iteritems():
            I = diff_view.data
            fmask = diff_view.pod.mask
            LL = np.zeros_like(diff_view.data)
            for name, pod in diff_view.pods.iteritems():
                LL += u.abs2(pod.fw(pod.probe * pod.object))

            error_dct[dname] = (np.sum(fmask * (LL - I) ** 2 / (I + 1.))
                            / np.prod(LL.shape))
        return error_dct


if __name__ == '__main__':
    unittest.main()
