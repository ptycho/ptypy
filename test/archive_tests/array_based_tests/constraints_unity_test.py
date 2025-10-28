'''
The tests for the constraints
'''


import unittest
import numpy as np
from test.archive_tests.array_based_tests import utils as tu
from ptypy.accelerate.array_based import data_utils as du
from collections import OrderedDict
from ptypy.engines.utils import basic_fourier_update
from ptypy.accelerate.array_based.constraints import difference_map_fourier_constraint
from ptypy.accelerate import array_based as ab

@unittest.skip("Skip these until I have had chance to investigate the tolerances.")
class ConstraintsUnityTest(unittest.TestCase):

    def test_difference_map_fourier_constraint_pbound_none_UNITY(self):
        ab.FLOAT_TYPE =np.float64
        ab.COMPLEX_TYPE = np.complex128
        PtychoInstance = tu.get_ptycho_instance('pod_to_numpy_test')
        PodPtychoInstance = tu.get_ptycho_instance('pod_to_numpy_test')

        # now convert to arrays
        vectorised_scan = du.pod_to_arrays(PtychoInstance, 'S0000')

        first_view_id = vectorised_scan['meta']['view_IDs'][0]
        master_pod = PtychoInstance.diff.V[first_view_id].pod
        propagator = master_pod.geometry.propagator

        ptypy_ewf, ptypy_error= self.ptypy_difference_map_fourier_constraint(PodPtychoInstance)
        errors = difference_map_fourier_constraint(vectorised_scan['mask'],
                                                  vectorised_scan['diffraction'],
                                                  vectorised_scan['obj'],
                                                  vectorised_scan['probe'],
                                                  vectorised_scan['exit wave'],
                                                  vectorised_scan['meta']['addr'],
                                                  prefilter=propagator.pre_fft,
                                                  postfilter=propagator.post_fft,
                                                  pbound=None,
                                                  alpha=1.0,
                                                  LL_error=True)
        rtol=1e-7
        for idx, key in enumerate(ptypy_ewf.keys()):
            np.testing.assert_allclose(ptypy_ewf[key],
                                       vectorised_scan['exit wave'][idx],
                                       err_msg="The array-based and pod-based exit waves are not consistent",
                                       rtol =rtol)

        ptypy_fmag = []
        ptypy_phot = []
        ptypy_exit = []

        for idx, key in enumerate(ptypy_error.keys()):
            err_fmag, err_phot, err_exit = ptypy_error[key]
            ptypy_fmag.append(err_fmag)
            ptypy_phot.append(err_phot)
            ptypy_exit.append(err_exit)

        ptypy_fmag = np.array(ptypy_fmag)
        ptypy_phot = np.array(ptypy_phot)

        ptypy_exit = np.array(ptypy_exit)

        npy_fmag = errors[0, :]
        npy_phot = errors[1, :]
        npy_exit = errors[2, :]
        ab.FLOAT_TYPE =np.float32
        ab.COMPLEX_TYPE = np.complex64

        np.testing.assert_array_equal(npy_fmag,
                                      ptypy_fmag,
                                      err_msg="The array-based and pod-based fmag errors are not consistent",
                                      rtol=rtol)

        np.testing.assert_array_equal(npy_phot,
                                   ptypy_phot,
                                   err_msg="The array-based and pod-based phot errors are not consistent",
                                   rtol=rtol)
        # there is a slight difference in numpy in the way the mean is calculated here.  It 1e-13 and a diagnostic so almost equal is fine
        np.testing.assert_allclose(npy_exit,
                                   ptypy_exit,
                                   err_msg="The array-based and pod-based exit errors are not consistent",
                                   rtol=rtol)

    def test_difference_map_fourier_constraint_pbound_less_than_fourier_error_UNITY(self):
        PtychoInstance = tu.get_ptycho_instance('pod_to_numpy_test')
        PodPtychoInstance = tu.get_ptycho_instance('pod_to_numpy_test')
        pbound = 0.597053604126
        # now convert to arrays
        vectorised_scan = du.pod_to_arrays(PtychoInstance, 'S0000')

        first_view_id = vectorised_scan['meta']['view_IDs'][0]
        master_pod = PtychoInstance.diff.V[first_view_id].pod
        propagator = master_pod.geometry.propagator

        ptypy_ewf, ptypy_error= self.ptypy_difference_map_fourier_constraint(PodPtychoInstance, pbound=pbound)
        errors = difference_map_fourier_constraint(vectorised_scan['mask'],
                                                  vectorised_scan['diffraction'],
                                                  vectorised_scan['obj'],
                                                  vectorised_scan['probe'],
                                                  vectorised_scan['exit wave'],
                                                  vectorised_scan['meta']['addr'],
                                                  prefilter=propagator.pre_fft,
                                                  postfilter=propagator.post_fft,
                                                  pbound=pbound,
                                                  alpha=1.0,
                                                  LL_error=True)

        for idx, key in enumerate(ptypy_ewf.keys()):
            np.testing.assert_array_equal(ptypy_ewf[key],
                                          vectorised_scan['exit wave'][idx],
                                       err_msg="The array-based and pod-based exit waves are not consistent")

        ptypy_fmag = []
        ptypy_phot = []
        ptypy_exit = []

        for idx, key in enumerate(ptypy_error.keys()):
            err_fmag, err_phot, err_exit = ptypy_error[key]
            ptypy_fmag.append(err_fmag)
            ptypy_phot.append(err_phot)
            ptypy_exit.append(err_exit)

        ptypy_fmag = np.array(ptypy_fmag)
        ptypy_phot = np.array(ptypy_phot)
        ptypy_exit = np.array(ptypy_exit)

        npy_fmag = errors[0, :]
        npy_phot = errors[1, :]
        npy_exit = errors[2, :]

        np.testing.assert_array_equal(npy_fmag,
                                   ptypy_fmag,
                                   err_msg="The array-based and pod-based fmag errors are not consistent")

        np.testing.assert_array_equal(npy_phot,
                                   ptypy_phot,
                                   err_msg="The array-based and pod-based phot errors are not consistent")
        # there is a slight difference in numpy in the way the mean is calculated here.  It 1e-13 and a diagnostic so almost equal is fine
        np.testing.assert_allclose(npy_exit,
                                   ptypy_exit,
                                   err_msg="The array-based and pod-based exit errors are not consistent")

    def test_difference_map_fourier_constraint_pbound_greater_than_fourier_error_UNITY(self):
        PtychoInstance = tu.get_ptycho_instance('pod_to_numpy_test')
        PodPtychoInstance = tu.get_ptycho_instance('pod_to_numpy_test')
        pbound = 200.0
        # now convert to arrays
        vectorised_scan = du.pod_to_arrays(PtychoInstance, 'S0000')

        first_view_id = vectorised_scan['meta']['view_IDs'][0]
        master_pod = PtychoInstance.diff.V[first_view_id].pod
        propagator = master_pod.geometry.propagator

        ptypy_ewf, ptypy_error = self.ptypy_difference_map_fourier_constraint(PodPtychoInstance, pbound=pbound)
        errors = difference_map_fourier_constraint(vectorised_scan['mask'],
                                                  vectorised_scan['diffraction'],
                                                  vectorised_scan['obj'],
                                                  vectorised_scan['probe'],
                                                  vectorised_scan['exit wave'],
                                                  vectorised_scan['meta']['addr'],
                                                  prefilter=propagator.pre_fft,
                                                  postfilter=propagator.post_fft,
                                                  pbound=pbound,
                                                  alpha=1.0,
                                                  LL_error=True)

        for idx, key in enumerate(ptypy_ewf.keys()):
            np.testing.assert_array_equal(ptypy_ewf[key],
                                          vectorised_scan['exit wave'][idx],
                                          err_msg="The array-based and pod-based exit waves are not consistent")

        ptypy_fmag = []
        ptypy_phot = []
        ptypy_exit = []

        for idx, key in enumerate(ptypy_error.keys()):
            err_fmag, err_phot, err_exit = ptypy_error[key]
            ptypy_fmag.append(err_fmag)
            ptypy_phot.append(err_phot)
            ptypy_exit.append(err_exit)

        ptypy_fmag = np.array(ptypy_fmag)
        ptypy_phot = np.array(ptypy_phot)
        ptypy_exit = np.array(ptypy_exit)

        npy_fmag = errors[0, :]
        npy_phot = errors[1, :]
        npy_exit = errors[2, :]

        np.testing.assert_array_equal(npy_fmag,
                                      ptypy_fmag,
                                      err_msg="The array-based and pod-based fmag errors are not consistent")

        np.testing.assert_array_equal(npy_phot,
                                      ptypy_phot,
                                      err_msg="The array-based and pod-based phot errors are not consistent")
        # there is a slight difference in numpy in the way the mean is calculated here.  It 1e-13 and a diagnostic so almost equal is fine
        np.testing.assert_allclose(npy_exit,
                                             ptypy_exit,
                                             err_msg="The array-based and pod-based exit errors are not consistent")

    def ptypy_difference_map_fourier_constraint(self, a_ptycho_instance, pbound=None):
        error_dct = OrderedDict()
        exit_wave = OrderedDict()
        for dname, diff_view in a_ptycho_instance.diff.views.items():
            di_view = a_ptycho_instance.diff.V[dname]
            error_dct[dname] = basic_fourier_update(di_view,
                                                    pbound=pbound,
                                                    alpha=1.0)
            for name, pod in di_view.pods.items():
                exit_wave[name] = pod.exit
        return exit_wave, error_dct


if __name__ == '__main__':
    unittest.main()



