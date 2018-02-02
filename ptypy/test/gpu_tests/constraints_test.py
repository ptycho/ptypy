'''
The tests for the constraints
'''


import unittest
import numpy as np
import utils as tu
from copy import deepcopy as copy
from ptypy.gpu import data_utils as du
from collections import OrderedDict
from ptypy.engines.utils import basic_fourier_update
from ptypy.gpu.constraints import difference_map_fourier_constraint

class ConstraintsTest(unittest.TestCase):
    def setUp(self):
        self.PtychoInstance = tu.get_ptycho_instance('pod_to_numpy_test')
        # now convert to arrays
        self.vectorised_scan = du.pod_to_arrays(self.PtychoInstance, 'S0000')
        self.addr = self.vectorised_scan['meta']['addr'] # probably want to extract these at a later date, but just to get stuff going...
        self.probe = self.vectorised_scan['probe']
        self.obj = self.vectorised_scan['obj']
        self.exit_wave = self.vectorised_scan['exit wave']
        self.diffraction = self.vectorised_scan['diffraction']
        self.mask = self.vectorised_scan['mask']
        view_names = self.PtychoInstance.diff.views.keys()
        self.error_dict = OrderedDict.fromkeys(view_names)
        first_view_id = self.vectorised_scan['meta']['view_IDs'][0]
        master_pod = self.PtychoInstance.diff.V[first_view_id].pod
        self.propagator = master_pod.geometry.propagator

    def test_difference_map_fourier_constraint(self):
        difference_map_fourier_constraint(self.mask,
                                          self.diffraction,
                                          self.obj,
                                          self.probe,
                                          self.exit_wave,
                                          self.addr,
                                          prefilter=self.propagator.pre_fft,
                                          postfilter=self.propagator.post_fft,
                                          pbound=None,
                                          alpha=1.0,
                                          LL_error=True)


    def test_difference_map_fourier_constraint_UNITY(self):
        ptypy_ewf, ptypy_error= self.ptypy_difference_map_fourier_constraint()
        exit_wave, errors = difference_map_fourier_constraint(self.mask,
                                                              self.diffraction,
                                                              self.obj,
                                                              self.probe,
                                                              self.exit_wave,
                                                              self.addr,
                                                              prefilter=self.propagator.pre_fft,
                                                              postfilter=self.propagator.post_fft,
                                                              pbound=None,
                                                              alpha=1.0,
                                                              LL_error=True)

        for idx, key in enumerate(ptypy_ewf.keys()):
            np.testing.assert_allclose(ptypy_ewf[key], exit_wave[idx])

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



        # import pylab as plt
        # x = np.arange(92)
        # plt.figure('fmag')
        # plt.plot(x, npy_fmag, x, ptypy_fmag)
        # plt.legend(['npy', 'ptypy'])
        # plt.show()







    def ptypy_difference_map_fourier_constraint(self):
        ptycho_instance = copy(self.PtychoInstance)
        error_dct = OrderedDict()
        exit_wave = OrderedDict()

        for dname in self.error_dict.keys():
            di_view = ptycho_instance.diff.V[dname]
            error_dct[dname] = basic_fourier_update(di_view,
                                                   pbound=None,
                                                   alpha=1.0)
            for name, pod in di_view.pods.iteritems():
                exit_wave[name] = pod.exit



        return exit_wave, error_dct


if __name__ == '__main__':
    unittest.main()



