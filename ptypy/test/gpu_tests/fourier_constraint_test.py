'''
Test for the propagation in numpy
'''

import unittest
import utils as tu
import numpy as np
from ptypy.gpu import data_utils as du
from ptypy.gpu.propagation import difference_map_fourier_constraint
from ptypy.engines.utils import basic_fourier_update

class FourierConstraintTest(unittest.TestCase):
    def setUp(self):
        self.PtychoInstance = tu.get_ptycho_instance('fourier_constraint_test')
        self.PtychoInstance.pbound = {}
        self.PtychoInstance.p.fourier_relax_factor = 1e-5
        self.PtychoInstance.p.alpha = 1.0#
        self.serialized_scan = du.pod_to_arrays(self.PtychoInstance, 'S0000')
        first_view_id = self.serialized_scan['meta']['view_IDs'][0]
        self.master_pod = self.PtychoInstance.diff.V[first_view_id].pod

        mean_power = 0.
        for name, s in self.PtychoInstance.diff.storages.iteritems():
            self.PtychoInstance.pbound[name] = (
                .25 * self.PtychoInstance.p.fourier_relax_factor**2 * s.pbound_stub)
            mean_power += s.tot_power/np.prod(s.shape)
        # now convert to arrays
        self.serialized_scan = du.pod_to_arrays(self.PtychoInstance, 'S0000')

    def test_fourier_update(self):
        error_dct = {}
        i = 0
        for name, di_view in self.PtychoInstance.diff.views.iteritems():
            # print i
            i+=1
            if not di_view.active:
                continue
            # print "view"
            pbound = self.PtychoInstance.pbound[di_view.storage.ID]
            error_dct[name] = basic_fourier_update(di_view,
                                                   pbound=pbound,
                                                   alpha=self.PtychoInstance.p.alpha)
        print np.array(error_dct.values())[:, 1]

    # def test_fourier_update_numpy(self):
    #     mask = self.serialized_scan['mask']
    #     Idata = self.serialized_scan['diffraction']
    #     obj = self.serialized_scan['obj']
    #     probe = self.serialized_scan['probe']
    #     exit_wave = self.serialized_scan['exit wave']
    #     addr = self.serialized_scan['meta']['addr']
    #     propagator = self.master_pod.geometry.propagator
    #
    #     error_phot = difference_map_fourier_constraint(mask, Idata, obj, probe, exit_wave, addr,
    #                                                           prefilter=propagator.pre_fft,
    #                                                           postfilter=propagator.post_fft,
    #                                                           pbound=None, alpha=1, LL_error=True)
    #     print error_phot



    def test_fourier_update_numpy_vectorised(self):
        pass


