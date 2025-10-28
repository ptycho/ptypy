'''
This is a unity test comparing to the pod based framework
'''

import unittest
import numpy as np
from test.archive_tests.array_based_tests import utils as tu
from ptypy.accelerate.array_based import data_utils as du
from ptypy.accelerate.array_based import object_probe_interaction as opi
from collections import OrderedDict


class ObjectProbeInteractionUnityTest(unittest.TestCase):

    def test_difference_map_realspace_constraint_UNITY(self):
        PtychoInstance = tu.get_ptycho_instance('pod_to_numpy_test')
        a_ptycho_instance = tu.get_ptycho_instance('pod_to_numpy_test')
        # now convert to arrays
        vectorised_scan = du.pod_to_arrays(PtychoInstance, 'S0000')
        addr_info = vectorised_scan['meta']['addr']  # probably want to extract these at a later date, but just to get stuff going...
        probe = vectorised_scan['probe']
        obj = vectorised_scan['obj']
        exit_wave = vectorised_scan['exit wave']
        probe_and_object = opi.scan_and_multiply(probe, obj, exit_wave.shape, addr_info)

        ptypy_dm_constraint = self.ptypy_apply_difference_map(a_ptycho_instance)
        numpy_dm_constraint = opi.difference_map_realspace_constraint(probe_and_object,
                                                                      exit_wave,
                                                                      alpha=1.0)
        for idx, key in enumerate(ptypy_dm_constraint):
            np.testing.assert_allclose(ptypy_dm_constraint[key], numpy_dm_constraint[idx])

    def ptypy_apply_difference_map(self, a_ptycho_instance):
        f = OrderedDict()
        alpha = 1.0
        for dname, diff_view in a_ptycho_instance.diff.views.items():
            for name, pod in diff_view.pods.items():
                if not pod.active:
                    continue
                f[name] = (1 + alpha) * pod.probe * pod.object - alpha * pod.exit
        return f

    if __name__ == "__main__":
        unittest.main()
