'''
tests for the object-probe interactions, including the specific DM, ePIE etc updates

'''

import unittest
import numpy as np
import utils as tu
from ptypy.gpu import data_utils as du
from ptypy.array_based import object_probe_interaction as opi
from ptypy.gpu import object_probe_interaction as gopi
from collections import OrderedDict

class ObjectProbeInteractionTest(unittest.TestCase):

    @unittest.skip("This method is not implemented yet")
    def test_scan_and_multiply_UNITY(self):
        PtychoInstance = tu.get_ptycho_instance('pod_to_numpy_test')
        # now convert to arrays
        vectorised_scan = du.pod_to_arrays(PtychoInstance, 'S0000')
        addr = vectorised_scan['meta']['addr'] # probably want to extract these at a later date, but just to get stuff going...
        probe = vectorised_scan['probe']
        obj = vectorised_scan['obj']
        exit_wave = vectorised_scan['exit wave']
        addr_info = addr[:, 0]
        po = opi.scan_and_multiply(probe, obj, exit_wave.shape, addr_info)
        gpo = gopi.scan_and_multiply(probe, obj, exit_wave.shape, addr_info)
        np.testing.assert_array_equal(po, gpo)

    @unittest.skip("This method is not implemented yet")
    def test_difference_map_realspace_constraint_UNITY(self):
        PtychoInstance = tu.get_ptycho_instance('pod_to_numpy_test')
        # now convert to arrays
        vectorised_scan = du.pod_to_arrays(PtychoInstance, 'S0000')
        addr = vectorised_scan['meta']['addr'] # probably want to extract these at a later date, but just to get stuff going...
        probe = vectorised_scan['probe']
        obj = vectorised_scan['obj']
        exit_wave = vectorised_scan['exit wave']
        po = opi.difference_map_realspace_constraint(obj, probe, exit_wave, addr, alpha=1.0)
        gpo = gopi.difference_map_realspace_constraint(obj, probe, exit_wave, addr, alpha=1.0)
        np.testing.assert_array_equal(po, gpo)


if __name__ == "__main__":
    unittest.main()
