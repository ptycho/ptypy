'''
tests for the object-probe interactions, including the specific DM, ePIE etc updates

'''

import unittest

import utils as tu
from ptypy.gpu import data_utils as du
from ptypy.gpu import object_probe_interaction as opi


class ObjectProbeInteractionTest(unittest.TestCase):
    def setUp(self):
        self.PtychoInstance = tu.get_ptycho_instance('pod_to_numpy_test')
        # now convert to arrays
        self.serialized_scan = du.pod_to_arrays(self.PtychoInstance, 'S0000')

    def test_get_exit_wave(self):
        opi.get_exit_wave(self.serialized_scan)
    
    def test_difference_map_realspace_constraint(self):
        opi.difference_map_realspace_constraint(self.serialized_scan, alpha=1.0)