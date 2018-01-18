'''
tests for the object-probe interactions, including the specific DM, ePIE etc updates

'''

import unittest

import utils as tu
from ptypy.gpu import data_utils as du
from ptypy.gpu.object_probe_interaction import get_exit_wave


class ObjectProbeInteractionTest(unittest.TestCase):
    def setUp(self):
        self.PtychoInstance = tu.get_ptycho_instance('pod_to_numpy_test')
        # now convert to arrays
        self.serialized_scan = du.pod_to_arrays(self.PtychoInstance, 'S0000')

    def test_get_exit_wave(self):
        get_exit_wave(self.serialized_scan)
        