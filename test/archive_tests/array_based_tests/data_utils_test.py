'''    
Created on 4 Jan 2018

@author: clb02321
'''
import unittest
from test.archive_tests.array_based_tests import utils as tu
import numpy as np

from ptypy.accelerate.array_based import data_utils as du



class DataUtilsTest(unittest.TestCase):
    '''
    tests the conversion between pods and numpy arrays
    '''

    def test_pod_to_numpy(self):
        '''
        tests if the vectorisation process works
        '''
        PtychoInstance = tu.get_ptycho_instance('pod_to_numpy_test')
        du.pod_to_arrays(PtychoInstance, 'S0000', scan_model='Full')

    def test_numpy_pod_consistency(self):
        '''
        vectorises the Ptycho instance.
        '''
        PtychoInstance = tu.get_ptycho_instance('pod_to_numpy_test')
        vectorised_scan = du.pod_to_arrays(PtychoInstance, 'S0000')
        addr = vectorised_scan['meta']['addr']
        view_IDS = vectorised_scan['meta']['view_IDs']

        # check the probe references match up
        vectorised_scan['probe'] *= np.random.rand(*vectorised_scan['probe'].shape)

        pa, oa, ea, da, ma = zip(*addr)

        for idx, vID in enumerate(view_IDS):
            np.testing.assert_array_equal(vectorised_scan['probe'][pa[idx][0]], PtychoInstance.pr.V[vID].data)



        vectorised_scan['exit wave'] *= np.random.rand(*vectorised_scan['exit wave'].shape)

        for idx, vID in enumerate(view_IDS):
            np.testing.assert_array_equal(vectorised_scan['exit wave'][ea[idx][0]], PtychoInstance.ex.V[vID].data)




if __name__ == "__main__":
    unittest.main()
