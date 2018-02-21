'''    
Created on 4 Jan 2018

@author: clb02321
'''
import unittest
import utils as tu
import numpy as np
from copy import deepcopy as copy

from ptypy.gpu import data_utils as du



class DataUtilsTest(unittest.TestCase):
    '''
    tests the conversion between pods and numpy arrays
    '''

    def test_pod_to_numpy(self):
        '''
        does this even run?
        '''
        PtychoInstance = tu.get_ptycho_instance('pod_to_numpy_test')
        du.pod_to_arrays(PtychoInstance, 'S0000', scan_model='Full')

    def test_numpy_to_pod(self):
        PtychoInstance = tu.get_ptycho_instance('pod_to_numpy_test')
        views = PtychoInstance.diff.V
        main_pod = views[views.keys()[0]].pod  # we will use this to get all the information
        obj_shape = main_pod.ob_view.storage.data.shape
        probe_shape = main_pod.pr_view.storage.data.shape
        exit_wave_shape = main_pod.ex_view.storage.data.shape
        exit_wave = np.ones((exit_wave_shape))
        obj = np.ones((obj_shape))
        probe = np.ones((probe_shape))

        array_dictionary = {'exit wave': exit_wave,
                            'obj': obj,
                            'probe': probe}
        diffraction_storages = PtychoInstance.diff.S.keys()
        dID = diffraction_storages[0]

        du.array_to_pods(copy(PtychoInstance), dID, array_dictionary, scan_model='Full')

    def test_numpy_pod_consistency(self):
        PtychoInstance = tu.get_ptycho_instance('pod_to_numpy_test')
        PtychoInstance2 = tu.get_ptycho_instance('pod_to_numpy_test')
        views = PtychoInstance.diff.V
        main_pod = views[views.keys()[0]].pod  # we will use this to get all the information
        obj_shape = main_pod.ob_view.storage.data.shape
        probe_shape = main_pod.pr_view.storage.data.shape
        exit_wave_shape = main_pod.ex_view.storage.data.shape
        foo = du.pod_to_arrays(PtychoInstance, 'S0000', scan_model='Full')
        # now make a change
        foo['exit wave'] = np.random.rand(*exit_wave_shape) + 1j * np.random.rand(*exit_wave_shape)
        foo['probe'] = np.ones((probe_shape), dtype=np.complex64)
        foo['object'] = np.ones((obj_shape), dtype=np.complex64)
        # and convert back
        bar = du.array_to_pods(PtychoInstance2, 'S0000', foo, scan_model='Full')

        # now need to iterate through all the pods and arrays and check that the right thing has been set to all of them
        k=0
        for name, pod in bar.pods.iteritems():
            np.testing.assert_array_equal(pod.exit, foo['exit wave'][k])
            k+=1




if __name__ == "__main__":
    unittest.main()
