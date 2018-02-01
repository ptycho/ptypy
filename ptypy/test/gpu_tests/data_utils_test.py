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

    def setUp(self):
        self.PtychoInstance = tu.get_ptycho_instance('pod_to_numpy_test')
        views = self.PtychoInstance.diff.V
        main_pod = views[views.keys()[0]].pod  # we will use this to get all the information

        self.obj_shape = main_pod.ob_view.storage.data.shape
        self.probe_shape = main_pod.pr_view.storage.data.shape
        self.exit_wave_shape = main_pod.ex_view.storage.data.shape


    def test_pod_to_numpy(self):
        '''
        does this even run?
        '''
        foo = du.pod_to_arrays(self.PtychoInstance, 'S0000', scan_model='Full')
        # replace these with a logger
        print 'diffraction', foo['diffraction'].shape
        print 'probe', foo['probe'].shape
        print 'obj', foo['obj'].shape
        print 'exit wave', foo['exit wave'].shape
        print 'mask', foo['mask'].shape
        print 'addr', foo['meta']['addr'].shape

    def test_numpy_to_pod(self):
        exit_wave = np.ones((self.exit_wave_shape))
        obj = np.ones((self.obj_shape))
        probe = np.ones((self.probe_shape))

        array_dictionary = {'exit wave': exit_wave,
                            'obj': obj,
                            'probe': probe}
        diffraction_storages = self.PtychoInstance.diff.S.keys()
        dID = diffraction_storages[0]

        _foo = du.array_to_pods(copy(self.PtychoInstance), dID, array_dictionary, scan_model='Full')



    def test_numpy_pod_consistency(self):

        foo = du.pod_to_arrays(self.PtychoInstance, 'S0000', scan_model='Full')
        # now make a change
        foo['exit wave'] = np.random.rand(*self.exit_wave_shape)
        foo['probe'] = np.ones((self.probe_shape))
        foo['object'] = np.ones((self.obj_shape))
        # and convert back
        bar = du.array_to_pods(copy(self.PtychoInstance), 'S0000', foo, scan_model='Full')

        # now need to iterate through all the pods and arrays and check that the right thing has been set to all of them
        k=0
        for name, pod in bar.pods.iteritems():
            np.testing.assert_allclose(pod.exit, foo['exit wave'][k])
            k+=1




if __name__ == "__main__":
    unittest.main()
