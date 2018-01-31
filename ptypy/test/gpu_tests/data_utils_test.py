'''    
Created on 4 Jan 2018

@author: clb02321
'''
import unittest

from ptypy.gpu import data_utils as du
import utils as tu



class DataUtilsTest(unittest.TestCase):
    '''
    tests the conversion between pods and numpy arrays
    '''

    def setUp(self):
        self.PtychoInstance = tu.get_ptycho_instance('pod_to_numpy_test')

    def test_pod_to_numpy(self):
        '''
        does this even run?
        '''
        foo = du.pod_to_arrays(self.PtychoInstance, 'S0000')
        print 'diffraction', foo['diffraction'].shape
        print 'probe', foo['probe'].shape
        print 'obj', foo['obj'].shape
        print 'exit wave', foo['exit wave'].shape
        print 'mask', foo['mask'].shape
        print 'addr', foo['meta']['addr'].shape

    def test_numpy_to_pod(self):
        pass

    def test_numpy_pod_consistency(self):
        foo = du.pod_to_arrays(self.PtychoInstance, 'S0000')
        bar = du.arrays_to_pods(self.PtychoInstance, 'S0000')



if __name__ == "__main__":
    unittest.main()
