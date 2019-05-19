'''
A test for the Storage class
'''

import unittest
import numpy as np
from ptypy.core import Storage, Container, View

class StorageTest(unittest.TestCase):
    def test_storage(self):
        """
        Tests that the Storage constructor runs.
        """
        cont = Container()
        a = Storage(cont)

    def test_storage_reformat(self):
        """
        Tests that storages reformat when adding views
        """
        # These values are chosen because of an earlier bug, issue #74
        psize = .1
        center = 54.75
        shape = (84,75)

        C = Container(data_dims=2)
        S = C.new_storage(shape=128, padonly=False, psize=psize)
        V = View(container=C, storageID=S.ID, coord=center, 
                             shape=shape, psize=psize)
        # repeated reformatting should not change these relations
        for i in range(10):
            S.reformat()
            assert S[V].shape == shape
            assert S.data.shape == (1,) + shape

    def test_storage_padding(self):
        """
        Test padding in storages
        """
        psize = 1.
        shape = (10, 10)
        padding = 5
        C = Container(data_dims=2)
        S = C.new_storage(psize=psize, padding=padding)
        V = View(container=C, storageID=S.ID, coord=(0., 0.), shape=shape)
        S.reformat()
        assert S.shape == (1, shape[0]+2*padding, shape[1] + 2*padding)

        # Check view access stability upon padding change
        S[V] = 1.
        S.padding = 10
        S.reformat()
        assert np.allclose(S[V], 1.)


if __name__ == '__main__':
    unittest.main()
