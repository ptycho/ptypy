'''
A test for the Storage class
'''

import unittest
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

if __name__ == '__main__':
    unittest.main()
