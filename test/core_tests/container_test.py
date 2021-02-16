'''
A test for the Container class
'''

import unittest
from ptypy.core import Container, Storage, View, Base
import numpy as np

class ContainerTest(unittest.TestCase):
    def test_container(self):
        a = Container()

    def test_container_copy(self):
        # make a container with a storage and two views
        B = Base()
        C = Container(B,data_type=float)
        V1 = View(container=C, shape=10, coord=(0, 0), storageID='S0')
        V2 = View(container=C, shape=10, coord=(3, 3), storageID='S0')
        C.reformat()

        # put some data in the views
        C[V1] += 1.0
        C[V2] += 1.0

        # some checks
        C2 = C.copy()
        assert np.allclose(
            C2.storages['S0'].data,
            C.storages['S0'].data)

        C3 = C.copy(fill=3.14)
        assert np.allclose(
            C3.storages['S0'].data,
            3.14)

        C4 = C.copy(dtype=int)
        assert C4.storages['S0'].data.dtype == int
        assert np.all(
            C4.storages['S0'].data == 0)
        # you can still use the old views here
        C4[V1] += 1
        C4[V2] += 1
        assert np.allclose(
            C4.storages['S0'].data,
            C.storages['S0'].data) 

        C5 = C.copy(dtype=int, fill=2)
        assert np.all(
            C5.storages['S0'].data == 2)

if __name__ == '__main__':
    unittest.main()