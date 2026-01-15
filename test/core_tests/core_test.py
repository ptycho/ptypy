'''
A test for the Base
'''

import unittest
import numpy as np
import ptypy
from ptypy import utils as u
from ptypy.core import View, Container, Storage, Base
from ptypy.core.classes import DEFAULT_ACCESSRULE


class CoreTest(unittest.TestCase):
    C1 = Container(data_type='real')
    S1 = C1.new_storage(shape=(1, 7, 7))
    ar = DEFAULT_ACCESSRULE.copy()
    ar.shape = (4, 4)  # ar.shape = 4 would have been also valid.
    ar.coord = 0.      # ar.coord = (0.,0.) would have been accepted, too.
    ar.storageID = S1.ID
    ar.psize = None
    S1.center = (2, 2)
    S1.psize = 0.1
    g = S1.grids()
    S1.origin -= 0.12
    y, x = S1.grids()
    S1.fill(x+y)
    V1 = View(C1, ID=None, accessrule=ar)
    data = S1[V1]
    V1.coord = (0.28, 0.28)
    S1.update_views(V1)
    mn = S1[V1].mean()
    S1.fill_value = mn
    S1.reformat()

    ar2 = ar.copy()
    ar2.coord = (-0.82, -0.82)
    V2 = View(C1, ID=None, accessrule=ar2)
    
    S1.fill_value = 0.
    S1.reformat()

    for i in range(1, 11):
        ar2 = ar.copy()
        ar2.coord = (-0.82+i*0.1, -0.82+i*0.1)
        View(C1, ID=None, accessrule=ar2)

    S1.data[:] = S1.get_view_coverage()

    ar = DEFAULT_ACCESSRULE.copy()
    ar.shape = 200
    ar.coord = 0.
    ar.storageID = 'S100'
    ar.psize = 1.0
    V3=View(C1, ID=None, accessrule=ar)


if __name__ == '__main__':
    unittest.main()
