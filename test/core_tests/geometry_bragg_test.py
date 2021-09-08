'''
A test for the Base
'''

import unittest
import ptypy.utils as u
import numpy as np
from ptypy.core.geometry_bragg import Geo_Bragg
from ptypy.core import Container, Storage, View

class BraggGeometryTest(unittest.TestCase):

    def testSetup(self):
        g = Geo_Bragg(
                psize=(0.005, 13e-6, 13e-6),
                shape=(9, 128, 128),
                energy=8.5,
                distance=2.0,
                theta_bragg=22.32)
        # some numerical checks
        assert np.round(g.dq1) == 279992
        assert np.round(g.dq2) == 279992
        assert np.round(g.dq3) == 2855229
        assert np.allclose(
            g.resolution,
            [2.64312974e-07, 1.89516090e-07, 1.75317015e-07], # (dr3, dr1, dr2)
            atol = 0, rtol = .001)

    def testViews(self):
        C = Container(data_dims=3)
        positions = np.array([np.arange(10), np.arange(10), np.arange(10)]).T
        for pos_ in positions:
            View(C, storageID='S0', psize=.2, coord=pos_, shape=12)
        S = list(C.storages.values())[0]
        S.reformat()
        cov = np.array(np.real(S.get_view_coverage()), dtype=int)
        # some numerical checks
        assert S.shape == (1, 57, 57, 57)
        assert cov.max() == 3
        assert cov.min() == 0
        assert cov.sum() == 17280

if __name__ == '__main__':
    unittest.main()