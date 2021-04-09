'''
A test for the Base
'''

import unittest
import ptypy.utils as u
import numpy as np
from ptypy.core import geometry
from ptypy.core import Base as theBase

# subclass for dictionary access
Base = type('Base',(theBase,),{})

class GeometryTest(unittest.TestCase):

    def set_up_farfield(self):
        P = Base()
        P.CType = np.complex128
        P.Ftype = np.float64
        g = u.Param()
        g.energy = None # u.keV2m(1.0)/6.32e-7
        g.lam = 5.32e-7
        g.distance = 15e-2
        g.psize = 24e-6
        g.shape = 256
        g.propagation = "farfield"
        G = geometry.Geo(owner=P, pars=g)
        return G

    def test_geometry_farfield_init(self):
        G = self.set_up_farfield()
        print(G.resolution)

    def test_geometry_farfield_resolution(self):
        G = self.set_up_farfield()
        print(G.resolution)
        assert (np.round(G.resolution*1e5,2) ==  np.array([1.30, 1.30])).all(), "geometry resolution incorrect for the far-field"

    def set_up_nearfield(self):
        P = Base()
        P.CType = np.complex128
        P.Ftype = np.float64
        g = u.Param()
        g.energy = None # u.keV2m(1.0)/6.32e-7
        g.lam = 1e-10
        g.distance = 1.0
        g.psize = 100e-9
        g.shape = 256
        g.propagation = "nearfield"
        G = geometry.Geo(owner=P, pars=g)
#         print G.resolution
        return G

    def test_geometry_near_field_init(self):
        G = self.set_up_nearfield()

    def test_geometry_nearfield_resolution(self):
        G = self.set_up_nearfield()
        assert (np.round(G.resolution*1e7) == [1.00, 1.00]).all(), "geometry resolution incorrect for the nearfield"

    def test_downsample(self):
        G = self.set_up_farfield()
        G.resample = 2

        B = np.indices((4,4), dtype=np.complex64)[0] + 1j * np.indices((4,4), dtype=np.complex64)[1]
        A = G.downsample(B)
        exp_A = np.array([[ 2. +2.j,  2.+10.j],
                          [10. +2.j, 10.+10.j]], dtype=np.complex64)
        np.testing.assert_almost_equal(A.sum(), B.sum())            
        np.testing.assert_array_almost_equal(A, exp_A)

    def test_upsample(self):
        G = self.set_up_farfield()
        G.resample = 2

        B = np.indices((2,2), dtype=np.complex64)[0] + 1j * np.indices((2,2), dtype=np.complex64)[1]
        A = G.upsample(B)
        exp_A = np.array([[0.  +0.j  , 0.  +0.j  , 0.  +0.25j, 0.  +0.25j],
                          [0.  +0.j  , 0.  +0.j  , 0.  +0.25j, 0.  +0.25j],
                          [0.25+0.j  , 0.25+0.j  , 0.25+0.25j, 0.25+0.25j],
                          [0.25+0.j  , 0.25+0.j  , 0.25+0.25j, 0.25+0.25j]], dtype=np.complex64)
        np.testing.assert_almost_equal(A.sum(), B.sum())            
        np.testing.assert_array_almost_equal(A, exp_A)

    def test_downsample_upsample(self):
        G = self.set_up_farfield()
        G.resample = 2

        A = np.random.random((4,4))
        B = G.downsample(G.upsample(A))
        np.testing.assert_almost_equal(A.sum(), B.sum())            
        np.testing.assert_array_almost_equal(A, B)

if __name__ == '__main__':
    unittest.main()
