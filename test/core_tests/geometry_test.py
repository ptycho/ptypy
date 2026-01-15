'''
A test for the Base
'''

import unittest
import ptypy.utils as u
import numpy as np
from ptypy.core import geometry
from ptypy.core import Base as theBase
from ptypy.core.geometry import BasicNearfieldPropagator
from ptypy.core.geometry import BasicFarfieldPropagator


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
        return G

    def test_geometry_near_field_init(self):
        G = self.set_up_nearfield()

    def test_geometry_nearfield_resolution(self):
        G = self.set_up_nearfield()
        assert (np.round(G.resolution*1e7) == [1.00, 1.00]).all(), "geometry resolution incorrect for the nearfield"

    def _basic_propagator_test(self, prop):

        # Create random 2D array
        S = (128,128)
        A = np.random.random(S) + 1j * np.random.random(S)

        # FFT and IFFT
        B = prop.fft(A)
        C = prop.ifft(B)

        # asserts
        assert (A.strides == B.strides), "FFT(x) has changed the strides of x, using {:s}".format(prop.FFTch.ffttype)
        assert (B.strides == C.strides), "IFFT(x) has changed the strides of x, using {:s}".format(prop.FFTch.ffttype)
        np.testing.assert_allclose(A,C, err_msg="IFFT(FFT(x) did not return the same as x, using {:s}".format(prop.FFTch.ffttype))

    def test_basic_nearfield_propagator_fftw(self):
        G = self.set_up_nearfield()
        P = BasicNearfieldPropagator(G.p,ffttype="fftw")
        self. _basic_propagator_test(P)

    def test_basic_nearfield_propagator_numpy(self):
        G = self.set_up_nearfield()
        P = BasicNearfieldPropagator(G.p,ffttype="numpy")
        self. _basic_propagator_test(P)

    def test_basic_nearfield_propagator_scipy(self):
        G = self.set_up_nearfield()
        P = BasicNearfieldPropagator(G.p,ffttype="scipy")
        self. _basic_propagator_test(P)

    def test_basic_farfield_propagator_fftw(self):
        G = self.set_up_farfield()
        P = BasicFarfieldPropagator(G.p,ffttype="fftw")
        self. _basic_propagator_test(P)

    def test_basic_farfield_propagator_numpy(self):
        G = self.set_up_farfield()
        P = BasicFarfieldPropagator(G.p,ffttype="numpy")
        self. _basic_propagator_test(P)

    def test_basic_farfield_propagator_scipy(self):
        G = self.set_up_farfield()
        P = BasicFarfieldPropagator(G.p,ffttype="scipy")
        self. _basic_propagator_test(P)
    


if __name__ == '__main__':
    unittest.main()
