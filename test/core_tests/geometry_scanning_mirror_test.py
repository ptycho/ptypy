'''
A test for the Base
'''

import unittest
import ptypy.utils as u
import numpy as np
from ptypy.core import geometry
from ptypy.core import Base as theBase
from ptypy.core.geometry import BasicNearfieldPropagator, BasicFarfieldPropagator
from ptypy.core.geometry_scanning_mirror import ScanningMirrorFarfieldPropagator
from ptypy.core import Container, Storage, View


# subclass for dictionary access
Base = type('Base',(theBase,),{})


class ScanningMirrorGeometryTest(unittest.TestCase):

    def set_up_farfield(self):
        P = Base()
        P.CType = np.complex128
        P.Ftype = np.float64
        g = u.Param()
        g.energy = None  # u.keV2m(1.0)/6.32e-7
        g.lam = 5.32e-7
        g.distance = 15e-2
        g.psize = 24e-6
        g.shape = (16, 16)
        g.propagation = "farfield"
        g.beam_shift = None
        G = geometry.Geo(owner=P, pars=g)
        return G

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

    def _test_standard_behaviour(self, prop):
        # Create random 2D array
        shape = prop.sh
        w = np.random.random(shape) + 1j * np.random.random(shape)

        farfield = prop.fw(w)
        bw = prop.bw(farfield)

        # asserts
        np.testing.assert_array_almost_equal(bw, w)

    def _test_equivalence(self, prop, prop_basic):
        """
        Test that the propagating with shift = [0, 0] gives the same result as
        the basic propagator.
        """
        # set the beam shift to [0, 0] to mimic no beam shft
        prop.beam_shift = [0, 0]

        # Create random 2D array
        np.testing.assert_array_equal(prop.sh, prop_basic.sh)
        shape = prop.sh
        w = np.random.random(shape) + 1j * np.random.random(shape)

        farfield = prop.fw(w.copy())
        bw = prop.bw(farfield.copy())
        farfield_basic = prop_basic.fw(w.copy())
        bw_basic = prop_basic.bw(farfield.copy())

        # asserts
        np.testing.assert_array_almost_equal(farfield, farfield_basic)
        np.testing.assert_array_almost_equal(bw, bw_basic)

    def _test_shifting_behaviour(self, prop):
        # Create random 2D array
        shape = prop.sh
        w = np.random.random(shape) + 1j * np.random.random(shape)
        #w = np.zeros(shape, dtype=complex); w[0, 1] = 64
        shift_i = int(np.random.rand() * (shape[0] - 1)) + 1
        shift_j = int(np.random.rand() * (shape[1] - 1)) + 1
        shift = np.array([shift_i, shift_j])

        # make shifted fourier transformed
        prop.beam_shift = [0, 0]
        farfield_no_shift = prop.fw(w.copy())
        farfield_rolled = np.roll(farfield_no_shift.copy(), shift, (0, 1))

        prop.beam_shift = shift
        farfield_shifted = prop.fw(w.copy())
        bw = prop.bw(farfield_shifted.copy())

        # asserts
        np.testing.assert_array_almost_equal(bw, w)
        np.testing.assert_array_almost_equal(np.abs(farfield_shifted), np.abs(farfield_rolled))
        np.testing.assert_array_almost_equal(np.sum(np.abs(bw)**2), np.sum(np.abs(w)**2))

    def test_geometry_farfield_init(self):
        G = self.set_up_farfield()

    def test_scanning_mirror_farfield_propagator_fftw(self):
        G = self.set_up_farfield()
        P = ScanningMirrorFarfieldPropagator(G.p, ffttype="fftw")
        P_basic = BasicFarfieldPropagator(G.p, ffttype="fftw")
        self._test_equivalence(P, P_basic)
        self._basic_propagator_test(P)
        self._test_standard_behaviour(P)
        self._test_shifting_behaviour(P)

    def test_scanning_mirror_farfield_propagator_numpy(self):
        G = self.set_up_farfield()
        P = ScanningMirrorFarfieldPropagator(G.p, ffttype="numpy")
        P_basic = BasicFarfieldPropagator(G.p, ffttype="fftw")
        self._test_equivalence(P, P_basic)
        self._basic_propagator_test(P)
        self._test_standard_behaviour(P)
        self._test_shifting_behaviour(P)

    def test_scanning_mirror_farfield_propagator_scipy(self):
        G = self.set_up_farfield()
        P = ScanningMirrorFarfieldPropagator(G.p, ffttype="scipy")
        P_basic = BasicFarfieldPropagator(G.p, ffttype="fftw")
        self._test_equivalence(P, P_basic)
        self._basic_propagator_test(P)
        self._test_standard_behaviour(P)
        self._test_shifting_behaviour(P)


if __name__ == '__main__':
    unittest.main()