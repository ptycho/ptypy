'''

'''

import numpy as np
import ptypy.utils as u
from . import CupyCudaTest, have_cupy

if have_cupy():
    import cupy as cp
    from ptypy.accelerate.cuda_cupy.kernels import PropagationKernel

from ptypy.core import geometry
from ptypy.core import Base as theBase

# subclass for dictionary access
Base = type('Base',(theBase,),{})

COMPLEX_TYPE = np.complex64
FLOAT_TYPE = np.float32
INT_TYPE = np.int32

class PropagationKernelTest(CupyCudaTest):

    def set_up_farfield(self,shape, resolution=None):
        P = Base()
        P.CType = COMPLEX_TYPE
        P.Ftype = FLOAT_TYPE
        g = u.Param()
        g.energy = None # u.keV2m(1.0)/6.32e-7
        g.lam = 5.32e-7
        g.distance = 15e-2
        g.psize = 24e-6
        g.shape = shape
        g.propagation = "farfield"
        if resolution is not None:
            g.resolution = resolution
        G = geometry.Geo(owner=P, pars=g)
        return G

    def set_up_nearfield(self, shape):
        P = Base()
        P.CType = COMPLEX_TYPE
        P.Ftype = FLOAT_TYPE
        g = u.Param()
        g.energy = None # u.keV2m(1.0)/6.32e-7
        g.lam = 1e-10
        g.distance = 1.0
        g.psize = 100e-9
        g.shape = shape
        g.propagation = "nearfield"
        G = geometry.Geo(owner=P, pars=g)
        return G

    def test_farfield_propagator_forward_UNITY(self):
        # setup
        SH = (2,16,16)
        aux = np.zeros((SH), dtype=COMPLEX_TYPE)
        aux[:,5:11,5:11] = 1. + 2j
        aux_d = cp.asarray(aux)
        geo = self.set_up_farfield(SH[1:])

        # test
        aux = geo.propagator.fw(aux)
        PropK = PropagationKernel(aux_d, geo.propagator, queue_thread=self.stream)
        PropK.allocate()
        PropK.fw(aux_d, aux_d)

        np.testing.assert_allclose(aux_d.get(), aux, atol=1e-06, rtol=5e-5, 
            err_msg="Numpy aux is \n%s, \nbut gpu aux is \n %s, \n " % (repr(aux), repr(aux_d.get())))

    def test_farfield_propagator_backward_UNITY(self):
        # setup
        SH = (2,16,16)
        aux = np.zeros((SH), dtype=COMPLEX_TYPE)
        aux[:,5:11,5:11] = 1. + 2j
        aux_d = cp.asarray(aux)
        geo = self.set_up_farfield(SH[1:])

        # test
        aux = geo.propagator.bw(aux)
        PropK = PropagationKernel(aux_d, geo.propagator, queue_thread=self.stream)
        PropK.allocate()
        PropK.bw(aux_d, aux_d)

        np.testing.assert_allclose(aux_d.get(), aux, atol=1e-06, rtol=5e-5, 
            err_msg="Numpy aux is \n%s, \nbut gpu aux is \n %s, \n " % (repr(aux), repr(aux_d.get())))

    def test_farfield_propagator_forward_crop_pad_UNITY(self):
        # setup
        SH = (2,16,16)
        aux = np.zeros((SH), dtype=COMPLEX_TYPE)
        aux[:,5:11,5:11] = 1. + 2j
        aux_d = cp.asarray(aux)
        geo = self.set_up_farfield(SH[1:])
        geo = self.set_up_farfield(SH[1:], resolution=0.5*geo.resolution)

        # test
        aux = geo.propagator.fw(aux)
        PropK = PropagationKernel(aux_d, geo.propagator, queue_thread=self.stream)
        PropK.allocate()
        PropK.fw(aux_d, aux_d)

        np.testing.assert_allclose(aux_d.get(), aux, atol=1e-06, rtol=5e-5, 
            err_msg="Numpy aux is \n%s, \nbut gpu aux is \n %s, \n " % (repr(aux), repr(aux_d.get())))

    def test_farfield_propagator_backward_crop_pad_UNITY(self):
        # setup
        SH = (2,16,16)
        aux = np.zeros((SH), dtype=COMPLEX_TYPE)
        aux[:,5:11,5:11] = 1. + 2j
        aux_d = cp.asarray(aux)
        geo = self.set_up_farfield(SH[1:])
        geo = self.set_up_farfield(SH[1:], resolution=0.5*geo.resolution)

        # test
        aux = geo.propagator.bw(aux)
        PropK = PropagationKernel(aux_d, geo.propagator, queue_thread=self.stream)
        PropK.allocate()
        PropK.bw(aux_d, aux_d)

        np.testing.assert_allclose(aux_d.get(), aux, atol=1e-06, rtol=5e-5, 
            err_msg="Numpy aux is \n%s, \nbut gpu aux is \n %s, \n " % (repr(aux), repr(aux_d.get())))

    def test_nearfield_propagator_forward_UNITY(self):
        # setup
        SH = (2,16,16)
        aux = np.zeros((SH), dtype=COMPLEX_TYPE)
        aux[:,5:11,5:11] = 1. + 2j
        aux_d = cp.asarray(aux)
        geo = self.set_up_nearfield(SH[1:])
        
        # test
        aux = geo.propagator.fw(aux)
        PropK = PropagationKernel(aux_d, geo.propagator, queue_thread=self.stream)
        PropK.allocate()
        PropK.fw(aux_d, aux_d)

        np.testing.assert_allclose(aux_d.get(), aux, atol=1e-06, rtol=5e-5, 
            err_msg="Numpy aux is \n%s, \nbut gpu aux is \n %s, \n " % (repr(aux), repr(aux_d.get())))

    def test_nearfield_propagator_backward_UNITY(self):
        # setup
        SH = (2,16,16)
        aux = np.zeros((SH), dtype=COMPLEX_TYPE)
        aux[:,5:11,5:11] = 1. + 2j
        aux_d = cp.asarray(aux)
        geo = self.set_up_nearfield(SH[1:])
    
        # test
        aux = geo.propagator.bw(aux)
        PropK = PropagationKernel(aux_d, geo.propagator, queue_thread=self.stream)
        PropK.allocate()
        PropK.bw(aux_d, aux_d)

        np.testing.assert_allclose(aux_d.get(), aux, atol=1e-06, rtol=5e-5, 
            err_msg="Numpy aux is \n%s, \nbut gpu aux is \n %s, \n " % (repr(aux), repr(aux_d.get())))