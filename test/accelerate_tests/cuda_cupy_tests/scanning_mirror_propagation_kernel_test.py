'''

'''
import pytest
import numpy as np
import ptypy.utils as u
from . import CupyCudaTest, have_cupy

if have_cupy():
    import cupy as cp
    from ptypy.accelerate.cuda_cupy.kernels import PropagationKernel
else:
    import numpy as cp

from ptypy.core.geometry_scanning_mirror import Geo_ScanningMirror
from ptypy.core import Base as theBase

# subclass for dictionary access
Base = type('Base',(theBase,),{})

COMPLEX_TYPE = np.complex64
FLOAT_TYPE = np.float32
INT_TYPE = np.int32


stream = cp.cuda.Stream()

def set_up_farfield(shape, resolution=None):
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
    g.beam_shift = [5, 10]
    if resolution is not None:
        g.resolution = resolution
    G = Geo_ScanningMirror(owner=P, pars=g)
    return G


def test_farfield_propagator_forward_UNITY():
    # setup
    SH = (2,16,16)
    aux = np.zeros((SH), dtype=COMPLEX_TYPE)
    aux[:,5:11,5:11] = 1. + 2j
    aux_d = cp.asarray(aux)
    geo = set_up_farfield(SH[1:])

    # test
    aux = geo.propagator.fw(aux)
    PropK = PropagationKernel(aux_d, geo.propagator, queue_thread=stream)
    PropK.allocate()
    PropK.fw(aux_d, aux_d)

    np.testing.assert_allclose(aux_d.get(), aux, atol=1e-06, rtol=5e-5,
        err_msg="Numpy aux is \n%s, \nbut gpu aux is \n %s, \n " % (repr(aux), repr(aux_d.get())))

def test_farfield_propagator_backward_UNITY():
    # setup
    SH = (2,16,16)
    aux = np.zeros((SH), dtype=COMPLEX_TYPE)
    aux[:,5:11,5:11] = 1. + 2j
    aux_d = cp.asarray(aux)
    geo = set_up_farfield(SH[1:])

    # test
    aux = geo.propagator.bw(aux)
    PropK = PropagationKernel(aux_d, geo.propagator, queue_thread=stream)
    PropK.allocate()
    PropK.bw(aux_d, aux_d)

    np.testing.assert_allclose(aux_d.get(), aux, atol=1e-06, rtol=5e-5,
        err_msg="Numpy aux is \n%s, \nbut gpu aux is \n %s, \n " % (repr(aux), repr(aux_d.get())))

def test_farfield_propagator_forward_crop_pad_UNITY():
    # setup
    SH = (2,16,16)
    aux = np.zeros((SH), dtype=COMPLEX_TYPE)
    aux[:,5:11,5:11] = 1. + 2j
    aux_d = cp.asarray(aux)
    geo = set_up_farfield(SH[1:])
    geo = set_up_farfield(SH[1:], resolution=0.5*geo.resolution)

    # test
    aux = geo.propagator.fw(aux)
    PropK = PropagationKernel(aux_d, geo.propagator, queue_thread=stream)
    PropK.allocate()
    PropK.fw(aux_d, aux_d)

    np.testing.assert_allclose(aux_d.get(), aux, atol=1e-06, rtol=5e-5,
        err_msg="Numpy aux is \n%s, \nbut gpu aux is \n %s, \n " % (repr(aux), repr(aux_d.get())))

def test_farfield_propagator_backward_crop_pad_UNITY():
    # setup
    SH = (2,16,16)
    aux = np.zeros((SH), dtype=COMPLEX_TYPE)
    aux[:,5:11,5:11] = 1. + 2j
    aux_d = cp.asarray(aux)
    geo = set_up_farfield(SH[1:])
    geo = set_up_farfield(SH[1:], resolution=0.5*geo.resolution)

    # test
    aux = geo.propagator.bw(aux)
    PropK = PropagationKernel(aux_d, geo.propagator, queue_thread=stream)
    PropK.allocate()
    PropK.bw(aux_d, aux_d)

    np.testing.assert_allclose(aux_d.get(), aux, atol=1e-06, rtol=5e-5,
        err_msg="Numpy aux is \n%s, \nbut gpu aux is \n %s, \n " % (repr(aux), repr(aux_d.get())))
