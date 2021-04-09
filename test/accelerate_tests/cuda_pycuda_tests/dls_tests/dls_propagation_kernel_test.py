'''
testing on real data
'''

import h5py
import unittest
import numpy as np
from parameterized import parameterized
from .. import PyCudaTest, have_pycuda

if have_pycuda():
    from pycuda import gpuarray
    from ptypy.accelerate.cuda_pycuda.kernels import PropagationKernel

import ptypy.utils as u
from ptypy.core import geometry
from ptypy.core import Base as theBase

# subclass for dictionary access
Base = type('Base',(theBase,),{})

COMPLEX_TYPE = np.complex64
FLOAT_TYPE = np.float32
INT_TYPE = np.int32

class DLsPropagationKernelTest(PyCudaTest):

    datadir = "/dls/science/users/iat69393/gpu-hackathon/test-data-%s/"
    rtol = 1e-6
    atol = 1e-6

    def set_up_farfield(self,shape):
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
        G = geometry.Geo(owner=P, pars=g)
        return G

    @parameterized.expand([
        ["base", 10],
        ["regul", 50],
        ["floating", 0],
    ])
    def test_forward_UNITY(self, name, iter):

        # Load data
        with h5py.File(self.datadir % name + "forward_%04d.h5" %iter, "r") as f:
            aux = f["aux"][0]

        # Copy data to device
        aux_dev = gpuarray.to_gpu(aux)

        # Geometry
        geo = self.set_up_farfield(aux.shape)

        # CPU kernel
        aux = geo.propagator.fw(aux)

        # GPU kernel
        PropK = PropagationKernel(aux_dev, geo.propagator, queue_thread=self.stream)
        PropK.allocate()
        PropK.fw(aux_dev, aux_dev)

        ## Assert
        np.testing.assert_allclose(aux, aux_dev.get(), atol=self.atol, rtol=self.rtol, 
            err_msg="Forward propagation was not as expected")

    @parameterized.expand([
        ["base", 10],
        ["regul", 50],
        ["floating", 0],
    ])
    def test_backward_UNITY(self, name, iter):

        # Load data
        with h5py.File(self.datadir % name + "backward_%04d.h5" %iter, "r") as f:
            aux = f["aux"][0]

        # Copy data to device
        aux_dev = gpuarray.to_gpu(aux)

        # Geometry
        geo = self.set_up_farfield(aux.shape)

        # CPU kernel
        aux = geo.propagator.bw(aux)

        # GPU kernel
        PropK = PropagationKernel(aux_dev, geo.propagator, queue_thread=self.stream)
        PropK.allocate()
        PropK.bw(aux_dev, aux_dev)

        ## Assert
        np.testing.assert_allclose(aux, aux_dev.get(), atol=self.atol, rtol=self.rtol, 
            err_msg="Backward propagation was not as expected")