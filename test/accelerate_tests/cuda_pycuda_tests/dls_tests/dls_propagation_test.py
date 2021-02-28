'''
testing on real data
'''

import h5py
import unittest
import numpy as np
import ptypy.utils as u
from .. import PyCudaTest, have_pycuda

if have_pycuda():
    from pycuda import gpuarray
    from ptypy.accelerate.cuda_pycuda.kernels import PropagationKernel

from ptypy.core import geometry
from ptypy.core import Base as theBase

# subclass for dictionary access
Base = type('Base',(theBase,),{})

COMPLEX_TYPE = np.complex64
FLOAT_TYPE = np.float32
INT_TYPE = np.int32

class DLsPropagationKernelTest(PyCudaTest):

    datadir = "/dls/science/users/iat69393/gpu-hackathon/test-data/"
    iter = 0
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

    def test_forward_UNITY(self):

        # Load data
        with h5py.File(self.datadir + "forward_%04d.h5" %self.iter, "r") as f:
            aux = f["aux"][:]

        # Copy data to device
        aux_dev = gpuarray.to_gpu(aux)

        # Geometry
        geo = self.set_up_farfield(aux.shape[1:])

        # CPU kernel
        aux = geo.propagator.fw(aux)

        # GPU kernel
        PropK = PropagationKernel(aux_dev, geo.propagator, queue_thread=self.stream)
        PropK.allocate()
        PropK.fw(aux_dev, aux_dev)

        ## Assert
        np.testing.assert_allclose(aux, aux_dev.get(), atol=self.atol, rtol=self.rtol, 
            err_msg="CPU aux is \n%s, \nbut GPU aux is \n %s, \n " % (repr(aux), repr(aux_d.get())))


    def test_ackward_UNITY(self):

        # Load data
        with h5py.File(self.datadir + "backward_%04d.h5" %self.iter, "r") as f:
            aux = f["aux"][:]

        # Copy data to device
        aux_dev = gpuarray.to_gpu(aux)

        # Geometry
        geo = self.set_up_farfield(aux.shape[1:])

        # CPU kernel
        aux = geo.propagator.bw(aux)

        # GPU kernel
        PropK = PropagationKernel(aux_dev, geo.propagator, queue_thread=self.stream)
        PropK.allocate()
        PropK.bw(aux_dev, aux_dev)

        ## Assert
        np.testing.assert_allclose(aux, aux_dev.get(), atol=self.atol, rtol=self.rtol, 
            err_msg="CPU aux is \n%s, \nbut GPU aux is \n %s, \n " % (repr(aux), repr(aux_d.get())))
