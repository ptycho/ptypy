'''
Testing based on real data
'''
import h5py
import unittest
import numpy as np
from parameterized import parameterized
from .. import PyCudaTest, have_pycuda

if have_pycuda():
    from pycuda import gpuarray
    from ptypy.accelerate.cuda_pycuda.kernels import PropagationKernel, GradientDescentKernel
from ptypy.accelerate.base.kernels import GradientDescentKernel as BaseGradientDescentKernel

import ptypy.utils as u
from ptypy.core import geometry
from ptypy.core import Base as theBase

# subclass for dictionary access
Base = type('Base',(theBase,),{})

class DLsFloatingIntensityTest(PyCudaTest):

    datadir = "/dls/science/users/iat69393/gpu-hackathon/test-data-base/"
    iter = 10
    rtol = 1e-6
    atol = 1e-6

    def set_up_geometry(self,shape):
        P = Base()
        P.CType = np.complex64
        P.Ftype = np.float32
        g = u.Param()
        g.energy = None # u.keV2m(1.0)/6.32e-7
        g.lam = 1.2781875567010311e-10
        g.distance = 14.65 
        g.psize = 5.5e-05
        g.shape = shape
        g.propagation = "farfield"
        G = geometry.Geo(owner=P, pars=g)
        return G

    @parameterized.expand([
        [False],
        [True],
    ])
    def test_floating_intensity_accuracy(self, do_floating):

        nmax = 10

        # Load data
        with h5py.File(self.datadir + "make_model_%04d.h5" %self.iter, "r") as f:
            aux = f["aux"][:nmax]
            addr = f["addr"][:nmax]
        with h5py.File(self.datadir + "floating_intensities_%04d.h5" %self.iter, "r") as f:
            w = f["w"][:nmax]
            I = f["I"][:nmax]

        # Create arrays for fic and err
        fic = np.ones(I.shape[0], dtype=np.float32)
        err = np.ones(I.shape[0], dtype=np.float32)

         # Geometry
        geo = self.set_up_geometry(aux.shape[1:])

        # Copy data to device
        aux_dev = gpuarray.to_gpu(aux)
        addr_dev = gpuarray.to_gpu(addr)
        w_dev = gpuarray.to_gpu(w)
        I_dev = gpuarray.to_gpu(I)
        fic_dev = gpuarray.to_gpu(fic)
        err_dev = gpuarray.to_gpu(err)

        # CPU Kernel
        BGDK = BaseGradientDescentKernel(aux, addr.shape[1])
        BGDK.allocate()

        # GPU Kernels
        GDK = GradientDescentKernel(aux_dev, addr.shape[1], math_type='double')
        GDK.allocate()
        PK = PropagationKernel(aux_dev, geo.propagator, queue_thread=self.stream)
        PK.allocate()

        # 1. Forward propagation
        aux = geo.propagator.fw(aux)
        PK.fw(aux_dev, aux_dev)
        # np.testing.assert_allclose(aux_dev.get(), aux, atol=self.atol, rtol=self.rtol, 
        #     verbose=False, err_msg="Forward propagation was not as expected")

        # 2. Make model
        BGDK.make_model(aux, addr)
        GDK.make_model(aux_dev, addr_dev)
        # np.testing.assert_allclose(GDK.gpu.Imodel.get(), BGDK.npy.Imodel, atol=self.atol, rtol=self.rtol,
        #     verbose=False, err_msg="`Imodel` buffer has not been updated as expected")

        # 3. Floating intensity update
        if do_floating:
            BGDK.floating_intensity(addr, w, I, fic)
            GDK.floating_intensity(addr_dev, w_dev, I_dev, fic_dev)
            # np.testing.assert_allclose(GDK.gpu.Imodel.get(), BGDK.npy.Imodel, atol=self.atol, rtol=self.rtol, 
            #     verbose=False, err_msg="`Imodel` buffer has not been updated as expected")
            # np.testing.assert_allclose(fic, fic_dev.get(), atol=self.atol, rtol=self.rtol, 
            #     verbose=True, err_msg="floating intensity coeff (fic) has not been updated as expected")

        # 4. Calculate gradients
        BGDK.main(aux, addr, w, I)
        GDK.main(aux_dev, addr_dev, w_dev, I_dev)
        # np.testing.assert_allclose(aux_dev.get(), aux, atol=self.atol, rtol=self.rtol, 
        #     verbose=False, err_msg="Auxiliary has not been updated as expected")
        # np.testing.assert_allclose(GDK.gpu.LLerr.get(), BGDK.npy.LLerr, atol=self.atol, rtol=self.rtol, 
        #     verbose=False, err_msg="Log-likelihood error has not been updated as expected")

        # 5. Reduce error
        BGDK.error_reduce(addr, err)
        GDK.error_reduce(addr_dev, err_dev)
        np.testing.assert_allclose(err_dev.get(), err, atol=self.atol, rtol=self.rtol, 
            verbose=False, err_msg="The error has not been reduced as expected")




        
