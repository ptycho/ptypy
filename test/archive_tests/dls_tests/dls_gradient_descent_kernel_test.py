'''
Testing on real data
'''

import h5py
import unittest
import numpy as np
from parameterized import parameterized
from .. import perfrun, PyCudaTest, have_pycuda

if have_pycuda():
    from pycuda import gpuarray
    from ptypy.accelerate.cuda_pycuda.kernels import GradientDescentKernel
from ptypy.accelerate.base.kernels import GradientDescentKernel as BaseGradientDescentKernel

COMPLEX_TYPE = np.complex64
FLOAT_TYPE = np.float32
INT_TYPE = np.int32

class DlsGradientDescentKernelTest(PyCudaTest):

    datadir = "/dls/science/users/iat69393/gpu-hackathon/test-data-%s/"
    rtol = 1e-6
    atol = 1e-6

    @parameterized.expand([
        ["base", 10],
        ["regul", 50],
        ["floating", 0],
    ])
    def test_make_model_UNITY(self, name, iter):

        # Load data
        with h5py.File(self.datadir %name + "make_model_%04d.h5" %iter, "r") as f:
            aux = f["aux"][:]
            addr = f["addr"][:]

        # Copy data to device
        aux_dev = gpuarray.to_gpu(aux)
        addr_dev = gpuarray.to_gpu(addr)

        # CPU Kernel
        BGDK = BaseGradientDescentKernel(aux, addr.shape[1])
        BGDK.allocate()
        BGDK.make_model(aux, addr)

        # GPU kernel
        GDK = GradientDescentKernel(aux_dev, addr.shape[1])
        GDK.allocate()
        GDK.make_model(aux_dev, addr_dev)

        ## Assert
        np.testing.assert_allclose(BGDK.npy.Imodel, GDK.gpu.Imodel.get(), atol=self.atol, rtol=self.rtol,
            err_msg="`Imodel` buffer has not been updated as expected")

    @parameterized.expand([
        ["base", 10],
        ["regul", 50],
        ["floating", 0],
    ])
    def test_floating_intensity_UNITY(self, name, iter):
        
        # Load data
        with h5py.File(self.datadir %name + "floating_intensities_%04d.h5" %iter, "r") as f:
            w = f["w"][:]
            addr = f["addr"][:]
            I = f["I"][:]
            fic = f["fic"][:]
            Imodel = f["Imodel"][:]
        with h5py.File(self.datadir %name + "make_model_%04d.h5" %iter, "r") as f:
            aux = f["aux"][:]
        
        # Copy data to device
        aux_dev = gpuarray.to_gpu(aux)
        w_dev = gpuarray.to_gpu(w)
        addr_dev = gpuarray.to_gpu(addr)
        I_dev = gpuarray.to_gpu(I)
        fic_dev = gpuarray.to_gpu(fic)
        Imodel_dev = gpuarray.to_gpu(np.ascontiguousarray(Imodel))

        # CPU Kernel
        BGDK = BaseGradientDescentKernel(aux, addr.shape[1])
        BGDK.allocate()
        BGDK.npy.Imodel = Imodel
        BGDK.floating_intensity(addr, w, I, fic)

        # GPU kernel
        GDK = GradientDescentKernel(aux_dev, addr.shape[1])
        GDK.allocate()
        GDK.gpu.Imodel = Imodel_dev
        GDK.floating_intensity(addr_dev, w_dev, I_dev, fic_dev)

        ## Assert
        np.testing.assert_allclose(BGDK.npy.LLerr, GDK.gpu.LLerr.get(), atol=self.atol, rtol=self.rtol, 
            verbose=False, equal_nan=False,
            err_msg="`LLerr` buffer has not been updated as expected")
        np.testing.assert_allclose(BGDK.npy.LLden, GDK.gpu.LLden.get(), atol=self.atol, rtol=self.rtol, 
            verbose=False, equal_nan=False,
            err_msg="`LLden` buffer has not been updated as expected")
        np.testing.assert_allclose(BGDK.npy.fic_tmp, GDK.gpu.fic_tmp.get(), atol=self.atol, rtol=self.rtol, 
            verbose=False, equal_nan=False,
            err_msg="`fic_tmp` buffer has not been updated as expected")

        np.testing.assert_allclose(fic, fic_dev.get(), atol=self.atol, rtol=self.rtol, 
            verbose=False, equal_nan=False, 
            err_msg="floating intensity coeff (fic) has not been updated as expected")

        np.testing.assert_allclose(BGDK.npy.Imodel, GDK.gpu.Imodel.get(), atol=self.atol, rtol=self.rtol, 
            verbose=False, equal_nan=False,
            err_msg="`Imodel` buffer has not been updated as expected")
        

    @parameterized.expand([
        ["base", 10],
        ["regul", 50],
        ["floating", 0],
    ])
    def test_main_and_error_reduce_UNITY(self, name, iter):

        # Load data
        with h5py.File(self.datadir %name + "main_%04d.h5" %iter, "r") as f:
            aux = f["aux"][:]
            addr = f["addr"][:]
            w = f["w"][:]
            I = f["I"][:]
        # Load data
        with h5py.File(self.datadir %name + "error_reduce_%04d.h5" %iter, "r") as f:
            err_phot = f["err_phot"][:]

        # Copy data to device
        aux_dev = gpuarray.to_gpu(aux)
        w_dev = gpuarray.to_gpu(w)
        addr_dev = gpuarray.to_gpu(addr)
        I_dev = gpuarray.to_gpu(I)
        err_phot_dev = gpuarray.to_gpu(err_phot)

        # CPU Kernel
        BGDK = BaseGradientDescentKernel(aux, addr.shape[1])
        BGDK.allocate()
        BGDK.main(aux, addr, w, I)
        BGDK.error_reduce(addr, err_phot)

        # GPU kernel
        GDK = GradientDescentKernel(aux_dev, addr.shape[1])
        GDK.allocate()
        GDK.main(aux_dev, addr_dev, w_dev, I_dev)
        GDK.error_reduce(addr_dev, err_phot_dev)

        ## Assert
        np.testing.assert_allclose(aux, aux_dev.get(), atol=self.atol, rtol=self.rtol, 
            err_msg="Auxiliary has not been updated as expected")
        np.testing.assert_allclose(BGDK.npy.LLerr, GDK.gpu.LLerr.get(), atol=self.atol, rtol=self.rtol, 
            err_msg="LogLikelihood error has not been updated as expected")
        np.testing.assert_allclose(err_phot, err_phot_dev.get(), atol=self.atol, rtol=self.rtol, 
            err_msg="`err_phot` has not been updated as expected")

    @parameterized.expand([
        ["base", 10],
        ["regul", 50],
        ["floating", 0],
    ])
    def test_make_a012_UNITY(self, name, iter):

        # Reduce the array size to make the tests run faster
        Nmax = 10 
        Ymax = 128
        Xmax = 128

        # Load data
        with h5py.File(self.datadir %name + "make_a012_%04d.h5" %iter, "r") as g:
            addr = g["addr"][:Nmax]
            I = g["I"][:Nmax,:Ymax,:Xmax]
            f = g["f"][:Nmax,:Ymax,:Xmax]
            a = g["a"][:Nmax,:Ymax,:Xmax]
            b = g["b"][:Nmax,:Ymax,:Xmax]
            fic = g["fic"][:Nmax]
        with h5py.File(self.datadir %name + "make_model_%04d.h5" %iter, "r") as h:
            aux = h["aux"][:Nmax,:Ymax,:Xmax]

        # Copy data to device
        aux_dev = gpuarray.to_gpu(aux)
        addr_dev = gpuarray.to_gpu(addr)
        I_dev = gpuarray.to_gpu(I)
        f_dev = gpuarray.to_gpu(f)
        a_dev = gpuarray.to_gpu(a)
        b_dev = gpuarray.to_gpu(b)
        fic_dev = gpuarray.to_gpu(fic)
        
        # CPU Kernel
        BGDK = BaseGradientDescentKernel(aux, addr.shape[1])
        BGDK.allocate()
        BGDK.make_a012(f, a, b, addr, I, fic)

        # GPU kernel        
        GDK = GradientDescentKernel(aux_dev, addr.shape[1], queue=self.stream)
        GDK.allocate()
        GDK.gpu.Imodel.fill(np.nan)
        GDK.gpu.LLerr.fill(np.nan)
        GDK.gpu.LLden.fill(np.nan)
        GDK.make_a012(f_dev, a_dev, b_dev, addr_dev, I_dev, fic_dev)

        ## Assert
        np.testing.assert_allclose(GDK.gpu.Imodel.get(), BGDK.npy.Imodel, atol=self.atol, rtol=self.rtol, 
            err_msg="Imodel error has not been updated as expected")
        np.testing.assert_allclose(GDK.gpu.LLerr.get(), BGDK.npy.LLerr, atol=self.atol, rtol=self.rtol, 
            err_msg="LLerr error has not been updated as expected")
        np.testing.assert_allclose(GDK.gpu.LLden.get(), BGDK.npy.LLden, atol=self.atol, rtol=self.rtol, 
            err_msg="LLden error has not been updated as expected")

    @parameterized.expand([
        ["base", 10],
        ["regul", 50],
        ["floating", 0],
    ])
    def test_fill_b_UNITY(self, name, iter):

        Nmax = 10
        Ymax = 128
        Xmax = 128

        # Load data
        with h5py.File(self.datadir %name + "fill_b_%04d.h5" %iter, "r") as f:
            w = f["w"][:Nmax, :Ymax, :Xmax]
            addr = f["addr"][:]
            B = f["B"][:]
            Brenorm = f["Brenorm"][...]
            A0 = f["A0"][:Nmax, :Ymax, :Xmax]
            A1 = f["A1"][:Nmax, :Ymax, :Xmax]
            A2 = f["A2"][:Nmax, :Ymax, :Xmax]
        with h5py.File(self.datadir %name + "make_model_%04d.h5" %iter, "r") as f:
            aux = f["aux"][:Nmax, :Ymax, :Xmax]

        # Copy data to device
        aux_dev = gpuarray.to_gpu(aux)
        w_dev = gpuarray.to_gpu(w)
        addr_dev = gpuarray.to_gpu(addr)
        B_dev = gpuarray.to_gpu(B.astype(np.float32))
        A0_dev = gpuarray.to_gpu(A0)
        A1_dev = gpuarray.to_gpu(A1)
        A2_dev = gpuarray.to_gpu(A2)

        # CPU Kernel
        BGDK = BaseGradientDescentKernel(aux, addr.shape[1])
        BGDK.allocate()
        BGDK.npy.Imodel = A0
        BGDK.npy.LLerr = A1
        BGDK.npy.LLden = A2
        BGDK.fill_b(addr, Brenorm, w, B)

        # GPU kernel
        GDK = GradientDescentKernel(aux_dev, addr.shape[1])
        GDK.allocate()
        GDK.gpu.Imodel = A0_dev
        GDK.gpu.LLerr = A1_dev
        GDK.gpu.LLden = A2_dev
        GDK.fill_b(addr_dev, Brenorm, w_dev, B_dev)

        ## Assert
        np.testing.assert_allclose(B, B_dev.get(), rtol=self.rtol, atol=self.atol, 
            err_msg="`B` has not been updated as expected")

