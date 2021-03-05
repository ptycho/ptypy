'''
Testing based on real data
'''
import h5py
import unittest
import numpy as np
from parameterized import parameterized

class DLsFloatingIntensityTest(unittest.TestCase):

    datadir = "/dls/science/users/iat69393/gpu-hackathon/test-data-%s/"
    rtol = 1e-6
    atol = 1e-6

    @parameterized.expand([
        ["base", 0],
        ["floating", 0],
    ])
    def test_before_fic_model_update(self, name, iter=0):

        # Load data ML_serial
        with h5py.File(self.datadir %name + "ml_serial_before_floating_%04d.h5" %iter, "r") as f:
            fic1 = f["fic"][:]
            Imodel1 = f["Imodel"][:]
        print(Imodel1.shape)

        # Load data ML_pycuda
        with h5py.File(self.datadir %name + "ml_pycuda_before_floating_%04d.h5" %iter, "r") as f:
            fic2 = f["fic"][:]
            Imodel2 = f["Imodel"][:]
        print(Imodel2.shape)

        np.testing.assert_allclose(fic2, fic1, atol=self.atol, rtol=self.rtol, verbose=False, 
            err_msg = "The calculated floating intensities are not matching")
        np.testing.assert_allclose(Imodel2, Imodel1, atol=self.atol, rtol=self.rtol, verbose=False,
            err_msg = "The updated Imodel is not matching")

    @parameterized.expand([
        ["base", 0],
        ["floating", 0],
    ])
    def test_after_fic_model_update(self, name, iter=0):

        # Load data ML_serial
        with h5py.File(self.datadir %name + "ml_serial_after_floating_%04d.h5" %iter, "r") as f:
            fic1 = f["fic"][:]
            Imodel1 = f["Imodel"][:]
        print(Imodel1.shape)

        # Load data ML_pycuda
        with h5py.File(self.datadir %name + "ml_pycuda_after_floating_%04d.h5" %iter, "r") as f:
            fic2 = f["fic"][:]
            Imodel2 = f["Imodel"][:]
        print(Imodel2.shape)

        np.testing.assert_allclose(fic2, fic1, atol=self.atol, rtol=self.rtol, verbose=False, 
            err_msg = "The calculated floating intensities are not matching")
        np.testing.assert_allclose(Imodel2, Imodel1, atol=self.atol, rtol=self.rtol, verbose=False, 
            err_msg = "The updated Imodel is not matching")

    @parameterized.expand([
        ["base", 0],
        ["floating", 0],
    ])
    def test_after_error_reduce(self, name, iter=0):

        # Load data ML_serial
        with h5py.File(self.datadir %name + "ml_serial_after_error_reduce_%04d.h5" %iter, "r") as f:
            LLerr1 = f["LLerr"][:]
            err_phot1 = f["err_phot"][:]

        # Load data ML_pycuda
        with h5py.File(self.datadir %name + "ml_pycuda_after_error_reduce_%04d.h5" %iter, "r") as f:
            LLerr2 = f["LLerr"][:]
            err_phot2 = f["err_phot"][:]

        #np.testing.assert_allclose(LLerr1, LLerr2, atol=self.atol, rtol=self.rtol, verbose=False, 
        #    err_msg = "The LLerr arrays are not matching")
        np.testing.assert_allclose(err_phot2, err_phot1, atol=self.atol, rtol=self.rtol, verbose=False, 
            err_msg = "The reduced likeligood error is not matching")