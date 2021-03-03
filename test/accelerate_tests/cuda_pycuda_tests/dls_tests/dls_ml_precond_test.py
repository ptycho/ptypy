'''
Testing on real data
'''

import h5py
import unittest
import numpy as np

class DlsPreconditionerTest(unittest.TestCase):

    datadir = "/dls/science/users/iat69393/gpu-hackathon/test-data-precond/"
    iter = 40
    rtol = 1e-6
    atol = 1e-6

    def test_object_probe_norm_serial(self):

        # Load data for ML
        with h5py.File(self.datadir + "ml_o_p_norm_%04d.h5" %self.iter, "r") as f:
            cn2_new_pr_grad1 = f["cn2_new_pr_grad"][...]
            cn2_new_ob_grad1 = f["cn2_new_ob_grad"][...]

        # Load data for ML_serial
        with h5py.File(self.datadir + "ml_serial_o_p_norm_%04d.h5" %self.iter, "r") as f:
            cn2_new_pr_grad2 = f["cn2_new_pr_grad"][...]
            cn2_new_ob_grad2 = f["cn2_new_ob_grad"][...]

        ## Assert
        np.testing.assert_allclose(cn2_new_pr_grad1,cn2_new_pr_grad2,  atol=self.atol, rtol=self.rtol, 
            err_msg="The probe gradient norm for ML_serial is not the same as for ML")
        np.testing.assert_allclose(cn2_new_ob_grad1,cn2_new_ob_grad2,  atol=self.atol, rtol=self.rtol, 
            err_msg="The object gradient norm for ML_serial is not the same as for ML")

    def test_object_probe_norm_pycuda(self):

        # Load data for ML
        with h5py.File(self.datadir + "ml_o_p_norm_%04d.h5" %self.iter, "r") as f:
            cn2_new_pr_grad1 = f["cn2_new_pr_grad"][...]
            cn2_new_ob_grad1 = f["cn2_new_ob_grad"][...]

        # Load data for ML_serial
        with h5py.File(self.datadir + "ml_pycuda_o_p_norm_%04d.h5" %self.iter, "r") as f:
            cn2_new_pr_grad2 = f["cn2_new_pr_grad"][...]
            cn2_new_ob_grad2 = f["cn2_new_ob_grad"][...]

        ## Assert
        np.testing.assert_allclose(cn2_new_pr_grad1,cn2_new_pr_grad2,  atol=self.atol, rtol=self.rtol, 
            err_msg="The probe gradient norm for ML_pycuda is not the same as for ML")
        np.testing.assert_allclose(cn2_new_ob_grad1,cn2_new_ob_grad2,  atol=self.atol, rtol=self.rtol, 
            err_msg="The object gradient norm for ML_pycuda is not the same as for ML")