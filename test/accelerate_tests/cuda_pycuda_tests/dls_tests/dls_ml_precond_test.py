'''
Testing on real data
'''

import h5py
import unittest
import numpy as np
from parameterized  import parameterized

class DlsPreconditionerTest(unittest.TestCase):

    datadir = "/dls/science/users/iat69393/gpu-hackathon/test-data-precond/"
    rtol = 1e-6
    atol = 1e-6

    @parameterized.expand([[i] for i in range(0,20,5)])
    @unittest.skip("")
    def test_object_probe_norm_serial(self, iter):

        # Load data for ML
        with h5py.File(self.datadir + "ml_o_p_norm_%04d.h5" %iter, "r") as f:
            cn2_new_pr_grad1 = f["cn2_new_pr_grad"][...]
            cn2_new_ob_grad1 = f["cn2_new_ob_grad"][...]

        # Load data for ML_serial
        with h5py.File(self.datadir + "ml_serial_o_p_norm_%04d.h5" %iter, "r") as f:
            cn2_new_pr_grad2 = f["cn2_new_pr_grad"][...]
            cn2_new_ob_grad2 = f["cn2_new_ob_grad"][...]

        ## Assert
        np.testing.assert_allclose(cn2_new_pr_grad1,cn2_new_pr_grad2,  atol=self.atol, rtol=self.rtol, 
            err_msg="The probe gradient norm for ML_serial is not the same as for ML")
        np.testing.assert_allclose(cn2_new_ob_grad1,cn2_new_ob_grad2,  atol=self.atol, rtol=self.rtol, 
            err_msg="The object gradient norm for ML_serial is not the same as for ML")

    @unittest.skip("need to first dump mlpycuda output")
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

    @parameterized.expand([[i] for i in range(0,5,5)])
    def test_object_gradient_mlserial(self, iter):

        # Load data for ML
        with h5py.File(self.datadir + "ml_grad_%04d.h5" %iter, "r") as f:
            pr_grad1 = f["pr_grad"][:]
            ob_grad1 = f["ob_grad"][:]
        print(pr_grad1.min(), pr_grad1.max())
        print(ob_grad1.min(), ob_grad1.max())

        # Load data for ML_serial
        with h5py.File(self.datadir + "ml_serial_grad_%04d.h5" %iter, "r") as f:
            pr_grad2 = f["pr_grad"][:]
            ob_grad2 = f["ob_grad"][:]
        print(pr_grad2.min(), pr_grad2.max())
        print(ob_grad2.min(), ob_grad2.max())
        print(1/0)
        ## Assert
        np.testing.assert_allclose(pr_grad1,pr_grad2,  atol=self.atol, rtol=self.rtol, 
            err_msg="The probe gradient ML_serial is not the same as for ML")
        np.testing.assert_allclose(ob_grad1,ob_grad2,  atol=self.atol, rtol=self.rtol, 
            err_msg="The object gradient ML_serial is not the same as for ML")