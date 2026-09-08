"""
Test for the ML engine.

This file is part of the PTYPY package.
    :copyright: Copyright 2014 by the PTYPY team, see AUTHORS.
    :license: see LICENSE for details.
"""
import unittest

from ptypy.experiment.p06 import P06Scan_scanning_mirror
from test import utils as tu
from ptypy import utils as u
import ptypy
ptypy.load_gpu_engines("serial")
import tempfile
import shutil
import numpy as np
import pytest

from ptypy.accelerate.base.engines.projectional_serial_scanning_mirror import DM_scanning_mirror_serial

class DMSerialTest(unittest.TestCase):

    def setUp(self):
        self.outpath = tempfile.mkdtemp(suffix="ML_serial_test")

    def tearDown(self):
        shutil.rmtree(self.outpath)

    def check_engine_output(self, output, plotting=False, debug=False, tol=0.1):
        P_DM, P_DM_serial = output
        numiter = len(P_DM.runtime["iter_info"])
        LL_ML = np.array([P_DM.runtime["iter_info"][i]["error"][1] for i in range(numiter)])
        LL_ML_serial = np.array([P_DM_serial.runtime["iter_info"][i]["error"][1] for i in range(numiter)])
        crop = 42
        OBJ_ML_serial, OBJ_ML = P_DM_serial.obj.S["SMFG00"].data[0,crop:-crop,crop:-crop], P_DM.obj.S["SMFG00"].data[0,crop:-crop,crop:-crop]
        PRB_ML_serial, PRB_ML = P_DM_serial.probe.S["SMFG00"].data[0], P_DM.probe.S["SMFG00"].data[0]
        eng_ML = P_DM.engines["engine00"]
        eng_ML_serial = P_DM_serial.engines["engine00"]
        # Normalize the outputs
        PRB_ML_max = np.abs(PRB_ML).max()
        PRB_ML_serial_max = np.abs(PRB_ML_serial).max()
        OBJ_ML_serial *= (PRB_ML_serial_max / PRB_ML_max)
        PRB_ML_serial /= (PRB_ML_serial_max / PRB_ML_max)
        if debug:
            import matplotlib.pyplot as plt
            plt.figure("ML debug")
            plt.imshow(np.abs(eng_ML.debug))
            plt.figure("ML serial debug")
            plt.imshow(np.abs(eng_ML_serial.debug))
            plt.show()

        if plotting:
            import matplotlib.pyplot as plt
            plt.figure("Errors")
            plt.plot(LL_ML, label="ML")
            plt.plot(LL_ML_serial, label="ML_serial")
            plt.legend()
            plt.show()
            plt.figure("Phase ML")
            plt.imshow(np.angle(OBJ_ML))
            plt.figure("Ampltitude ML")
            plt.imshow(np.abs(OBJ_ML))
            plt.figure("Phase ML serial")
            plt.imshow(np.angle(OBJ_ML_serial))
            plt.figure("Amplitude ML serial")
            plt.imshow(np.abs(OBJ_ML_serial))
            plt.figure("Phase difference")
            plt.imshow(np.angle(OBJ_ML_serial) - np.angle(OBJ_ML), vmin=-0.1, vmax=0.1)
            plt.colorbar()
            plt.figure("Amplitude difference")
            plt.imshow(np.abs(OBJ_ML_serial) - np.abs(OBJ_ML), vmin=-0.1, vmax=0.1)
            plt.colorbar()
            plt.show()
        # np.testing.assert_allclose(eng_ML.debug, eng_ML_serial.debug, atol=1e-7, rtol=1e-7,
        #                             err_msg="The debug arrays are not matching as expected")
        RMSE_ob = (np.mean(np.abs(OBJ_ML_serial - OBJ_ML)**2))
        RMSE_pr = (np.mean(np.abs(PRB_ML_serial - PRB_ML)**2))        
        #MSE_LL = (np.mean(np.abs(LL_ML_serial - LL_ML)**2))
        np.testing.assert_allclose(RMSE_ob, 0.0, atol=tol, 
                                    err_msg="The object arrays are not matching as expected")
        np.testing.assert_allclose(RMSE_pr, 0.0, atol=tol, 
                                    err_msg="The probe arrays are not matching as expected")
        #p.testing.assert_allclose(RMSE_LL, 0.0, atol=1e-7,
        #                           err_msg="The log-likelihood errors are not matching as expected")


    def test_DM_serial_base(self):
        out = []
        for eng in ["DM_scanning_mirror_serial"]:
            engine_params = u.Param()
            engine_params.name = eng
            engine_params.numiter = 100
            out.append(tu.EngineTestRunner(engine_params, output_path=self.outpath, init_correct_probe=True,
                                           scanmodel="BlockFull", autosave=False, verbose_level="critical", frames_per_block=100))
        self.check_engine_output(out, plotting=False, debug=False)


if __name__ == "__main__":
    unittest.main()
