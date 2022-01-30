"""
Test for the ML engine.

This file is part of the PTYPY package.
    :copyright: Copyright 2014 by the PTYPY team, see AUTHORS.
    :license: GPLv2, see LICENSE for details.
"""
import unittest
from test import utils as tu
from ptypy import utils as u
import ptypy
ptypy.load_gpu_engines("serial")
import tempfile
import shutil
import numpy as np

class MLSerialTest(unittest.TestCase):

    def setUp(self):
        self.outpath = tempfile.mkdtemp(suffix="ML_serial_test")

    def tearDown(self):
        shutil.rmtree(self.outpath)

    def check_engine_output(self, output, plotting=False):
        P_ML, P_ML_serial = output
        numiter = len(P_ML.runtime["iter_info"])
        LL_ML = np.array([P_ML.runtime["iter_info"][i]["error"][1] for i in range(numiter)])
        LL_ML_serial = np.array([P_ML_serial.runtime["iter_info"][i]["error"][1] for i in range(numiter)])
        crop = 82
        OBJ_ML_serial, OBJ_ML = P_ML_serial.obj.S["SMFG00"].data[0,crop:-crop,crop:-crop], P_ML.obj.S["SMFG00"].data[0,crop:-crop,crop:-crop]
        PRB_MK_serial, PRB_ML = P_ML_serial.probe.S["SMFG00"].data[0], P_ML.probe.S["SMFG00"].data[0]
        if plotting:
            import matplotlib.pyplot as plt
            plt.figure("Errors")
            plt.plot(LL_ML)
            plt.plot(LL_ML_serial)
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
            plt.imshow(np.angle(OBJ_ML_serial) - np.angle(OBJ_ML))
            plt.figure("Amplitude difference")
            plt.imshow(np.abs(OBJ_ML_serial) - np.abs(OBJ_ML))
            plt.show()
        np.testing.assert_allclose(OBJ_ML_serial, OBJ_ML, atol=0.1,
                                    err_msg="The object arrays are not matching as expected")
        np.testing.assert_allclose(PRB_MK_serial, PRB_ML, atol=0.1,
                                    err_msg="The object arrays are not matching as expected")
        np.testing.assert_allclose(LL_ML_serial, LL_ML, rtol=0.0001,
                                    err_msg="The log-likelihood errors are not matching as expected")
    

    def test_ML_serial_base(self):
        out = []
        for eng in ["ML", "ML_serial"]:
            engine_params = u.Param()
            engine_params.name = eng
            engine_params.numiter = 300
            engine_params.floating_intensities = False
            engine_params.reg_del2 = False
            engine_params.reg_del2_amplitude = 1.
            # engine_params.smooth_gradient = 20
            # engine_params.smooth_gradient_decay = 1/50.
            engine_params.scale_precond = False
            out.append(tu.EngineTestRunner(engine_params, output_path=self.outpath, init_correct_probe=True,
                                           scanmodel="BlockFull", autosave=False, verbose_level="critical"))
        self.check_engine_output(out, plotting=True)


if __name__ == "__main__":
    unittest.main()
