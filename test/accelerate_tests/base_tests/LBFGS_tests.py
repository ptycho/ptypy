"""
Test for the LBFGS engine.

This file is part of the PTYPY package.
    :copyright: Copyright 2014 by the PTYPY team, see AUTHORS.
    :license: GPLv2, see LICENSE for details.
"""
import unittest

from test import utils as tu
from ptypy import utils as u
import ptypy
from ptypy.custom import LBFGS, LBFGS_serial
import tempfile
import shutil
import numpy as np

class LBFGSSerialTest(unittest.TestCase):

    def setUp(self):
        self.outpath = tempfile.mkdtemp(suffix="LBFGS_serial_test")

    def tearDown(self):
        shutil.rmtree(self.outpath)

    def check_engine_output(self, output, plotting=False, debug=False):
        P_LBFGS, P_LBFGS_serial = output
        numiter = len(P_LBFGS.runtime["iter_info"])
        LL_LBFGS = np.array([P_LBFGS.runtime["iter_info"][i]["error"][1] for i in range(numiter)])
        LL_LBFGS_serial = np.array([P_LBFGS_serial.runtime["iter_info"][i]["error"][1] for i in range(numiter)])
        crop = 42
        OBJ_LBFGS_serial, OBJ_LBFGS = P_LBFGS_serial.obj.S["SMFG00"].data[0,crop:-crop,crop:-crop], P_LBFGS.obj.S["SMFG00"].data[0,crop:-crop,crop:-crop]
        PRB_LBFGS_serial, PRB_LBFGS = P_LBFGS_serial.probe.S["SMFG00"].data[0], P_LBFGS.probe.S["SMFG00"].data[0]
        eng_LBFGS = P_LBFGS.engines["engine00"]
        eng_LBFGS_serial = P_LBFGS_serial.engines["engine00"]
        if debug:
            import matplotlib.pyplot as plt
            plt.figure("LBFGS debug")
            plt.imshow(np.abs(eng_LBFGS.debug))
            plt.figure("LBFGS serial debug")
            plt.imshow(np.abs(eng_LBFGS_serial.debug))
            plt.show()

        if plotting:
            import matplotlib.pyplot as plt
            plt.figure("Errors")
            plt.plot(LL_LBFGS, label="LBFGS")
            plt.plot(LL_LBFGS_serial, label="LBFGS_serial")
            plt.legend()
            plt.show()
            plt.figure("Phase LBFGS")
            plt.imshow(np.angle(OBJ_LBFGS))
            plt.figure("Ampltitude LBFGS")
            plt.imshow(np.abs(OBJ_LBFGS))
            plt.figure("Phase LBFGS serial")
            plt.imshow(np.angle(OBJ_LBFGS_serial))
            plt.figure("Amplitude LBFGS serial")
            plt.imshow(np.abs(OBJ_LBFGS_serial))
            plt.figure("Phase difference")
            plt.imshow(np.angle(OBJ_LBFGS_serial) - np.angle(OBJ_LBFGS), vmin=-0.1, vmax=0.1)
            plt.colorbar()
            plt.figure("Amplitude difference")
            plt.imshow(np.abs(OBJ_LBFGS_serial) - np.abs(OBJ_LBFGS), vmin=-0.1, vmax=0.1)
            plt.colorbar()
            plt.show()
        # np.testing.assert_allclose(eng_LBFGS.debug, eng_LBFGS_serial.debug, atol=1e-7, rtol=1e-7,
        #                             err_msg="The debug arrays are not matching as expected")
        RMSE_ob = (np.mean(np.abs(OBJ_LBFGS_serial - OBJ_LBFGS)**2))
        RMSE_pr = (np.mean(np.abs(PRB_LBFGS_serial - PRB_LBFGS)**2))
        # RMSE_LL = (np.mean(np.abs(LL_LBFGS_serial - LL_LBFGS)**2))
        np.testing.assert_allclose(RMSE_ob, 0.0, atol=1e-1,
                                    err_msg="The object arrays are not matching as expected")
        np.testing.assert_allclose(RMSE_pr, 0.0, atol=1e-1,
                                    err_msg="The probe arrays are not matching as expected")
        # np.testing.assert_allclose(RMSE_LL, 0.0, atol=1e-7,
                                    # err_msg="The log-likelihood errors are not matching as expected")


    def test_LBFGS_serial_base(self):
        out = []
        for eng in ["LBFGS", "LBFGS_serial"]:
            engine_params = u.Param()
            engine_params.name = eng
            engine_params.numiter = 100
            engine_params.floating_intensities = False
            engine_params.reg_del2 = False
            engine_params.reg_del2_amplitude = 1.
            engine_params.scale_precond = False
            out.append(tu.EngineTestRunner(engine_params, output_path=self.outpath, init_correct_probe=True,
                                           scanmodel="BlockFull", autosave=False, verbose_level="critical"))
        self.check_engine_output(out, plotting=False, debug=False)

    def test_LBFGS_serial_regularizer(self):
        out = []
        for eng in ["LBFGS", "LBFGS_serial"]:
            engine_params = u.Param()
            engine_params.name = eng
            engine_params.numiter = 90
            engine_params.floating_intensities = False
            engine_params.reg_del2 = True
            engine_params.reg_del2_amplitude = 1.
            engine_params.scale_precond = False
            out.append(tu.EngineTestRunner(engine_params, output_path=self.outpath, init_correct_probe=True,
                                           scanmodel="BlockFull", autosave=False, verbose_level="critical"))
        self.check_engine_output(out, plotting=False, debug=False)

    def test_LBFGS_serial_preconditioner(self):
        out = []
        for eng in ["LBFGS", "LBFGS_serial"]:
            engine_params = u.Param()
            engine_params.name = eng
            engine_params.numiter = 100
            engine_params.floating_intensities = False
            engine_params.reg_del2 = False
            engine_params.reg_del2_amplitude = 1.
            engine_params.scale_precond = True
            engine_params.scale_probe_object = 1e-6
            out.append(tu.EngineTestRunner(engine_params, output_path=self.outpath, init_correct_probe=True,
                                           scanmodel="BlockFull", autosave=False, verbose_level="critical"))
        self.check_engine_output(out, plotting=False, debug=False)

    def test_LBFGS_serial_floating(self):
        out = []
        # fail at iter num 80
        for eng in ["LBFGS", "LBFGS_serial"]:
            engine_params = u.Param()
            engine_params.name = eng
            engine_params.numiter = 100
            engine_params.floating_intensities = True
            engine_params.reg_del2 = False
            engine_params.reg_del2_amplitude = 1.
            engine_params.scale_precond = False
            out.append(tu.EngineTestRunner(engine_params, output_path=self.outpath, init_correct_probe=True,
                                           scanmodel="BlockFull", autosave=False, verbose_level="critical"))
        self.check_engine_output(out, plotting=False, debug=False)

    def test_LBFGS_serial_smoothing_regularizer(self):
        out = []
        for eng in ["LBFGS", "LBFGS_serial"]:
            engine_params = u.Param()
            engine_params.name = eng
            engine_params.numiter = 100
            engine_params.floating_intensities = False
            engine_params.reg_del2 = False
            engine_params.reg_del2_amplitude = 1.
            engine_params.smooth_gradient = 20
            engine_params.smooth_gradient_decay = 1/10.
            engine_params.scale_precond = False
            out.append(tu.EngineTestRunner(engine_params, output_path=self.outpath, init_correct_probe=True,
                                           scanmodel="BlockFull", autosave=False, verbose_level="critical"))
        self.check_engine_output(out, plotting=False, debug=False)

    def test_LBFGS_serial_all(self):
        out = []
        for eng in ["LBFGS", "LBFGS_serial"]:
            engine_params = u.Param()
            engine_params.name = eng
            engine_params.numiter = 100
            engine_params.floating_intensities = False
            engine_params.reg_del2 = True
            engine_params.reg_del2_amplitude = 1.
            engine_params.smooth_gradient = 20
            engine_params.smooth_gradient_decay = 1/10.
            engine_params.scale_precond = True
            engine_params.scale_probe_object = 1e-6
            out.append(tu.EngineTestRunner(engine_params, output_path=self.outpath, init_correct_probe=True,
                                           scanmodel="BlockFull", autosave=False, verbose_level="critical"))
        self.check_engine_output(out, plotting=False, debug=False)

if __name__ == "__main__":
    unittest.main()
