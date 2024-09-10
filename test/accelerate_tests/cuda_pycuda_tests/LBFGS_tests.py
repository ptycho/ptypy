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
from ptypy.custom import LBFGS_serial, LBFGS_pycuda
import tempfile
import shutil
import numpy as np

class LBFGSPycudaTest(unittest.TestCase):

    def setUp(self):
        self.outpath = tempfile.mkdtemp(suffix="LBFGS_pycuda_test")

    def tearDown(self):
        shutil.rmtree(self.outpath)

    def check_engine_output(self, output, plotting=False, debug=False):
        P_LBFGS_serial, P_LBFGS_pycuda = output
        numiter = len(P_LBFGS_serial.runtime["iter_info"])
        LL_LBFGS_serial = np.array([P_LBFGS_serial.runtime["iter_info"][i]["error"][1] for i in range(numiter)])
        LL_LBFGS_pycuda = np.array([P_LBFGS_pycuda.runtime["iter_info"][i]["error"][1] for i in range(numiter)])
        crop = 42
        OBJ_LBFGS_pycuda, OBJ_LBFGS_serial = P_LBFGS_pycuda.obj.S["SMFG00"].data[0,crop:-crop,crop:-crop], P_LBFGS_serial.obj.S["SMFG00"].data[0,crop:-crop,crop:-crop]
        PRB_LBFGS_pycuda, PRB_LBFGS_serial = P_LBFGS_pycuda.probe.S["SMFG00"].data[0], P_LBFGS_serial.probe.S["SMFG00"].data[0]
        eng_LBFGS_serial = P_LBFGS_serial.engines["engine00"]
        eng_LBFGS_pycuda = P_LBFGS_pycuda.engines["engine00"]
        if debug:
            import matplotlib.pyplot as plt
            plt.figure("LBFGS serial debug")
            plt.imshow(np.abs(eng_LBFGS_serial.debug))
            plt.figure("LBFGS pycuda debug")
            plt.imshow(np.abs(eng_LBFGS_pycuda.debug))
            plt.show()

        if plotting:
            import matplotlib.pyplot as plt
            plt.figure("Errors")
            plt.plot(LL_LBFGS_serial, label="LBFGS_serial")
            plt.plot(LL_LBFGS_pycuda, label="LBFGS_pycuda")
            plt.legend()
            plt.show()
            plt.figure("Phase LBFGS serial")
            plt.imshow(np.angle(OBJ_LBFGS_serial))
            plt.figure("Ampltitude LBFGS serial")
            plt.imshow(np.abs(OBJ_LBFGS_serial))
            plt.figure("Phase LBFGS pycuda")
            plt.imshow(np.angle(OBJ_LBFGS_pycuda))
            plt.figure("Amplitude LBFGS pycuda")
            plt.imshow(np.abs(OBJ_LBFGS_pycuda))
            plt.figure("Phase difference")
            plt.imshow(np.angle(OBJ_LBFGS_pycuda) - np.angle(OBJ_LBFGS_serial), vmin=-0.1, vmax=0.1)
            plt.colorbar()
            plt.figure("Amplitude difference")
            plt.imshow(np.abs(OBJ_LBFGS_pycuda) - np.abs(OBJ_LBFGS_serial), vmin=-0.1, vmax=0.1)
            plt.colorbar()
            plt.show()
        # np.testing.assert_allclose(eng_LBFGS.debug, eng_LBFGS_serial.debug, atol=1e-7, rtol=1e-7,
        #                             err_msg="The debug arrays are not matching as expected")
        RMSE_ob = (np.mean(np.abs(OBJ_LBFGS_pycuda - OBJ_LBFGS_serial)**2))
        RMSE_pr = (np.mean(np.abs(PRB_LBFGS_pycuda - PRB_LBFGS_serial)**2))
        # RMSE_LL = (np.mean(np.abs(LL_LBFGS_pycuda - LL_LBFGS_serial)**2))
        np.testing.assert_allclose(RMSE_ob, 0.0, atol=1e-2,
                                    err_msg="The object arrays are not matching as expected")
        np.testing.assert_allclose(RMSE_pr, 0.0, atol=1e-2,
                                    err_msg="The probe arrays are not matching as expected")
        # np.testing.assert_allclose(RMSE_LL, 0.0, atol=1e-7,
                                    # err_msg="The log-likelihood errors are not matching as expected")


    def test_LBFGS_pycuda_base(self):
        out = []
        for eng in ["LBFGS_serial", "LBFGS_pycuda"]:
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

    def test_LBFGS_pycuda_regularizer(self):
        out = []
        for eng in ["LBFGS_serial", "LBFGS_pycuda"]:
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

    @unittest.skip("LBFGS may not work with preconditioner")
    def test_LBFGS_pycuda_preconditioner(self):
        out = []
        for eng in ["LBFGS_serial", "LBFGS_pycuda"]:
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

    @unittest.skip("LBFGS may not work with floating intensities")
    def test_LBFGS_pycuda_floating(self):
        out = []
        # fail at iter num 80
        for eng in ["LBFGS_serial", "LBFGS_pycuda"]:
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

    def test_LBFGS_pycuda_smoothing_regularizer(self):
        out = []
        for eng in ["LBFGS_serial", "LBFGS_pycuda"]:
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

    def test_LBFGS_pycuda_all(self):
        out = []
        for eng in ["LBFGS_serial", "LBFGS_pycuda"]:
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
