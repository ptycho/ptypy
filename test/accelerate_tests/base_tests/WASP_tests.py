"""
Test for the WASP engine.

This file is part of the PTYPY package.
    :copyright: Copyright 2014 by the PTYPY team, see AUTHORS.
    :license: see LICENSE for details.
"""
import tempfile
import shutil
import unittest

import numpy as np

from test import utils as tu
from ptypy import utils as u
from ptypy.custom import WASP, WASP_serial
from ptypy.utils import parallel


class WASPSerialTest(unittest.TestCase):

    def setUp(self):
        self.outpath = tempfile.mkdtemp(suffix="WASP_serial_test")

    def tearDown(self):
        shutil.rmtree(self.outpath)

    def check_engine_output(self, output, plotting=False, debug=False):
        P_WASP, P_WASP_serial = output
        numiter = len(P_WASP.runtime["iter_info"])
        LL_WASP = np.array([P_WASP.runtime["iter_info"][i]["error"][1] for i in range(numiter)])
        LL_WASP_serial = np.array([P_WASP_serial.runtime["iter_info"][i]["error"][1] for i in range(numiter)])
        crop = 42
        OBJ_WASP_serial, OBJ_WASP = P_WASP_serial.obj.S["SMFG00"].data[0,crop:-crop,crop:-crop], P_WASP.obj.S["SMFG00"].data[0,crop:-crop,crop:-crop]
        PRB_WASP_serial, PRB_WASP = P_WASP_serial.probe.S["SMFG00"].data[0], P_WASP.probe.S["SMFG00"].data[0]
        eng_WASP = P_WASP.engines["engine00"]
        eng_WASP_serial = P_WASP_serial.engines["engine00"]
        if debug:
            import matplotlib.pyplot as plt
            plt.figure("WASP debug")
            plt.imshow(np.abs(eng_WASP.debug))
            plt.figure("WASP serial debug")
            plt.imshow(np.abs(eng_WASP_serial.debug))
            plt.show()

        if plotting:
            import matplotlib.pyplot as plt
            plt.figure("Errors")
            plt.plot(LL_WASP, label="WASP")
            plt.plot(LL_WASP_serial, label="WASP_serial")
            plt.legend()
            plt.show()
            plt.figure("Phase WASP")
            plt.imshow(np.angle(OBJ_WASP))
            plt.figure("Ampltitude WASP")
            plt.imshow(np.abs(OBJ_WASP))
            plt.figure("Phase WASP serial")
            plt.imshow(np.angle(OBJ_WASP_serial))
            plt.figure("Amplitude WASP serial")
            plt.imshow(np.abs(OBJ_WASP_serial))
            plt.figure("Phase difference")
            plt.imshow(np.angle(OBJ_WASP_serial) - np.angle(OBJ_WASP), vmin=-0.1, vmax=0.1)
            plt.colorbar()
            plt.figure("Amplitude difference")
            plt.imshow(np.abs(OBJ_WASP_serial) - np.abs(OBJ_WASP), vmin=-0.1, vmax=0.1)
            plt.colorbar()
            plt.show()

            plt.figure("Phase WASP")
            plt.imshow(np.angle(PRB_WASP))
            plt.figure("Ampltitude WASP")
            plt.imshow(np.abs(PRB_WASP))
            plt.figure("Phase WASP serial")
            plt.imshow(np.angle(PRB_WASP_serial))
            plt.figure("Amplitude WASP serial")
            plt.imshow(np.abs(PRB_WASP_serial))
            plt.figure("Phase difference")
            plt.imshow(np.angle(PRB_WASP_serial) - np.angle(PRB_WASP), vmin=-0.1, vmax=0.1)
            plt.colorbar()
            plt.figure("Amplitude difference")
            plt.imshow(np.abs(PRB_WASP_serial) - np.abs(PRB_WASP), vmin=-0.1, vmax=0.1)
            plt.colorbar()
            plt.show()

        RMSE_ob = (np.mean(np.abs(OBJ_WASP_serial - OBJ_WASP)**2))
        RMSE_pr = (np.mean(np.abs(PRB_WASP_serial - PRB_WASP)**2))
        np.testing.assert_allclose(RMSE_ob, 0.0, atol=1e-1,
                                    err_msg="The object arrays are not matching as expected")

        # the extremly high tolerance for probe is a result of precision
        # difference between the normal and serial version, which is
        # deliberate(?) to match GPU version
        np.testing.assert_allclose(RMSE_pr, 0.0, atol=1e3,
                                    err_msg="The probe arrays are not matching as expected")

    def test_WASP_serial_base(self):
        out = []
        for eng in ["WASP", "WASP_serial"]:
            engine_params = u.Param()
            engine_params.name = eng
            engine_params.numiter = 10
            engine_params.random_seed = 721
            out.append(tu.EngineTestRunner(engine_params, output_path=self.outpath, init_correct_probe=True,
                                           scanmodel="BlockFull", autosave=False, verbose_level="critical"))

        if parallel.master:
            self.check_engine_output(out, plotting=False, debug=False)

    def test_WASP_serial_clip(self):
        out = []
        for eng in ["WASP", "WASP_serial"]:
            engine_params = u.Param()
            engine_params.name = eng
            engine_params.numiter = 10
            engine_params.clip_object = (0, 2)
            engine_params.random_seed = 721
            out.append(tu.EngineTestRunner(engine_params, output_path=self.outpath, init_correct_probe=True,
                                           scanmodel="BlockFull", autosave=False, verbose_level="critical"))

        if parallel.master:
            self.check_engine_output(out, plotting=False, debug=False)

    def test_WASP_serial_alpha_beta(self):
        out = []
        for eng in ["WASP", "WASP_serial"]:
            engine_params = u.Param()
            engine_params.name = eng
            engine_params.numiter = 10
            engine_params.alpha = 0.64
            engine_params.beta = 0.94
            engine_params.random_seed = 721
            out.append(tu.EngineTestRunner(engine_params, output_path=self.outpath, init_correct_probe=True,
                                           scanmodel="BlockFull", autosave=False, verbose_level="critical"))

        if parallel.master:
            self.check_engine_output(out, plotting=False, debug=False)

    def test_WASP_serial_all(self):
        out = []
        for eng in ["WASP", "WASP_serial"]:
            engine_params = u.Param()
            engine_params.name = eng
            engine_params.numiter = 10
            engine_params.clip_object = (0, 2)
            engine_params.alpha = 0.64
            engine_params.beta = 0.94
            engine_params.random_seed = 721
            out.append(tu.EngineTestRunner(engine_params, output_path=self.outpath, init_correct_probe=True,
                                           scanmodel="BlockFull", autosave=False, verbose_level="critical"))

        if parallel.master:
            self.check_engine_output(out, plotting=False, debug=False)

if __name__ == "__main__":
    unittest.main()
