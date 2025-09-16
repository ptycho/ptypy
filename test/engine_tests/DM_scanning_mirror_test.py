"""
Test for the DM_scanning_mirror engine.
"""
import unittest
import tempfile
import shutil

import numpy as np

from test import utils as tu
from ptypy import utils as u
from ptypy.utils import parallel
from ptypy.core import Ptycho


from ptypy.custom import DM_scanning_mirror
from ptypy.experiment import scanning_mirror_sim


def _EngineTestRunner(engine_params, output_path='./', output_file=None,
                    autosave=True, verbose_level="info", init_correct_probe=False,
                    frames_per_block=100000):
    p = u.param_from_json(
        'test/scanning_mirror_test_data/scanning_mirror_test_config.json')
    p.frames_per_block = frames_per_block
    p.verbose_level = verbose_level
    p.io = u.Param()
    p.io.home = output_path
    p.io.rfile = "%s.ptyr" % output_file
    p.io.interaction = u.Param()
    p.io.interaction.active = False
    p.io.autosave = u.Param(active=autosave)
    p.io.autoplot = u.Param(active=False)
    p.scans.scan00.data = u.Param()
    p.scans.scan00.data.name = 'ScanningMirrorMoonFlower'
    p.scans.scan00.data.shift_pos_factor = 0.0  # make zero
    p.scans.scan00.data.num_frames = 10
    p.scans.scan00.data.save = None
    p.scans.scan00.data.density = 0.05
    p.scans.scan00.data.photons = 1e10
    p.scans.scan00.data.add_poisson_noise = False
    p.scans.scan00.data.psf = 0
    p.engines = u.Param()
    p.engines.engine00 = engine_params
    p.engines.engine00.numiter = 4

    p.engines.engine00.position_refinement = u.Param()
    p.engines.engine00.position_refinement.method = "GridSearch"
    p.engines.engine00.position_refinement.start = 1
    p.engines.engine00.position_refinement.stop = 200
    p.engines.engine00.position_refinement.interval = 10
    p.engines.engine00.position_refinement.nshifts = 8
    p.engines.engine00.position_refinement.amplitude = 50.0e-9
    p.engines.engine00.position_refinement.max_shift = 100.0e-9
    p.engines.engine00.position_refinement.record = False
    P = Ptycho(p, level=4)

    P.run()

    # important for subdividing data, ensure a fresh start if a test will be
    # run afterwards
    parallel.loadmanager.reset()

    return P


class DMScanningMirror_Test(unittest.TestCase):

    def setUp(self):
        self.outpath = tempfile.mkdtemp(suffix="DM_scanning_mirror_test")

    def tearDown(self):
        shutil.rmtree(self.outpath)

    def test_DM_scanning_mirror_position_refinement(self):
        engine_params = u.Param()
        engine_params.name = 'DM_scanning_mirror'
        engine_params.alpha =1
        engine_params.probe_update_start = 2
        engine_params.overlap_converge_factor = 0.05
        engine_params.overlap_max_iterations = 10
        engine_params.probe_inertia = 1e-3
        engine_params.object_inertia = 0.1
        engine_params.fourier_relax_factor = 0.01
        engine_params.obj_smooth_std = 20
        engine_params.position_refinement = True
        _EngineTestRunner(engine_params, output_path=self.outpath)

    def test_DM_scnning_mirror(self):
        engine_params = u.Param()
        engine_params.name = 'DM_scanning_mirror'
        engine_params.alpha =1
        engine_params.probe_update_start = 2
        engine_params.overlap_converge_factor = 0.05
        engine_params.overlap_max_iterations = 10
        engine_params.probe_inertia = 1e-3
        engine_params.object_inertia = 0.1
        engine_params.fourier_relax_factor = 0.01
        engine_params.obj_smooth_std = 20
        _EngineTestRunner(engine_params, output_path=self.outpath)

    def test_reconstruction_equvalence(self):
        """
        Run engine with
        Returns shift_pos_factors = 0. It should be equivalent to the standard
        propagator case.
        """
        # Start reconstruction with DM_scanning_mirror and
        # ScanningMirrorMoonFlower, with shift_pos_factor = 0.0.
        p = u.param_from_json(
            'test/scanning_mirror_test_data/scanning_mirror_test_config.json')
        p.scans.scan00.data = u.Param()
        p.scans.scan00.data.name = 'ScanningMirrorMoonFlower'
        p.scans.scan00.data.shift_pos_factor = 0.0
        p.scans.scan00.data.num_frames = 5
        p.scans.scan00.data.save = None
        p.scans.scan00.data.density = 0.01
        p.scans.scan00.data.photons = 1e10
        p.scans.scan00.data.add_poisson_noise = False
        p.scans.scan00.data.psf = 0
        p.engines = u.Param()
        p.engines.engine00 = u.Param()
        p.engines.engine00.name = "DM_scanning_mirror"
        p.engines.engine00.numiter = 4
        p.engines.engine00.numiter_contiguous = 2

        p.engines.engine00.position_refinement = u.Param()
        p.engines.engine00.position_refinement.method = "GridSearch"
        p.engines.engine00.position_refinement.start = 1
        p.engines.engine00.position_refinement.stop = 200
        p.engines.engine00.position_refinement.interval = 10
        p.engines.engine00.position_refinement.nshifts = 8
        p.engines.engine00.position_refinement.amplitude = 50.0e-9
        p.engines.engine00.position_refinement.max_shift = 100.0e-9
        p.engines.engine00.position_refinement.record = False
        P = Ptycho(p, level=5)
        obj = P.obj.storages['Sscan00G00'].data[0, :, :]
        probe = P.probe.storages['Sscan00G00'].data[0, :, :]

        # Replace scanning mirror related with standard and run again.
        p.scans.scan00.data.name = 'MoonFlowerScan'
        p.scans.scan00.name = 'Full'
        p.engines.engine00.name = "DM"
        p.scans.scan00.data.pop('shift_pos_factor')
        P_basic = Ptycho(p, level=5)
        for key, storage in P.obj.storages.items():
            storage_basic = P_basic.obj.storages[key]
            np.testing.assert_array_almost_equal(
                storage_basic.origin, storage.origin)

        for key, view in P.obj.views.items():
            view_basic = P_basic.obj.views[key]
            np.testing.assert_array_almost_equal(
                view_basic.psize, view.psize)
            np.testing.assert_array_almost_equal(
                view_basic.coord, view.coord)
            np.testing.assert_array_almost_equal(
                view_basic.pcoord, view.pcoord)
            np.testing.assert_array_almost_equal(
                view_basic.dcoord, view.dcoord)
            np.testing.assert_array_almost_equal(
                view_basic.dlow, view.dlow)
            np.testing.assert_array_almost_equal(
                view_basic.dhigh, view.dhigh)

        # asserts
        obj_basic = P_basic.obj.storages['Sscan00G00'].data[0, :, :]
        probe_basic = P_basic.probe.storages['Sscan00G00'].data[0, :, :]
        np.testing.assert_array_almost_equal(obj, obj_basic, decimal=2)
        np.testing.assert_array_almost_equal(probe, probe_basic, decimal=2)

        # important for subdividing data, ensure a fresh start if a test will be
        # run afterwards
        parallel.loadmanager.reset()



if __name__ == "__main__":
    unittest.main()
