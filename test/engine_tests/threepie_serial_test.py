"""
Tests of ptypy.custom.threepie_serial.ThreePIE_serial on the CPU.

Three properties are checked on the MoonFlower synthetic scan, with the same
seeded view order for every engine so that the comparisons do not depend on
the draw:

  A. number_of_slices=1  ->  reconstruction matches EPIE_serial
  B. number_of_slices=2  ->  the Fourier error decreases over the iterations
  C. 2-slice product object matches the pod/view CPU reference
     ptypy.custom.threepie.ThreePIE

This file is part of the PTYPY package.

    :copyright: Copyright 2014 by the PTYPY team, see AUTHORS.
    :license: see LICENSE for details.
"""
import os
import shutil
import tempfile
import unittest

import numpy as np

import ptypy
import ptypy.custom.threepie  # noqa: F401  (registers ThreePIE)
import ptypy.custom.threepie_serial  # noqa: F401  (registers ThreePIE_serial)
from ptypy import utils as u
from ptypy.core import Ptycho
from test.utils import seeded_view_order, ncorr, aligned_ncorr

NUMITER = 60
# Two slices need enough data to be determined. With 100 frames the two
# engines settle on visibly different solutions on some noise draws (their
# objects correlated between 0.66 and 0.95 over five seeds); with 200 they
# agree to 0.98 and better on every seed. Running longer does not help, which
# is what tells the two apart: 120 iterations on 100 frames still gave 0.70.
NFRAMES = 200
SHAPE = 64
THICK = 5e-7
SEED = 5


class ThreePIESerialTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        ptypy.load_gpu_engines("serial")
        cls.outdir = tempfile.mkdtemp(prefix="threepie_serial_test_")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.outdir, ignore_errors=True)

    def _params(self, engine_name, nslices):
        p = u.Param()
        p.verbose_level = "error"
        p.io = u.Param()
        p.io.home = self.outdir
        p.io.autosave = u.Param(active=False)
        p.io.interaction = u.Param(active=False)
        p.io.autoplot = u.Param(active=False)
        p.scans = u.Param()
        p.scans.MF = u.Param()
        p.scans.MF.name = "BlockFull"
        p.scans.MF.data = u.Param()
        p.scans.MF.data.name = "MoonFlowerScan"
        p.scans.MF.data.shape = SHAPE
        p.scans.MF.data.num_frames = NFRAMES
        p.scans.MF.data.density = 0.2
        p.scans.MF.data.photons = 1e8
        p.scans.MF.data.psf = 0.0
        p.scans.MF.data.save = None
        p.engines = u.Param()
        p.engines.e0 = u.Param()
        p.engines.e0.name = engine_name
        p.engines.e0.numiter = NUMITER
        p.engines.e0.probe_center_tol = None
        p.engines.e0.compute_log_likelihood = True
        if engine_name != "ThreePIE":
            p.engines.e0.compute_fourier_error = True
        p.engines.e0.object_norm_is_global = True
        p.engines.e0.alpha = 1
        p.engines.e0.beta = 1
        p.engines.e0.probe_update_start = 0
        if nslices is not None:
            p.engines.e0.number_of_slices = nslices
            p.engines.e0.slice_thickness = THICK
            p.engines.e0.fslices = os.path.join(
                self.outdir, "slices_%s_%d.h5" % (engine_name, nslices))
        return p

    def _run(self, engine_name, nslices):
        np.random.seed(SEED)
        with seeded_view_order(SEED):
            P = Ptycho(self._params(engine_name, nslices), level=5)
        ob = list(P.obj.storages.values())[0].data[0].copy()
        pr = list(P.probe.storages.values())[0].data[0].copy()
        return P, ob, pr

    def test_single_slice_matches_epie_serial(self):
        _, ob_e, pr_e = self._run("EPIE_serial", None)
        _, ob_1, pr_1 = self._run("ThreePIE_serial", 1)
        self.assertGreater(aligned_ncorr(pr_e, pr_1), 0.85)
        self.assertGreater(aligned_ncorr(ob_e, ob_1), 0.85)

    def test_two_slices_converge(self):
        P, _, _ = self._run("ThreePIE_serial", 2)
        errs = []
        for it in P.runtime["iter_info"]:
            e = np.asarray(it.get("error"), dtype=float).ravel()
            if e.size and e[0] > 0:
                errs.append(float(e[0]))
        self.assertGreater(len(errs), 2)
        self.assertLess(errs[-1] / errs[0], 0.8)

    def test_two_slices_match_cpu_reference(self):
        _, ob_r, pr_r = self._run("ThreePIE", 2)
        _, ob_s, pr_s = self._run("ThreePIE_serial", 2)
        # A ptychographic solution is fixed only up to a joint probe/object
        # shift, and the two engines settle on different ones, so the fields
        # are registered against each other before they are compared.
        self.assertGreater(aligned_ncorr(pr_r, pr_s), 0.8)
        self.assertGreater(aligned_ncorr(ob_r, ob_s), 0.8)


if __name__ == "__main__":
    unittest.main()
