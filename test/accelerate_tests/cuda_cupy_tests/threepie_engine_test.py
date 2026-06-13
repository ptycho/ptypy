"""
End-to-end tests for the GPU multislice engine ``ThreePIE_cupy``.

The engine implements 3PIE (multi-slice ePIE) from

    A. M. Maiden, M. J. Humphry and J. M. Rodenburg,
    "Ptychographic transmission microscopy in three dimensions using a
     multi-slice approach", J. Opt. Soc. Am. A 29, 1606 (2012).

These tests run on a synthetic MoonFlower scan and check that

  * with a single slice the engine reduces to ordinary ePIE and agrees with
    the CPU-serialized ``EPIE_serial`` engine,
  * a multi-slice run reconstructs the sample (error drops substantially), and
  * the GPU multi-slice result agrees with the CPU reference engine
    ``ptypy.custom.threepie.ThreePIE`` within a tolerance that accounts for the
    stochastic view ordering of ePIE-type algorithms.

The whole module is skipped when cupy is not available.

This file is part of the PTYPY package.
    :copyright: Copyright 2014 by the PTYPY team, see AUTHORS.
    :license: see LICENSE for details.
"""
import os
import shutil
import tempfile
import unittest

import numpy as np

from . import have_cupy

if have_cupy():
    from test import utils as tu
    from ptypy import utils as u

    import ptypy
    ptypy.load_gpu_engines("cupy")        # registers ThreePIE_cupy, EPIE_cupy, ...
    import ptypy.custom.threepie          # registers the CPU reference ThreePIE


def _similarity(a, b, crop=30):
    """Normalised complex correlation |<a, b>| / (||a|| ||b||) of two images.

    This is invariant to the global phase and amplitude-scale ambiguity that
    ptychography leaves undetermined, and -- unlike a plain RMSE -- it is robust
    to the run-to-run variation of stochastic (ePIE-type) engines: two correct
    reconstructions of the same sample correlate at ~0.94, whereas an
    unrelated / broken reconstruction correlates near 0.

    A border is cropped because the object edges are weakly constrained (few or
    no probe visits there); ptypy's own engine tests crop similarly.
    """
    a = a[crop:-crop, crop:-crop].ravel()
    b = b[crop:-crop, crop:-crop].ravel()
    return np.abs(np.vdot(a, b)) / (np.linalg.norm(a) * np.linalg.norm(b))


@unittest.skipIf(not have_cupy(), "no cupy available")
class ThreePIECupyTest(unittest.TestCase):

    def setUp(self):
        self.outpath = tempfile.mkdtemp(suffix="ThreePIE_cupy_test")

    def tearDown(self):
        shutil.rmtree(self.outpath, ignore_errors=True)

    # -- helpers ---------------------------------------------------------
    def _run(self, name, numiter=100, slices=None, thickness=1e-7,
             start=0, scanmodel="BlockFull", fpb=100):
        ep = u.Param()
        ep.name = name
        ep.numiter = numiter
        ep.probe_update_start = 0
        if slices is not None:
            ep.number_of_slices = slices
            ep.slice_thickness = thickness
            ep.slice_start_iteration = start
            # keep the per-run slice dump out of the way
            ep.fslices = os.path.join(self.outpath, "%s_slices.h5" % name)
        return tu.EngineTestRunner(
            ep, output_path=self.outpath, output_file=name,
            init_correct_probe=True, scanmodel=scanmodel,
            autosave=False, verbose_level="critical", frames_per_block=fpb)

    @staticmethod
    def _summed_error(P):
        info = P.runtime["iter_info"]
        return np.array([np.sum(info[i]["error"]) for i in range(len(info))])

    @staticmethod
    def _obj(P, key="SMFG00"):
        return P.obj.S[key].data[0]

    @staticmethod
    def _probe(P, key="SMFG00"):
        return P.probe.S[key].data[0]

    # -- tests -----------------------------------------------------------
    def test_single_slice_reduces_to_epie(self):
        """ThreePIE_cupy with one slice must behave like ordinary ePIE."""
        P_ref = self._run("EPIE_serial", numiter=100)
        P_gpu = self._run("ThreePIE_cupy", numiter=100, slices=1)

        sim = _similarity(self._obj(P_ref), self._obj(P_gpu))
        self.assertGreater(sim, 0.85,
                           "1-slice ThreePIE_cupy disagrees with EPIE_serial "
                           "(correlation=%.3f)" % sim)

    def test_multislice_converges(self):
        """A two-slice run must reduce the error by a large factor."""
        P = self._run("ThreePIE_cupy", numiter=100, slices=2, thickness=1e-7)
        err = self._summed_error(P)
        self.assertTrue(np.all(np.isfinite(err)), "non-finite error values")
        self.assertLess(err[-1], 0.3 * err[0],
                        "two-slice ThreePIE_cupy did not converge "
                        "(first=%.3g last=%.3g)" % (err[0], err[-1]))

    def test_multislice_matches_cpu_reference(self):
        """GPU multi-slice product object must match the CPU 3PIE engine."""
        P_cpu = self._run("ThreePIE", numiter=120, slices=2, thickness=1e-7,
                          scanmodel="Full", fpb=100000)
        P_gpu = self._run("ThreePIE_cupy", numiter=120, slices=2, thickness=1e-7)

        sim = _similarity(self._obj(P_cpu), self._obj(P_gpu))
        self.assertGreater(sim, 0.85,
                           "GPU and CPU 3PIE product objects disagree "
                           "(correlation=%.3f)" % sim)

    def test_slice_start_iteration_list(self):
        """A per-slice start-iteration list must be accepted and run."""
        P = self._run("ThreePIE_cupy", numiter=60, slices=2, thickness=1e-7,
                      start=[0, 10])
        err = self._summed_error(P)
        self.assertTrue(np.all(np.isfinite(err)))
        self.assertLess(err[-1], err[0])


if __name__ == "__main__":
    unittest.main()
