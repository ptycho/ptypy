"""
Ground-truth validation of the ThreePIE multislice engines.

Every other ThreePIE test compares an engine either against another engine or
against itself. This one compares against *known truth*: the diffraction data
are generated with a real two-slice forward model

    probe -> x obj0 -> near-field(slice_sep) -> x obj1 -> far field

from two DISTINCT, known phantoms (flowers upstream, a spoke star
downstream). That makes it possible to assert what multislice is actually
for -- that information ends up in the CORRECT slice -- rather than only that
the backends agree with each other.

The scan is self-contained (it generates its own data, no external files) and
small enough to run in a normal test suite.

Two properties are checked per engine:

  recovery   ncorr(reconstructed slice i, ground-truth phantom i)  -- high
  crosstalk  ncorr(reconstructed slice i, the OTHER phantom)       -- low

plus per-slice agreement between the backends.

Two settings matter for this to be a fair test and are deliberate:

  * The beam focus is placed midway between the slices. A quasi-collimated
    probe has almost no depth discrimination over the slice separation, and
    the reconstruction then splits the two layers arbitrarily -- for every
    backend alike.
  * slice_bandlimit is switched OFF for the serialized/GPU engines. The data
    are generated with the exact near-field propagator at a separation below
    the angular-spectrum critical distance, so the anti-alias band limit --
    correct protection for real data above z_crit -- would here discard true
    signal and bias the comparison against those engines.
"""
import importlib
import shutil
import tempfile
import unittest

import numpy as np

import ptypy
from ptypy import utils as u
from ptypy.core import Ptycho, geometry

# --- scan / reconstruction size -------------------------------------------- #
# Chosen by a sweep over frame size, position count and iterations. Bigger is
# NOT better here: at a fixed number of positions the larger grids are more
# weakly constrained, and by shape 64 the split stops being meaningful at all
# (recovery 0.42 against crosstalk 0.57 -- i.e. inverted). This configuration
# separates cleanly and runs in ~16 s for the CPU and serial engines together.
SHAPE = 32           # frame size in pixels
NFRAMES = 150        # scan positions
NUMITER = 80         # iterations per engine
SEED = 7
DENSITY = 0.15
SEP_FRAC = 0.85      # slice separation as a fraction of z_crit = N*dx^2/lambda
SPOKES = 24

# --- assertion thresholds --------------------------------------------------- #
# Calibrated over six seeds (7, 11, 23, 42, 77, 101) on the configuration
# above. Worst value seen across all of them, for both the CPU and the serial
# engine:
#
#     recovery      >= 0.805      (threshold 0.60)
#     crosstalk     <= 0.119      (threshold 0.30)
#     backend agree >= 0.818      (threshold 0.60)
#
# The thresholds keep ~0.2 of headroom against the worst observed draw while
# still asserting something strong: each phantom correlates with its own slice
# at least twice as well as with the other one.
RECOVERY_MIN = 0.60
CROSSTALK_MAX = 0.30
CROSS_BACKEND_MIN = 0.60


def have_cupy():
    if importlib.util.find_spec("cupy") is None:
        return False
    try:
        import cupy as cp
        cp.cuda.Device(0).compute_capability
        return True
    except Exception:
        return False


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
def ncorr(a, b):
    """Phase- and scale-invariant normalized correlation of two complex fields."""
    a = np.asarray(a).ravel()
    b = np.asarray(b).ravel()
    a = a - a.mean()
    b = b - b.mean()
    den = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.abs(np.vdot(a, b)) / den) if den else 0.0


def _register_shift(a, b):
    A = np.fft.fft2(a - a.mean())
    B = np.fft.fft2(b - b.mean())
    cc = np.fft.ifft2(A * np.conj(B))
    idx = np.unravel_index(np.argmax(np.abs(cc)), cc.shape)
    return [int(s) if s <= n // 2 else int(s - n) for s, n in zip(idx, cc.shape)]


def _common_central(a, b, frac=0.75):
    n0 = int(min(a.shape[0], b.shape[0]) * frac)
    n1 = int(min(a.shape[1], b.shape[1]) * frac)

    def crop(x):
        c0 = (x.shape[0] - n0) // 2
        c1 = (x.shape[1] - n1) // 2
        return x[c0:c0 + n0, c1:c1 + n1]
    return crop(a), crop(b)


def aligned_ncorr(a, b, margin_frac=0.1):
    """
    ncorr after removing the joint-translation gauge.

    A ptychographic solution is only defined up to a common shift of probe and
    object, so the fields are registered against each other before comparison
    and the wrap margins are trimmed.
    """
    a, b = _common_central(a, b)
    b = np.roll(b, _register_shift(a, b), axis=(-2, -1))
    m = max(1, int(min(a.shape) * margin_frac))
    return ncorr(a[m:-m, m:-m], b[m:-m, m:-m])


def spoke_star(shape, spokes=SPOKES, phase=0.4, rmax=0.95):
    """Siemens-star-like phantom for the downstream slice."""
    n0, n1 = shape
    y, x = np.mgrid[0:n0, 0:n1]
    y = y - n0 / 2.0
    x = x - n1 / 2.0
    theta = np.arctan2(y, x)
    r = np.hypot(y / (n0 / 2.0), x / (n1 / 2.0))
    ph = phase * np.tanh(4 * np.sin(spokes * theta)) * (r < rmax)
    amp = 1.0 - 0.15 * (np.cos(spokes * theta) * (r < rmax)) ** 2
    return (amp * np.exp(1j * ph)).astype(np.complex64)


# --------------------------------------------------------------------------- #
# the ground-truth scan
# --------------------------------------------------------------------------- #
_SCAN_NAME = "ThreePIEGroundTruthScan"
_scan_registered = False


def _register_scan():
    """Register the two-slice ground-truth PtyScan exactly once."""
    global _scan_registered
    if _scan_registered:
        return
    from ptypy import defaults_tree
    from ptypy.core.data import MoonFlowerScan
    from ptypy.experiment import register
    from ptypy.utils import Param

    @register()
    @defaults_tree.parse_doc('scandata.' + _SCAN_NAME, True)
    class ThreePIEGroundTruthScan(MoonFlowerScan):
        """
        MoonFlower-style scan whose data come from a true two-slice forward
        model, with both object slices known.

        Defaults:

        [name]
        default = ThreePIEGroundTruthScan
        type = str
        help =
        doc =

        [slice_sep]
        default = 1e-3
        type = float
        help = Separation of the two object slices in meters
        doc =

        [spokes]
        default = 24
        type = int
        help = Number of spokes of the downstream star phantom
        doc =
        """

        def __init__(self, pars=None, **kwargs):
            super().__init__(pars, **kwargs)
            # upstream slice: the flower object built by the parent class
            self.obj0 = self.obj
            # downstream slice: a distinct, known phantom on the same frame
            self.obj1 = spoke_star(self.obj.shape, spokes=self.p.spokes)

            g = Param()
            g.energy = self.geo.energy
            g.psize = self.geo.resolution
            g.shape = self.geo.shape
            g.propagation = "nearfield"
            g.distance = self.p.slice_sep
            self._slice_prop = geometry.Geo(owner=None, pars=g).propagator

            # put the focus midway between the slices: the moon field is
            # treated as the focal plane and back-propagated by half the
            # separation, so the beam converges at slice 0 and diverges at
            # slice 1. That curvature difference is what carries the depth
            # information the reconstruction needs.
            g.distance = self.p.slice_sep / 2.0
            half = geometry.Geo(owner=None, pars=g).propagator
            self.pr = half.bw(self.pr)

        def load(self, indices):
            p = self.pixel
            s = self.geo.shape
            raw = {}
            for k in indices:
                sl = (slice(p[k][0], p[k][0] + s[0]),
                      slice(p[k][1], p[k][1] + s[1]))
                wave = self._slice_prop.fw(self.pr * self.obj0[sl]) * self.obj1[sl]
                inten = u.abs2(self.geo.propagator.fw(wave))
                raw[k] = (np.random.poisson(inten).astype(np.int32)
                          if self.p.add_poisson_noise
                          else inten.astype(np.int32))
            return raw, {}, {}

    _scan_registered = True


def _read_slices(path):
    """Per-slice object arrays from the engine's fslices output."""
    from ptypy import io
    content = io.h5read(path, "content")["content"]
    objects = content["objects"]
    out = {}
    for name, storages in objects.items():
        idx = int(name.rsplit("_", 1)[-1])
        storage = list(storages.values())[0]
        out[idx] = np.asarray(storage["data"])[0]
    return out


class ThreePIEGroundTruthTest(unittest.TestCase):
    """Do the engines put each phantom in the slice it belongs to?"""

    @classmethod
    def setUpClass(cls):
        _register_scan()
        cls.outdir = tempfile.mkdtemp(suffix="_threepie_gt")
        cls._cache = {}
        cls._gt = None

        # slice separation from the scan's own geometry, inside the
        # alias-free window: sep = SEP_FRAC * z_crit
        from ptypy.experiment import PTYSCANS
        probe_pars = u.Param()
        probe_pars.shape = SHAPE
        probe_pars.num_frames = 8
        probe_pars.density = DENSITY
        probe_pars.slice_sep = 1e-3
        tmp = PTYSCANS[_SCAN_NAME](probe_pars)
        dx = float(np.mean(tmp.geo.resolution))
        lam = geometry.Geo._keV2m / float(tmp.geo.energy)
        cls.zsep = SEP_FRAC * SHAPE * dx * dx / lam
        del tmp

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.outdir, ignore_errors=True)

    # -- machinery ---------------------------------------------------------- #
    def _params(self, engine_name):
        import os
        p = u.Param()
        p.verbose_level = "error"
        p.io = u.Param()
        p.io.autosave = u.Param(active=False)
        p.io.interaction = u.Param(active=False)
        p.io.autoplot = u.Param(active=False)

        p.scans = u.Param()
        p.scans.MF = u.Param()
        p.scans.MF.name = "BlockFull"
        p.scans.MF.data = u.Param()
        p.scans.MF.data.name = _SCAN_NAME
        p.scans.MF.data.shape = SHAPE
        p.scans.MF.data.num_frames = NFRAMES
        p.scans.MF.data.density = DENSITY
        p.scans.MF.data.photons = 1e8
        p.scans.MF.data.psf = 0.0
        p.scans.MF.data.save = None
        p.scans.MF.data.slice_sep = self.zsep
        # start from a probe focused midway between the slices, mirroring the
        # defocus that a real multislice experiment is set up with
        p.scans.MF.illumination = u.Param()
        p.scans.MF.illumination.propagation = u.Param()
        p.scans.MF.illumination.propagation.parallel = -self.zsep / 2.0

        p.engines = u.Param()
        p.engines.e0 = u.Param()
        p.engines.e0.name = engine_name
        p.engines.e0.numiter = NUMITER
        p.engines.e0.numiter_contiguous = max(10, NUMITER // 10)
        p.engines.e0.probe_center_tol = 1
        p.engines.e0.compute_log_likelihood = False
        p.engines.e0.number_of_slices = 2
        p.engines.e0.slice_thickness = self.zsep
        if engine_name != "ThreePIE":
            # serialized/GPU-only options; see the module docstring on why the
            # band limit is off for exactly-simulated data
            p.engines.e0.compute_fourier_error = True
            p.engines.e0.slice_bandlimit = False
        p.engines.e0.fslices = os.path.join(
            self.outdir, "slices_%s.h5" % engine_name)
        return p

    def _run(self, engine_name):
        """Reconstruct once per engine; cache the slices and the ground truth."""
        if engine_name in self._cache:
            return self._cache[engine_name]

        ptypy.load_gpu_engines("serial")
        if have_cupy():
            ptypy.load_gpu_engines("cupy")
        importlib.import_module("ptypy.custom.threepie")
        importlib.import_module("ptypy.custom.threepie_serial")

        np.random.seed(SEED)
        pars = self._params(engine_name)
        P = Ptycho(pars, level=5)
        if type(self)._gt is None:
            ptyscan = list(P.model.scans.values())[0].ptyscan
            type(self)._gt = {0: np.array(ptyscan.obj0),
                              1: np.array(ptyscan.obj1)}
        del P

        slices = _read_slices(pars.engines.e0.fslices)
        self._cache[engine_name] = slices
        return slices

    def _assert_separates(self, engine_name):
        rec = self._run(engine_name)
        gt = type(self)._gt

        for i in (0, 1):
            recovery = aligned_ncorr(gt[i], rec[i])
            self.assertGreater(
                recovery, RECOVERY_MIN,
                "%s did not recover phantom %d into slice %d "
                "(ncorr %.3f, expected > %.2f)"
                % (engine_name, i, i, recovery, RECOVERY_MIN))

        for i, j in ((0, 1), (1, 0)):
            crosstalk = aligned_ncorr(gt[i], rec[j])
            self.assertLess(
                crosstalk, CROSSTALK_MAX,
                "%s leaked phantom %d into slice %d "
                "(ncorr %.3f, expected < %.2f)"
                % (engine_name, i, j, crosstalk, CROSSTALK_MAX))

    # -- the phantoms really are distinct ----------------------------------- #
    def test_phantoms_are_independent(self):
        """The premise: a reconstruction cannot score on both by accident."""
        self._run("ThreePIE_serial")
        gt = type(self)._gt
        self.assertLess(aligned_ncorr(gt[0], gt[1]), 0.2)

    # -- per-engine separation ---------------------------------------------- #
    def test_cpu_separates_slices(self):
        self._assert_separates("ThreePIE")

    def test_serial_separates_slices(self):
        self._assert_separates("ThreePIE_serial")

    @unittest.skipIf(not have_cupy(), "no cupy available")
    def test_cupy_separates_slices(self):
        self._assert_separates("ThreePIE_cupy")

    # -- the backends agree slice by slice ---------------------------------- #
    def test_backends_agree_per_slice(self):
        cpu = self._run("ThreePIE")
        serial = self._run("ThreePIE_serial")
        pairs = [("ThreePIE", cpu, "ThreePIE_serial", serial)]
        if have_cupy():
            gpu = self._run("ThreePIE_cupy")
            pairs.append(("ThreePIE_cupy", gpu, "ThreePIE_serial", serial))

        for na, a, nb, b in pairs:
            for i in (0, 1):
                agreement = aligned_ncorr(a[i], b[i])
                self.assertGreater(
                    agreement, CROSS_BACKEND_MIN,
                    "%s and %s disagree on slice %d "
                    "(ncorr %.3f, expected > %.2f)"
                    % (na, nb, i, agreement, CROSS_BACKEND_MIN))


if __name__ == "__main__":
    unittest.main()
