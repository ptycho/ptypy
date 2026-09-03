#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Ground-truth two-slice ThreePIE simulation with a per-slice comparison figure.

The data are generated with a real two-slice forward model

    probe -> x obj0 -> near-field(slice_sep) -> x obj1 -> far field

from two distinct, known phantoms (flowers upstream, a spoke star downstream),
so every reconstructed slice can be scored against truth and not only against
another backend. The plain MoonFlower comparison cannot provide that: there
the data come from a single object plane, so the 2-slice split is a free gauge
and any division of the object between the slices is "right".

The scan is self-contained and needs no external data files.

Reported per engine (``ThreePIE`` / ``ThreePIE_serial`` / ``ThreePIE_cupy``):

  recovery    aligned ncorr(reconstructed slice i, ground-truth phantom i)
  swap check  aligned ncorr(reconstructed slice i, the other phantom)
  distinctness   aligned ncorr(phantom 0, phantom 1), the premise of the test
  cross-backend  aligned ncorr of the same slice between two backends

plus a figure whose first column is the ground truth and whose remaining
columns are the engines, one row per slice.

This is the exploratory, figure-producing sibling of the pass-fail gate in
``test/engine_tests/threepie_groundtruth_slices_test.py``. The phantom, the
focus-midway probe treatment, the ``slice_bandlimit=False`` choice for the
serialized/GPU engines and the metric conventions are kept identical to that
test. Two of those settings matter:

  * The beam focus is placed midway between the slices. A quasi-collimated
    probe has almost no depth discrimination over the slice separation, and
    the reconstruction then splits the two layers arbitrarily, for every
    backend alike. Use ``--no-focus-mid`` to see that failure mode.
  * ``slice_bandlimit`` is switched off for the serialized/GPU engines. The
    data are generated with the exact near-field propagator below the
    angular-spectrum critical distance. There the anti-alias band limit
    (correct protection for real data above z_crit) would discard true
    signal and bias the comparison against those engines.

Outputs, written into ``--outdir``:

    sim_gt_slices.npz     ground truth + every engine's slices
    sim_gt_slices.txt     the printed report
    sim_gt_slices.png     figure, rows = slice, cols = ground truth + engines
    slices_gt_<engine>.h5 the raw per-engine ``fslices`` dumps

Typical use (from the repo root, with the ptypy_v8 environment):

    python -m ptypy.debug.run_threepie_groundtruth_sim --outdir /tmp/gt
    CUDA_VISIBLE_DEVICES=1 python -m ptypy.debug.run_threepie_groundtruth_sim \
        --shape 64 --nframes 300 --numiter 300 --outdir /tmp/gt

A quick smoke run that finishes in a couple of minutes:

    python -m ptypy.debug.run_threepie_groundtruth_sim \
        --shape 32 --nframes 100 --numiter 40 --outdir /tmp/gt_smoke

The ``-m`` form puts the repo root on sys.path; calling the file by path works
too when ptypy is installed or PYTHONPATH points at the checkout.

``ThreePIE_cupy`` is skipped with a printed note when cupy (or a usable GPU)
is not available, so the script still produces its report and figure on a
CPU-only machine.

This file is part of the PTYPY package.

    :copyright: Copyright 2014 by the PTYPY team, see AUTHORS.
    :license: see LICENSE for details.
"""

import argparse
import importlib
import importlib.util
import os

import matplotlib
matplotlib.use("Agg")            # these runs are headless; before pyplot
import matplotlib.pyplot as plt  # noqa: E402

import numpy as np               # noqa: E402

# ptypy itself is imported inside the functions that need it, so that --help
# still works when the repo root is not on sys.path (this script does not
# touch sys.path; run it as a module from the repo root, or with the
# package installed).

# Name of the PtyScan registered below. Kept in sync with the pytest sibling.
SCAN_NAME = "ThreePIEGroundTruthScan"

ENGINE_LABEL = {"ThreePIE": "cpu",
                "ThreePIE_serial": "serial",
                "ThreePIE_cupy": "gpu"}
DEFAULT_ENGINES = "ThreePIE,ThreePIE_serial,ThreePIE_cupy"

# Metric conventions, identical to threepie_groundtruth_slices_test.py: both
# fields are first reduced to a common central 75 % region, then registered
# against each other (a ptychographic solution is only defined up to a joint
# probe/object shift) and the 10 % wrap-around margin is trimmed.
CROP_FRAC = 0.75
MARGIN_FRAC = 0.1

# Iterations per contiguous engine block, clamped to --numiter so that short
# smoke runs still execute at least one block.
NUMITER_CONTIGUOUS = 20

# Fraction of each panel kept in the figure.
FIGURE_CROP_FRAC = 0.8


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
def gt_ncorr(a, b):
    """Aligned normalized correlation; ``aligned_ncorr`` returns (shift, value)."""
    from ptypy.debug.threepie_compare import aligned_ncorr
    _shift, value = aligned_ncorr(a, b, margin_frac=MARGIN_FRAC,
                                  crop_frac=CROP_FRAC)
    return value


def have_cupy():
    """True when cupy is importable and a GPU is reachable."""
    if importlib.util.find_spec("cupy") is None:
        return False
    try:
        import cupy as cp
        cp.cuda.Device(0).compute_capability
        return True
    except Exception:
        return False


# --------------------------------------------------------------------------- #
# the phantoms and the ground-truth two-slice scan
# --------------------------------------------------------------------------- #
def spoke_star(shape, spokes=24, phase=0.4, rmax=0.95):
    """Siemens-star-like phase phantom for the downstream slice, amplitude ~1."""
    n0, n1 = shape
    y, x = np.mgrid[0:n0, 0:n1]
    y = y - n0 / 2.0
    x = x - n1 / 2.0
    theta = np.arctan2(y, x)
    r = np.hypot(y / (n0 / 2.0), x / (n1 / 2.0))
    ph = phase * np.tanh(4 * np.sin(spokes * theta)) * (r < rmax)
    amp = 1.0 - 0.15 * (np.cos(spokes * theta) * (r < rmax)) ** 2
    return (amp * np.exp(1j * ph)).astype(np.complex64)


def register_scan():
    """
    Register the two-slice ground-truth PtyScan and return its class.

    Registration happens exactly once per process; calling this again (or after
    the pytest sibling has registered the same scan) returns the existing class.
    """
    from ptypy.experiment import PTYSCANS
    if SCAN_NAME in PTYSCANS:
        return PTYSCANS[SCAN_NAME]

    from ptypy import defaults_tree
    from ptypy import utils as u
    from ptypy.core import geometry
    from ptypy.core.data import MoonFlowerScan
    from ptypy.experiment import register
    from ptypy.utils import Param

    @register()
    @defaults_tree.parse_doc('scandata.' + SCAN_NAME, True)
    class ThreePIEGroundTruthScan(MoonFlowerScan):
        """
        MoonFlower-style test scan whose data come from a true two-slice
        forward model: probe * flowers -> near-field(z = slice_sep) ->
        * spoke-star -> far field. Both object slices are known.

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

        [focus_mid]
        default = True
        type = bool
        help = Place the beam focus midway between the two slices
        doc = Mirrors the real experiment (slices at +-0.75 mm around focus); the curvature difference between the planes gives the reconstruction its depth discrimination.
        """

        def __init__(self, pars=None, **kwargs):
            super().__init__(pars, **kwargs)
            # upstream slice: the flower object built by the parent class
            self.obj0 = self.obj
            # downstream slice: a distinct, known phantom on the same frame
            self.obj1 = spoke_star(self.obj.shape, spokes=self.p.spokes)
            # inter-slice near-field propagator on the probe frame
            g = Param()
            g.energy = self.geo.energy
            g.distance = self.p.slice_sep
            g.psize = self.geo.resolution
            g.shape = self.geo.shape
            g.propagation = "nearfield"
            self._slice_prop = geometry.Geo(owner=None, pars=g).propagator
            if self.p.focus_mid:
                # treat the moon field as the focal plane and back-propagate
                # it by slice_sep/2, so the focus sits midway between the
                # slices: converging at slice 0, diverging at slice 1. That
                # curvature difference carries the depth information.
                g.distance = self.p.slice_sep / 2.0
                half = geometry.Geo(owner=None, pars=g).propagator
                self.pr = half.bw(self.pr)

        def load(self, indices):
            p = self.pixel
            s = self.geo.shape
            raw = {}
            for k in indices:
                o0 = self.obj0[p[k][0]:p[k][0] + s[0], p[k][1]:p[k][1] + s[1]]
                o1 = self.obj1[p[k][0]:p[k][0] + s[0], p[k][1]:p[k][1] + s[1]]
                wave = self._slice_prop.fw(self.pr * o0) * o1
                intensity = u.abs2(self.geo.propagator.fw(wave))
                if self.p.psf > 0.:
                    intensity = u.gf(intensity, self.p.psf)
                if self.p.add_poisson_noise:
                    raw[k] = np.random.poisson(intensity).astype(np.int32)
                else:
                    raw[k] = intensity.astype(np.int32)
            return raw, {}, {}

    return ThreePIEGroundTruthScan


# --------------------------------------------------------------------------- #
# geometry / reconstruction
# --------------------------------------------------------------------------- #
def slice_separation(scan_cls, args):
    """
    Slice separation in meters, taken from the scan's own geometry.

    The separation must sit inside the optically separable, alias-free window
    DOF << sep < z_crit, with z_crit = N*dx^2/lambda the angular-spectrum
    critical distance. Returns ``(zsep, dx, zcrit, dof)``.
    """
    from ptypy import utils as u
    from ptypy.core import geometry
    probe_pars = u.Param()
    probe_pars.shape = args.shape
    probe_pars.num_frames = 8
    probe_pars.density = args.density
    probe_pars.slice_sep = 1e-3        # dummy for the throwaway instance
    tmp = scan_cls(probe_pars)
    dx = float(np.mean(tmp.geo.resolution))
    lam = geometry.Geo._keV2m / float(tmp.geo.energy)
    del tmp
    zcrit = args.shape * dx * dx / lam
    dof = 5.2 * dx * dx / lam
    # sep/DOF = 0.164 * shape at the default --sep-frac, i.e. ~11 at the
    # default --shape 64, matching the real crop-128 case that separates
    # cleanly.
    return args.sep_frac * zcrit, dx, zcrit, dof


def build_params(engine_name, zsep, args):
    """Ptycho parameter tree for one engine of the ground-truth simulation."""
    from ptypy import utils as u
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
    p.scans.MF.data.name = SCAN_NAME
    p.scans.MF.data.shape = args.shape
    p.scans.MF.data.num_frames = args.nframes
    p.scans.MF.data.density = args.density
    p.scans.MF.data.photons = 1e8
    p.scans.MF.data.psf = 0.0
    p.scans.MF.data.save = None
    p.scans.MF.data.slice_sep = zsep
    p.scans.MF.data.spokes = args.spokes
    p.scans.MF.data.focus_mid = args.focus_mid
    # initial probe hint, like the real runner's --defocus-um: start from a
    # probe whose focus sits midway between the slices
    p.scans.MF.illumination = u.Param()
    p.scans.MF.illumination.propagation = u.Param()
    p.scans.MF.illumination.propagation.parallel = -zsep / 2.0

    p.engines = u.Param()
    p.engines.e0 = u.Param()
    p.engines.e0.name = engine_name
    p.engines.e0.numiter = args.numiter
    p.engines.e0.numiter_contiguous = max(1, min(NUMITER_CONTIGUOUS,
                                                 args.numiter))
    p.engines.e0.probe_center_tol = 1
    p.engines.e0.compute_log_likelihood = True
    if engine_name != "ThreePIE":
        # serialized/GPU-only options; see the module docstring on why the
        # band limit is off for exactly-simulated data
        p.engines.e0.compute_fourier_error = True
        p.engines.e0.slice_bandlimit = False
    p.engines.e0.number_of_slices = 2
    p.engines.e0.slice_thickness = zsep
    p.engines.e0.fslices = os.path.join(args.outdir,
                                        "slices_gt_%s.h5" % engine_name)
    return p


def load_engines(engines):
    """Import the ThreePIE engine variants needed for ``engines``."""
    import ptypy
    importlib.import_module("ptypy.custom.threepie")
    if any(e != "ThreePIE" for e in engines):
        ptypy.load_gpu_engines("serial")
        importlib.import_module("ptypy.custom.threepie_serial")
    if "ThreePIE_cupy" in engines:
        ptypy.load_gpu_engines("cupy")
        importlib.import_module("ptypy.custom.threepie_cupy")


def run_engine(engine_name, zsep, args):
    """
    Reconstruct once with ``engine_name``.

    Returns ``({slice_index: array}, {slice_index: phantom})``; the ground
    truth is read straight off the PtyScan instance that generated the data.
    """
    from ptypy.core import Ptycho
    from ptypy.debug.threepie_compare import read_slices
    np.random.seed(args.seed)   # identical positions + noise realization
    pars = build_params(engine_name, zsep, args)
    P = Ptycho(pars, level=5)
    ptyscan = list(P.model.scans.values())[0].ptyscan
    truth = {0: np.array(ptyscan.obj0), 1: np.array(ptyscan.obj1)}
    del P
    return read_slices(pars.engines.e0.fslices), truth


# --------------------------------------------------------------------------- #
# report and figure
# --------------------------------------------------------------------------- #
def build_report(engines, gt, results, zsep, args):
    """The printed/saved comparison report, as one string."""
    lines = ["ground-truth two-slice simulation: shape %d, %d frames, %d it, "
             "slice_sep %.2f mm" % (args.shape, args.nframes, args.numiter,
                                    zsep * 1e3),
             "",
             "engine slice vs GROUND TRUTH (aligned ncorr):",
             "engine            slice0        slice1"]
    for eng in engines:
        vals = ["%.4f" % gt_ncorr(gt[i], results[(eng, "slice%d" % i)])
                for i in (0, 1)]
        lines.append("%-14s   %s        %s"
                     % (ENGINE_LABEL[eng], vals[0], vals[1]))

    lines += ["", "swap check: engine slice vs the OTHER GT slice (aligned):"]
    for eng in engines:
        c01 = gt_ncorr(gt[0], results[(eng, "slice1")])
        c10 = gt_ncorr(gt[1], results[(eng, "slice0")])
        lines.append("%-14s   GT0-vs-rec1 %.4f   GT1-vs-rec0 %.4f"
                     % (ENGINE_LABEL[eng], c01, c10))

    lines += ["", "GT slice0-vs-slice1 (phantom distinctness): %.4f"
              % gt_ncorr(gt[0], gt[1])]
    pairs = [(a, b, tag)
             for a, b, tag in (("ThreePIE_serial", "ThreePIE", "serial-vs-cpu"),
                               ("ThreePIE_cupy", "ThreePIE", "gpu-vs-cpu"),
                               ("ThreePIE_cupy", "ThreePIE_serial",
                                "gpu-vs-serial"))
             if a in engines and b in engines]
    if pairs:
        lines += ["cross-backend per slice (aligned):"]
    for a, b, tag in pairs:
        vals = ["%.4f" % gt_ncorr(results[(a, "slice%d" % i)],
                                  results[(b, "slice%d" % i)])
                for i in (0, 1)]
        lines.append("%-16s   slice0 %s   slice1 %s" % (tag, vals[0], vals[1]))
    return "\n".join(lines)


def common_crop(panels, frac=FIGURE_CROP_FRAC):
    """Crop every panel centrally to the same (smallest) shape."""
    n0 = int(min(p.shape[-2] for p in panels) * frac)
    n1 = int(min(p.shape[-1] for p in panels) * frac)
    out = []
    for x in panels:
        c0 = (x.shape[-2] - n0) // 2
        c1 = (x.shape[-1] - n1) // 2
        out.append(x[..., c0:c0 + n0, c1:c1 + n1])
    return out


def make_figure(engines, gt, results, zsep, args, path):
    """Rows = slice, columns = ground truth followed by each engine."""
    from ptypy.debug.threepie_compare import gauge_phase
    cols = ["ground truth"] + ["%s (%s)" % (e, ENGINE_LABEL[e])
                               for e in engines]
    fig, axes = plt.subplots(2, len(cols), figsize=(3.9 * len(cols), 3.9 * 2))
    for i in (0, 1):
        panels = [gt[i]] + [results[(e, "slice%d" % i)] for e in engines]
        # crop every panel (ground truth included) to the same central FOV
        phases = [gauge_phase(p) for p in common_crop(panels)]
        pooled = np.concatenate([p.ravel() for p in phases])
        vmin, vmax = np.percentile(pooled, [1, 99])
        im = None
        for j, ph in enumerate(phases):
            ax = axes[i, j]
            im = ax.imshow(ph, cmap="gray", vmin=vmin, vmax=vmax)
            ax.set_xticks([])
            ax.set_yticks([])
            if i == 0:
                ax.set_title(cols[j], fontsize=11)
            if j == 0:
                ax.set_ylabel("slice %d" % i, fontsize=12)
        cb = fig.colorbar(im, ax=axes[i, -1], fraction=0.046, pad=0.03)
        cb.set_label("phase (rad)", fontsize=8)
        cb.ax.tick_params(labelsize=7)
    fig.suptitle("Two-slice ground-truth simulation, shape %d, %d it, "
                 "slice_sep %.2f mm" % (args.shape, args.numiter, zsep * 1e3),
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# command line
# --------------------------------------------------------------------------- #
def build_argparser():
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n")[1],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--outdir", default=".",
                        help="Directory for sim_gt_slices.npz/.txt/.png and "
                             "the per-engine slices_gt_<engine>.h5 dumps; "
                             "created if missing.")
    parser.add_argument("--shape", type=int, default=64,
                        help="Diffraction frame size in pixels.")
    parser.add_argument("--nframes", type=int, default=300,
                        help="Number of scan positions.")
    parser.add_argument("--numiter", type=int, default=300,
                        help="Iterations per engine.")
    parser.add_argument("--density", type=float, default=0.15,
                        help="MoonFlower scan position density.")
    parser.add_argument("--sep-frac", type=float, default=0.85,
                        help="Slice separation as a fraction of the "
                             "angular-spectrum critical distance "
                             "z_crit = N*dx^2/lambda. Keep it below 1 so the "
                             "near-field forward model stays alias-free.")
    parser.add_argument("--spokes", type=int, default=24,
                        help="Number of spokes of the downstream star phantom.")
    parser.add_argument("--seed", type=int, default=7,
                        help="Seed for the scan positions and the Poisson "
                             "noise; every engine gets the same realization.")
    parser.add_argument("--engines", default=DEFAULT_ENGINES,
                        help="Comma-separated engine list. ThreePIE_cupy is "
                             "dropped automatically when cupy is unavailable.")
    parser.add_argument("--no-focus-mid", dest="focus_mid",
                        action="store_false",
                        help="Do not place the beam focus midway between the "
                             "slices. The quasi-collimated probe then has "
                             "almost no depth discrimination. Use this to see "
                             "the failure mode; it is not a valid comparison.")
    parser.set_defaults(focus_mid=True)
    return parser


def main():
    args = build_argparser().parse_args()

    engines = [e.strip() for e in args.engines.split(",") if e.strip()]
    unknown = [e for e in engines if e not in ENGINE_LABEL]
    if unknown:
        raise SystemExit("unknown engine(s): %s (known: %s)"
                         % (", ".join(unknown), ", ".join(ENGINE_LABEL)))
    if "ThreePIE_cupy" in engines and not have_cupy():
        print("cupy unavailable -> skipping the ThreePIE_cupy engine",
              flush=True)
        engines = [e for e in engines if e != "ThreePIE_cupy"]
    if not engines:
        raise SystemExit("no engines left to run")

    os.makedirs(args.outdir, exist_ok=True)

    load_engines(engines)
    scan_cls = register_scan()

    zsep, dx, zcrit, dof = slice_separation(scan_cls, args)
    print("dx=%.1f nm, z_crit=%.2f mm, DOF~%.1f um -> slice_sep=%.2f mm "
          "(sep/DOF=%.0f)" % (dx * 1e9, zcrit * 1e3, dof * 1e6,
                              zsep * 1e3, zsep / dof), flush=True)

    results = {}
    gt = None
    for eng in engines:
        print("=== %s ===" % eng, flush=True)
        slices, truth = run_engine(eng, zsep, args)
        if gt is None:
            gt = truth
        for idx, arr in slices.items():
            results[(eng, "slice%d" % idx)] = arr

    report = build_report(engines, gt, results, zsep, args)
    print(report, flush=True)
    with open(os.path.join(args.outdir, "sim_gt_slices.txt"), "w") as fh:
        fh.write(report + "\n")
    np.savez(os.path.join(args.outdir, "sim_gt_slices.npz"),
             gt_slice0=gt[0], gt_slice1=gt[1],
             **{"%s_%s" % (ENGINE_LABEL[e], k): v
                for (e, k), v in results.items()})

    make_figure(engines, gt, results, zsep, args,
                os.path.join(args.outdir, "sim_gt_slices.png"))
    print("saved sim_gt_slices.png/.npz/.txt in %s"
          % os.path.abspath(args.outdir), flush=True)


if __name__ == "__main__":
    main()
