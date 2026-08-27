#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Compare the three ThreePIE backends on the MoonFlower simulation, at several
frame shapes.

The frame shape is the simulation analogue of detector cropping: it sets the
number of pixels per frame and, with it, the reconstruction grid. For every
requested shape this script runs the CPU (``ThreePIE``), serialized
(``ThreePIE_serial``) and GPU (``ThreePIE_cupy``) engines on an *identical*
MoonFlowerScan -- same RNG seed, same frames, same iteration count -- and then
compares their objects, probes and per-slice objects with the phase- and
scale-invariant normalized correlation from ``ptypy.debug.threepie_compare``.

Because the data is generated on the fly there is nothing to stage: the script
runs anywhere. The GPU engine is skipped with a warning (not a crash) when
cupy or a usable device is missing, so a CPU-only machine still gets the
CPU-vs-serial half of the comparison.

Outputs, all written into ``--outdir``:

    sim_crop_comparison_<numiter>it.txt   correlation tables (also printed)
    sim_crop_comparison_<numiter>it.npz   objects, probes and per-slice objects
    sim_crop_comparison_<numiter>it.png   object-phase panels, rows = shape,
                                          cols = engine
    slices_<engine>_<shape>_<numiter>it.h5   raw per-slice dumps from each run

Typical use (from the repository root, with the ptypy_v8 environment):

    python -m ptypy.debug.run_threepie_sim_comparison --outdir /tmp/simcmp

    CUDA_VISIBLE_DEVICES=1 python -m ptypy.debug.run_threepie_sim_comparison \
        --outdir /tmp/simcmp --shapes 32,64,128 --numiter 60 --nframes 100

A quick smoke run that finishes in a couple of minutes:

    python -m ptypy.debug.run_threepie_sim_comparison \
        --outdir /tmp/simcmp --shapes 32 --numiter 20 --nframes 60

Run it as ``python -m ptypy.debug.run_threepie_sim_comparison`` from the
repository root (or with the root on ``PYTHONPATH``) so that ``import ptypy``
picks up this checkout; ``--help`` works from anywhere.

This file is part of the PTYPY package.

    :copyright: Copyright 2014 by the PTYPY team, see AUTHORS.
    :license: see LICENSE for details.
"""

import argparse
import importlib
import os

import matplotlib
matplotlib.use("Agg")            # headless: must happen before pyplot loads
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np               # noqa: E402

# Engine name -> (argument for ptypy.load_gpu_engines, module that registers it).
# ThreePIE itself is pure numpy and needs no GPU-engine bundle.
ENGINE_SETUP = {
    "ThreePIE": (None, "ptypy.custom.threepie"),
    "ThreePIE_serial": ("serial", "ptypy.custom.threepie_serial"),
    "ThreePIE_cupy": ("cupy", "ptypy.custom.threepie_cupy"),
}

# Engine pairs reported in the correlation tables, in the historical order.
# Pairs whose engines did not both run are dropped.
PAIRS = (
    ("ThreePIE_serial", "ThreePIE", "serial-vs-cpu"),
    ("ThreePIE_cupy", "ThreePIE", "gpu-vs-cpu"),
    ("ThreePIE_cupy", "ThreePIE_serial", "gpu-vs-serial"),
)

DEFAULT_SHAPES = ["32", "64", "128"]
DEFAULT_ENGINES = ["ThreePIE", "ThreePIE_serial", "ThreePIE_cupy"]
DEFAULT_NUMITER = 60
DEFAULT_NFRAMES = 100
DEFAULT_NSLICES = 2
DEFAULT_SLICE_THICKNESS = 5e-7
DEFAULT_SEED = 7

# MoonFlower scan settings shared by every run; kept fixed so the only thing
# that varies across a comparison is the engine and the frame shape.
SCAN_DENSITY = 0.2
SCAN_PHOTONS = 1e8
SCAN_PSF = 0.0

# Fraction of the object kept in the figure panels: the original crop was
# ob[n // 6 : n - n // 6], i.e. the central two thirds.
FIGURE_CROP_FRAC = 2.0 / 3.0


# --------------------------------------------------------------------------- #
# argument parsing
# --------------------------------------------------------------------------- #
def positive_int(value):
    """argparse type: strictly positive integer."""
    ivalue = int(value)
    if ivalue < 1:
        raise argparse.ArgumentTypeError(
            "expected a positive integer, got %r" % (value,))
    return ivalue


def parse_int_list(values, what="value"):
    """
    Flatten ``--shapes 32 64 128`` and ``--shapes 32,64,128`` into ``[32, 64, 128]``.

    ``values`` is the list argparse collected for an ``nargs="+"`` option.
    """
    out = []
    for chunk in values:
        for part in str(chunk).replace(",", " ").split():
            number = int(part)
            if number < 1:
                raise argparse.ArgumentTypeError(
                    "%s must be a positive integer, got %r" % (what, part))
            out.append(number)
    if not out:
        raise argparse.ArgumentTypeError("no %s given" % (what,))
    return out


def parse_str_list(values):
    """Same flattening as :func:`parse_int_list`, for name lists."""
    out = []
    for chunk in values:
        out.extend(part for part in str(chunk).replace(",", " ").split())
    return out


def default_numiter():
    """
    Default for ``--numiter``.

    The beamtime version of this script read the iteration count from the
    ``SIM_NUMITER`` environment variable. That still works -- it is now just
    the default of a proper command-line argument, so ``--numiter`` wins.
    """
    try:
        return positive_int(os.environ.get("SIM_NUMITER", DEFAULT_NUMITER))
    except (ValueError, argparse.ArgumentTypeError):
        return DEFAULT_NUMITER


def build_argparser():
    parser = argparse.ArgumentParser(
        description="Compare the ThreePIE CPU/serial/GPU backends on the "
                    "MoonFlower simulation at several frame shapes.")
    parser.add_argument("--outdir", default=".",
                        help="Directory for the .txt/.npz/.png report and the "
                             "per-engine slice dumps; created if missing "
                             "(default: %(default)s, i.e. the current "
                             "directory).")
    parser.add_argument("--shapes", nargs="+", default=DEFAULT_SHAPES,
                        help="Frame shapes to run, space- or comma-separated. "
                             "Each shape sets the pixels per frame and the "
                             "reconstruction grid (default: 32 64 128).")
    parser.add_argument("--engines", nargs="+", default=DEFAULT_ENGINES,
                        help="Engines to run, space- or comma-separated. "
                             "Choose from ThreePIE, ThreePIE_serial, "
                             "ThreePIE_cupy (default: all three). Engines "
                             "whose backend is unavailable are skipped.")
    parser.add_argument("--numiter", type=positive_int, default=default_numiter(),
                        help="Iterations per reconstruction (default: "
                             "%(default)s; the SIM_NUMITER environment "
                             "variable sets this default).")
    parser.add_argument("--nframes", type=positive_int, default=DEFAULT_NFRAMES,
                        help="Number of simulated diffraction frames "
                             "(default: %(default)s).")
    parser.add_argument("--nslices", type=positive_int, default=DEFAULT_NSLICES,
                        help="Number of object slices for the multislice "
                             "engines (default: %(default)s).")
    parser.add_argument("--slice-thickness", type=float,
                        default=DEFAULT_SLICE_THICKNESS,
                        help="Slice separation in meters (default: %(default)s).")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help="numpy RNG seed re-applied before every run so all "
                             "engines see the same MoonFlower scan positions "
                             "(default: %(default)s).")
    parser.add_argument("--aligned", action="store_true",
                        help="Report aligned_ncorr (registers out the joint "
                             "probe/object translation gauge) instead of the "
                             "plain ncorr. Off by default so the numbers stay "
                             "comparable with earlier runs.")
    parser.add_argument("--aligned-crop-frac", type=float, default=None,
                        help="With --aligned, first reduce both inputs to this "
                             "central fraction. Only needed when comparing "
                             "different-sized grids, which this script does "
                             "not do (default: %(default)s).")
    return parser


# --------------------------------------------------------------------------- #
# engine setup
# --------------------------------------------------------------------------- #
def import_ptypy():
    """
    Import ptypy, with a pointer to the usual cause when it is not on the path.

    This module lives inside the package, so it deliberately does no sys.path
    surgery: run it as ``python -m ptypy.debug.run_threepie_sim_comparison``
    from the repository root instead.
    """
    try:
        import ptypy
    except ImportError as err:
        raise SystemExit(
            "cannot import ptypy (%s).\nRun this from the repository root as "
            "'python -m ptypy.debug.run_threepie_sim_comparison', or put the "
            "root on PYTHONPATH." % (err,))
    return ptypy


def have_cupy():
    """
    True only when cupy is importable AND a GPU is actually reachable.

    Importing cupy and calling ``load_gpu_engines("cupy")`` both succeed on a
    machine that has cupy installed but no visible device (e.g. under
    ``CUDA_VISIBLE_DEVICES=""``), and the failure then surfaces much later as a
    CUDARuntimeError in the middle of a reconstruction. Touching the device
    here turns that into a clean skip.
    """
    if importlib.util.find_spec("cupy") is None:
        return False
    try:
        import cupy as cp
        cp.cuda.Device(0).compute_capability
        return True
    except Exception:                # noqa: BLE001 - no usable device
        return False


def load_engines(engine_names):
    """
    Register the requested ThreePIE backends and return the ones that loaded.

    A backend that cannot be used -- in practice ``ThreePIE_cupy`` on a machine
    without cupy or without a visible device -- is reported and left out
    instead of taking the whole comparison down with it.
    """
    ptypy = import_ptypy()

    available = []
    for name in engine_names:
        bundle, module = ENGINE_SETUP[name]
        if bundle == "cupy" and not have_cupy():
            print("WARNING: skipping %s -- no usable GPU (cupy missing or no "
                  "visible device)" % name, flush=True)
            continue
        try:
            if bundle is not None:
                ptypy.load_gpu_engines(bundle)
            importlib.import_module(module)
        except Exception as err:     # noqa: BLE001 - any backend problem skips
            print("WARNING: skipping %s -- backend unavailable (%s: %s)"
                  % (name, type(err).__name__, err), flush=True)
            continue
        available.append(name)
    return available


def base_params(u, engine_name, shape, args):
    """Ptycho parameter tree for one (engine, shape) reconstruction."""
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
    p.scans.MF.data.name = "MoonFlowerScan"
    p.scans.MF.data.shape = shape
    p.scans.MF.data.num_frames = args.nframes
    p.scans.MF.data.density = SCAN_DENSITY
    p.scans.MF.data.photons = SCAN_PHOTONS
    p.scans.MF.data.psf = SCAN_PSF
    p.scans.MF.data.save = None

    p.engines = u.Param()
    p.engines.e0 = u.Param()
    p.engines.e0.name = engine_name
    p.engines.e0.numiter = args.numiter
    p.engines.e0.numiter_contiguous = 10
    p.engines.e0.probe_center_tol = None
    p.engines.e0.compute_log_likelihood = True
    if engine_name != "ThreePIE":  # serial/GPU-only option
        p.engines.e0.compute_fourier_error = True
    p.engines.e0.number_of_slices = args.nslices
    p.engines.e0.slice_thickness = args.slice_thickness
    p.engines.e0.fslices = os.path.join(
        args.outdir, "slices_%s_%d_%dit.h5" % (engine_name, shape, args.numiter))
    return p


def run_reconstructions(shapes, engines, args):
    """
    Run every (shape, engine) combination and collect the reconstructed fields.

    Returns ``{(shape, engine, key): ndarray}`` with ``key`` one of ``"obj"``,
    ``"probe"`` or ``"slice<i>"``.
    """
    from ptypy import utils as u
    from ptypy.core import Ptycho
    from ptypy.debug.threepie_compare import read_slices

    results = {}
    for shape in shapes:
        for engine in engines:
            np.random.seed(args.seed)  # same moonflower positions every run
            print("=== %s shape=%d ===" % (engine, shape), flush=True)
            pars = base_params(u, engine, shape, args)
            P = Ptycho(pars, level=5)
            sname = list(P.obj.storages.keys())[0]
            results[(shape, engine, "obj")] = np.array(
                P.obj.storages[sname].data[0])
            results[(shape, engine, "probe")] = np.array(
                P.probe.storages[list(P.probe.storages.keys())[0]].data[0])
            del P
            for idx, arr in read_slices(pars.engines.e0.fslices).items():
                results[(shape, engine, "slice%d" % idx)] = arr
    return results


# --------------------------------------------------------------------------- #
# comparison
# --------------------------------------------------------------------------- #
def make_metric(args):
    """
    The similarity metric used throughout the report.

    Default is the plain ``ncorr`` the original comparison used. ``--aligned``
    switches to ``aligned_ncorr``, which registers ``b`` onto ``a`` first and
    returns ``(shift, value)`` -- only the value is tabulated.
    """
    from ptypy.debug.threepie_compare import aligned_ncorr, ncorr

    if not args.aligned:
        return ncorr

    def metric(a, b):
        _shift, value = aligned_ncorr(a, b, crop_frac=args.aligned_crop_frac)
        return value
    return metric


def active_pairs(engines):
    """The historical engine pairs, restricted to engines that actually ran."""
    return [(a, b, tag) for a, b, tag in PAIRS if a in engines and b in engines]


def build_report(results, shapes, engines, args, metric):
    """Correlation tables as a single text block."""
    pairs = active_pairs(engines)

    lines = ["pair                          shape   object   probe"]
    for shape in shapes:
        for a, b, tag in pairs:
            co = metric(results[(shape, a, "obj")], results[(shape, b, "obj")])
            cp = metric(results[(shape, a, "probe")],
                        results[(shape, b, "probe")])
            lines.append("%-28s %5d   %.4f   %.4f" % (tag, shape, co, cp))
    lines.append("")
    lines.append("per-slice (product-independent) comparison:")
    lines.append("pair                          shape  slice   ncorr")
    for shape in shapes:
        for a, b, tag in pairs:
            for i in range(args.nslices):
                key_a, key_b = (shape, a, "slice%d" % i), (shape, b, "slice%d" % i)
                if key_a not in results or key_b not in results:
                    continue
                lines.append("%-28s %5d  %5d   %.4f"
                             % (tag, shape, i, metric(results[key_a],
                                                      results[key_b])))
    lines.append("separation within engine ncorr(slice0, slice1):")
    for shape in shapes:
        row = ["  shape %d:" % shape]
        for engine in engines:
            k0, k1 = (shape, engine, "slice0"), (shape, engine, "slice1")
            if k0 in results and k1 in results:
                row.append("%s %.4f" % (engine.replace("ThreePIE", "") or "cpu",
                                        metric(results[k0], results[k1])))
        lines.append("  ".join(row))
    return "\n".join(lines)


def save_figure(results, shapes, engines, args, path):
    """Object-phase panels, one row per shape and one column per engine."""
    from ptypy.debug.threepie_compare import central

    fig, axes = plt.subplots(len(shapes), len(engines), squeeze=False,
                             figsize=(3.2 * len(engines), 3.2 * len(shapes)))
    for i, shape in enumerate(shapes):
        for j, engine in enumerate(engines):
            ax = axes[i, j]
            # trim the poorly covered border before showing the phase
            ax.imshow(np.angle(central(results[(shape, engine, "obj")],
                                       frac=FIGURE_CROP_FRAC)), cmap="gray")
            ax.set_xticks([])
            ax.set_yticks([])
            if i == 0:
                ax.set_title(engine, fontsize=11)
            if j == 0:
                ax.set_ylabel("shape %d" % shape, fontsize=11)
    fig.suptitle("MoonFlower %d-slice ThreePIE: object phase (%d it, %d frames)"
                 % (args.nslices, args.numiter, args.nframes), fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
def main():
    parser = build_argparser()
    args = parser.parse_args()

    # Validate everything that can be validated without importing ptypy, so a
    # typo is reported as a clean argparse error rather than a traceback.
    try:
        shapes = parse_int_list(args.shapes, what="shape")
    except (ValueError, argparse.ArgumentTypeError) as err:
        parser.error("--shapes: %s" % (err,))
    requested = parse_str_list(args.engines)
    unknown = [name for name in requested if name not in ENGINE_SETUP]
    if unknown:
        parser.error("--engines: unknown engine(s) %s; choose from %s"
                     % (", ".join(unknown), ", ".join(sorted(ENGINE_SETUP))))

    args.outdir = os.path.abspath(args.outdir)
    os.makedirs(args.outdir, exist_ok=True)

    engines = load_engines(requested)
    if not engines:
        raise SystemExit("no ThreePIE backend could be loaded -- nothing to run")
    if len(engines) < len(requested):
        print("running with engines: %s" % ", ".join(engines), flush=True)

    results = run_reconstructions(shapes, engines, args)

    report = build_report(results, shapes, engines, args, make_metric(args))
    print(report, flush=True)

    stem = os.path.join(args.outdir, "sim_crop_comparison_%dit" % args.numiter)
    with open(stem + ".txt", "w") as fh:
        fh.write(report + "\n")
    np.savez(stem + ".npz",
             **{"%s_%s_%s" % (s, e, k): v for (s, e, k), v in results.items()})
    save_figure(results, shapes, engines, args, stem + ".png")
    print("saved %s.png/.npz/.txt" % stem, flush=True)


if __name__ == "__main__":
    main()
