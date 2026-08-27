#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Plot ThreePIE engine speed versus raw detector crop for the three backends.

Every ptypy reconstruction carries its own timing: the engine writes a
``runtime/iter_info`` group into each saved ``.ptyr``, with one ``duration``
entry per saved iteration block. This script therefore needs no benchmark log
at all -- it opens the newest ``rec/rec_*.ptyr`` of each run directory, sums
those durations and divides by the *declared* iteration count of the run
family. (The number of ``iter_info`` entries is the number of save blocks, not
the number of iterations, so the declared count is what makes the numbers
comparable -- hence ``--family SUFFIX:NUMITER``.)

Runs are located by the naming convention of the beamtime runner:

    <scan-dir>/ptycho_ptypy_Mslice_<tag>_crop-<crop>_bin-<bin>_
        slices-<S>_pmodes-<P><suffix>/rec/rec_*.ptyr

with ``<tag>`` one of ``gpu`` / ``serial`` / ``cpu``. Two or more run families
(same physics, different iteration counts) can be given; the family with the
most iterations draws the line, the others are plotted as open triangles, which
shows at a glance that the per-iteration cost does not drift with run length.

The figure has two panels, both with a log-scaled crop axis:

  left   seconds per iteration vs crop (log-log), with a ``crop^2`` guide line
         anchored on the smallest crop of the reference (serial) engine
  right  GPU speedup factor over the two CPU-side engines

Typical use (with the ptypy_v8 environment, from the repo root):

    python -m ptypy.debug.plot_threepie_speed \\
        --scan-dir <beamtime>/process/0002_multislice/scan_000434 \\
        --family _cmp100:100 --family _speed20:20 \\
        --crops 128,256,512 --outdir ./figures

    python ptypy/debug/plot_threepie_speed.py --scan-dir <scan-dir> --help

The colours are a fixed, colourblind-safe palette (blue / orange / green) and
are deliberately not configurable, so that the same engine keeps the same hue
across every figure in this comparison set.

This file is part of the PTYPY package.

    :copyright: Copyright 2014 by the PTYPY team, see AUTHORS.
    :license: see LICENSE for details.
"""

import argparse
import os
import re

import numpy as np

import matplotlib
matplotlib.use("Agg")            # these plots are made headless
import matplotlib.pyplot as plt  # noqa: E402  (must follow the backend choice)

try:                            # normal in-package import
    from ptypy.debug.threepie_compare import find_latest, iteration_seconds
except ImportError:             # executed as a bare script from this folder
    from threepie_compare import find_latest, iteration_seconds

# Fixed palette order: engine identity -> hue (colourblind-safe).
ENGINES = [
    ("gpu", "ThreePIE_cupy (GPU)", "#2a78d6"),
    ("serial", "ThreePIE_serial", "#eb6834"),
    ("cpu", "ThreePIE (CPU)", "#1baf7a"),
]
GPU_TAG = "gpu"       # engine the speedup panel divides by
GUIDE_TAG = "serial"  # engine the crop^2 guide line is anchored on

RUN_TEMPLATE = ("ptycho_ptypy_Mslice_{tag}_crop-{crop}_bin-{binning}_"
                "slices-{slices}_pmodes-{pmodes}{suffix}")
REC_GLOB = os.path.join("rec", "rec_*.ptyr")

DEFAULT_FAMILIES = [("_speed20", 20), ("_cmp100", 100)]
DEFAULT_CROPS = "128,256,512"
DEFAULT_GPU_NOTE = "GPU is launch-overhead-bound, flat at 0.85 s/it"

TEXT1, TEXT2 = "#1a1a19", "#5f5e56"
GRID, SPINE = "#e7e6e0", "#c9c8c0"


# --------------------------------------------------------------------------- #
# argument types
# --------------------------------------------------------------------------- #
def family_spec(value):
    """argparse type for ``--family``: ``SUFFIX:NUMITER`` -> (suffix, n)."""
    if ":" not in value:
        raise argparse.ArgumentTypeError(
            "expected SUFFIX:NUMITER (e.g. _cmp100:100), got %r" % (value,))
    suffix, _, count = value.rpartition(":")
    try:
        numiter = int(count)
    except ValueError:
        raise argparse.ArgumentTypeError(
            "iteration count of %r is not an integer" % (value,))
    if numiter < 1:
        raise argparse.ArgumentTypeError(
            "iteration count of %r must be strictly positive" % (value,))
    return suffix, numiter


def crop_list(value):
    """argparse type for ``--crops``: comma-separated positive integers."""
    crops = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            crop = int(part)
        except ValueError:
            raise argparse.ArgumentTypeError(
                "crop %r is not an integer" % part)
        if crop < 1:
            raise argparse.ArgumentTypeError("crop %r must be positive" % part)
        crops.append(crop)
    if not crops:
        raise argparse.ArgumentTypeError("no crops given")
    return crops


# --------------------------------------------------------------------------- #
# timing extraction
# --------------------------------------------------------------------------- #
def run_pattern(args, tag, crop, suffix):
    """Glob for the saved reconstructions of one (engine, crop, family) run."""
    return os.path.join(args.scan_dir,
                        RUN_TEMPLATE.format(tag=tag, crop=crop,
                                            binning=args.binning,
                                            slices=args.slices,
                                            pmodes=args.probe_modes,
                                            suffix=suffix),
                        REC_GLOB)


def per_iter_seconds(args, tag, crop, suffix, numiter):
    """
    Mean engine seconds per iteration for one run, or None if it is missing.

    ``iteration_seconds`` returns ``(total_seconds, n_entries)``; ``n_entries``
    counts saved blocks, not iterations, so the run's declared ``numiter`` is
    the divisor.
    """
    path = find_latest(run_pattern(args, tag, crop, suffix))
    if path is None:
        return None
    total, _ = iteration_seconds(path)
    return total / numiter


def collect(args, families):
    """``{(tag, suffix): {crop: seconds_per_iter}}`` for every run found."""
    data = {}
    for tag, _, _ in ENGINES:
        for suffix, numiter in families:
            vals = {}
            for crop in args.crops:
                value = per_iter_seconds(args, tag, crop, suffix, numiter)
                if value is not None:
                    vals[crop] = value
            data[(tag, suffix)] = vals
    return data


def print_table(data, crops):
    """The same per-engine/per-family table the beamtime script printed."""
    print("%-8s %-9s " % ("engine", "family") +
          " ".join("crop%4d" % c for c in crops))
    for (tag, suffix), vals in data.items():
        row = " ".join("%8.2f" % vals.get(c, float("nan")) for c in crops)
        print("%-8s %-9s %s" % (tag, suffix, row))


# --------------------------------------------------------------------------- #
# figure
# --------------------------------------------------------------------------- #
def scan_label(scan_dir, override=None):
    """``.../scan_000434`` -> ``scan 434``; ``--scan-label`` overrides it."""
    if override:
        return override
    name = os.path.basename(os.path.normpath(scan_dir))
    match = re.match(r"^scan[_-]?0*(\d+)$", name)
    return "scan %s" % match.group(1) if match else name


def marker_note(main, others):
    """``circles 100 it, triangles 20 it`` for the plot title."""
    note = "circles %d it" % main[1]
    if others:
        note += ", triangles %s it" % ", ".join(str(n) for _, n in others)
    return note


def style_axis(ax, crops, xlabel, ylabel, title):
    """Shared light-theme styling of both panels."""
    ax.set_xscale("log", base=2)
    ax.set_xticks(crops)
    ax.set_xticklabels([str(c) for c in crops])
    ax.set_xlabel(xlabel, color=TEXT1)
    ax.set_ylabel(ylabel, color=TEXT1)
    ax.set_title(title, fontsize=11, color=TEXT1)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    ax.grid(True, which="major", color=GRID, lw=0.8, zorder=0)
    ax.tick_params(colors=TEXT2)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(SPINE)


def make_figure(args, data, main, others):
    """Build the two-panel speed figure and return it."""
    crops = args.crops
    main_suffix = main[0]
    fig, (ax, ax2) = plt.subplots(
        1, 2, figsize=(10.4, 4.4), facecolor="white",
        gridspec_kw={"width_ratios": [1.5, 1]})

    # --- left: seconds per iteration vs crop (log-log) -------------------- #
    for tag, label, color in ENGINES:
        main_vals = data[(tag, main_suffix)]
        if not main_vals:
            continue
        x = sorted(main_vals)
        y = [main_vals[c] for c in x]
        ax.plot(x, y, "-o", color=color, lw=2, ms=7, label=label, zorder=3)
        for suffix, _ in others:
            chk = data[(tag, suffix)]
            cx = sorted(chk)
            ax.plot(cx, [chk[c] for c in cx], "^", color=color, ms=6,
                    mfc="white", mew=1.6, zorder=3)
        ax.annotate("%.2f s" % y[-1], (x[-1], y[-1]),
                    textcoords="offset points", xytext=(8, -3),
                    fontsize=9, color=TEXT1)

    # crop^2 guide through the reference engine's smallest crop
    ref = data.get((GUIDE_TAG, main_suffix), {})
    anchor = crops[0]
    if anchor in ref:
        g = np.array(crops, float)
        ax.plot(g, ref[anchor] * (g / anchor) ** 2, "--", color=TEXT2, lw=1,
                zorder=2, alpha=0.7)
        ax.annotate(r"$\propto$ crop$^2$",
                    (crops[-1], ref[anchor] * (crops[-1] / anchor) ** 2),
                    textcoords="offset points", xytext=(-2, 8),
                    fontsize=9, color=TEXT2, ha="right")

    ax.set_yscale("log")
    style_axis(ax, crops, "raw detector crop (pixels)",
               "engine time per iteration (s)",
               "ThreePIE speed vs crop — %s, %d frames, bin %d,\n"
               "%d slices, %d probe modes (%s)"
               % (scan_label(args.scan_dir, args.scan_label), args.frames,
                  args.binning, args.slices, args.probe_modes,
                  marker_note(main, others)))

    # --- right: GPU speedup factor ---------------------------------------- #
    gpu = data.get((GPU_TAG, main_suffix), {})
    for tag, label, color in ENGINES:
        if tag == GPU_TAG:
            continue
        vals = data[(tag, main_suffix)]
        x = sorted(set(vals) & set(gpu))
        if not x:
            continue
        y = [vals[c] / gpu[c] for c in x]
        ax2.plot(x, y, "-o", color=color, lw=2, ms=7, label="vs %s" % label)
        ax2.annotate("%.1fx" % y[-1], (x[-1], y[-1]),
                     textcoords="offset points", xytext=(6, -3),
                     fontsize=10, color=TEXT1)
    ax2.axhline(1.0, color=TEXT2, lw=1, ls=":")
    style_axis(ax2, crops, "raw detector crop (pixels)", "GPU speedup factor",
               "GPU advantage grows with crop\n(%s)" % args.gpu_note)

    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# cli
# --------------------------------------------------------------------------- #
def build_argparser():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--scan-dir", required=True,
                        help="Scan directory holding the "
                             "ptycho_ptypy_Mslice_* run folders, e.g. "
                             "<beamtime>/process/<sample>/scan_000434. "
                             "Required: there is no sensible default.")
    parser.add_argument("--outdir", default=".",
                        help="Directory the figure is written to, created if "
                             "missing (default: the working directory).")
    parser.add_argument("--filename", default="speed_vs_crop.png",
                        help="File name of the figure inside --outdir "
                             "(default: speed_vs_crop.png).")
    parser.add_argument("--crops", type=crop_list, default=DEFAULT_CROPS,
                        help="Comma-separated raw detector crops to read "
                             "(default: %s)." % DEFAULT_CROPS)
    parser.add_argument("--family", type=family_spec, action="append",
                        dest="families", default=None,
                        metavar="SUFFIX:NUMITER",
                        help="Run family: output-dir suffix plus the declared "
                             "iteration count of that run, repeatable, e.g. "
                             "--family _cmp100:100 --family _speed20:20 "
                             "(default: %s)."
                             % " ".join("%s:%d" % f for f in DEFAULT_FAMILIES))
    parser.add_argument("--main-family", default=None, metavar="SUFFIX",
                        help="Family drawing the solid line; every other "
                             "family becomes open triangles (default: the "
                             "family with the most iterations).")
    parser.add_argument("--binning", type=int, default=2,
                        help="Detector binning of the runs to read "
                             "(default: 2).")
    parser.add_argument("--slices", type=int, default=2,
                        help="Number of object slices of the runs to read "
                             "(default: 2).")
    parser.add_argument("--probe-modes", type=int, default=2,
                        help="Number of probe modes of the runs to read "
                             "(default: 2).")
    parser.add_argument("--frames", type=int, default=670,
                        help="Frame count quoted in the plot title; it is "
                             "annotation only and is not measured "
                             "(default: 670).")
    parser.add_argument("--gpu-note", default=DEFAULT_GPU_NOTE,
                        help="Parenthesised second line of the speedup panel "
                             "title; the default describes the scan-434 "
                             "reference dataset (default: %s)."
                             % DEFAULT_GPU_NOTE)
    parser.add_argument("--scan-label", default=None,
                        help="Scan name used in the plot title "
                             "(default: derived from --scan-dir).")
    parser.add_argument("--dpi", type=int, default=170,
                        help="Resolution of the saved figure (default: 170).")
    return parser


def main():
    args = build_argparser().parse_args()
    if isinstance(args.crops, str):          # untouched string default
        args.crops = crop_list(args.crops)
    families = args.families or list(DEFAULT_FAMILIES)

    known = [suffix for suffix, _ in families]
    if args.main_family is not None:
        if args.main_family not in known:
            raise SystemExit("--main-family %r is not one of the given "
                             "families %s" % (args.main_family, known))
        main_family = [f for f in families if f[0] == args.main_family][0]
    else:
        main_family = max(families, key=lambda f: f[1])
    others = [f for f in families if f[0] != main_family[0]]

    data = collect(args, families)
    print_table(data, args.crops)
    if not any(data.values()):
        raise SystemExit("no reconstructions found under %s -- check "
                         "--scan-dir, --crops, --family, --binning, --slices "
                         "and --probe-modes" % args.scan_dir)

    fig = make_figure(args, data, main_family, others)
    if args.outdir:
        os.makedirs(args.outdir, exist_ok=True)
    out = os.path.join(args.outdir, args.filename)
    fig.savefig(out, dpi=args.dpi)
    print("saved %s" % out)


if __name__ == "__main__":
    main()
