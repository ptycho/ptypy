#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Per-slice comparison of ThreePIE reconstructions across CPU/serial/GPU backends.

The point of multislice is to separate information between the slices, so this
compares the individual slice objects (not their product):

  * cross-backend agreement per slice:  ncorr(A slice i, B slice i)
  * slice-swap check:                   ncorr(A slice 0, B slice 1) and the
    reverse -- a high value here means the backends ordered the layers
    differently, or that the layers are duplicates of each other
  * separation within one engine:       ncorr(slice 0, slice 1). Low means the
    two layers carry distinct structure, high means leakage/duplication.

Slice objects are read from the ``fslices`` files the engines write at
``<run-dir>/rec/slices_*_iter-*.h5``; the newest match wins. Run directories are
the systematically named outputs of ``run_threepie_realdata_matrix.py``:

    <base-dir>/ptycho_ptypy_Mslice_<tag>_crop-<crop>_bin-<bin>_
        slices-<S>_pmodes-<P><suffix>/

Every comparison goes through ``aligned_ncorr`` from ``ptypy.debug.threepie_compare``,
because a ptychographic solution is only defined up to a global phase and a joint
probe/object translation, and stochastic engines shuffle their view order
independently.

Two outputs land in ``--outdir``: ``slice_comparison_<tag>.txt`` (the report,
also printed) and ``slice_comparison_<tag>.png`` (rows = crop x slice,
cols = engine; object phase with the global phase gauge removed).

Typical use (from the repo root, with the ptypy_v8 environment):

    python -m ptypy.debug.compare_threepie_slices \\
        --base-dir <beamtime>/process/0002_multislice/scan_000434 \\
        --suffix _cmp100 --outdir /tmp/slicecmp

    python ptypy/debug/compare_threepie_slices.py --base-dir <scan-dir> --help

This file is part of the PTYPY package.

    :copyright: Copyright 2014 by the PTYPY team, see AUTHORS.
    :license: see LICENSE for details.
"""

import argparse
import os

import numpy as np

import matplotlib
matplotlib.use("Agg")           # these run headless; must precede pyplot
import matplotlib.pyplot as plt  # noqa: E402

try:                            # normal in-package import
    from ptypy.debug.threepie_compare import (
        ncorr, aligned_ncorr, central, gauge_phase, read_slices, find_latest)
except ImportError:             # executed as a bare script from this folder
    from threepie_compare import (
        ncorr, aligned_ncorr, central, gauge_phase, read_slices, find_latest)


# Engine class name -> the tag used in the reconstruction directory name.
ENGINE_DIRTAG = {"ThreePIE": "cpu",
                 "ThreePIE_serial": "serial",
                 "ThreePIE_cupy": "gpu"}

# Backend pairs for the per-slice cross-backend table, (a, b, label).
CROSS_PAIRS = (("ThreePIE_serial", "ThreePIE", "serial-vs-cpu"),
               ("ThreePIE_cupy", "ThreePIE", "gpu-vs-cpu"),
               ("ThreePIE_cupy", "ThreePIE_serial", "gpu-vs-serial"))

# The swap check only needs the two comparisons against the CPU reference.
SWAP_PAIRS = CROSS_PAIRS[:2]


def run_dir(args, dirtag, crop):
    """Reconstruction directory for one (engine tag, crop) combination."""
    name = ("ptycho_ptypy_Mslice_%s_crop-%d_bin-%d_slices-%d_pmodes-%d%s"
            % (dirtag, crop, args.binning, args.slices, args.pmodes,
               args.suffix))
    return os.path.join(args.base_dir, name)


def load_slices(args, dirtag, crop):
    """
    Per-slice objects of the newest ``fslices`` file, or None if there is none.

    Returns ``{slice_index: complex ndarray}``.
    """
    pattern = os.path.join(run_dir(args, dirtag, crop),
                           "rec", "slices_*_iter-*.h5")
    path = find_latest(pattern)
    if path is None:
        return None
    return read_slices(path)


def collect(args, crops):
    """Load every available (crop, engine) slice set; report what is missing."""
    data = {}
    for crop in crops:
        for engine, dirtag in ENGINE_DIRTAG.items():
            slices = load_slices(args, dirtag, crop)
            if slices is None:
                print("missing slices file: %s crop %d%s"
                      % (dirtag, crop, args.suffix))
                continue
            data[(crop, engine)] = slices
    return data


def build_report(args, data, crops, scan_label):
    """The three text tables, as a list of lines."""
    lines = ["per-slice comparison, %s, suffix %s" % (scan_label, args.suffix),
             "",
             "cross-backend per slice (central 60%, raw / aligned):",
             "pair                          crop  slice   raw     aligned(shift)"]
    for crop in crops:
        for a, b, ptag in CROSS_PAIRS:
            if (crop, a) not in data or (crop, b) not in data:
                continue
            for i in range(args.slices):
                oa = central(data[(crop, a)][i])
                ob = central(data[(crop, b)][i])
                shift, ca = aligned_ncorr(oa, ob)
                lines.append("%-28s %5d  %5d   %.4f  %.4f %9s"
                             % (ptag, crop, i, ncorr(oa, ob), ca, str(shift)))

    lines += ["", "slice-swap check ncorr(A slice0, B slice1) (aligned):"]
    for crop in crops:
        for a, b, ptag in SWAP_PAIRS:
            if (crop, a) not in data or (crop, b) not in data:
                continue
            _, c01 = aligned_ncorr(central(data[(crop, a)][0]),
                                   central(data[(crop, b)][1]))
            _, c10 = aligned_ncorr(central(data[(crop, a)][1]),
                                   central(data[(crop, b)][0]))
            lines.append("%-28s %5d   0-vs-1 %.4f   1-vs-0 %.4f"
                         % (ptag, crop, c01, c10))

    lines += ["",
              "separation within engine ncorr(slice0, slice1) (low = separated):"]
    for crop in crops:
        row = ["crop %d: " % crop]
        for engine in ENGINE_DIRTAG:
            if (crop, engine) not in data:
                continue
            c = ncorr(central(data[(crop, engine)][0]),
                      central(data[(crop, engine)][1]))
            row.append("%s %.4f" % (ENGINE_DIRTAG[engine], c))
        lines.append("  ".join(row))
    return lines


def render_figure(args, data, crops, scan_label, out_png):
    """
    Per-slice object-phase panel: rows = crop x slice, cols = engine.

    Each panel's global phase gauge is removed (rotate by the phase of the
    complex mean -- wrap-safe), and the engines in a row share robust percentile
    color limits so large particles keep their internal contrast.
    """
    engines = list(ENGINE_DIRTAG)
    nrows = len(crops) * args.slices
    fig, axes = plt.subplots(nrows, len(engines), squeeze=False,
                             figsize=(3.9 * len(engines), 3.6 * nrows))
    for ic, crop in enumerate(crops):
        for isl in range(args.slices):
            r = ic * args.slices + isl
            phases = {eng: gauge_phase(central(data[(crop, eng)][isl]))
                      for eng in engines if (crop, eng) in data}
            if phases:
                pooled = np.concatenate([p.ravel() for p in phases.values()])
                vmin, vmax = np.percentile(pooled, [1, 99])
            im = None
            for j, eng in enumerate(engines):
                ax = axes[r, j]
                ax.set_xticks([])
                ax.set_yticks([])
                if eng not in phases:
                    ax.text(0.5, 0.5, "missing", ha="center", va="center",
                            transform=ax.transAxes)
                    continue
                im = ax.imshow(phases[eng], cmap="gray", vmin=vmin, vmax=vmax)
                if r == 0:
                    ax.set_title("%s (%s)" % (eng, ENGINE_DIRTAG[eng]),
                                 fontsize=11)
                if j == 0:
                    ax.set_ylabel("crop %d\nslice %d" % (crop, isl), fontsize=11)
            if im is not None:
                cb = fig.colorbar(im, ax=axes[r, -1], fraction=0.046, pad=0.03)
                cb.set_label("phase (rad)", fontsize=8)
                cb.ax.tick_params(labelsize=7)
    fig.suptitle("%s per-slice object phase (slice 0 = upstream layer, "
                 "slice 1 = downstream) [%s]" % (scan_label, args.suffix),
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_png, dpi=130)
    return out_png


def build_argparser():
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n")[1],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--base-dir", required=True,
                        help="Scan directory holding the "
                             "ptycho_ptypy_Mslice_* reconstruction folders, "
                             "e.g. <beamtime>/process/<sample>/scan_000434.")
    parser.add_argument("--suffix", default="_cmp100",
                        help="Output-suffix of the reconstruction folders to "
                             "compare (the trailing part of the directory "
                             "name after pmodes-<P>).")
    parser.add_argument("--crops", default="128,256,512",
                        help="Comma-separated raw detector crops to compare.")
    parser.add_argument("--binning", type=int, default=2,
                        help="Binning encoded in the directory names (bin-<N>).")
    parser.add_argument("--slices", type=int, default=2,
                        help="Number of slices: used both in the directory "
                             "name (slices-<S>) and as the number of slice "
                             "objects compared per reconstruction.")
    parser.add_argument("--pmodes", type=int, default=2,
                        help="Probe modes encoded in the directory names "
                             "(pmodes-<P>).")
    parser.add_argument("--outdir", default=".",
                        help="Directory for slice_comparison_<tag>.txt/.png.")
    parser.add_argument("--tag", default=None,
                        help="Label for the output filenames "
                             "(default: the suffix without underscores).")
    return parser


def main():
    args = build_argparser().parse_args()
    tag = args.tag or args.suffix.strip("_")
    crops = [int(c) for c in args.crops.split(",") if c.strip()]
    scan_label = os.path.basename(os.path.normpath(args.base_dir))
    os.makedirs(args.outdir, exist_ok=True)

    data = collect(args, crops)
    if not data:
        raise SystemExit("no slices files found under %s for suffix %s"
                         % (args.base_dir, args.suffix))

    report = "\n".join(build_report(args, data, crops, scan_label))
    print(report)
    out_txt = os.path.join(args.outdir, "slice_comparison_%s.txt" % tag)
    with open(out_txt, "w") as fh:
        fh.write(report + "\n")
    print("saved %s" % out_txt)

    out_png = os.path.join(args.outdir, "slice_comparison_%s.png" % tag)
    render_figure(args, data, crops, scan_label, out_png)
    print("saved %s" % out_png)


if __name__ == "__main__":
    main()
