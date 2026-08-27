#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Compare the final ThreePIE reconstructions of one scan across backends and crops.

For a given scan folder and run family (the ``--suffix`` that was appended to
the output directory names), this loads the last ``rec_*.ptyr`` of every
(engine, crop) reconstruction, prints a backend-agreement table and renders an
object-phase panel figure (rows = crop, cols = engine).

Agreement is measured with the phase- and scale-invariant normalized
correlation ``ncorr``; the ``obj-aligned`` column additionally removes the joint
probe/object translation gauge, which stochastic ePIE-type engines are free to
pick differently on every run. Both metrics are taken on the central region
only, because the border of the object is barely covered by the scan.

With the binning fixed, every crop reconstructs the same field of view and only
the pixel size changes (dx ~ 1/crop), so all panels of the figure share the same
physical extent and can be compared directly.

The reconstruction folders are expected to be named the way
``run_threepie_realdata_matrix.py`` writes them::

    <base>/ptycho_ptypy_Mslice_<tag>_crop-<crop>_bin-<bin>_slices-<S>_pmodes-<P><suffix>/rec/rec_*.ptyr

Typical use (from the repo root, with the ptypy_v8 environment). Run it as a
module so ``import ptypy`` resolves to this checkout without any path hacking;
if ptypy is installed, or PYTHONPATH points at the repo root, calling the file
by path works just as well::

    python -m ptypy.debug.compare_threepie_crops \
        --base <beamtime>/process/0002_multislice/scan_000434 \
        --outdir /tmp/threepie_crops --suffix _cmp100

    python -m ptypy.debug.compare_threepie_crops \
        --base <beamtime>/process/0002_multislice/scan_000434 \
        --outdir /tmp/threepie_crops --suffix _speed20 --tag 20it

This file is part of the PTYPY package.

    :copyright: Copyright 2014 by the PTYPY team, see AUTHORS.
    :license: see LICENSE for details.
"""

import argparse
import os
import re

import numpy as np

import matplotlib
matplotlib.use("Agg")  # headless: must be set before pyplot is imported
import matplotlib.pyplot as plt  # noqa: E402

try:                             # normal in-package import
    from ptypy.debug.threepie_compare import (  # noqa: E402
        ncorr,
        central,
        gauge_phase,
        aligned_ncorr,
        read_recon,
        find_latest,
    )
except ImportError:              # executed as a bare script from this folder
    from threepie_compare import (  # noqa: E402
        ncorr,
        central,
        gauge_phase,
        aligned_ncorr,
        read_recon,
        find_latest,
    )

# Engine name -> the tag that appears in the output directory name.
ENGINE_DIRTAG = {
    "ThreePIE": "cpu",
    "ThreePIE_serial": "serial",
    "ThreePIE_cupy": "gpu",
}

# Backend pairs of the agreement table, in printing order.
PAIRS = (
    ("ThreePIE_serial", "ThreePIE", "serial-vs-cpu"),
    ("ThreePIE_cupy", "ThreePIE", "gpu-vs-cpu"),
    ("ThreePIE_cupy", "ThreePIE_serial", "gpu-vs-serial"),
)

DEFAULT_CROPS = "128,256,512"


def crop_list(value):
    """argparse type: comma-separated crops -> list of ints."""
    crops = [int(c) for c in str(value).split(",") if c.strip()]
    if not crops:
        raise argparse.ArgumentTypeError(
            "expected at least one crop, got %r" % (value,))
    return crops


def scan_label(base):
    """
    Short scan label for titles, derived from the scan folder name.

    ``.../scan_000434`` -> ``434``; anything else falls back to the basename.
    """
    name = os.path.basename(os.path.normpath(base))
    match = re.match(r"^scan[_-]?(\d+)$", name)
    if match:
        return str(int(match.group(1)))
    return name


def load_rec(base, dirtag, crop, args):
    """
    Latest ``.ptyr`` of one (engine, crop) reconstruction, or None if missing.

    Returns the ``read_recon`` dict with ``probe`` reduced to the first probe
    mode, which is what the agreement table compares.
    """
    pattern = os.path.join(
        base,
        "ptycho_ptypy_Mslice_%s_crop-%d_bin-%d_slices-%d_pmodes-%d%s"
        % (dirtag, crop, args.bin, args.slices, args.pmodes, args.suffix),
        "rec", "rec_*.ptyr")
    path = find_latest(pattern)
    if path is None:
        return None
    rec = read_recon(path)
    rec["probe"] = rec["probe"][0]
    return rec


def collect(args):
    """Load every available (crop, engine) reconstruction into a dict."""
    data = {}
    for crop in args.crops:
        for engine, dirtag in ENGINE_DIRTAG.items():
            rec = load_rec(args.base, dirtag, crop, args)
            if rec is None:
                print("missing: %s crop %d%s" % (dirtag, crop, args.suffix))
                continue
            data[(crop, engine)] = rec
    return data


def agreement_report(data, args, label):
    """Backend-agreement table for every crop, as a single string."""
    lines = ["real data scan %s, suffix %s" % (label, args.suffix),
             "pair                          crop   object  "
             "obj-aligned(shift)   probe"]
    for crop in args.crops:
        for a, b, ptag in PAIRS:
            if (crop, a) not in data or (crop, b) not in data:
                continue
            oa = central(data[(crop, a)]["obj"])
            ob = central(data[(crop, b)]["obj"])
            co = ncorr(oa, ob)
            shift, ca = aligned_ncorr(oa, ob)
            cp = ncorr(data[(crop, a)]["probe"], data[(crop, b)]["probe"])
            lines.append("%-28s %5d   %.4f  %.4f %9s   %.4f"
                         % (ptag, crop, co, ca, str(shift), cp))
    return "\n".join(lines)


def phase_panel(data, args, label, out_png):
    """
    Object-phase panel figure: rows = crop, cols = engine.

    The global phase gauge is removed per panel (wrap-safely, via
    ``gauge_phase``) and robust 1/99 percentile color limits are shared across
    each row so the backends of one crop are on one scale.
    """
    engines = list(ENGINE_DIRTAG)
    fig, axes = plt.subplots(len(args.crops), len(engines),
                             figsize=(3.9 * len(engines),
                                      3.6 * len(args.crops)),
                             squeeze=False)
    for i, crop in enumerate(args.crops):
        phases = {eng: gauge_phase(central(data[(crop, eng)]["obj"]))
                  for eng in engines if (crop, eng) in data}
        vmin = vmax = None
        if phases:
            pooled = np.concatenate([p.ravel() for p in phases.values()])
            vmin, vmax = np.percentile(pooled, [1, 99])
        im = None
        for j, eng in enumerate(engines):
            ax = axes[i, j]
            ax.set_xticks([])
            ax.set_yticks([])
            if eng not in phases:
                ax.text(0.5, 0.5, "missing", ha="center", va="center",
                        transform=ax.transAxes)
                continue
            rec = data[(crop, eng)]
            half = phases[eng].shape[-1] / 2 * rec["psize"] * 1e6
            im = ax.imshow(phases[eng], cmap="gray", vmin=vmin, vmax=vmax,
                           extent=(-half, half, -half, half))
            ax.set_xticks([])
            ax.set_yticks([])
            if i == 0:
                ax.set_title("%s (%s)" % (eng, ENGINE_DIRTAG[eng]), fontsize=11)
            if j == 0:
                ax.set_ylabel("crop %d\n(dx %.0f nm)"
                              % (crop, rec["psize"] * 1e9), fontsize=11)
        if im is not None:
            cb = fig.colorbar(im, ax=axes[i, -1], fraction=0.046, pad=0.03)
            cb.set_label("phase (rad)", fontsize=8)
            cb.ax.tick_params(labelsize=7)
    fig.suptitle("Scan %s ThreePIE object phase, bin %d, %d slices [%s]"
                 % (label, args.bin, args.slices, args.suffix), fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def build_argparser():
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n")[1],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--base", required=True,
                        help="Scan process directory holding the "
                             "ptycho_ptypy_Mslice_* reconstruction folders, "
                             "e.g. <beamtime>/process/<sample>/scan_000434.")
    parser.add_argument("--outdir", default=".",
                        help="Directory for the comparison table and figure; "
                             "created if missing.")
    parser.add_argument("--suffix", default="_cmp100",
                        help="Output-directory suffix selecting the run "
                             "family, e.g. _cmp100 or _speed20.")
    parser.add_argument("--crops", type=crop_list, default=DEFAULT_CROPS,
                        help="Comma-separated crops to compare.")
    parser.add_argument("--bin", type=int, default=2,
                        help="Binning factor appearing in the folder names.")
    parser.add_argument("--slices", type=int, default=2,
                        help="Number of slices appearing in the folder names.")
    parser.add_argument("--pmodes", type=int, default=2,
                        help="Number of probe modes appearing in the folder "
                             "names.")
    parser.add_argument("--tag", default=None,
                        help="Label for output filenames "
                             "(default: the suffix without underscores).")
    parser.add_argument("--scan-label", default=None,
                        help="Scan label for the table header and figure "
                             "title (default: derived from --base).")
    return parser


def main():
    args = build_argparser().parse_args()
    tag = args.tag or args.suffix.strip("_")
    label = args.scan_label or scan_label(args.base)

    outdir = os.path.abspath(args.outdir)
    os.makedirs(outdir, exist_ok=True)

    data = collect(args)

    # Backend agreement per crop (same grid), central region; the "aligned"
    # column removes the joint probe+object translation gauge mode.
    report = agreement_report(data, args, label)
    print(report)
    out_txt = os.path.join(outdir, "real_crop_comparison_%s.txt" % tag)
    with open(out_txt, "w") as fh:
        fh.write(report + "\n")

    out_png = os.path.join(outdir, "real_crop_comparison_%s.png" % tag)
    phase_panel(data, args, label, out_png)
    print("saved %s" % out_png)


if __name__ == "__main__":
    main()
