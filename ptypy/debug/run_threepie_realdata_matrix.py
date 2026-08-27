#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Launch a small ThreePIE real-data crop/bin matrix on the NanoMAX multislice scan.

This is a thin orchestrator around the beamtime runner
``<beamtime-basedir>/scripts/run_threepie_cupy_nanomax.py``: it builds one
runner command per (engine, crop) combination, so CPU (``ThreePIE``),
serialized (``ThreePIE_serial``) and GPU (``ThreePIE_cupy``) reconstructions of
the same scan land in separate, systematically named output folders:

    process/<sample>/scan_<scan>/ptycho_ptypy_Mslice_<tag>_crop-<crop>_bin-<bin>_
        slices-<S>_pmodes-<P><output-suffix>/

The helpers in this module (``output_suffix``, ``detector_pixel``,
``diagnostic_crops``, ``add_common_run_args``, ``positive_int``) are unit-tested
by ``test/util_tests/threepie_matrix_runner_test.py`` -- keep their behaviour in
sync with that contract and with the argparser of the beamtime runner script.

Typical use (from the repo root, with the ptypy_v8 environment):

    python ptypy/debug/run_threepie_realdata_matrix.py --dry-run
    python ptypy/debug/run_threepie_realdata_matrix.py \
        --engines ThreePIE_serial,ThreePIE_cupy --crop 256 --slice-pad 2
"""

import argparse
import copy
import os
import subprocess
import sys

# Reference slice spacing of the NanoMAX 0002_multislice scan 434:
# both sample layers sit ~750 um up/downstream of focus -> 1.5 mm spacing.
DEFAULT_SLICE_THICKNESS = 1500.0e-6

# Detector pixel sizes matched to the defaults of run_threepie_cupy_nanomax.py.
DETECTOR_PIXEL = {
    "eiger4m": 75e-6,
    "merlin": 55e-6,
    "pilatus": 172e-6,
}


def positive_int(value):
    """argparse type: strictly positive integer (used for --slice-pad)."""
    ivalue = int(value)
    if ivalue < 1:
        raise argparse.ArgumentTypeError(
            "expected a positive integer, got %r" % (value,))
    return ivalue


def detector_pixel(args):
    """Detector pixel size in meters; --detector-pixel overrides the lookup."""
    if getattr(args, "detector_pixel", None) is not None:
        return args.detector_pixel
    return DETECTOR_PIXEL[args.detector]


def output_suffix(args):
    """
    Output-directory suffix for one matrix run.

    An explicit --output-suffix always wins. The default encodes the raw crop
    plus any non-default physics options so control runs never overwrite the
    baselines:  _LT_debug<crop>[_z<um>um][_pad<N>]
    """
    if getattr(args, "output_suffix", None) is not None:
        return args.output_suffix
    suffix = "_LT_debug%d" % args.crop
    thickness = args.slice_thickness
    if abs(thickness - DEFAULT_SLICE_THICKNESS) > 1e-12:
        suffix += "_z%dum" % int(round(thickness * 1e6))
    if args.slice_pad > 1:
        suffix += "_pad%d" % args.slice_pad
    return suffix


def diagnostic_crops(args):
    """
    Crops for the geometry diagnostic, as a comma-separated string.

    Defaults to the requested crop plus the 128/512 transition context
    (crop 128 converges on this scan, larger crops historically did not).
    """
    if getattr(args, "diagnostic_crops", None):
        return args.diagnostic_crops
    crops = sorted(set([128, args.crop, 512]))
    return ",".join(str(c) for c in crops)


def add_common_run_args(cmd, args):
    """
    Append the runner arguments shared by every engine in the matrix to ``cmd``.

    The flag names must match the argparser of run_threepie_cupy_nanomax.py.
    """
    cmd += ["--ptypy-path", str(args.ptypy_path)]
    cmd += ["--beamtime-basedir", str(args.beamtime_basedir)]
    cmd += ["--sample", str(args.sample)]
    cmd += ["--detector", str(args.detector)]
    cmd += ["--scan", str(args.scan)]
    cmd += ["--distance", str(args.distance)]
    cmd += ["--defocus-um", str(args.defocus_um)]
    cmd += ["--energy-kev", str(args.energy_kev)]
    cmd += ["--crop", str(args.crop)]
    cmd += ["--center-y", str(args.center_y)]
    cmd += ["--center-x", str(args.center_x)]
    cmd += ["--binning", str(args.binning)]
    cmd += ["--probe-modes", str(args.probe_modes)]
    cmd += ["--numiter", str(args.numiter)]
    cmd += ["--save-every", str(args.save_every)]
    cmd += ["--number-of-slices", str(args.number_of_slices)]
    cmd += ["--slice-thickness", str(args.slice_thickness)]
    cmd += ["--slice-start-iteration", str(args.slice_start_iteration)]
    cmd += ["--output-suffix", output_suffix(args)]
    if getattr(args, "frames_per_block", None) is not None:
        cmd += ["--frames-per-block", str(args.frames_per_block)]
    if getattr(args, "no_slice_bandlimit", False):
        cmd += ["--no-slice-bandlimit"]
    cmd += ["--slice-pad", str(args.slice_pad)]
    return cmd


def build_argparser():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--engines", default="ThreePIE_cupy",
                        help="Comma-separated engine list, e.g. "
                             "ThreePIE,ThreePIE_serial,ThreePIE_cupy.")
    parser.add_argument("--runner", default=None,
                        help="Path to run_threepie_cupy_nanomax.py "
                             "(default: <beamtime-basedir>/scripts/).")
    # Common physics/geometry arguments, defaults matched to the runner.
    parser.add_argument("--ptypy-path", default="/home/litang/ptypy_v8/dev-work/")
    parser.add_argument("--beamtime-basedir", default="/home/litang/multislice")
    parser.add_argument("--sample", default="0002_multislice")
    parser.add_argument("--detector", default="eiger4m")
    parser.add_argument("--detector-pixel", type=float, default=None,
                        help="Override the per-detector pixel size in meters.")
    parser.add_argument("--scan", type=int, default=434)
    parser.add_argument("--distance", type=float, default=4.150)
    parser.add_argument("--defocus-um", type=float, default=-750.0)
    parser.add_argument("--energy-kev", type=float, default=8.0)
    parser.add_argument("--crop", type=int, default=256)
    parser.add_argument("--center-y", type=float, default=1281.0)
    parser.add_argument("--center-x", type=float, default=772.0)
    parser.add_argument("--binning", type=int, default=2)
    parser.add_argument("--probe-modes", type=int, default=2)
    parser.add_argument("--numiter", type=int, default=100)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--number-of-slices", type=int, default=2)
    parser.add_argument("--slice-thickness", type=float,
                        default=DEFAULT_SLICE_THICKNESS)
    parser.add_argument("--slice-start-iteration", default="0")
    parser.add_argument("--frames-per-block", type=int, default=None)
    parser.add_argument("--no-slice-bandlimit", action="store_true",
                        help="Disable the serial engine's inter-slice "
                             "band limit (band limit is ON by default).")
    parser.add_argument("--slice-pad", type=positive_int, default=1,
                        help="Zero-pad factor for the inter-slice propagator "
                             "(serial engine only).")
    parser.add_argument("--output-suffix", default=None,
                        help="Explicit output-dir suffix; default is derived "
                             "from crop / thickness / padding.")
    parser.add_argument("--diagnostic-crops", default=None,
                        help="Crops for the geometry diagnostic printout.")
    parser.add_argument("--matrix-crops", default=None,
                        help="Comma-separated crops to run per engine "
                             "(default: just --crop).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the commands without launching anything.")
    return parser


def build_matrix_commands(args):
    """One runner command per (engine, crop); returns a list of (label, cmd)."""
    runner = args.runner or os.path.join(
        args.beamtime_basedir, "scripts", "run_threepie_cupy_nanomax.py")
    engines = [e.strip() for e in args.engines.split(",") if e.strip()]
    crops = ([int(c) for c in args.matrix_crops.split(",")]
             if args.matrix_crops else [args.crop])
    commands = []
    for crop in crops:
        for engine in engines:
            run_args = copy.copy(args)
            run_args.crop = crop
            cmd = [sys.executable, runner, "--engine", engine]
            add_common_run_args(cmd, run_args)
            commands.append(("%s crop=%d" % (engine, crop), cmd))
    return commands


def main():
    args = build_argparser().parse_args()
    print("diagnostic crops: %s" % diagnostic_crops(args))
    print("detector pixel:   %.3e m" % detector_pixel(args))
    commands = build_matrix_commands(args)
    results = []
    for label, cmd in commands:
        print("\n=== %s ===" % label)
        print(" ".join(cmd))
        if args.dry_run:
            continue
        proc = subprocess.run(cmd)
        results.append((label, proc.returncode))
        print("--- %s finished with exit code %d" % (label, proc.returncode))
    if not args.dry_run:
        print("\n=== matrix summary ===")
        for label, code in results:
            print("%-32s %s" % (label, "OK" if code == 0 else "FAIL(%d)" % code))
        if any(code != 0 for _, code in results):
            sys.exit(1)


if __name__ == "__main__":
    main()
