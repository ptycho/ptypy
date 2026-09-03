#!/usr/bin/env python3
"""
Diagnose crop-dependent ThreePIE multislice propagation limits.

The critical condition for angular-spectrum propagation between slices is:

    z <= N * dx**2 / lambda

where N and dx are the prepared real-space wavefront size and pixel size after
detector cropping/rebinning. Larger detector crops reduce dx and therefore make
the safe inter-slice propagation distance smaller.
"""

import argparse
import numpy as np


HC_KEV_M = 12.398419843320026e-10


def build_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--energy-kev", type=float, default=8.0)
    parser.add_argument("--detector-distance", type=float, default=4.150)
    parser.add_argument("--detector-pixel", type=float, default=75e-6)
    parser.add_argument("--binning", type=int, default=2)
    parser.add_argument("--suggest-binnings", default="1,2,4,8",
                        help="Comma-separated binnings to evaluate for each crop.")
    parser.add_argument("--slice-thickness", type=float, default=1500.0e-6,
                        help="Physical spacing between slices in meters. This "
                             "is diagnosed as a fixed experimental value, not "
                             "optimized as a reconstruction knob.")
    parser.add_argument("--crops", default="64,128,256,512",
                        help="Comma-separated raw detector crop sizes.")
    return parser


def bandlimit_keep_fraction(n, dx, wavelength, distance):
    vlim = 1.0 / np.sqrt((2.0 * distance / (n * dx)) ** 2 + 1.0)
    coord = np.arange(n)
    coord = ((coord + n // 2) % n) - n // 2
    v = coord * (wavelength / (n * dx))
    V, W = np.meshgrid(v, v, indexing="ij")
    return float(((np.abs(V) <= vlim) & (np.abs(W) <= vlim)).mean()), float(vlim), float(np.max(np.abs(v)))


def crop_sampling_stats(raw_crop, binning, wavelength, detector_distance,
                        detector_pixel, slice_thickness):
    """Return sampling diagnostics for one raw detector crop size."""
    if raw_crop % binning != 0:
        raise ValueError(f"crop {raw_crop} is incompatible with binning {binning}")
    n = raw_crop // binning
    det_psize = detector_pixel * binning
    dx = wavelength * detector_distance / (n * det_psize)
    zcrit = n * dx * dx / wavelength
    keep, vlim, vmax = bandlimit_keep_fraction(
        n, dx, wavelength, abs(slice_thickness))
    ratio = abs(slice_thickness) / zcrit
    # slice_pad="auto" picks ceil(ratio), capped at 4 in the engines
    auto_pad = min(max(1, int(np.ceil(ratio))), 4)
    if ratio <= 1.0:
        status = "unaliased"
    elif keep > 0.25:
        status = "needs-bandlimit"
    else:
        status = "strongly-bandlimited"
    return {
        "raw_crop": raw_crop,
        "prepared_n": n,
        "dx": dx,
        "zcrit": zcrit,
        "ratio": ratio,
        "keep_fraction": keep,
        "vlim": vlim,
        "vmax": vmax,
        "auto_pad": auto_pad,
        "status": status,
    }


def binning_suggestion(raw_crop, binnings, wavelength, detector_distance,
                       detector_pixel, slice_thickness):
    """
    Return the smallest listed binning that satisfies z <= zcrit.

    The slice distance is treated as fixed. This only asks which detector
    binning gives enough sampled pixels for ordinary ASM. With PtyPy's crop then
    rebin path, lowering binning can help; increasing binning generally does not
    raise the formal critical distance for a fixed raw crop.
    """
    checked = []
    for binning in sorted(set(int(b) for b in binnings)):
        if raw_crop % binning != 0:
            continue
        stats = crop_sampling_stats(
            raw_crop, binning, wavelength, detector_distance,
            detector_pixel, slice_thickness)
        checked.append(stats)
        if stats["ratio"] <= 1.0:
            return binning, stats, checked
    return None, None, checked


def padding_suggestion(stats):
    """Smallest integer zero-padding factor that makes z <= zcrit."""
    if stats["ratio"] <= 1.0:
        return 1
    return int(np.ceil(stats["ratio"]))


def recommendation_text(stats, current_binning, suggested_binning=None):
    """Return a concise fixed-distance recommendation for one crop row."""
    if stats["ratio"] <= 1.0:
        return (
            f"crop {stats['raw_crop']}: current binning {current_binning} is "
            "within the ordinary ASM sampling limit."
        )

    pad = padding_suggestion(stats)
    text = (
        f"crop {stats['raw_crop']}: keep the physical slice distance fixed; "
        f"use --slice-pad {pad} to preserve resolution"
    )
    if stats["keep_fraction"] < 0.5:
        text += (
            f", or keep slice_bandlimit=True knowing it keeps only "
            f"{100.0 * stats['keep_fraction']:.1f}% of frequencies"
        )
    if suggested_binning is not None and suggested_binning != current_binning:
        if suggested_binning < current_binning:
            text += f"; lower binning to {suggested_binning} is also sampled safely if memory allows"
        else:
            text += f"; binning {suggested_binning} is also sampled safely"
    return text + "."


def main():
    args = build_argparser().parse_args()
    wavelength = HC_KEV_M / args.energy_kev
    crops = [int(value.strip()) for value in args.crops.split(",") if value.strip()]
    suggest_binnings = [
        int(value.strip()) for value in args.suggest_binnings.split(",")
        if value.strip()
    ]

    print("ThreePIE crop propagation diagnostic")
    print(f"energy: {args.energy_kev:g} keV  lambda: {wavelength:.6e} m")
    print(f"detector distance: {args.detector_distance:g} m")
    print(f"detector pixel after binning: {args.detector_pixel * args.binning:.6e} m")
    print(f"slice thickness: {args.slice_thickness:.6e} m (fixed physical spacing)")
    print()
    print("raw_crop prepared_N dx_nm zcrit_mm slice/zcrit bandlimit_keep min_pad auto_pad suggested_binning status")

    recommendations = []
    for crop in crops:
        try:
            stats = crop_sampling_stats(
                crop, args.binning, wavelength, args.detector_distance,
                args.detector_pixel, args.slice_thickness)
        except ValueError:
            print(f"{crop:8d} incompatible with binning {args.binning}")
            continue
        print(
            f"{crop:8d} {stats['prepared_n']:10d} {stats['dx']*1e9:7.3f} "
            f"{stats['zcrit']*1e3:8.3f} {stats['ratio']:11.3f} "
            f"{100*stats['keep_fraction']:13.2f}% "
            f"{padding_suggestion(stats):7d} "
            f"{stats['auto_pad']:8d} ",
            end="",
        )
        if stats["ratio"] <= 1.0:
            print(f"{'current':>17s} {stats['status']}")
            recommendations.append(recommendation_text(stats, args.binning))
        else:
            binning, safe_stats, _ = binning_suggestion(
                crop, suggest_binnings, wavelength, args.detector_distance,
                args.detector_pixel, args.slice_thickness)
            if binning is None:
                print(f"{'none':>17s} {stats['status']}")
                recommendations.append(recommendation_text(stats, args.binning))
            elif binning == args.binning:
                print(f"{'current':>17s} {stats['status']}")
                recommendations.append(
                    recommendation_text(stats, args.binning, binning))
            else:
                print(
                    f"{('bin%d' % binning):>17s} "
                    f"{stats['status']} (safe dx={safe_stats['dx']*1e9:.1f} nm)"
                )
                recommendations.append(
                    recommendation_text(stats, args.binning, binning))

    print()
    print("Interpretation:")
    print("  slice/zcrit <= 1: ordinary ASM propagation is sampled safely.")
    print("  slice/zcrit > 1: wrap-around aliasing is expected without bandlimit.")
    print("  very small bandlimit_keep means high frequencies are heavily clipped.")
    print("  min_pad keeps the fixed slice distance and preserves resolution, but")
    print("  FFT memory/time grow roughly as min_pad^2.")
    print("  auto_pad is what slice_pad=\"auto\" picks (min_pad capped at 4; the")
    print("  band limit cleans any residual beyond the cap).")
    print("  suggested_binning is the smallest tested binning that makes ordinary")
    print("  ASM safe for the same fixed slice distance. For PtyPy crop-then-rebin,")
    print("  this may mean less binning, not more binning.")
    print()
    print("Recommendations:")
    for text in recommendations:
        print(f"  {text}")


if __name__ == "__main__":
    main()
