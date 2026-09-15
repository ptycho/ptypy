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

plus ``product``, the aligned ncorr of the product of the reconstructed
slices against the product of the two phantoms (the thin-sample projection,
which stays well defined even when the slices themselves are not separable),
and the probes: every engine's reconstructed illumination against the true
probe at slice 0 (free-space propagation is unitary, so the propagated
probe at slice 1 would score identically and is only drawn, not scored),
and the slice-1 wave the engine holds after its last view (the incident
wave of that view, as updated by the last-slice probe update) against the
true wave at that scan position (true probe times the true slice-0 patch,
propagated to slice 1; the engines record which view they processed last)
and against the free-space propagation of its own illumination.

Figures: ``sim_gt_slices.png`` whose first column is the ground truth and
whose remaining columns are the engines, one row per slice, and
``sim_gt_probes.png`` with the same column layout for the probes (amplitude
and phase at slice 0, free-space propagated to slice 1, and the last-view
incident wave at slice 1).

The probe that generates the data is either the MoonFlower "moon" probe
(default; with ``focus_mid`` it is back-propagated so that its focus sits
midway between the slices) or, with ``--probe gaussian``, an analytic beam
with a Gaussian intensity profile and a converging spherical phase of
numerical aperture ``--probe-na`` (the sine of the half-angle of convergence
at the 1/e^2 intensity radius). Its size at slice 0 is either given as
``--probe-fwhm`` (then the focus lies ``w / NA`` downstream of slice 0, ``w``
the 1/e^2 radius) or, when omitted, chosen so that the focus is midway
between the slices. The script prints the resulting beam sizes and warns
when the NA exceeds the angular sampling of the frame, ``lambda / (2 dx)``.

The slice separation is chosen either as a fraction of the angular-spectrum
critical distance (``--sep-frac``, default) or as a multiple of the depth of
field DOF = 5.2 dx^2/lambda (``--sep-dof``). Values of ``--sep-dof`` at or
below 1 put the whole two-layer sample inside one depth of field: there the
slices are not expected to separate, but the product must still be recovered
and the backends must still agree.

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

    sim_gt_slices.npz     ground truth + every engine's slices and probes
    sim_gt_slices.txt     the printed report
    sim_gt_slices.png     figure, rows = slice, cols = ground truth + engines
    sim_gt_probes.png     probe figure, same column layout
    sim_gt_gradient.png   phase gradient (d/dx, d/dy) of every slice
    sim_gt_probes_complex.png  complex colour-coded probes per slice
    sim_gt_overview.png   recovered probes (complex colour) above the
                          reconstructed slices (phase)
    slices_gt_<engine>.h5 the raw per-engine ``fslices`` dumps
    recons/<run>/<run>_<engine>_<iter>.ptyr   the ptypy reconstruction files

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

--seed-engines N gives every engine the same seeded view order. Without it
each engine shuffles its views with its own unseeded generator, and the
panels then differ by the draw as much as by the backend: one engine run
twice agrees with itself at only 0.6 to 0.8 per slice at shape 64. Set it
for any run that compares backends.
"""

import argparse
import contextlib
import importlib
import importlib.util
import os
from unittest import mock

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


def make_phantom(spec, shape, phase_max=1.0, amp_depth=0.15, spokes=24,
                 flowers=None):
    """
    Object phantom for one slice on the object frame ``shape``.

    ``spec``: "flowers" (the MoonFlower object, passed in as ``flowers``),
    "star" (the spoke star), "flowers-fine[:zoom]" (the flower image shrunk
    by ``zoom`` so that more of its structure falls into the field) or
    "image:<path>[:zoom]" (any image file: its grey level g in [0, 1] becomes
    phase ``phase_max * (g - mean)`` and amplitude ``1 - amp_depth * (1 - g)``,
    shrunk by ``zoom`` and mirror-padded or cropped to the frame). Returns a
    complex64 array of shape ``shape``.
    """
    from ptypy import utils as u
    shape = tuple(int(v) for v in shape)
    if spec == "flowers":
        if flowers is None:
            from ptypy import resources
            flowers = resources.flower_obj(shape)
        return np.asarray(flowers, dtype=np.complex64)
    if spec == "star":
        return spoke_star(shape, spokes=spokes)
    parts = spec.split(":")
    kind = parts[0]
    if kind == "flowers-fine":
        from ptypy import resources
        zoom = float(parts[1]) if len(parts) > 1 else 2.0
        im = resources.flower_obj(None)
        im = u.zoom(im, 1.0 / zoom)
        d = np.array(shape) - np.array(im.shape[:2])
        im = u.crop_pad(im, d, axes=[0, 1], cen=None, fillpar=0.0,
                        filltype='mirror')
        return np.asarray(im, dtype=np.complex64)
    if kind == "image":
        from PIL import Image
        rest = spec.split(":", 1)[1] if ":" in spec else ""
        if not rest:
            raise ValueError("image phantom needs a path: image:<path>[:zoom]")
        path, _, tail = rest.rpartition(":")
        try:
            zoom = float(tail)
        except ValueError:
            path, zoom = rest, 1.0
        if not path:
            path, zoom = rest, 1.0
        g = np.asarray(Image.open(path).convert("L"), dtype=np.float64) / 255.0
        if zoom != 1.0:
            g = u.zoom(g, 1.0 / zoom)
        d = np.array(shape) - np.array(g.shape)
        g = u.crop_pad(g, d, axes=[0, 1], cen=None, fillpar=0.0,
                       filltype='mirror')
        g = np.clip(g, 0.0, 1.0)
        phase = phase_max * (g - g.mean())
        amp = 1.0 - amp_depth * (1.0 - g)
        return (amp * np.exp(1j * phase)).astype(np.complex64)
    raise ValueError("unknown phantom %r" % spec)


FWHM_PER_W = np.sqrt(2.0 * np.log(2.0))   # intensity FWHM / 1/e^2 radius of a Gaussian


def gaussian_probe_geometry(na, fwhm, slice_sep, focus=None):
    """
    ``(w, fwhm, focus)`` of the Gaussian/NA probe at slice 0: 1/e^2 intensity
    radius, intensity FWHM and the distance to the focus (positive
    downstream, negative for a beam diverging from a focus upstream of
    slice 0). With ``fwhm`` None the focus is put midway between the
    slices and the size follows from the NA; with ``fwhm`` given the focus
    follows from ``w / NA`` unless ``focus`` is given explicitly, in which
    case the NA follows from ``w / |focus|`` and ``na`` is ignored.
    """
    if focus is not None:
        if fwhm is None:
            w = na * abs(focus)
            fwhm = w * FWHM_PER_W
        else:
            w = fwhm / FWHM_PER_W
        return w, fwhm, float(focus)
    if fwhm is None:
        focus = slice_sep / 2.0
        w = na * focus
        fwhm = w * FWHM_PER_W
    else:
        w = fwhm / FWHM_PER_W
        focus = w / na
    return w, fwhm, focus


def effective_na(na, fwhm, slice_sep, focus=None):
    """NA actually used by the beam (w / |focus| when the focus is given)."""
    w, fwhm, f = gaussian_probe_geometry(na, fwhm, slice_sep, focus)
    return w / abs(f) if f else na


def parse_foci(value):
    """Focus positions as fractions of the slice separation, from a string."""
    if isinstance(value, str):
        return [float(v) for v in value.split(",") if v.strip()]
    return [float(v) for v in value]


def gaussian_na_probe(shape, resolution, energy, na, fwhm, slice_sep,
                      model="gaussian", charge=1, foci="0.4,0.5,0.6",
                      focus=None):
    """
    Structured probe with a Gaussian envelope and a converging phase.

    ``model``: "gaussian", a Gaussian intensity profile with the spherical
    phase ``exp(-i k (sqrt(r^2 + focus^2) - focus))`` of a wave converging to
    ``focus`` downstream (sign matching ptypy's near-field propagator);
    "vortex", the same with an orbital angular momentum ``charge``: a
    Laguerre-Gauss-like ring ``(r/w)^|l| exp(-r^2/w^2) exp(i l theta)``;
    "multifocal", the equal-weight coherent sum of converging waves with foci
    at the fractions ``foci`` of the slice separation (an idealised
    multi-focus zone plate), sharing one envelope of 1/e^2 radius
    ``w = NA * mean focus``. Returns the complex field on the probe frame.
    """
    from ptypy.core import geometry
    lam = geometry.Geo._keV2m / float(energy)
    n0, n1 = (int(v) for v in shape)
    dy, dx = (float(v) for v in resolution)
    y = (np.arange(n0) - n0 / 2.0) * dy
    x = (np.arange(n1) - n1 / 2.0) * dx
    r2 = y[:, None] ** 2 + x[None, :] ** 2
    k = 2 * np.pi / lam
    if model == "multifocal":
        fs = [f * slice_sep for f in parse_foci(foci)]
        if fwhm is None:
            w = na * float(np.mean(fs))
        else:
            w = fwhm / FWHM_PER_W
        amp = np.exp(-r2 / w ** 2)
        field = np.zeros_like(r2, dtype=np.complex128)
        for f in fs:
            field += amp * np.exp(-1j * k * (np.sqrt(r2 + f ** 2) - f))
        field /= np.sqrt(len(fs))
        return field.astype(np.complex64)
    w, fwhm, focus = gaussian_probe_geometry(na, fwhm, slice_sep, focus)
    amp = np.exp(-r2 / w ** 2)
    # converging for focus > 0, diverging for focus < 0
    phase = -k * np.sign(focus) * (np.sqrt(r2 + focus ** 2) - abs(focus))
    if model == "vortex":
        l = int(charge)
        theta = np.arctan2(y[:, None], x[None, :])
        amp = amp * (np.sqrt(r2) / w) ** abs(l)
        phase = phase + l * theta
    elif model != "gaussian":
        raise ValueError("unknown probe model %r" % model)
    return (amp * np.exp(1j * phase)).astype(np.complex64)


def probe_summary(args, zsep, dx, lam):
    """Printed description of the structured probe and its sampling checks."""
    w, fwhm, focus = gaussian_probe_geometry(args.probe_na, args.probe_fwhm,
                                             zsep, args.probe_focus)
    na_used = effective_na(args.probe_na, args.probe_fwhm, zsep, args.probe_focus)
    na_limit = lam / (2 * dx)
    frame = args.shape * dx
    w1 = abs(focus - zsep) * na_used
    na_max = na_used
    if args.probe == "multifocal":
        fs = [f * zsep for f in parse_foci(args.probe_foci)]
        w = (args.probe_na * float(np.mean(fs)) if args.probe_fwhm is None
             else args.probe_fwhm / FWHM_PER_W)
        fwhm = w * FWHM_PER_W
        focus = float(np.mean(fs))
        nas = [w / f for f in fs]
        na_max = max(nas)
        w1 = max(abs(f - zsep) * n for f, n in zip(fs, nas))
    lines = ["%s probe: NA %.2e, 1/e^2 radius %.3f um, FWHM %.3f um at "
             "slice 0, focus %.3f mm downstream of slice 0 (slice 1 at %.3f mm),"
             " geometric 1/e^2 radius at slice 1 %.3f um, diffraction-limited "
             "spot ~ %.3f um"
             % (args.probe, na_used, w * 1e6, fwhm * 1e6, focus * 1e3,
                zsep * 1e3, w1 * 1e6, lam / (2 * na_used) * 1e6)]
    if args.probe == "vortex":
        lines.append("vortex charge %d: ring of peak radius %.3f um at slice 0"
                     % (args.probe_charge, w * np.sqrt(abs(args.probe_charge)
                                                       / 2.0) * 1e6))
    if args.probe == "multifocal":
        fs = [f * zsep for f in parse_foci(args.probe_foci)]
        wm = args.probe_na * float(np.mean(fs)) if args.probe_fwhm is None \
            else args.probe_fwhm / FWHM_PER_W
        nas = [wm / f for f in fs]
        zr = lam / (np.pi * np.mean(nas) ** 2)
        spacing = min(np.diff(sorted(fs))) if len(fs) > 1 else float("inf")
        lines.append("multifocal: foci at %s mm downstream of slice 0, envelope "
                     "1/e^2 radius %.3f um, per-focus NA %s; Rayleigh range of "
                     "one focus %.3f mm vs focus spacing %.3f mm%s"
                     % (", ".join("%.3f" % (f * 1e3) for f in fs), wm * 1e6,
                        ", ".join("%.2e" % v for v in nas), zr * 1e3,
                        spacing * 1e3,
                        " (foci NOT resolved along z)" if spacing < 2 * zr
                        else ""))
    lines.append("angular sampling limit lambda/(2 dx) = %.2e; the steepest NA "
                 "%.2e uses %.0f %% of it (the beam edge at 2 w aliases beyond "
                 "50 %%)" % (na_limit, na_max, 100 * na_max / na_limit))
    if na_max > 0.9 * na_limit:
        lines.append("WARNING: NA %.2e is at the angular sampling limit of this "
                     "frame; reduce the NA or the pixel size" % na_max)
    if fwhm > frame / 3.0 or w1 * FWHM_PER_W > frame / 3.0:
        lines.append("WARNING: the beam covers more than a third of the %.2f um "
                     "frame at one of the slices; wrap-around in the near-field "
                     "step" % (frame * 1e6))
    if w < 2 * dx or w1 < 2 * dx:
        lines.append("WARNING: the beam is narrower than two pixels at one of "
                     "the slices (w0 %.0f nm, w1 %.0f nm, dx %.0f nm)"
                     % (w * 1e9, w1 * 1e9, dx * 1e9))
    return "\n".join(lines)


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

        [phantom0]
        default = flowers
        type = str
        help = Phantom of slice 0: flowers, star, flowers-fine[:zoom] or image:<path>[:zoom]
        doc =

        [phantom1]
        default = star
        type = str
        help = Phantom of slice 1: flowers, star, flowers-fine[:zoom] or image:<path>[:zoom]
        doc =

        [phantom_phase]
        default = 1.0
        type = float
        help = Peak-to-peak phase range in radians of an image phantom (phase = phantom_phase * (g - mean g)); keep it below 2 pi
        doc =

        [phantom_amp]
        default = 0.15
        type = float
        help = Amplitude modulation depth of an image phantom
        doc =

        [focus_mid]
        default = True
        type = bool
        help = Place the beam focus midway between the two slices
        doc = Mirrors the real experiment (slices at +-0.75 mm around focus); the curvature difference between the planes gives the reconstruction its depth discrimination. Applies to the moon probe only.

        [probe_model]
        default = moon
        type = str
        help = Probe that generates the data: "moon", "gaussian", "vortex" or "multifocal"
        doc = "gaussian" is a Gaussian intensity profile with a converging spherical phase of numerical aperture probe_na, defined at slice 0; "vortex" adds an orbital angular momentum probe_charge (ring-shaped beam); "multifocal" is the coherent sum of converging waves with foci at the fractions probe_foci of the slice separation.

        [probe_charge]
        default = 1
        type = int
        help = Orbital angular momentum of the vortex probe
        doc =

        [probe_foci]
        default = 0.2,0.5,0.8
        type = str
        help = Focus positions of the multifocal probe as fractions of the slice separation, comma-separated
        doc =

        [probe_na]
        default = 3e-4
        type = float
        help = Numerical aperture of the gaussian probe
        doc = Sine of the half-angle of convergence at the 1/e^2 intensity radius.

        [probe_fwhm]
        default = None
        type = float
        help = Intensity FWHM of the gaussian probe at slice 0 in meters
        doc = None puts the focus midway between the slices and derives the size from probe_na.

        [probe_focus]
        default = None
        type = float
        help = Distance from slice 0 to the focus in meters (negative: focus upstream, diverging beam)
        doc = None: the focus follows from probe_fwhm and probe_na, or sits midway between the slices.
        """

        def __init__(self, pars=None, **kwargs):
            super().__init__(pars, **kwargs)
            # upstream and downstream slices on the same frame (the parent
            # class built the flower object in self.obj)
            self.obj0 = make_phantom(self.p.phantom0, self.obj.shape,
                                     self.p.phantom_phase, self.p.phantom_amp,
                                     self.p.spokes, flowers=self.obj)
            self.obj1 = make_phantom(self.p.phantom1, self.obj.shape,
                                     self.p.phantom_phase, self.p.phantom_amp,
                                     self.p.spokes, flowers=self.obj)
            # inter-slice near-field propagator on the probe frame
            g = Param()
            g.energy = self.geo.energy
            g.distance = self.p.slice_sep
            g.psize = self.geo.resolution
            g.shape = self.geo.shape
            g.propagation = "nearfield"
            self._slice_prop = geometry.Geo(owner=None, pars=g).propagator
            if self.p.probe_model in ("gaussian", "vortex", "multifocal"):
                self.pr = gaussian_na_probe(
                    self.geo.shape, self.geo.resolution, self.geo.energy,
                    self.p.probe_na, self.p.probe_fwhm, self.p.slice_sep,
                    model=self.p.probe_model, charge=self.p.probe_charge,
                    foci=self.p.probe_foci, focus=self.p.probe_focus)
                # same photon count as the moon probe (the parent normalised
                # its own probe before this replacement)
                self.pr /= np.sqrt(u.abs2(self.pr).sum() / self.p.photons)
            elif self.p.probe_model != "moon":
                raise ValueError("probe_model must be 'moon', 'gaussian', "
                                 "'vortex' or 'multifocal'")
            elif self.p.focus_mid:
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
    from ptypy.core import geometry
    probe_pars = u_param_for_probe(args)
    tmp = scan_cls(probe_pars)
    dx = float(np.mean(tmp.geo.resolution))
    lam = geometry.Geo._keV2m / float(tmp.geo.energy)
    del tmp
    zcrit = args.shape * dx * dx / lam
    dof = 5.2 * dx * dx / lam
    if getattr(args, "sep_dof", None) is not None:
        return args.sep_dof * dof, dx, zcrit, dof
    # sep/DOF = 0.164 * shape at the default --sep-frac, i.e. ~11 at the
    # default --shape 64, matching the real crop-128 case that separates
    # cleanly.
    return args.sep_frac * zcrit, dx, zcrit, dof


def free_space(wave, distance, meta):
    """``wave`` propagated by ``distance`` in free space on its own frame."""
    from ptypy.core import geometry
    from ptypy.utils import Param
    g = Param()
    g.energy = meta["energy"]
    g.distance = distance
    g.psize = meta["resolution"]
    g.shape = tuple(int(v) for v in wave.shape[-2:])
    g.propagation = "nearfield"
    prop = geometry.Geo(owner=None, pars=g).propagator
    if wave.ndim == 2:
        return prop.fw(wave)
    return np.array([prop.fw(w) for w in wave])


def u_param_for_probe(args):
    """Minimal scan parameters for a throwaway instance (geometry only)."""
    from ptypy import utils as u
    pars = u.Param()
    pars.shape = args.shape
    pars.num_frames = 8
    pars.density = args.density
    pars.slice_sep = 1e-3
    apply_geometry(pars, args)
    return pars


def apply_geometry(pars, args):
    """Copy the optional --energy-kev / --distance / --psize overrides."""
    if getattr(args, "energy", None) is not None:
        pars.energy = args.energy
    if getattr(args, "distance", None) is not None:
        pars.distance = args.distance
    if getattr(args, "psize", None) is not None:
        pars.psize = args.psize


def build_params(engine_name, zsep, args):
    """Ptycho parameter tree for one engine of the ground-truth simulation."""
    from ptypy import utils as u
    p = u.Param()
    p.verbose_level = "error"
    p.io = u.Param()
    # every run keeps its .ptyr under its own --outdir; with the default
    # io.home="./" parallel runs of this script would all write
    # ./recons/<run>/<run>_<engine>_<it>.ptyr (run = this script's name)
    # and collide on the HDF5 file lock
    p.io.home = args.outdir
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
    p.scans.MF.data.phantom0 = args.phantom0
    p.scans.MF.data.phantom1 = args.phantom1
    p.scans.MF.data.phantom_phase = args.phantom_phase
    p.scans.MF.data.phantom_amp = args.phantom_amp
    p.scans.MF.data.focus_mid = args.focus_mid
    apply_geometry(p.scans.MF.data, args)
    p.scans.MF.data.probe_model = args.probe
    p.scans.MF.data.probe_na = args.probe_na
    p.scans.MF.data.probe_fwhm = args.probe_fwhm
    p.scans.MF.data.probe_charge = args.probe_charge
    p.scans.MF.data.probe_foci = args.probe_foci
    p.scans.MF.data.probe_focus = args.probe_focus
    structured = args.probe in ("gaussian", "vortex", "multifocal")
    if args.probe == "vortex" and args.probe_init == "aperture":
        # a plain aperture is orthogonal to a vortex beam (no angular
        # momentum); start from the analytic ring instead
        print("note: --probe-init aperture carries no angular momentum, using "
              "the scaled analytic vortex beam as initial probe", flush=True)
        args.probe_init = "gaussian"
    # initial probe hint, like the real runner's --defocus-um: start from a
    # probe whose focus sits where the true one is
    p.scans.MF.illumination = u.Param()
    p.scans.MF.illumination.propagation = u.Param()
    if structured and args.probe_init == "aperture":
        from ptypy.core import geometry
        w, fwhm, focus = gaussian_probe_geometry(args.probe_na, args.probe_fwhm,
                                                 zsep, args.probe_focus)
        na = effective_na(args.probe_na, args.probe_fwhm, zsep, args.probe_focus)
        lam = geometry.Geo._keV2m / float(args.energy_kev)
        # a circular aperture at the focus of diameter lambda / NA, propagated
        # to slice 0, has about the right angle (and sign of curvature)
        p.scans.MF.illumination.aperture = u.Param()
        p.scans.MF.illumination.aperture.form = "circ"
        p.scans.MF.illumination.aperture.size = lam / na
        p.scans.MF.illumination.propagation.parallel = -focus
    elif structured:
        # the analytic beam as the initial probe: the true one ("truth") or
        # one with the FWHM scaled by --probe-init-scale ("gaussian"), i.e.
        # what a user knowing the optics and a rough size would give
        scale = 1.0 if args.probe_init == "truth" else args.probe_init_scale
        if args.probe == "multifocal":
            fs = [f * zsep for f in parse_foci(args.probe_foci)]
            wm = (args.probe_na * float(np.mean(fs)) if args.probe_fwhm is None
                  else args.probe_fwhm / FWHM_PER_W)
            fwhm = wm * FWHM_PER_W
        else:
            w, fwhm, focus = gaussian_probe_geometry(args.probe_na,
                                                     args.probe_fwhm, zsep,
                                                     args.probe_focus)
        # scale NA together with the FWHM so that the guess keeps the true
        # focus position (focus = w / NA); only the beam size is off
        init = gaussian_na_probe((args.shape, args.shape),
                                 (args.dx, args.dx), args.energy_kev,
                                 args.probe_na * scale, fwhm * scale, zsep,
                                 model=args.probe, charge=args.probe_charge,
                                 foci=args.probe_foci, focus=args.probe_focus)
        p.scans.MF.illumination.model = init[np.newaxis]
        p.scans.MF.illumination.aperture = u.Param()
        p.scans.MF.illumination.aperture.form = "circ"
        p.scans.MF.illumination.aperture.size = 2 * args.shape * args.dx
        p.scans.MF.illumination.propagation.parallel = None
    else:
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

    Returns ``(slices, probes, truth, meta)``: the per-slice objects and
    incident waves read back from the engine's ``fslices`` file, the two
    phantoms, and ``meta`` with the true probe at slice 0 (``probe``) and the
    frame geometry (``energy``, ``resolution``, ``shape``). The ground truth
    is read straight off the PtyScan instance that generated the data.
    """
    from ptypy.core import Ptycho
    from ptypy.debug.threepie_compare import (read_slices, read_slice_probes,
                                              read_last_view)
    np.random.seed(args.seed)   # identical positions + noise realization
    pars = build_params(engine_name, zsep, args)
    if args.seed_engines is None:
        seeded = contextlib.nullcontext()
    else:
        # every engine draws its view order from numpy.random.default_rng();
        # give them all the same seeded generator
        seeded = mock.patch(
            "numpy.random.default_rng",
            lambda *a, **k: np.random.Generator(
                np.random.PCG64(args.seed_engines)))
    with seeded:
        P = Ptycho(pars, level=5)
    ptyscan = list(P.model.scans.values())[0].ptyscan
    truth = {0: np.array(ptyscan.obj0), 1: np.array(ptyscan.obj1)}
    meta = {"probe": np.array(ptyscan.pr),
            "energy": float(ptyscan.geo.energy),
            "resolution": np.array(ptyscan.geo.resolution, dtype=float),
            "shape": tuple(int(v) for v in ptyscan.geo.shape),
            "pixel": np.array(ptyscan.pixel),
            "pos": np.array(ptyscan.pos, dtype=float)}
    del P
    fslices = pars.engines.e0.fslices
    return (read_slices(fslices), read_slice_probes(fslices), truth, meta,
            read_last_view(fslices))


def probe_results(probes, zsep, meta, last=None, truth=None):
    """
    ``{"probe0", "probe1_free", "probe1_view", "probe1_truth", "last_view"}``
    for one engine, mode 0: the reconstructed illumination, its free-space
    propagation to slice 1, the slice-1 wave the engine holds after its
    last view, the true wave at that scan position (true probe times the
    true slice-0 patch, propagated to slice 1) and the frame index of that
    position. ``{}`` when the engine did not save probes; the last two
    entries are missing when it did not record its last view.
    """
    if not probes:
        return {}
    out = {"probe0": np.array(probes[0][0])}
    out["probe1_free"] = free_space(out["probe0"], zsep, meta)
    if 1 in probes:
        out["probe1_view"] = np.array(probes[1][0])
    if last is not None and truth is not None and 1 in probes:
        pos = meta["pos"]
        k = int(last["layer"])
        if not (0 <= k < len(pos) and np.allclose(pos[k], last["coord"],
                                                  atol=1e-10)):
            k = int(np.argmin(np.sum((pos - last["coord"][None, :]) ** 2,
                                     axis=1)))
        py, px = (int(v) for v in meta["pixel"][k])
        s0, s1 = meta["shape"]
        patch = truth[0][py:py + s0, px:px + s1]
        out["probe1_truth"] = free_space(meta["probe"] * patch, zsep, meta)
        out["last_view"] = k
    return out


# --------------------------------------------------------------------------- #
# report and figure
# --------------------------------------------------------------------------- #
def build_report(engines, gt, results, gtp, pres, geom, args):
    """The printed/saved comparison report, as one string."""
    zsep, dx, zcrit, dof = geom
    order = ("engines unseeded" if args.seed_engines is None
             else "same seeded view order (seed %d)" % args.seed_engines)
    alias = "  (ALIASED: sep > z_crit)" if zsep > zcrit else ""
    if args.probe in ("gaussian", "vortex", "multifocal"):
        w, fwhm, focus = gaussian_probe_geometry(args.probe_na, args.probe_fwhm,
                                                 zsep, args.probe_focus)
        probe = ("%s probe NA %.1e, FWHM %.2f um at slice 0, focus %.2f mm"
                 % (args.probe, effective_na(args.probe_na, args.probe_fwhm,
                                             zsep, args.probe_focus),
                    fwhm * 1e6, focus * 1e3))
        if args.probe == "vortex":
            probe += ", charge %d" % args.probe_charge
        if args.probe == "multifocal":
            probe = ("multifocal probe NA %.1e, foci at %s of the separation"
                     % (args.probe_na, args.probe_foci))
    else:
        probe = "moon probe" + (", focus midway" if args.focus_mid else "")
    lines = ["ground-truth two-slice simulation: shape %d, %d frames, %d it, "
             "slice_sep %.3f mm, %s, %s" % (args.shape, args.nframes,
                                            args.numiter, zsep * 1e3, probe,
                                            order),
             "dx = %.1f nm, DOF = 5.2 dx^2/lambda = %.1f um, "
             "z_crit = N dx^2/lambda = %.2f mm" % (dx * 1e9, dof * 1e6,
                                                   zcrit * 1e3),
             "sep/DOF = %.2f, sep/z_crit = %.3f%s" % (zsep / dof, zsep / zcrit,
                                                      alias),
             "",
             "engine slice vs GROUND TRUTH (aligned ncorr):",
             "%-14s   %-13s %-13s %s" % ("engine", "slice0", "slice1",
                                         "product")]
    for eng in engines:
        vals = ["%.4f" % gt_ncorr(gt[i], results[(eng, "slice%d" % i)])
                for i in (0, 1)]
        prod = gt_ncorr(gt[0] * gt[1],
                        results[(eng, "slice0")] * results[(eng, "slice1")])
        lines.append("%-14s   %-13s %-13s %.4f"
                     % (ENGINE_LABEL[eng], vals[0], vals[1], prod))

    lines += ["", "phantoms: slice 0 = %s, slice 1 = %s" % (args.phantom0,
                                                             args.phantom1),
              "", "swap check: engine slice vs the OTHER GT slice (aligned):"]
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

    with_probes = [e for e in engines if pres.get(e)]
    if with_probes:
        # Free-space propagation is unitary, so the propagated probe at
        # slice 1 scores exactly like the illumination at slice 0; only the
        # illumination is reported. The last column measures how much the
        # slice-0 structure of the last view modulates the wave that enters
        # slice 1 (1.0 would mean an empty slice 0 at that position).
        lines += ["",
                  "probe vs GROUND TRUTH probe (aligned ncorr, mode 0):",
                  "%-14s   %-24s %-34s %-30s %s"
                  % ("engine", "illumination (slice 0)",
                     "slice-1 wave vs true wave at pos.",
                     "slice-1 wave vs free-space probe", "last view")]
        for eng in with_probes:
            p0 = gt_ncorr(gtp[0], pres[eng]["probe0"])
            pv = pt = "n/a"
            view = "n/a"
            if "probe1_view" in pres[eng]:
                pv = "%.4f" % gt_ncorr(pres[eng]["probe1_free"],
                                       pres[eng]["probe1_view"])
            if "probe1_truth" in pres[eng]:
                pt = "%.4f" % gt_ncorr(pres[eng]["probe1_truth"],
                                       pres[eng]["probe1_view"])
                view = "frame %d" % pres[eng]["last_view"]
            lines.append("%-14s   %-24s %-34s %-30s %s"
                         % (ENGINE_LABEL[eng], "%.4f" % p0, pt, pv, view))
        cross = [(a, b, tag) for a, b, tag in pairs
                 if a in with_probes and b in with_probes]
        if cross:
            lines += ["cross-backend illumination (slice 0, aligned):"]
        for a, b, tag in cross:
            lines.append("%-16s   %.4f"
                         % (tag, gt_ncorr(pres[a]["probe0"],
                                          pres[b]["probe0"])))
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
    order = ("engines unseeded" if args.seed_engines is None
             else "same seeded view order, seed %d" % args.seed_engines)
    fig.suptitle("Two-slice ground-truth simulation, shape %d, %d it, "
                 "slice_sep %.2f mm, %s"
                 % (args.shape, args.numiter, zsep * 1e3, order), fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def probe_panels(p, floor=0.05):
    """
    Normalized amplitude and phase of a probe for display: the global phase
    is removed with the intensity-weighted mean phasor, and the phase is
    blanked (NaN) where the amplitude is below ``floor`` of its maximum.
    """
    amp = np.abs(p)
    peak = amp.max() if amp.max() > 0 else 1.0
    amp = amp / peak
    ref = np.sum(amp ** 2 * np.exp(1j * np.angle(p)))
    ph = np.angle(p * np.exp(-1j * np.angle(ref)))
    ph = np.where(amp > floor, ph, np.nan)
    return amp, ph


def phase_gradient(field):
    """
    ``(d/dy, d/dx)`` of the phase of a complex field in radians per pixel,
    from Im(conj(u) grad u) / |u|^2, which needs no unwrapping. Exact for
    small phase steps; it under-estimates gradients above about 0.5 rad per
    pixel (the phasor difference saturates), which is fine for the phantoms
    used here.
    """
    field = np.asarray(field)
    gy, gx = np.gradient(field)
    den = np.abs(field) ** 2 + 1e-12 * np.max(np.abs(field)) ** 2
    return (np.imag(np.conj(field) * gy) / den,
            np.imag(np.conj(field) * gx) / den)


def complex_rgb(field, floor=0.0, gauge=True):
    """
    Hue = phase, brightness = amplitude / max (HSV to RGB). With ``gauge``
    the intensity-weighted mean phase of the field is removed first, so
    that cyan (hue 0.5) marks the mean phase of every panel; the colour
    wheel legend is drawn with ``gauge=False``.
    """
    from matplotlib.colors import hsv_to_rgb
    field = np.asarray(field)
    amp = np.abs(field)
    peak = amp.max() if amp.max() > 0 else 1.0
    if gauge:
        ref = np.sum(amp ** 2 * np.exp(1j * np.angle(field)))
        ph = np.angle(field * np.exp(-1j * np.angle(ref)))
    else:
        ph = np.angle(field)
    hsv = np.stack([(ph + np.pi) / (2 * np.pi), np.ones_like(amp),
                    np.clip(amp / peak, floor, 1.0)], axis=-1)
    return hsv_to_rgb(hsv)


def make_gradient_figure(engines, gt, results, geom, args, path):
    """
    Phase gradient of every slice: rows d/dx and d/dy for slice 0 and
    slice 1, columns ground truth followed by the engines. The gradient is
    insensitive to the phase offset and to wrapping, so slices can be
    compared without any gauge alignment beyond the common crop.
    """
    zsep, dx, zcrit, dof = geom
    cols = ["ground truth"] + ["%s (%s)" % (e, ENGINE_LABEL[e])
                               for e in engines]
    rows = [(0, 1, "slice 0, d phase/dx"), (0, 0, "slice 0, d phase/dy"),
            (1, 1, "slice 1, d phase/dx"), (1, 0, "slice 1, d phase/dy")]
    fig, axes = plt.subplots(len(rows), len(cols),
                             figsize=(3.4 * len(cols), 3.4 * len(rows)))
    for r, (i, comp, label) in enumerate(rows):
        panels = common_crop([gt[i]] + [results[(e, "slice%d" % i)]
                                        for e in engines])
        grads = [phase_gradient(p)[comp] for p in panels]
        lim = np.percentile(np.abs(np.concatenate([g.ravel() for g in grads])),
                            99)
        im = None
        for c, g in enumerate(grads):
            ax = axes[r, c]
            im = ax.imshow(g, cmap="RdBu_r", vmin=-lim, vmax=lim)
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(cols[c], fontsize=11)
            if c == 0:
                ax.set_ylabel(label, fontsize=10)
        cb = fig.colorbar(im, ax=axes[r, :].tolist(), fraction=0.02, pad=0.01)
        cb.set_label("rad / pixel", fontsize=8)
        cb.ax.tick_params(labelsize=7)
    fig.suptitle("Phase gradient per slice, shape %d, %d it, slice_sep %.3f mm "
                 "(sep/DOF %.2f)" % (args.shape, args.numiter, zsep * 1e3,
                                     zsep / dof), fontsize=12)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def make_complex_probe_figure(engines, gtp, pres, geom, args, path):
    """
    Complex colour-coded probes (hue = phase, brightness = amplitude): rows
    slice 0, slice 1 free-space propagated, slice 1 wave after the last
    view; columns ground truth followed by the engines.
    """
    zsep, dx, zcrit, dof = geom
    cols = ["ground truth"] + ["%s (%s)" % (e, ENGINE_LABEL[e])
                               for e in engines]
    rows = [("slice 0", "probe0", 0), ("slice 1, free space", "probe1_free", 1),
            ("slice 1, wave after the last view", "probe1_view", None),
            ("slice 1, true wave at that position", "probe1_truth", None)]
    fig, axes = plt.subplots(len(rows), len(cols) + 1,
                             figsize=(3.2 * (len(cols) + 1), 3.2 * len(rows)),
                             gridspec_kw={"width_ratios": [1] * len(cols) + [0.5]})
    for r, (label, key, gt_index) in enumerate(rows):
        panels = [None if gt_index is None else gtp[gt_index]]
        panels += [pres.get(e, {}).get(key) for e in engines]
        for c, p in enumerate(panels):
            ax = axes[r, c]
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(cols[c], fontsize=11)
            if c == 0:
                ax.set_ylabel(label, fontsize=10)
            if p is None:
                ax.text(0.5, 0.5, "per engine,\nsee its column" if c == 0
                        else "n/a", ha="center", va="center",
                        transform=ax.transAxes, fontsize=9, color="gray")
                continue
            ax.imshow(complex_rgb(p))
        axes[r, -1].axis("off")
    # colour wheel legend
    n = 128
    yy, xx = np.mgrid[-1:1:n * 1j, -1:1:n * 1j]
    wheel = (np.hypot(yy, xx) <= 1) * np.exp(1j * np.arctan2(yy, xx))
    ax = axes[0, -1]
    ax.axis("on")
    ax.imshow(complex_rgb(wheel * np.hypot(yy, xx), gauge=False))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("hue = phase (cyan = panel mean)\nbrightness = |P|", fontsize=8)
    fig.suptitle("Complex probes per slice, shape %d, %d it, slice_sep %.3f mm "
                 "(sep/DOF %.2f)" % (args.shape, args.numiter, zsep * 1e3,
                                     zsep / dof), fontsize=12)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def phantom_label(spec):
    """Short label of a phantom spec: image specs keep only the file name."""
    if spec.startswith("image:"):
        rest = spec.split(":", 1)[1]
        path, _, tail = rest.rpartition(":")
        try:
            float(tail)
        except ValueError:
            path, tail = rest, ""
        return "image:%s%s" % (os.path.basename(path or rest),
                               (":" + tail) if tail else "")
    return spec


def make_overview_figure(engines, gt, results, gtp, pres, geom, args, path):
    """
    Recovered probes above the reconstructed sample: rows are the complex
    colour-coded probe at slice 0, its free-space propagation to slice 1,
    then the phase of slice 0 and of slice 1; columns are the ground truth
    followed by the engines.
    """
    from ptypy.debug.threepie_compare import gauge_phase
    zsep, dx, zcrit, dof = geom
    cols = ["ground truth"] + ["%s (%s)" % (e, ENGINE_LABEL[e])
                               for e in engines]
    fig, axes = plt.subplots(4, len(cols), figsize=(3.4 * len(cols), 3.4 * 4))
    # probe rows
    for r, (label, key, gt_index) in enumerate(
            [("probe at slice 0", "probe0", 0),
             ("probe at slice 1 (free space)", "probe1_free", 1)]):
        panels = [gtp[gt_index]] + [pres.get(e, {}).get(key) for e in engines]
        for c, p in enumerate(panels):
            ax = axes[r, c]
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(cols[c], fontsize=11)
            if c == 0:
                ax.set_ylabel(label, fontsize=10)
            if p is None:
                ax.text(0.5, 0.5, "n/a", ha="center", va="center",
                        transform=ax.transAxes, color="gray")
                continue
            ax.imshow(complex_rgb(p))
    # sample rows
    for i in (0, 1):
        r = 2 + i
        panels = [gt[i]] + [results[(e, "slice%d" % i)] for e in engines]
        phases = [gauge_phase(p) for p in common_crop(panels)]
        pooled = np.concatenate([p.ravel() for p in phases])
        vmin, vmax = np.percentile(pooled, [1, 99])
        im = None
        for c, ph in enumerate(phases):
            ax = axes[r, c]
            im = ax.imshow(ph, cmap="gray", vmin=vmin, vmax=vmax)
            ax.set_xticks([])
            ax.set_yticks([])
            if c == 0:
                ax.set_ylabel("slice %d phase" % i, fontsize=10)
        cb = fig.colorbar(im, ax=axes[r, :].tolist(), fraction=0.02, pad=0.01)
        cb.set_label("phase (rad)", fontsize=8)
        cb.ax.tick_params(labelsize=7)
    probe = args.probe if args.probe != "moon" else "moon probe"
    fig.suptitle("%s, slice_sep %.3f mm (sep/DOF %.2f): recovered probes above "
                 "the reconstructed sample (%s / %s), %d it"
                 % (probe, zsep * 1e3, zsep / dof, phantom_label(args.phantom0),
                    phantom_label(args.phantom1), args.numiter), fontsize=12)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def make_probe_figure(engines, gtp, pres, geom, args, path):
    """
    Rows: slice-0 amplitude and phase, free-space slice-1 amplitude and
    phase, last-view incident wave at slice 1 (amplitude). Columns: ground
    truth followed by each engine; the ground truth has no last-view wave.
    """
    zsep, dx, zcrit, dof = geom
    cols = ["ground truth"] + ["%s (%s)" % (e, ENGINE_LABEL[e])
                               for e in engines]
    rows = [("slice 0 |P|", "probe0", 0, "amp"),
            ("slice 0 phase", "probe0", 0, "ph"),
            ("slice 1 |P|, free space", "probe1_free", 1, "amp"),
            ("slice 1 phase, free space", "probe1_free", 1, "ph"),
            ("slice 1 |P|, wave after the last view", "probe1_view", None,
             "amp"),
            ("slice 1 |P|, true wave at that position", "probe1_truth", None,
             "amp")]
    fig, axes = plt.subplots(len(rows), len(cols),
                             figsize=(3.2 * len(cols), 3.2 * len(rows)))
    for r, (label, key, gt_index, kind) in enumerate(rows):
        panels = []
        if gt_index is None:
            panels.append(None)
        else:
            panels.append(gtp[gt_index])
        for e in engines:
            panels.append(pres.get(e, {}).get(key))
        im = None
        for c, p in enumerate(panels):
            ax = axes[r, c]
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(cols[c], fontsize=11)
            if c == 0:
                ax.set_ylabel(label, fontsize=9)
            if p is None:
                ax.text(0.5, 0.5, "per engine,\nsee its column" if c == 0
                        else "n/a", ha="center", va="center",
                        transform=ax.transAxes, fontsize=9, color="gray")
                continue
            amp, ph = probe_panels(p)
            if kind == "amp":
                im = ax.imshow(amp, cmap="viridis", vmin=0, vmax=1)
            else:
                im = ax.imshow(ph, cmap="twilight", vmin=-np.pi, vmax=np.pi)
        if im is not None:
            cb = fig.colorbar(im, ax=axes[r, -1], fraction=0.046, pad=0.03)
            cb.set_label("|P| / max" if kind == "amp" else "phase (rad)",
                         fontsize=8)
            cb.ax.tick_params(labelsize=7)
    order = ("engines unseeded" if args.seed_engines is None
             else "same seeded view order, seed %d" % args.seed_engines)
    fig.suptitle("Probes per slice, shape %d, %d it, slice_sep %.3f mm "
                 "(sep/DOF %.2f), %s"
                 % (args.shape, args.numiter, zsep * 1e3, zsep / dof, order),
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
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
    parser.add_argument("--sep-dof", type=float, default=None,
                        help="Slice separation as a multiple of the depth of "
                             "field DOF = 5.2 dx^2/lambda; overrides "
                             "--sep-frac. At or below 1 the whole two-layer "
                             "sample sits inside one depth of field.")
    parser.add_argument("--energy-kev", dest="energy", type=float, default=None,
                        help="Photon energy in keV (default: the MoonFlower "
                             "geometry, 7.2 keV).")
    parser.add_argument("--distance", type=float, default=None,
                        help="Sample-detector distance in m (default 7.19).")
    parser.add_argument("--psize", type=float, default=None,
                        help="Detector pixel size in m (default 172e-6); the "
                             "reconstructed pixel is lambda*distance/(shape*psize).")
    parser.add_argument("--probe",
                        choices=("moon", "gaussian", "vortex", "multifocal"),
                        default="moon",
                        help="Probe that generates the data: the MoonFlower "
                             "probe; a Gaussian intensity profile with a "
                             "converging spherical phase of numerical "
                             "aperture --probe-na; the same with an orbital "
                             "angular momentum --probe-charge (ring beam); or "
                             "a multifocal beam, the coherent sum of converging "
                             "waves with foci at --probe-foci.")
    parser.add_argument("--probe-focus", type=float, default=None,
                        help="Distance from slice 0 to the focus in meters, "
                             "negative for a beam diverging from a focus "
                             "upstream of slice 0. Fixes the probe at slice 0 "
                             "independently of the separation (with "
                             "--probe-fwhm the NA follows as w/|focus|). "
                             "Write negative values as --probe-focus=-0.00075 "
                             "(argparse does not accept -0.75e-3 as a number).")
    parser.add_argument("--probe-charge", type=int, default=1,
                        help="Orbital angular momentum of the vortex probe.")
    parser.add_argument("--probe-foci", default="0.2,0.5,0.8",
                        help="Foci of the multifocal probe as fractions of "
                             "the slice separation, comma-separated.")
    parser.add_argument("--probe-na", type=float, default=3e-4,
                        help="Numerical aperture of the gaussian probe: sine "
                             "of the convergence half-angle at the 1/e^2 "
                             "intensity radius. Must stay below the angular "
                             "sampling lambda/(2 dx) of the frame.")
    parser.add_argument("--probe-fwhm", type=float, default=None,
                        help="Intensity FWHM of the gaussian probe at slice 0 "
                             "in meters; the focus then lies w/NA downstream. "
                             "Default: focus midway between the slices, size "
                             "from the NA.")
    parser.add_argument("--probe-init", choices=("aperture", "gaussian", "truth"),
                        default="aperture",
                        help="Initial probe of the reconstruction for the "
                             "gaussian probe: a circular aperture of diameter "
                             "lambda/NA at the focus propagated back to slice 0 "
                             "(default), the analytic Gaussian/NA beam with the "
                             "FWHM scaled by --probe-init-scale, or the true "
                             "probe (diagnostic only).")
    parser.add_argument("--probe-init-scale", type=float, default=1.3,
                        help="FWHM factor of the --probe-init gaussian guess.")
    parser.add_argument("--phantom0", default="flowers",
                        help="Phantom of slice 0: flowers, star, "
                             "flowers-fine[:zoom] (the flower image shrunk by "
                             "zoom, more structure per field) or "
                             "image:<path>[:zoom] (any image, grey level to "
                             "phase and amplitude).")
    parser.add_argument("--phantom1", default="star",
                        help="Phantom of slice 1, same choices.")
    parser.add_argument("--phantom-phase", type=float, default=1.0,
                        help="Peak-to-peak phase range of an image phantom "
                             "(rad), phase = phantom_phase * (g - mean g); "
                             "keep it below 2 pi.")
    parser.add_argument("--phantom-amp", type=float, default=0.15,
                        help="Amplitude modulation depth of an image phantom.")
    parser.add_argument("--spokes", type=int, default=24,
                        help="Number of spokes of the downstream star phantom.")
    parser.add_argument("--seed", type=int, default=7,
                        help="Seed for the scan positions and the Poisson "
                             "noise; every engine gets the same realization.")
    parser.add_argument("--seed-engines", type=int, default=None,
                        help="Seed for the engines' view order. Without it "
                             "each engine draws its own order from an "
                             "unseeded generator, so two runs differ. With it, "
                             "all engines see the views in the same order. "
                             "Default: unseeded.")
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
    geom = (zsep, dx, zcrit, dof)
    from ptypy.core import geometry as _geometry
    tmp = scan_cls(u_param_for_probe(args))
    args.energy_kev = float(tmp.geo.energy)
    args.dx = dx
    del tmp
    lam = _geometry.Geo._keV2m / args.energy_kev
    if args.probe in ("gaussian", "vortex", "multifocal"):
        print(probe_summary(args, zsep, dx, lam), flush=True)
    print("dx=%.1f nm, z_crit=%.2f mm, DOF=%.1f um -> slice_sep=%.3f mm "
          "(sep/DOF=%.2f, sep/z_crit=%.3f)"
          % (dx * 1e9, zcrit * 1e3, dof * 1e6, zsep * 1e3, zsep / dof,
             zsep / zcrit), flush=True)
    if zsep > zcrit:
        print("WARNING: slice_sep exceeds z_crit, the near-field forward "
              "model is aliased", flush=True)

    results = {}
    pres = {}
    gt = None
    gtp = None
    for eng in engines:
        print("=== %s ===" % eng, flush=True)
        slices, probes, truth, meta, last = run_engine(eng, zsep, args)
        if gt is None:
            gt = truth
            gtp = {0: meta["probe"], 1: free_space(meta["probe"], zsep, meta)}
        for idx, arr in slices.items():
            results[(eng, "slice%d" % idx)] = arr
        pres[eng] = probe_results(probes, zsep, meta, last, truth)
        if not pres[eng]:
            print("note: %s saved no per-slice probes" % eng, flush=True)

    report = build_report(engines, gt, results, gtp, pres, geom, args)
    print(report, flush=True)
    with open(os.path.join(args.outdir, "sim_gt_slices.txt"), "w") as fh:
        fh.write(report + "\n")
    arrays = {"%s_%s" % (ENGINE_LABEL[e], k): v
              for (e, k), v in results.items()}
    for e, d in pres.items():
        for k, v in d.items():
            arrays["%s_%s" % (ENGINE_LABEL[e], k)] = np.asarray(v)
    np.savez(os.path.join(args.outdir, "sim_gt_slices.npz"),
             gt_slice0=gt[0], gt_slice1=gt[1],
             gt_probe0=gtp[0], gt_probe1_free=gtp[1],
             geom=np.array(geom, dtype=float), **arrays)

    make_figure(engines, gt, results, zsep, args,
                os.path.join(args.outdir, "sim_gt_slices.png"))
    make_gradient_figure(engines, gt, results, geom, args,
                         os.path.join(args.outdir, "sim_gt_gradient.png"))
    if any(pres.values()):
        make_probe_figure(engines, gtp, pres, geom, args,
                          os.path.join(args.outdir, "sim_gt_probes.png"))
        make_complex_probe_figure(
            engines, gtp, pres, geom, args,
            os.path.join(args.outdir, "sim_gt_probes_complex.png"))
        make_overview_figure(
            engines, gt, results, gtp, pres, geom, args,
            os.path.join(args.outdir, "sim_gt_overview.png"))
    print("saved sim_gt_slices.png/.npz/.txt, sim_gt_gradient.png, "
          "sim_gt_probes.png and sim_gt_probes_complex.png in %s"
          % os.path.abspath(args.outdir), flush=True)


if __name__ == "__main__":
    main()
