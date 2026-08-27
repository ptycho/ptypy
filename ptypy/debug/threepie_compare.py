#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Shared helpers for comparing ThreePIE reconstructions.

The multislice comparison tools in this directory all need the same handful of
operations, and getting them subtly different is how two "comparisons" end up
disagreeing about the same pair of reconstructions. They live here once:

  ncorr / aligned_ncorr   phase-, scale- and translation-invariant similarity
  central                 crop away the poorly covered border
  gauge_phase             remove the global phase before plotting
  read_slices             per-slice objects from an engine's ``fslices`` file
  read_recon              object/probe/pixel size from a ``.ptyr``

Why the invariances matter: a ptychographic solution is only defined up to a
global phase and a joint probe/object translation, and stochastic (ePIE-type)
engines shuffle their view order independently. Comparing two correct
reconstructions elementwise therefore fails no matter how correct they are.
Every comparison in this directory goes through ``aligned_ncorr``.

This module is intentionally numpy-only at import time; ``h5py`` is imported
lazily by the readers.

This file is part of the PTYPY package.

    :copyright: Copyright 2014 by the PTYPY team, see AUTHORS.
    :license: see LICENSE for details.
"""

import glob
import os

import numpy as np

__all__ = [
    "ncorr",
    "register_shift",
    "aligned_ncorr",
    "central",
    "common_central",
    "gauge_phase",
    "read_slices",
    "read_recon",
    "find_latest",
]


# --------------------------------------------------------------------------- #
# similarity metrics
# --------------------------------------------------------------------------- #
def ncorr(a, b):
    """
    Normalized correlation of two complex fields, invariant to a global phase
    and to an overall scale.

    Returns a value in [0, 1]: 1 means identical up to phase and scale, 0 means
    uncorrelated.
    """
    a = np.asarray(a).ravel()
    b = np.asarray(b).ravel()
    a = a - a.mean()
    b = b - b.mean()
    den = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.abs(np.vdot(a, b)) / den) if den else 0.0


def register_shift(a, b):
    """Integer-pixel shift of ``b`` relative to ``a``, by phase correlation."""
    A = np.fft.fft2(a - a.mean())
    B = np.fft.fft2(b - b.mean())
    cc = np.fft.ifft2(A * np.conj(B))
    idx = np.unravel_index(np.argmax(np.abs(cc)), cc.shape)
    return [int(s) if s <= n // 2 else int(s - n) for s, n in zip(idx, cc.shape)]


def common_central(a, b, frac=0.75):
    """Crop both arrays centrally to a common shape, keeping ``frac`` of it."""
    n0 = int(min(a.shape[-2], b.shape[-2]) * frac)
    n1 = int(min(a.shape[-1], b.shape[-1]) * frac)

    def crop(x):
        c0 = (x.shape[-2] - n0) // 2
        c1 = (x.shape[-1] - n1) // 2
        return x[..., c0:c0 + n0, c1:c1 + n1]
    return crop(a), crop(b)


def central(arr, frac=0.6):
    """Central ``frac`` of an array, dropping the thinly scanned border."""
    n0, n1 = arr.shape[-2:]
    c0 = int(n0 * (1 - frac) / 2)
    c1 = int(n1 * (1 - frac) / 2)
    return arr[..., c0:n0 - c0, c1:n1 - c1]


def aligned_ncorr(a, b, margin_frac=0.08, crop_frac=None):
    """
    ``ncorr`` after removing the joint-translation gauge.

    ``b`` is registered onto ``a``, the wrap-around margins are trimmed, and
    the correlation is taken on what remains. Returns ``(shift, ncorr)``.

    Set ``crop_frac`` to first reduce both inputs to a common central region
    (useful when the two reconstructions sit on different-sized grids).
    """
    if crop_frac is not None:
        a, b = common_central(a, b, frac=crop_frac)
    shift = register_shift(a, b)
    b = np.roll(b, shift, axis=(-2, -1))
    m = max(1, int(min(a.shape[-2:]) * margin_frac))
    return shift, ncorr(a[..., m:-m, m:-m], b[..., m:-m, m:-m])


# --------------------------------------------------------------------------- #
# display
# --------------------------------------------------------------------------- #
def gauge_phase(obj):
    """
    Phase of ``obj`` with the global phase gauge removed, wrap-safely.

    Rotating by the phase of the complex mean (rather than subtracting a mean
    phase) keeps the result away from the +-pi branch cut, which is what makes
    an otherwise featureless panel render as solid black-and-white noise.
    """
    ref = np.asarray(obj).mean()
    if np.abs(ref) > 0:
        obj = obj * np.exp(-1j * np.angle(ref))
    return np.angle(obj)


# --------------------------------------------------------------------------- #
# readers
# --------------------------------------------------------------------------- #
def find_latest(pattern):
    """Newest path matching a glob, or None."""
    hits = sorted(glob.glob(pattern))
    return hits[-1] if hits else None


def read_slices(path):
    """
    Per-slice object arrays from an engine's ``fslices`` output.

    Returns ``{slice_index: complex ndarray}``. The slice containers are named
    ``<container>_o_<index>``, one storage each.
    """
    import h5py
    out = {}
    with h5py.File(path, "r") as f:
        objects = f["content/objects"]
        for name in objects:
            idx = int(name.rsplit("_", 1)[-1])
            group = objects[name]
            storage = group[list(group.keys())[0]]
            out[idx] = np.array(storage["data"])[0]
    return out


def read_recon(path):
    """
    Object, probe and pixel size from a ``.ptyr`` reconstruction.

    Returns a dict with ``obj``, ``probe`` (all modes), ``psize`` and ``path``.
    """
    import h5py
    with h5py.File(path, "r") as f:
        oid = list(f["content/obj"].keys())[0]
        pid = list(f["content/probe"].keys())[0]
        obj = np.array(f["content/obj/%s/data" % oid])[0]
        probe = np.array(f["content/probe/%s/data" % pid])
        pkey = "content/obj/%s/_psize" % oid
        if pkey not in f:
            pkey = "content/obj/%s/psize" % oid
        psize = float(np.mean(np.array(f[pkey])))
    return {"obj": obj, "probe": probe, "psize": psize, "path": path}


def iteration_seconds(path):
    """
    Mean seconds per iteration recorded inside a ``.ptyr``.

    The engine stores a ``runtime/iter_info`` group with one ``duration`` per
    saved iteration block, so a reconstruction carries its own timing and no
    separate benchmark log is needed. Returns ``(total_seconds, n_entries)``.
    """
    import h5py
    with h5py.File(path, "r") as f:
        info = f["content/runtime/iter_info"]
        durations = [float(info[k]["duration"][()])
                     for k in info if "duration" in info[k]]
    return float(np.sum(durations)), len(durations)


if __name__ == "__main__":
    print(__doc__)
    print("Exports: " + ", ".join(__all__))
