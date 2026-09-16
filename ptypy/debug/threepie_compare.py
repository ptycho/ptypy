#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Shared helpers for comparing ThreePIE reconstructions.

The multislice comparison tools in this directory all need the same handful of
operations. If each tool carried its own slightly different copy, two
comparisons of the same pair of reconstructions could disagree, so the
operations live here once:

  ncorr / aligned_ncorr   phase-, scale- and translation-invariant similarity
  central                 crop away the poorly covered border
  gauge_phase             remove the global phase before plotting
  read_slices             per-slice objects from an engine's ``fslices`` file
  read_recon              object/probe/pixel size from a ``.ptyr``

The invariances are needed because a ptychographic solution is only defined
up to a global phase and a joint probe/object translation, and stochastic
(ePIE-type) engines shuffle their view order independently. An elementwise
comparison of two correct reconstructions therefore fails. Every comparison
in this directory goes through ``aligned_ncorr``.

This module is numpy-only at import time; ``h5py`` is imported lazily by the
readers.

This file is part of the PTYPY package.

    :copyright: Copyright 2014 by the PTYPY team, see AUTHORS.
    :license: see LICENSE for details.
"""

import argparse
import glob
import importlib.util

import numpy as np

__all__ = [
    "ncorr",
    "register_shift",
    "aligned_ncorr",
    "central",
    "common_central",
    "common_crop",
    "gauge_phase",
    "read_slices",
    "read_last_view",
    "read_slice_probes",
    "read_recon",
    "find_latest",
    "iteration_seconds",
    "ENGINE_DIRTAG",
    "PAIRS",
    "positive_int",
    "crop_list",
    "have_cupy",
]

# Engine class name -> tag used in the reconstruction directory names.
ENGINE_DIRTAG = {"ThreePIE": "cpu",
                 "ThreePIE_serial": "serial",
                 "ThreePIE_cupy": "gpu"}

# Backend pairs of the agreement tables, (a, b, label), in printing order.
PAIRS = (("ThreePIE_serial", "ThreePIE", "serial-vs-cpu"),
         ("ThreePIE_cupy", "ThreePIE", "gpu-vs-cpu"),
         ("ThreePIE_cupy", "ThreePIE_serial", "gpu-vs-serial"))


def positive_int(value):
    """argparse type: strictly positive integer."""
    ivalue = int(value)
    if ivalue < 1:
        raise argparse.ArgumentTypeError(
            "expected a positive integer, got %r" % (value,))
    return ivalue


def crop_list(value):
    """argparse type: comma-separated positive integers -> list of ints."""
    crops = []
    for part in str(value).split(","):
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


def have_cupy():
    """
    True only when cupy is importable and a GPU is reachable.

    Importing cupy and calling ``load_gpu_engines("cupy")`` both succeed on a
    machine that has cupy installed but no visible device (for example under
    ``CUDA_VISIBLE_DEVICES=""``); the failure would only show up inside a
    reconstruction. Touching the device here turns that into a skip.
    """
    if importlib.util.find_spec("cupy") is None:
        return False
    try:
        import cupy as cp
        cp.cuda.Device(0).compute_capability
        return True
    except Exception:
        return False


def common_crop(panels, frac):
    """Crop every panel centrally to the same fraction of the smallest shape."""
    n0 = int(min(p.shape[-2] for p in panels) * frac)
    n1 = int(min(p.shape[-1] for p in panels) * frac)
    out = []
    for x in panels:
        c0 = (x.shape[-2] - n0) // 2
        c1 = (x.shape[-1] - n1) // 2
        out.append(x[..., c0:c0 + n0, c1:c1 + n1])
    return out


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


def gauge_phase(obj):
    """
    Phase of ``obj`` with the global phase gauge removed, wrap-safely.

    The field is rotated by the phase of its complex mean, which keeps the
    result away from the +-pi branch cut. Subtracting a mean phase does not:
    near the cut an otherwise featureless panel renders as solid
    black-and-white noise.
    """
    ref = np.asarray(obj).mean()
    if np.abs(ref) > 0:
        obj = obj * np.exp(-1j * np.angle(ref))
    return np.angle(obj)


def find_latest(pattern):
    """Newest path matching a glob, or None."""
    hits = sorted(glob.glob(pattern))
    return hits[-1] if hits else None


def _read_slice_group(path, group):
    """
    ``{slice_index: data array}`` for one group of an ``fslices`` file.

    The containers in a group are named ``<container>_o_<index>`` (objects)
    or ``<container>_p_<index>`` (probes), one storage each; the full storage
    data (layers/modes first) is returned. An absent group gives ``{}``.
    """
    import h5py
    out = {}
    with h5py.File(path, "r") as f:
        key = "content/%s" % group
        if key not in f:
            return out
        for name in f[key]:
            idx = int(name.rsplit("_", 1)[-1])
            storage = f[key][name]
            storage = storage[list(storage.keys())[0]]
            out[idx] = np.array(storage["data"])
    return out


def read_slices(path):
    """
    Per-slice object arrays from an engine's ``fslices`` output.

    Returns ``{slice_index: complex ndarray}`` (first object layer). The slice
    containers are named ``<container>_o_<index>``, one storage each.
    """
    return {idx: data[0] for idx, data in _read_slice_group(path, "objects").items()}


def read_last_view(path):
    """
    ``{"ID", "layer", "coord"}`` of the last view the engine processed, as
    saved with the per-slice probes (``layer`` is the frame index, ``coord``
    the scan position in metres), or ``None`` for files without it.
    """
    import h5py
    with h5py.File(path, "r") as f:
        if "content/last_view" not in f:
            return None
        g = f["content/last_view"]
        ID = g["ID"][()]
        if isinstance(ID, bytes):
            ID = ID.decode()
        return {"ID": str(ID), "layer": int(g["layer"][()]),
                "coord": np.array(g["coord"], dtype=float)}


def read_slice_probes(path):
    """
    Per-slice incident waves from an engine's ``fslices`` output.

    Returns ``{slice_index: complex ndarray (modes, ny, nx)}``: index 0 is the
    reconstructed illumination, index ``s > 0`` the wave that entered slice
    ``s`` for the last view the engine processed (it carries the object
    structure of that scan position, so it is not a free-space propagated
    probe). Files written before the engines saved probes give ``{}``.
    """
    return _read_slice_group(path, "probes")


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
    Total engine seconds and number of recorded iteration blocks in a ``.ptyr``.

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
