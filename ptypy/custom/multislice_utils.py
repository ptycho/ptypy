# -*- coding: utf-8 -*-
"""
Helpers shared by the multislice (ThreePIE) engines: normalisation of the
slice padding option, the angular-spectrum band limit for alias-free
propagation between slices, and a centred crop/pad on the last two axes.

This file is part of the PTYPY package.

    :copyright: Copyright 2014 by the PTYPY team, see AUTHORS.
    :license: see LICENSE for details.
"""
import numpy as np

from ptypy.core import geometry

__all__ = ["normalize_slice_pad", "slice_bandlimit", "crop_pad_last2",
           "PaddedSlicePropagator", "PaddedSlicePropagationKernel"]


def normalize_slice_pad(value, shape, resolution, energy, slice_thickness):
    """
    Normalize the ThreePIE slice padding option to a positive integer.

    ``"auto"`` chooses the smallest pad factor that satisfies the angular
    spectrum sampling limit for the largest requested slice spacing, capped at
    four to keep memory growth bounded.
    """
    if value is None:
        return 1
    if isinstance(value, str):
        if value.lower() != "auto":
            raise ValueError('slice_pad must be a positive integer or "auto"')
        if isinstance(slice_thickness, (list, tuple)):
            distance = max(abs(float(d)) for d in slice_thickness)
        else:
            distance = abs(float(slice_thickness))
        n = int(min(shape[-2:]))
        dx = float(np.mean(resolution))
        wavelength = geometry.Geo._keV2m / float(energy)
        ratio = distance / (n * dx * dx / wavelength)
        return min(max(1, int(np.ceil(ratio))), 4)
    pad = int(value)
    if pad < 1:
        raise ValueError("slice_pad must be a positive integer")
    return pad


def slice_bandlimit(shape, resolution, energy, distance):
    """Angular-spectrum support mask for alias-free multislice propagation."""
    nrows, ncols = shape[-2:]
    dy, dx = resolution
    wavelength = geometry.Geo._keV2m / float(energy)
    distance = abs(float(distance))
    if distance == 0.0:
        return np.ones((nrows, ncols), dtype=np.complex64)
    vlim_y = 1.0 / np.sqrt((2.0 * distance / (nrows * dy)) ** 2 + 1.0)
    vlim_x = 1.0 / np.sqrt((2.0 * distance / (ncols * dx)) ** 2 + 1.0)
    y = ((np.arange(nrows) + nrows // 2) % nrows) - nrows // 2
    x = ((np.arange(ncols) + ncols // 2) % ncols) - ncols // 2
    vy = y * (wavelength / (nrows * dy))
    vx = x * (wavelength / (ncols * dx))
    VY, VX = np.meshgrid(vy, vx, indexing="ij")
    keep = (np.abs(VY) <= vlim_y) & (np.abs(VX) <= vlim_x)
    return keep.astype(np.complex64)


def _array_module(array):
    """``numpy`` or ``cupy``, whichever the array belongs to."""
    if type(array).__module__.split(".")[0] == "cupy":
        import cupy
        return cupy
    return np


def crop_pad_last2(array, target_shape, out=None):
    """
    Centered crop/pad on the last two axes, for a numpy or a cupy array.

    ``out`` is an optional buffer of the target shape to write into. The GPU
    engine passes one because allocating inside the update of a scan position
    would stop that update being captured as a CUDA graph.
    """
    target_shape = tuple(int(v) for v in target_shape)
    if out is None:
        out = _array_module(array).zeros(
            array.shape[:-2] + target_shape, dtype=array.dtype)
    else:
        out.fill(0)
    src_slices = []
    dst_slices = []
    for src_n, dst_n in zip(array.shape[-2:], target_shape):
        n = min(src_n, dst_n)
        src0 = (src_n - n) // 2
        dst0 = (dst_n - n) // 2
        src_slices.append(slice(src0, src0 + n))
        dst_slices.append(slice(dst0, dst0 + n))
    out[(...,) + tuple(dst_slices)] = array[(...,) + tuple(src_slices)]
    return out


class PaddedSlicePropagator:
    """
    Inter-slice propagator with optional centered zero-padding, for a
    propagator that takes a wave and returns one (``ptypy.core.geometry``).

    Padding keeps the real-space pixel size and enlarges the propagation grid,
    which raises the angular-spectrum sampling limit for a given slice
    spacing. ``pad=1`` passes the wave straight through.
    """

    def __init__(self, propagator, shape, pad=1):
        self.propagator = propagator
        self._pad_factor = int(pad)
        self._shape = tuple(int(v) for v in shape)
        self._padded_shape = tuple(int(v) * self._pad_factor for v in self._shape)

    def _run(self, wave, direction):
        if self._pad_factor == 1:
            return direction(wave)
        padded = crop_pad_last2(wave, self._padded_shape)
        return crop_pad_last2(direction(padded), self._shape)

    def fw(self, wave):
        return self._run(wave, self.propagator.fw)

    def bw(self, wave):
        return self._run(wave, self.propagator.bw)


class PaddedSlicePropagationKernel:
    """
    The same padding for an accelerate ``PropagationKernel``, which writes
    into an output buffer instead of returning: ``fw(x, y)`` / ``bw(x, y)``.

    The kernel it wraps is allocated on the padded grid. The intermediate
    buffer is kept between calls, so a padded propagation allocates nothing
    once it has run at least once.
    """

    def __init__(self, prop_kernel, shape, pad=1):
        self.propagator = prop_kernel
        self._pad_factor = int(pad)
        self._shape = tuple(int(v) for v in shape)
        self._padded_shape = tuple(int(v) * self._pad_factor for v in self._shape)
        self._buffer = None

    def _run(self, x, y, direction):
        if self._pad_factor == 1:
            direction(x, y)
            return
        wanted = x.shape[:-2] + self._padded_shape
        if self._buffer is None or self._buffer.shape != wanted:
            self._buffer = _array_module(x).zeros(wanted, dtype=x.dtype)
        crop_pad_last2(x, self._padded_shape, out=self._buffer)
        direction(self._buffer, self._buffer)
        crop_pad_last2(self._buffer, self._shape, out=y)

    def fw(self, x, y):
        self._run(x, y, self.propagator.fw)

    def bw(self, x, y):
        self._run(x, y, self.propagator.bw)
