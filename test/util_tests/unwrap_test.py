# -*- coding: utf-8 -*-
"""
Tests for the quality-map phase unwrapping.

This file is part of the PTYPY package.

    :copyright: Copyright 2014 by the PTYPY team, see AUTHORS.
    :license: see LICENSE for details.
"""

import unittest
import numpy as np

import pytest

# Skip everything if ptypy was installed without the C extension.
pytest.importorskip("ptypy.utils.unwrap._qmunwrap")

from ptypy.utils.unwrap import unwrap, qualitymap


def wrap(a):
    """Wrap `a` into [-pi, pi]."""
    return a - 2 * np.pi * np.round(a / (2 * np.pi))


def smooth_phase(shape=(128, 128), sigma=10., max_gradient=2.5, seed=1234):
    """
    A random smooth phase, scaled so that the largest gradient between
    neighbouring pixels stays below pi and the phase is therefore unwrappable.
    """
    from scipy.ndimage import gaussian_filter
    rng = np.random.default_rng(seed)
    a = gaussian_filter(rng.normal(size=shape), sigma)
    g = max(np.abs(np.diff(a, axis=0)).max(), np.abs(np.diff(a, axis=1)).max())
    return a * (max_gradient / g)


class UnwrapTest(unittest.TestCase):

    def test_roundtrip(self):
        """Unwrapping a wrapped smooth phase gives the phase back."""
        phase = smooth_phase()
        assert np.ptp(phase) > 4 * np.pi, "test phase does not wrap"
        out = unwrap(wrap(phase))
        np.testing.assert_allclose(out - out[0, 0], phase - phase[0, 0],
                                   atol=1e-9)

    def test_roundtrip_many_wraps(self):
        """The same, over a couple of dozen wraps."""
        phase = smooth_phase((256, 256), sigma=20., max_gradient=2.8)
        out = unwrap(wrap(phase))
        np.testing.assert_allclose(out - out[0, 0], phase - phase[0, 0],
                                   atol=1e-9)

    def test_starting_point_is_untouched(self):
        """The seed pixel keeps its input value."""
        phase = wrap(smooth_phase())
        for start in [(0, 0), (37, 61), (127, 127)]:
            out = unwrap(phase, start=start)
            self.assertEqual(out[start], phase[start])

    def test_starting_point_shifts_by_multiple_of_2pi(self):
        """Changing the seed only shifts the result by a multiple of 2 pi."""
        phase = wrap(smooth_phase())
        a = unwrap(phase, start=(0, 0))
        b = unwrap(phase, start=(37, 61))
        offsets = a - b
        np.testing.assert_allclose(offsets, offsets[0, 0], atol=1e-9)
        self.assertAlmostEqual(offsets[0, 0] % (2 * np.pi), 0., places=9)

    def test_congruent_to_input(self):
        """The output always stays congruent to the input modulo 2 pi."""
        rng = np.random.default_rng(0)
        for phase in [wrap(smooth_phase()),
                      rng.uniform(-np.pi, np.pi, size=(40, 40))]:
            out = unwrap(phase)
            np.testing.assert_allclose(wrap(out - phase), 0., atol=1e-9)

    def test_num_levels(self):
        """Any number of quality levels gives a valid unwrapping."""
        phase = smooth_phase()
        for num_levels in [1, 2, 8, 32, 100]:
            out = unwrap(wrap(phase), num_levels=num_levels)
            np.testing.assert_allclose(out - out[0, 0], phase - phase[0, 0],
                                       atol=1e-9)

    def test_constant_array(self):
        """A constant phase has a degenerate quality map."""
        phase = np.full((32, 32), 0.3)
        np.testing.assert_allclose(unwrap(phase), phase, atol=1e-9)

    def test_single_row_and_column(self):
        """Arrays with a single row or column have no interior."""
        ramp = np.linspace(0, 40, 64)
        for phase in [wrap(ramp)[None, :], wrap(ramp)[:, None]]:
            out = unwrap(phase)
            self.assertEqual(out.shape, phase.shape)
            np.testing.assert_allclose(out.ravel() - out.ravel()[0],
                                       ramp - ramp[0], atol=1e-9)

    def test_single_pixel(self):
        phase = np.array([[1.5]])
        np.testing.assert_allclose(unwrap(phase), phase)

    def test_input_is_not_modified(self):
        phase = wrap(smooth_phase((32, 32)))
        before = phase.copy()
        unwrap(phase)
        np.testing.assert_array_equal(phase, before)

    def test_accepts_awkward_input(self):
        """Non-contiguous, non-float64 and list inputs are converted."""
        phase = wrap(smooth_phase((64, 64)))
        expected = unwrap(phase)

        non_contiguous = np.zeros((64, 128))[:, ::2]
        non_contiguous[:] = phase
        np.testing.assert_allclose(unwrap(non_contiguous), expected)

        np.testing.assert_allclose(unwrap(phase.tolist()), expected)
        np.testing.assert_allclose(unwrap(phase.astype(np.float32)),
                                   expected, atol=1e-5)

    def test_invalid_arguments(self):
        phase = np.zeros((8, 8))
        with self.assertRaises(ValueError):
            unwrap(np.zeros(8))
        with self.assertRaises(ValueError):
            unwrap(np.zeros((2, 2, 2)))
        with self.assertRaises(ValueError):
            unwrap(phase, num_levels=0)
        with self.assertRaises(ValueError):
            unwrap(phase, num_levels=-3)
        with self.assertRaises(ValueError):
            unwrap(phase, start=(8, 0))
        with self.assertRaises(ValueError):
            unwrap(phase, start=(0, -1))


class QualityMapTest(unittest.TestCase):

    def test_constant_phase_has_zero_quality_map(self):
        np.testing.assert_allclose(qualitymap(np.full((16, 16), 2.)), 0.)

    def test_edges_are_included(self):
        """Every pixel takes part in at least two edges, so nothing is zero."""
        rng = np.random.default_rng(2)
        qmap = qualitymap(rng.uniform(-np.pi, np.pi, size=(16, 16)))
        self.assertTrue((qmap > 0).all())

    def test_matches_numpy(self):
        """Compare against the equivalent numpy expression."""
        phase = wrap(smooth_phase((32, 32)))
        qmap = qualitymap(phase)

        expected = np.zeros_like(phase)
        d0 = wrap(np.diff(phase, axis=0))
        expected[:-1, :] += d0 ** 2
        expected[1:, :] += d0 ** 2
        d1 = wrap(np.diff(phase, axis=1))
        expected[:, :-1] += d1 ** 2
        expected[:, 1:] += d1 ** 2

        np.testing.assert_allclose(qmap, expected, atol=1e-12)

    def test_smooth_phase_has_better_quality_than_noise(self):
        rng = np.random.default_rng(3)
        smooth = qualitymap(wrap(smooth_phase((64, 64))))
        noise = qualitymap(rng.uniform(-np.pi, np.pi, size=(64, 64)))
        self.assertLess(smooth.mean(), noise.mean())


if __name__ == '__main__':
    unittest.main()
