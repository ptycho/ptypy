"""
Tests for the ThreePIE crop-dependent propagation diagnostic.

This file is part of the PTYPY package.

    :copyright: Copyright 2014 by the PTYPY team, see AUTHORS.
    :license: see LICENSE for details.
"""
import unittest

from ptypy.debug import diagnose_threepie_crop as diag
from ptypy.accelerate.base.multislice import normalize_slice_pad
from test.accelerate_tests.cuda_cupy_tests import have_cupy


class ThreePIECropDiagnosticTest(unittest.TestCase):

    def setUp(self):
        self.diag = diag
        self.wavelength = self.diag.HC_KEV_M / 8.0
        self.detector_distance = 4.150
        self.detector_pixel_after_binning = 75e-6 * 2

    def _stats(self, raw_crop, slice_thickness):
        return self.diag.crop_sampling_stats(
            raw_crop=raw_crop,
            binning=2,
            wavelength=self.wavelength,
            detector_distance=self.detector_distance,
            detector_pixel=self.detector_pixel_after_binning / 2,
            slice_thickness=slice_thickness,
        )

    def test_default_nanomax_crop_128_is_unaliased(self):
        stats = self._stats(128, 1500e-6)
        self.assertEqual(stats["prepared_n"], 64)
        self.assertAlmostEqual(stats["dx"] * 1e9, 66.997, places=3)
        self.assertAlmostEqual(stats["zcrit"] * 1e3, 1.854, places=3)
        self.assertLess(stats["ratio"], 1.0)
        self.assertAlmostEqual(stats["keep_fraction"], 1.0)
        self.assertEqual(stats["status"], "unaliased")

    def test_default_nanomax_crop_256_needs_bandlimit(self):
        stats = self._stats(256, 1500e-6)
        self.assertEqual(stats["prepared_n"], 128)
        self.assertAlmostEqual(stats["dx"] * 1e9, 33.498, places=3)
        self.assertAlmostEqual(stats["zcrit"] * 1e3, 0.927, places=3)
        self.assertGreater(stats["ratio"], 1.0)
        self.assertAlmostEqual(stats["keep_fraction"], 0.3809, places=4)
        self.assertEqual(stats["status"], "needs-bandlimit")

    def test_crop_256_with_900um_slice_is_unaliased_control(self):
        stats = self._stats(256, 900e-6)
        self.assertLess(stats["ratio"], 1.0)
        self.assertAlmostEqual(stats["keep_fraction"], 1.0)
        self.assertEqual(stats["status"], "unaliased")

    def test_crop_256_fixed_distance_prefers_less_binning_or_padding(self):
        stats = self._stats(256, 1500e-6)
        self.assertEqual(self.diag.padding_suggestion(stats), 2)

        binning, safe_stats, checked = self.diag.binning_suggestion(
            raw_crop=256,
            binnings=[1, 2, 4],
            wavelength=self.wavelength,
            detector_distance=self.detector_distance,
            detector_pixel=75e-6,
            slice_thickness=1500e-6,
        )
        self.assertEqual(binning, 1)
        self.assertLess(safe_stats["ratio"], 1.0)
        self.assertEqual([item["prepared_n"] for item in checked], [256])

    def test_more_binning_does_not_raise_zcrit_for_fixed_raw_crop(self):
        bin2 = self._stats(256, 1500e-6)
        bin4 = self.diag.crop_sampling_stats(
            raw_crop=256,
            binning=4,
            wavelength=self.wavelength,
            detector_distance=self.detector_distance,
            detector_pixel=75e-6,
            slice_thickness=1500e-6,
        )
        self.assertAlmostEqual(bin4["dx"], bin2["dx"])
        self.assertLess(bin4["zcrit"], bin2["zcrit"])
        self.assertGreater(bin4["ratio"], bin2["ratio"])

    def test_recommendation_names_padding_for_crop_256_bin2(self):
        stats = self._stats(256, 1500e-6)
        text = self.diag.recommendation_text(
            stats, current_binning=2, suggested_binning=1)
        self.assertIn("keep the physical slice distance fixed", text)
        self.assertIn("--slice-pad 2", text)
        self.assertIn("lower binning to 1", text)

    def test_recommendation_names_larger_padding_for_crop_256_bin4(self):
        stats = self.diag.crop_sampling_stats(
            raw_crop=256,
            binning=4,
            wavelength=self.wavelength,
            detector_distance=self.detector_distance,
            detector_pixel=75e-6,
            slice_thickness=1500e-6,
        )
        text = self.diag.recommendation_text(
            stats, current_binning=4, suggested_binning=1)
        self.assertIn("--slice-pad 4", text)
        self.assertIn("keeps only 8.8% of frequencies", text)

    def test_normalize_slice_pad_rejects_non_positive_values(self):
        self.assertEqual(normalize_slice_pad(None, (64, 64), (1e-8, 1e-8), 8.0, 1e-6), 1)
        self.assertEqual(normalize_slice_pad(2, (64, 64), (1e-8, 1e-8), 8.0, 1e-6), 2)
        for bad in (0, -1):
            with self.assertRaises(ValueError):
                normalize_slice_pad(bad, (64, 64), (1e-8, 1e-8), 8.0, 1e-6)
        with self.assertRaises(ValueError):
            normalize_slice_pad("sideways", (64, 64), (1e-8, 1e-8), 8.0, 1e-6)

    def test_auto_slice_pad_grows_with_the_slice_spacing(self):
        thin = normalize_slice_pad("auto", (64, 64), (1e-8, 1e-8), 8.0, 1e-9)
        thick = normalize_slice_pad("auto", (64, 64), (1e-8, 1e-8), 8.0, 1e-3)
        self.assertEqual(thin, 1)
        self.assertEqual(thick, 4)

    @unittest.skipUnless(have_cupy(), "cupy and a GPU are required")
    def test_gpu_padding_wrapper_is_not_registered(self):
        import ptypy
        from ptypy.engines import ENGINES
        ptypy.load_gpu_engines("cupy")
        self.assertIn("ThreePIE_cupy", ENGINES)
        self.assertNotIn("_PaddedSlicePROP", ENGINES)


if __name__ == "__main__":
    unittest.main()
