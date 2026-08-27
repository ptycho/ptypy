"""Tests for the ThreePIE crop-dependent propagation diagnostic."""

import importlib.util
import os
import unittest


def _load_diagnostic_module():
    test_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(os.path.dirname(test_dir))
    path = os.path.join(repo_root, "ptypy", "debug", "diagnose_threepie_crop.py")
    spec = importlib.util.spec_from_file_location("diagnose_threepie_crop", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ThreePIECropDiagnosticTest(unittest.TestCase):

    def setUp(self):
        self.diag = _load_diagnostic_module()
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

    def test_gpu_padding_wrapper_is_not_registered_or_shadowed(self):
        test_dir = os.path.dirname(os.path.abspath(__file__))
        repo_root = os.path.dirname(os.path.dirname(test_dir))
        gpu_path = os.path.join(
            repo_root, "ptypy", "accelerate", "cuda_cupy", "engines",
            "stochastic.py")
        serial_path = os.path.join(
            repo_root, "ptypy", "custom", "threepie_serial.py")
        with open(gpu_path, "r") as stream:
            source = stream.read()

        marker = "class _PaddedSlicePROP:"
        class_start = source.index(marker)
        prefix_lines = [
            line.strip() for line in source[:class_start].splitlines()
            if line.strip()
        ]
        self.assertNotEqual(prefix_lines[-1], "@register()")
        self.assertIn("self._pad_factor = int(pad)", source)
        self.assertNotIn("self._pad = int(pad)", source)
        self.assertIn("normalize_slice_pad", source)

        with open(serial_path, "r") as stream:
            serial_source = stream.read()
        self.assertIn('"normalize_slice_pad"', serial_source)
        self.assertIn("def normalize_slice_pad", serial_source)
        self.assertIn("slice_pad must be a positive integer", serial_source)


if __name__ == "__main__":
    unittest.main()
