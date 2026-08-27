"""Tests for the ThreePIE real-data matrix runner helpers."""

import importlib.util
import os
import unittest


def _load_matrix_module():
    test_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(os.path.dirname(test_dir))
    path = os.path.join(repo_root, "ptypy", "debug", "run_threepie_realdata_matrix.py")
    spec = importlib.util.spec_from_file_location("run_threepie_realdata_matrix", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Args:
    pass


class ThreePIEMatrixRunnerTest(unittest.TestCase):

    def setUp(self):
        self.matrix = _load_matrix_module()

    def test_default_output_suffix_follows_crop(self):
        args = Args()
        args.crop = 256
        args.output_suffix = None
        args.slice_thickness = 1500e-6
        args.slice_pad = 1
        self.assertEqual(self.matrix.output_suffix(args), "_LT_debug256")

        args.crop = 512
        self.assertEqual(self.matrix.output_suffix(args), "_LT_debug512")

    def test_default_output_suffix_includes_nondefault_slice_thickness(self):
        args = Args()
        args.crop = 256
        args.output_suffix = None
        args.slice_thickness = 900e-6
        args.slice_pad = 1
        self.assertEqual(self.matrix.output_suffix(args), "_LT_debug256_z900um")

    def test_default_output_suffix_includes_padding(self):
        args = Args()
        args.crop = 256
        args.output_suffix = None
        args.slice_thickness = 1500e-6
        args.slice_pad = 2
        self.assertEqual(self.matrix.output_suffix(args), "_LT_debug256_pad2")

    def test_explicit_output_suffix_is_preserved(self):
        args = Args()
        args.crop = 256
        args.output_suffix = "_custom"
        args.slice_thickness = 900e-6
        args.slice_pad = 2
        self.assertEqual(self.matrix.output_suffix(args), "_custom")

    def test_detector_pixel_defaults_match_nanomax_runner(self):
        args = Args()
        args.detector_pixel = None

        args.detector = "eiger4m"
        self.assertEqual(self.matrix.detector_pixel(args), 75e-6)

        args.detector = "merlin"
        self.assertEqual(self.matrix.detector_pixel(args), 55e-6)

        args.detector = "pilatus"
        self.assertEqual(self.matrix.detector_pixel(args), 172e-6)

    def test_detector_pixel_override(self):
        args = Args()
        args.detector = "eiger4m"
        args.detector_pixel = 1.23e-6
        self.assertEqual(self.matrix.detector_pixel(args), 1.23e-6)

    def test_diagnostic_crops_default_shows_transition_context(self):
        args = Args()
        args.crop = 256
        args.diagnostic_crops = None
        self.assertEqual(self.matrix.diagnostic_crops(args), "128,256,512")

        args.crop = 128
        self.assertEqual(self.matrix.diagnostic_crops(args), "128,512")

    def test_diagnostic_crops_override(self):
        args = Args()
        args.crop = 256
        args.diagnostic_crops = "64,128,256"
        self.assertEqual(self.matrix.diagnostic_crops(args), "64,128,256")

    def test_common_run_args_forward_fixed_distance_sampling_controls(self):
        args = Args()
        args.ptypy_path = "/ptypy"
        args.beamtime_basedir = "/beamtime"
        args.sample = "sample"
        args.detector = "eiger4m"
        args.scan = 434
        args.distance = 4.15
        args.defocus_um = -750.0
        args.energy_kev = 8.0
        args.crop = 256
        args.center_y = 1281.0
        args.center_x = 772.0
        args.binning = 2
        args.probe_modes = 2
        args.numiter = 10
        args.save_every = 10
        args.number_of_slices = 2
        args.slice_thickness = 1500e-6
        args.slice_start_iteration = "0"
        args.output_suffix = None
        args.frames_per_block = None
        args.no_slice_bandlimit = True
        args.slice_pad = 2

        cmd = ["python", "run_threepie_cupy_nanomax.py"]
        self.matrix.add_common_run_args(cmd, args)

        self.assertIn("--slice-thickness", cmd)
        self.assertEqual(cmd[cmd.index("--slice-thickness") + 1], "0.0015")
        self.assertIn("--no-slice-bandlimit", cmd)
        self.assertIn("--slice-pad", cmd)
        self.assertEqual(cmd[cmd.index("--slice-pad") + 1], "2")
        self.assertIn("--output-suffix", cmd)
        self.assertEqual(cmd[cmd.index("--output-suffix") + 1], "_LT_debug256_pad2")

    def test_positive_int_rejects_invalid_slice_pad(self):
        self.assertEqual(self.matrix.positive_int("2"), 2)
        with self.assertRaises(Exception):
            self.matrix.positive_int("0")
        with self.assertRaises(Exception):
            self.matrix.positive_int("-1")


if __name__ == "__main__":
    unittest.main()
