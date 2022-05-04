'''


'''

import unittest
import numpy as np
from ptypy.accelerate.base.kernels import PoUpdateKernel

COMPLEX_TYPE = np.complex64
FLOAT_TYPE = np.float32
INT_TYPE = np.int32


class PoUpdateKernelTest(unittest.TestCase):

    def setUp(self):
        import sys
        np.set_printoptions(threshold=sys.maxsize, linewidth=np.inf)

    def tearDown(self):
        np.set_printoptions()

    def test_init(self):
        POUK = PoUpdateKernel()

        np.testing.assert_equal(POUK.kernels,
                                ['pr_update', 'ob_update'],
                                err_msg='PoUpdateKernel does not have the correct functions registered.')

    def prepare_arrays(self):
        B = 5  # frame size y
        C = 5  # frame size x

        D = 2  # number of probe modes
        E = B  # probe size y
        F = C  # probe size x

        npts_greater_than = 2  # how many points bigger than the probe the object is.
        G = 2  # number of object modes
        H = B + npts_greater_than  # object size y
        I = C + npts_greater_than  # object size x

        scan_pts = 2  # one dimensional scan point number

        total_number_scan_positions = scan_pts ** 2
        total_number_modes = G * D
        A = total_number_scan_positions * total_number_modes  # this is a 16 point scan pattern (4x4 grid) over all the modes

        probe = np.empty(shape=(D, E, F), dtype=COMPLEX_TYPE)
        for idx in range(D):
            probe[idx] = np.ones((E, F)) * (idx + 1) + 1j * np.ones((E, F)) * (idx + 1)

        object_array = np.empty(shape=(G, H, I), dtype=COMPLEX_TYPE)
        for idx in range(G):
            object_array[idx] = np.ones((H, I)) * (3 * idx + 1) + 1j * np.ones((H, I)) * (3 * idx + 1)

        exit_wave = np.empty(shape=(A, B, C), dtype=COMPLEX_TYPE)
        for idx in range(A):
            exit_wave[idx] = np.ones((B, C)) * (idx + 1) + 1j * np.ones((B, C)) * (idx + 1)

        X, Y = np.meshgrid(range(scan_pts), range(scan_pts))
        X = X.reshape((total_number_scan_positions))
        Y = Y.reshape((total_number_scan_positions))

        addr = np.zeros((total_number_scan_positions, total_number_modes, 5, 3), dtype=INT_TYPE)

        exit_idx = 0
        position_idx = 0
        for xpos, ypos in zip(X, Y):  #
            mode_idx = 0
            for pr_mode in range(D):
                for ob_mode in range(G):
                    addr[position_idx, mode_idx] = np.array([[pr_mode, 0, 0],
                                                             [ob_mode, ypos, xpos],
                                                             [exit_idx, 0, 0],
                                                             [0, 0, 0],
                                                             [0, 0, 0]], dtype=INT_TYPE)
                    mode_idx += 1
                    exit_idx += 1
            position_idx += 1

        object_array_denominator = np.empty_like(object_array, dtype=FLOAT_TYPE)
        for idx in range(G):
            object_array_denominator[idx] = np.ones((H, I)) * (5 * idx + 2)  # + 1j * np.ones((H, I)) * (5 * idx + 2)

        probe_denominator = np.empty_like(probe, dtype=FLOAT_TYPE)
        for idx in range(D):
            probe_denominator[idx] = np.ones((E, F)) * (5 * idx + 2)  # + 1j * np.ones((E, F)) * (5 * idx + 2)

        return addr, object_array, object_array_denominator, probe, exit_wave, probe_denominator

    def test_ob_update(self):
        # setup
        addr, object_array, object_array_denominator, probe, exit_wave, probe_denominator = self.prepare_arrays()

        # test
        POUK = PoUpdateKernel()
        POUK.allocate()  # doesn't do anything but is the call signature
        POUK.ob_update(addr, object_array, object_array_denominator, probe, exit_wave)

        # assert
        expected_object_array = np.array([[[15. + 1.j, 53. + 1.j, 53. + 1.j, 53. + 1.j, 53. + 1.j, 39. + 1.j, 1. + 1.j],
                                           [77. + 1.j, 201. + 1.j, 201. + 1.j, 201. + 1.j, 201. + 1.j, 125. + 1.j,
                                            1. + 1.j],
                                           [77. + 1.j, 201. + 1.j, 201. + 1.j, 201. + 1.j, 201. + 1.j, 125. + 1.j,
                                            1. + 1.j],
                                           [77. + 1.j, 201. + 1.j, 201. + 1.j, 201. + 1.j, 201. + 1.j, 125. + 1.j,
                                            1. + 1.j],
                                           [77. + 1.j, 201. + 1.j, 201. + 1.j, 201. + 1.j, 201. + 1.j, 125. + 1.j,
                                            1. + 1.j],
                                           [63. + 1.j, 149. + 1.j, 149. + 1.j, 149. + 1.j, 149. + 1.j, 87. + 1.j,
                                            1. + 1.j],
                                           [1. + 1.j, 1. + 1.j, 1. + 1.j, 1. + 1.j, 1. + 1.j, 1. + 1.j, 1. + 1.j]],
                                          [[24. + 4.j, 68. + 4.j, 68. + 4.j, 68. + 4.j, 68. + 4.j, 48. + 4.j, 4. + 4.j],
                                           [92. + 4.j, 228. + 4.j, 228. + 4.j, 228. + 4.j, 228. + 4.j, 140. + 4.j,
                                            4. + 4.j],
                                           [92. + 4.j, 228. + 4.j, 228. + 4.j, 228. + 4.j, 228. + 4.j, 140. + 4.j,
                                            4. + 4.j],
                                           [92. + 4.j, 228. + 4.j, 228. + 4.j, 228. + 4.j, 228. + 4.j, 140. + 4.j,
                                            4. + 4.j],
                                           [92. + 4.j, 228. + 4.j, 228. + 4.j, 228. + 4.j, 228. + 4.j, 140. + 4.j,
                                            4. + 4.j],
                                           [72. + 4.j, 164. + 4.j, 164. + 4.j, 164. + 4.j, 164. + 4.j, 96. + 4.j,
                                            4. + 4.j],
                                           [4. + 4.j, 4. + 4.j, 4. + 4.j, 4. + 4.j, 4. + 4.j, 4. + 4.j, 4. + 4.j]]],
                                         dtype=COMPLEX_TYPE)
        np.testing.assert_array_equal(object_array, expected_object_array,
                                      err_msg="The object array has not been updated as expected")

        # assert
        expected_object_array_denominator = np.array([[[12., 22., 22., 22., 22., 12., 2.],
                                                       [22., 42., 42., 42., 42., 22., 2.],
                                                       [22., 42., 42., 42., 42., 22., 2.],
                                                       [22., 42., 42., 42., 42., 22., 2.],
                                                       [22., 42., 42., 42., 42., 22., 2.],
                                                       [12., 22., 22., 22., 22., 12., 2.],
                                                       [2., 2., 2., 2., 2., 2., 2.]],

                                                      [[17., 27., 27., 27., 27., 17., 7.],
                                                       [27., 47., 47., 47., 47., 27., 7.],
                                                       [27., 47., 47., 47., 47., 27., 7.],
                                                       [27., 47., 47., 47., 47., 27., 7.],
                                                       [27., 47., 47., 47., 47., 27., 7.],
                                                       [17., 27., 27., 27., 27., 17., 7.],
                                                       [7., 7., 7., 7., 7., 7., 7.]]], dtype=FLOAT_TYPE)
        np.testing.assert_array_equal(object_array_denominator, expected_object_array_denominator,
                                      err_msg="The object array denominatorhas not been updated as expected")

    def test_pr_update(self):
        # setup
        addr, object_array, object_array_denominator, probe, exit_wave, probe_denominator = self.prepare_arrays()

        # test
        POUK = PoUpdateKernel()
        POUK.allocate()  # this doesn't do anything, but is the call pattern.
        POUK.pr_update(addr, probe, probe_denominator, object_array, exit_wave)

        # assert
        expected_probe = np.array([[[313. + 1.j, 313. + 1.j, 313. + 1.j, 313. + 1.j, 313. + 1.j],
                                    [313. + 1.j, 313. + 1.j, 313. + 1.j, 313. + 1.j, 313. + 1.j],
                                    [313. + 1.j, 313. + 1.j, 313. + 1.j, 313. + 1.j, 313. + 1.j],
                                    [313. + 1.j, 313. + 1.j, 313. + 1.j, 313. + 1.j, 313. + 1.j],
                                    [313. + 1.j, 313. + 1.j, 313. + 1.j, 313. + 1.j, 313. + 1.j]],

                                   [[394. + 2.j, 394. + 2.j, 394. + 2.j, 394. + 2.j, 394. + 2.j],
                                    [394. + 2.j, 394. + 2.j, 394. + 2.j, 394. + 2.j, 394. + 2.j],
                                    [394. + 2.j, 394. + 2.j, 394. + 2.j, 394. + 2.j, 394. + 2.j],
                                    [394. + 2.j, 394. + 2.j, 394. + 2.j, 394. + 2.j, 394. + 2.j],
                                    [394. + 2.j, 394. + 2.j, 394. + 2.j, 394. + 2.j, 394. + 2.j]]],
                                  dtype=COMPLEX_TYPE)
        np.testing.assert_array_equal(probe, expected_probe,
                                      err_msg="The probe has not been updated as expected")

        # assert
        expected_probe_denominator = np.array([[[138., 138., 138., 138., 138.],
                                                [138., 138., 138., 138., 138.],
                                                [138., 138., 138., 138., 138.],
                                                [138., 138., 138., 138., 138.],
                                                [138., 138., 138., 138., 138.]],

                                               [[143., 143., 143., 143., 143.],
                                                [143., 143., 143., 143., 143.],
                                                [143., 143., 143., 143., 143.],
                                                [143., 143., 143., 143., 143.],
                                                [143., 143., 143., 143., 143.]]], dtype=FLOAT_TYPE)
        np.testing.assert_array_equal(probe_denominator, expected_probe_denominator,
                                      err_msg="The probe denominatorhas not been updated as expected")

    def test_pr_update_ML(self):
        # setup  
        addr, object_array, object_array_denominator, probe, exit_wave, probe_denominator = self.prepare_arrays()

        # test
        POUK = PoUpdateKernel()
        POUK.allocate()  # this doesn't do anything, but is the call pattern.
        POUK.pr_update_ML(addr, probe, object_array, exit_wave)

        # assert
        expected_probe = np.array([[[625. + 1.j, 625. + 1.j, 625. + 1.j, 625. + 1.j, 625. + 1.j],
                                    [625. + 1.j, 625. + 1.j, 625. + 1.j, 625. + 1.j, 625. + 1.j],
                                    [625. + 1.j, 625. + 1.j, 625. + 1.j, 625. + 1.j, 625. + 1.j],
                                    [625. + 1.j, 625. + 1.j, 625. + 1.j, 625. + 1.j, 625. + 1.j],
                                    [625. + 1.j, 625. + 1.j, 625. + 1.j, 625. + 1.j, 625. + 1.j]],

                                   [[786. + 2.j, 786. + 2.j, 786. + 2.j, 786. + 2.j, 786. + 2.j],
                                    [786. + 2.j, 786. + 2.j, 786. + 2.j, 786. + 2.j, 786. + 2.j],
                                    [786. + 2.j, 786. + 2.j, 786. + 2.j, 786. + 2.j, 786. + 2.j],
                                    [786. + 2.j, 786. + 2.j, 786. + 2.j, 786. + 2.j, 786. + 2.j],
                                    [786. + 2.j, 786. + 2.j, 786. + 2.j, 786. + 2.j, 786. + 2.j]]],
                                  dtype=COMPLEX_TYPE)
        np.testing.assert_array_equal(probe, expected_probe,
                                      err_msg="The probe has not been updated as expected")

    def test_ob_update_ML(self):
        # setup
        addr, object_array, object_array_denominator, probe, exit_wave, probe_denominator = self.prepare_arrays()

        # test
        POUK = PoUpdateKernel()
        POUK.allocate()  # this doesn't do anything, but is the call pattern.
        POUK.ob_update_ML(addr, object_array, probe, exit_wave)

        # assert
        expected_object_array = np.array(
            [[[29. + 1.j, 105. + 1.j, 105. + 1.j, 105. + 1.j, 105. + 1.j, 77. + 1.j, 1. + 1.j],
              [153. + 1.j, 401. + 1.j, 401. + 1.j, 401. + 1.j, 401. + 1.j, 249. + 1.j, 1. + 1.j],
              [153. + 1.j, 401. + 1.j, 401. + 1.j, 401. + 1.j, 401. + 1.j, 249. + 1.j, 1. + 1.j],
              [153. + 1.j, 401. + 1.j, 401. + 1.j, 401. + 1.j, 401. + 1.j, 249. + 1.j, 1. + 1.j],
              [153. + 1.j, 401. + 1.j, 401. + 1.j, 401. + 1.j, 401. + 1.j, 249. + 1.j, 1. + 1.j],
              [125. + 1.j, 297. + 1.j, 297. + 1.j, 297. + 1.j, 297. + 1.j, 173. + 1.j, 1. + 1.j],
              [1. + 1.j, 1. + 1.j, 1. + 1.j, 1. + 1.j, 1. + 1.j, 1. + 1.j, 1. + 1.j]],

             [[44. + 4.j, 132. + 4.j, 132. + 4.j, 132. + 4.j, 132. + 4.j, 92. + 4.j, 4. + 4.j],
              [180. + 4.j, 452. + 4.j, 452. + 4.j, 452. + 4.j, 452. + 4.j, 276. + 4.j, 4. + 4.j],
              [180. + 4.j, 452. + 4.j, 452. + 4.j, 452. + 4.j, 452. + 4.j, 276. + 4.j, 4. + 4.j],
              [180. + 4.j, 452. + 4.j, 452. + 4.j, 452. + 4.j, 452. + 4.j, 276. + 4.j, 4. + 4.j],
              [180. + 4.j, 452. + 4.j, 452. + 4.j, 452. + 4.j, 452. + 4.j, 276. + 4.j, 4. + 4.j],
              [140. + 4.j, 324. + 4.j, 324. + 4.j, 324. + 4.j, 324. + 4.j, 188. + 4.j, 4. + 4.j],
              [4. + 4.j, 4. + 4.j, 4. + 4.j, 4. + 4.j, 4. + 4.j, 4. + 4.j, 4. + 4.j]]],
            dtype=COMPLEX_TYPE)
        np.testing.assert_array_equal(object_array, expected_object_array,
                                      err_msg="The object array has not been updated as expected")


    def test_pr_update_local(self):
        # setup
        B = 5  # frame size y
        C = 5  # frame size x

        D = 1  # number of probe modes
        E = B  # probe size y
        F = C  # probe size x

        npts_greater_than = 2  # how many points bigger than the probe the object is.
        G = 1  # number of object modes
        H = B + npts_greater_than  # object size y
        I = C + npts_greater_than  # object size x

        scan_pts = 1  # one dimensional scan point number

        total_number_scan_positions = scan_pts ** 2
        total_number_modes = G * D
        A = total_number_scan_positions * total_number_modes  # this is a 16 point scan pattern (4x4 grid) over all the modes

        probe = np.empty(shape=(D, E, F), dtype=COMPLEX_TYPE)
        for idx in range(D):
            probe[idx] = np.ones((E, F)) * (idx + 1) + 1j * np.ones((E, F)) * (idx + 1)

        object_array = np.empty(shape=(G, H, I), dtype=COMPLEX_TYPE)
        for idx in range(G):
            object_array[idx] = np.ones((H, I)) * (3 * idx + 1) + 1j * np.ones((H, I)) * (3 * idx + 1)

        exit_wave = np.empty(shape=(A, B, C), dtype=COMPLEX_TYPE)
        for idx in range(A):
            exit_wave[idx] = np.ones((B, C)) * (idx + 1) + 1j * np.ones((B, C)) * (idx + 1)
        auxiliary_wave = exit_wave.copy() * 1.5

        object_norm = np.empty(shape=(1,B,C), dtype=FLOAT_TYPE)

        X, Y = np.meshgrid(range(scan_pts), range(scan_pts))
        X = X.reshape((total_number_scan_positions))
        Y = Y.reshape((total_number_scan_positions))

        addr = np.zeros((total_number_scan_positions, total_number_modes, 5, 3), dtype=INT_TYPE)

        exit_idx = 0
        position_idx = 0
        for xpos, ypos in zip(X, Y):  #
            mode_idx = 0
            for pr_mode in range(D):
                for ob_mode in range(G):
                    addr[position_idx, mode_idx] = np.array([[pr_mode, 0, 0],
                                                             [ob_mode, ypos, xpos],
                                                             [exit_idx, 0, 0],
                                                             [0, 0, 0],
                                                             [0, 0, 0]], dtype=INT_TYPE)
                    mode_idx += 1
                    exit_idx += 1
            position_idx += 1

        # test
        POUK = PoUpdateKernel()
        POUK.allocate()  # this doesn't do anything, but is the call pattern.
        POUK.ob_norm_local(addr, object_array, object_norm)
        POUK.pr_update_local(addr, probe, object_array, exit_wave, auxiliary_wave, object_norm, object_norm.max())

        # assert
        expected_probe = np.array([[[0.5+1.j, 0.5+1.j, 0.5+1.j, 0.5+1.j, 0.5+1.j],
                                    [0.5+1.j, 0.5+1.j, 0.5+1.j, 0.5+1.j, 0.5+1.j],
                                    [0.5+1.j, 0.5+1.j, 0.5+1.j, 0.5+1.j, 0.5+1.j],
                                    [0.5+1.j, 0.5+1.j, 0.5+1.j, 0.5+1.j, 0.5+1.j],
                                    [0.5+1.j, 0.5+1.j, 0.5+1.j, 0.5+1.j, 0.5+1.j]]], dtype=COMPLEX_TYPE)
        np.testing.assert_array_equal(probe, expected_probe,
                                      err_msg="The probe has not been updated as expected")

    def test_ob_update_local(self):
        # setup
        B = 5  # frame size y
        C = 5  # frame size x

        D = 1  # number of probe modes
        E = B  # probe size y
        F = C  # probe size x

        npts_greater_than = 2  # how many points bigger than the probe the object is.
        G = 1  # number of object modes
        H = B + npts_greater_than  # object size y
        I = C + npts_greater_than  # object size x

        scan_pts = 1  # one dimensional scan point number

        total_number_scan_positions = scan_pts ** 2
        total_number_modes = G * D
        A = total_number_scan_positions * total_number_modes  # this is a 16 point scan pattern (4x4 grid) over all the modes

        probe = np.empty(shape=(D, E, F), dtype=COMPLEX_TYPE)
        for idx in range(D):
            probe[idx] = np.ones((E, F)) * (idx + 1) + 1j * np.ones((E, F)) * (idx + 1)

        object_array = np.empty(shape=(G, H, I), dtype=COMPLEX_TYPE)
        for idx in range(G):
            object_array[idx] = np.ones((H, I)) * (3 * idx + 1) + 1j * np.ones((H, I)) * (3 * idx + 1)

        exit_wave = np.empty(shape=(A, B, C), dtype=COMPLEX_TYPE)
        for idx in range(A):
            exit_wave[idx] = np.ones((B, C)) * (idx + 1) + 1j * np.ones((B, C)) * (idx + 1)
        auxiliary_wave = exit_wave.copy() * 2

        probe_norm = np.empty(shape=(1,B,C), dtype=FLOAT_TYPE)

        X, Y = np.meshgrid(range(scan_pts), range(scan_pts))
        X = X.reshape((total_number_scan_positions))
        Y = Y.reshape((total_number_scan_positions))

        addr = np.zeros((total_number_scan_positions, total_number_modes, 5, 3), dtype=INT_TYPE)

        exit_idx = 0
        position_idx = 0
        for xpos, ypos in zip(X, Y):  #
            mode_idx = 0
            for pr_mode in range(D):
                for ob_mode in range(G):
                    addr[position_idx, mode_idx] = np.array([[pr_mode, 0, 0],
                                                             [ob_mode, ypos, xpos],
                                                             [exit_idx, 0, 0],
                                                             [0, 0, 0],
                                                             [0, 0, 0]], dtype=INT_TYPE)
                    mode_idx += 1
                    exit_idx += 1
            position_idx += 1

        # test
        POUK = PoUpdateKernel()
        POUK.allocate()  # this doesn't do anything, but is the call pattern.
        POUK.pr_norm_local(addr, probe, probe_norm)
        POUK.ob_update_local(addr, object_array, probe, exit_wave, auxiliary_wave, probe_norm)

        # assert
        expected_object_array = np.array([[[0.+1.j, 0.+1.j, 0.+1.j, 0.+1.j, 0.+1.j, 1.+1.j, 1.+1.j],
                                           [0.+1.j, 0.+1.j, 0.+1.j, 0.+1.j, 0.+1.j, 1.+1.j, 1.+1.j],
                                           [0.+1.j, 0.+1.j, 0.+1.j, 0.+1.j, 0.+1.j, 1.+1.j, 1.+1.j],
                                           [0.+1.j, 0.+1.j, 0.+1.j, 0.+1.j, 0.+1.j, 1.+1.j, 1.+1.j],
                                           [0.+1.j, 0.+1.j, 0.+1.j, 0.+1.j, 0.+1.j, 1.+1.j, 1.+1.j],
                                           [1.+1.j, 1.+1.j, 1.+1.j, 1.+1.j, 1.+1.j, 1.+1.j, 1.+1.j],
                                           [1.+1.j, 1.+1.j, 1.+1.j, 1.+1.j, 1.+1.j, 1.+1.j, 1.+1.j]]], dtype=COMPLEX_TYPE)
        np.testing.assert_array_equal(object_array, expected_object_array,
                                      err_msg="The object array has not been updated as expected")

    def test_pr_norm_local(self):
        # setup
        B = 5  # frame size y
        C = 5  # frame size x

        D = 2  # number of probe modes
        E = B  # probe size y
        F = C  # probe size x

        npts_greater_than = 2  # how many points bigger than the probe the object is.
        G = 2  # number of object modes
        H = B + npts_greater_than  # object size y
        I = C + npts_greater_than  # object size x

        scan_pts = 1  # one dimensional scan point number

        total_number_scan_positions = scan_pts ** 2
        total_number_modes = G * D
        A = total_number_scan_positions * total_number_modes  # this is a 16 point scan pattern (4x4 grid) over all the modes

        probe = np.empty(shape=(D, E, F), dtype=COMPLEX_TYPE)
        for idx in range(D):
            probe[idx] = np.ones((E, F)) * (idx + 1) + 1j * np.ones((E, F)) * (idx + 1)
        probe_norm = np.empty(shape=(1,B,C), dtype=FLOAT_TYPE)

        X, Y = np.meshgrid(range(scan_pts), range(scan_pts))
        X = X.reshape((total_number_scan_positions))
        Y = Y.reshape((total_number_scan_positions))

        addr = np.zeros((total_number_scan_positions, total_number_modes, 5, 3), dtype=INT_TYPE)

        exit_idx = 0
        position_idx = 0
        for xpos, ypos in zip(X, Y):  #
            mode_idx = 0
            for pr_mode in range(D):
                for ob_mode in range(G):
                    addr[position_idx, mode_idx] = np.array([[pr_mode, 0, 0],
                                                             [ob_mode, ypos, xpos],
                                                             [exit_idx, 0, 0],
                                                             [0, 0, 0],
                                                             [0, 0, 0]], dtype=INT_TYPE)
                    mode_idx += 1
                    exit_idx += 1
            position_idx += 1

        # test
        POUK = PoUpdateKernel()
        POUK.allocate()  # this doesn't do anything, but is the call pattern.
        POUK.pr_norm_local(addr, probe, probe_norm)

        # assert
        expected_probe_norm = np.array([[[10., 10., 10., 10., 10.],
                                         [10., 10., 10., 10., 10.],
                                         [10., 10., 10., 10., 10.],
                                         [10., 10., 10., 10., 10.],
                                         [10., 10., 10., 10., 10.]]], dtype=FLOAT_TYPE)
        np.testing.assert_array_equal(probe_norm, expected_probe_norm,
                                      err_msg="The probe norm has not been updated as expected")


    def test_ob_norm_local(self):
        # setup
        B = 5  # frame size y
        C = 5  # frame size x

        D = 2  # number of probe modes
        E = B  # probe size y
        F = C  # probe size x

        npts_greater_than = 2  # how many points bigger than the probe the object is.
        G = 2  # number of object modes
        H = B + npts_greater_than  # object size y
        I = C + npts_greater_than  # object size x

        scan_pts = 1  # one dimensional scan point number

        total_number_scan_positions = scan_pts ** 2
        total_number_modes = G * D
        A = total_number_scan_positions * total_number_modes  # this is a 16 point scan pattern (4x4 grid) over all the modes

        object_array = np.empty(shape=(G, H, I), dtype=COMPLEX_TYPE)
        for idx in range(G):
            object_array[idx] = np.ones((H, I)) * (3 * idx + 1) + 1j * np.ones((H, I)) * (3 * idx + 1)
        object_norm = np.empty(shape=(1,B,C), dtype=FLOAT_TYPE)

        X, Y = np.meshgrid(range(scan_pts), range(scan_pts))
        X = X.reshape((total_number_scan_positions))
        Y = Y.reshape((total_number_scan_positions))

        addr = np.zeros((total_number_scan_positions, total_number_modes, 5, 3), dtype=INT_TYPE)

        exit_idx = 0
        position_idx = 0
        for xpos, ypos in zip(X, Y):  #
            mode_idx = 0
            for pr_mode in range(D):
                for ob_mode in range(G):
                    addr[position_idx, mode_idx] = np.array([[pr_mode, 0, 0],
                                                             [ob_mode, ypos, xpos],
                                                             [exit_idx, 0, 0],
                                                             [0, 0, 0],
                                                             [0, 0, 0]], dtype=INT_TYPE)
                    mode_idx += 1
                    exit_idx += 1
            position_idx += 1

        # test
        POUK = PoUpdateKernel()
        POUK.allocate()  # this doesn't do anything, but is the call pattern.
        POUK.ob_norm_local(addr, object_array, object_norm)

        # assert
        expected_object_norm = np.array([[[34., 34., 34., 34., 34.],
                                           [34., 34., 34., 34., 34.],
                                           [34., 34., 34., 34., 34.],
                                           [34., 34., 34., 34., 34.],
                                           [34., 34., 34., 34., 34.]]], dtype=FLOAT_TYPE)
        np.testing.assert_array_equal(object_norm, expected_object_norm,
                                      err_msg="The object norm has not been updated as expected")

if __name__ == '__main__':
    unittest.main()
