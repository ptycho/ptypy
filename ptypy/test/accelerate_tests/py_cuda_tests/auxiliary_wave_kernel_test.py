'''


'''

import unittest
import numpy as np
from ptypy.accelerate.array_based.auxiliary_wave_kernel import AuxiliaryWaveKernel

COMPLEX_TYPE = np.complex64
FLOAT_TYPE = np.float32
INT_TYPE = np.int32


class AuxiliaryWaveKernelTest(unittest.TestCase):

    def setUp(self):
        import sys
        np.set_printoptions(threshold=sys.maxsize, linewidth=np.inf)

    def tearDown(self):
        np.set_printoptions()

    def test_init(self):
        attrs = ["_offset",
                 "ob_shape",
                 "pr_shape",
                 "nviews",
                 "nmodes",
                 "ncoords",
                 "naxes",
                 "num_pods"]

        AWK = AuxiliaryWaveKernel()
        for attr in attrs:
            self.assertTrue(hasattr(AWK, attr), msg="AuxiliaryWaveKernel does not have attribute: %s" % attr)

        np.testing.assert_equal(AWK.kernels,
                                ['build_aux', 'build_exit'],
                                err_msg='AuxiliaryWaveKernel does not have the correct functions registered.')


    def test_batch_offset(self):
        AWK = AuxiliaryWaveKernel()
        self.assertEqual(AWK.batch_offset, None)

        set_batch = [INT_TYPE(10), np.int64(10), np.float32(10.0), np.float64(10.0)]

        for batch_set in set_batch:
            AWK.batch_offset = batch_set
            self.assertTrue(isinstance(AWK.batch_offset, INT_TYPE),
                            msg="AuxiliaryWaveKernel batch set has not converted type: %s, %s"
                                % (batch_set, type(batch_set)))

    def test_configure(self):
        '''
        setup
        '''
        B = 5  # frame size y
        C = 5  # frame size x

        D = 2  # number of probe modes
        E = B  # probe size y
        F = C  # probe size x

        npts_greater_than = 2  # how many points bigger than the probe the object is.
        G = 2  # number of object modes
        H = B + npts_greater_than  #  object size y
        I = C + npts_greater_than  #  object size x

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

        X, Y = np.meshgrid(range(scan_pts), range(scan_pts))
        X = X.reshape((total_number_scan_positions))
        Y = Y.reshape((total_number_scan_positions))

        addr = np.zeros((total_number_scan_positions, total_number_modes, 5, 3))

        exit_idx = 0
        position_idx = 0
        for xpos, ypos in zip(X, Y):#
            mode_idx = 0
            for pr_mode in range(D):
                for ob_mode in range(G):
                    addr[position_idx, mode_idx] = np.array([[pr_mode, 0, 0],
                                                             [ob_mode, ypos, xpos],
                                                             [exit_idx, 0, 0],
                                                             [0, 0, 0],
                                                             [0, 0, 0]])
                    mode_idx += 1
                    exit_idx += 1
            position_idx += 1

        '''
        test
        '''
        AWK = AuxiliaryWaveKernel()
        alpha_set = 0.9
        AWK.configure(object_array, probe, addr, alpha=alpha_set)


        expected_batch_offset = INT_TYPE(0)
        expected_ob_shape = tuple([INT_TYPE(G), INT_TYPE(H), INT_TYPE(I)])
        expected_pr_shape = tuple([INT_TYPE(D), INT_TYPE(E), INT_TYPE(F)])
        expected_nviews = INT_TYPE(total_number_scan_positions)
        expected_nmodes = INT_TYPE(total_number_modes)
        expected_ncoords = INT_TYPE(5)
        expected_naxes = INT_TYPE(3)
        expected_num_pods = INT_TYPE(A)
        expected_alpha = FLOAT_TYPE(alpha_set)

        np.testing.assert_equal(AWK.batch_offset, expected_batch_offset)
        np.testing.assert_equal(AWK.ob_shape, expected_ob_shape)
        np.testing.assert_equal(AWK.pr_shape, expected_pr_shape)
        np.testing.assert_equal(AWK.nviews, expected_nviews)
        np.testing.assert_equal(AWK.nmodes, expected_nmodes)
        np.testing.assert_equal(AWK.ncoords, expected_ncoords)
        np.testing.assert_equal(AWK.naxes, expected_naxes)
        np.testing.assert_equal(AWK.num_pods, expected_num_pods)

    def test_build_aux_same_as_exit(self):
        '''
        setup
        '''
        B = 3  # frame size y
        C = 3  # frame size x

        D = 2  # number of probe modes
        E = B  # probe size y
        F = C  # probe size x

        npts_greater_than = 2  # how many points bigger than the probe the object is.
        G = 2  # number of object modes
        H = B + npts_greater_than  #  object size y
        I = C + npts_greater_than  #  object size x

        scan_pts = 2  # one dimensional scan point number

        total_number_scan_positions = scan_pts ** 2
        total_number_modes = G * D
        A = total_number_scan_positions * total_number_modes # this is a 16 point scan pattern (4x4 grid) over all the modes


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
        for xpos, ypos in zip(X, Y):#
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

        '''
        test
        '''
        auxiliary_wave = np.zeros_like(exit_wave)

        AWK = AuxiliaryWaveKernel()
        alpha_set = 1.0
        AWK.configure(object_array, probe, addr, alpha=alpha_set)

        AWK.build_aux(auxiliary_wave, object_array, probe, exit_wave, addr)

        # print("auxiliary_wave after")
        # print(repr(auxiliary_wave))

        expected_auxiliary_wave = np.array([[[-1. + 3.j,  -1. + 3.j,  -1. + 3.j],
                                             [-1. + 3.j,  -1. + 3.j,  -1. + 3.j],
                                             [-1. + 3.j,  -1. + 3.j,  -1. + 3.j]],
                                            [[-2.+14.j,  -2.+14.j,  -2.+14.j],
                                             [-2.+14.j,  -2.+14.j,  -2.+14.j],
                                             [-2.+14.j,  -2.+14.j,  -2.+14.j]],
                                            [[-3. + 5.j,  -3. + 5.j,  -3. + 5.j],
                                             [-3. + 5.j,  -3. + 5.j,  -3. + 5.j],
                                             [-3. + 5.j,  -3. + 5.j,  -3. + 5.j]],
                                            [[-4.+28.j,  -4.+28.j,  -4.+28.j],
                                             [-4.+28.j,  -4.+28.j,  -4.+28.j],
                                             [-4.+28.j,  -4.+28.j,  -4.+28.j]],
                                            [[-5. - 1.j,  -5. - 1.j,  -5. - 1.j],
                                             [-5. - 1.j,  -5. - 1.j,  -5. - 1.j],
                                             [-5. - 1.j,  -5. - 1.j,  -5. - 1.j]],
                                            [[-6.+10.j,  -6.+10.j,  -6.+10.j],
                                             [-6.+10.j,  -6.+10.j,  -6.+10.j],
                                             [-6.+10.j,  -6.+10.j,  -6.+10.j]],
                                            [[-7. + 1.j,  -7. + 1.j,  -7. + 1.j],
                                             [-7. + 1.j,  -7. + 1.j,  -7. + 1.j],
                                             [-7. + 1.j,  -7. + 1.j,  -7. + 1.j]],
                                            [[-8.+24.j,  -8.+24.j,  -8.+24.j],
                                             [-8.+24.j,  -8.+24.j,  -8.+24.j],
                                             [-8.+24.j,  -8.+24.j,  -8.+24.j]],
                                            [[-9. - 5.j,  -9. - 5.j,  -9. - 5.j],
                                             [-9. - 5.j,  -9. - 5.j,  -9. - 5.j],
                                             [-9. - 5.j,  -9. - 5.j,  -9. - 5.j]],
                                            [[-10. + 6.j, -10. + 6.j, -10. + 6.j],
                                             [-10. + 6.j, -10. + 6.j, -10. + 6.j],
                                             [-10. + 6.j, -10. + 6.j, -10. + 6.j]],
                                            [[-11. - 3.j, -11. - 3.j, -11. - 3.j],
                                             [-11. - 3.j, -11. - 3.j, -11. - 3.j],
                                             [-11. - 3.j, -11. - 3.j, -11. - 3.j]],
                                            [[-12.+20.j, -12.+20.j, -12.+20.j],
                                             [-12.+20.j, -12.+20.j, -12.+20.j],
                                             [-12.+20.j, -12.+20.j, -12.+20.j]],
                                            [[-13. - 9.j, -13. - 9.j, -13. - 9.j],
                                             [-13. - 9.j, -13. - 9.j, -13. - 9.j],
                                             [-13. - 9.j, -13. - 9.j, -13. - 9.j]],
                                            [[-14. + 2.j, -14. + 2.j, -14. + 2.j],
                                             [-14. + 2.j, -14. + 2.j, -14. + 2.j],
                                             [-14. + 2.j, -14. + 2.j, -14. + 2.j]],
                                            [[-15. - 7.j, -15. - 7.j, -15. - 7.j],
                                             [-15. - 7.j, -15. - 7.j, -15. - 7.j],
                                             [-15. - 7.j, -15. - 7.j, -15. - 7.j]],
                                            [[-16.+16.j, -16.+16.j, -16.+16.j],
                                             [-16.+16.j, -16.+16.j, -16.+16.j],
                                             [-16.+16.j, -16.+16.j, -16.+16.j]]], dtype=COMPLEX_TYPE)

        np.testing.assert_array_equal(expected_auxiliary_wave, expected_auxiliary_wave,
                                      err_msg="The auxiliary_wave has not been updated as expected")

    def test_build_exit_aux_same_as_exit(self):
        '''
        setup
        '''
        B = 3  # frame size y
        C = 3  # frame size x

        D = 2  # number of probe modes
        E = B  # probe size y
        F = C  # probe size x

        npts_greater_than = 2  # how many points bigger than the probe the object is.
        G = 2  # number of object modes
        H = B + npts_greater_than  #  object size y
        I = C + npts_greater_than  #  object size x

        scan_pts = 2  # one dimensional scan point number

        total_number_scan_positions = scan_pts ** 2
        total_number_modes = G * D
        A = total_number_scan_positions * total_number_modes # this is a 16 point scan pattern (4x4 grid) over all the modes


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
        for xpos, ypos in zip(X, Y):#
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

        '''
        test
        '''
        auxiliary_wave = np.zeros_like(exit_wave)

        AWK = AuxiliaryWaveKernel()
        alpha_set = 1.0
        AWK.configure(object_array, probe, addr, alpha=alpha_set)

        AWK.build_exit(auxiliary_wave, object_array, probe, exit_wave, addr)
        #
        # print("auxiliary_wave after")
        # print(repr(auxiliary_wave))
        #
        # print("exit_wave after")
        # print(repr(exit_wave))

        expected_auxiliary_wave = np.array([[[0. - 2.j, 0. - 2.j, 0. - 2.j],
                                             [0. - 2.j, 0. - 2.j, 0. - 2.j],
                                             [0. - 2.j, 0. - 2.j, 0. - 2.j]],
                                            [[0. - 8.j, 0. - 8.j, 0. - 8.j],
                                             [0. - 8.j, 0. - 8.j, 0. - 8.j],
                                             [0. - 8.j, 0. - 8.j, 0. - 8.j]],
                                            [[0. - 4.j, 0. - 4.j, 0. - 4.j],
                                             [0. - 4.j, 0. - 4.j, 0. - 4.j],
                                             [0. - 4.j, 0. - 4.j, 0. - 4.j]],
                                            [[0.-16.j, 0.-16.j, 0.-16.j],
                                             [0.-16.j, 0.-16.j, 0.-16.j],
                                             [0.-16.j, 0.-16.j, 0.-16.j]],
                                            [[0. - 2.j, 0. - 2.j, 0. - 2.j],
                                             [0. - 2.j, 0. - 2.j, 0. - 2.j],
                                             [0. - 2.j, 0. - 2.j, 0. - 2.j]],
                                            [[0. - 8.j, 0. - 8.j, 0. - 8.j],
                                             [0. - 8.j, 0. - 8.j, 0. - 8.j],
                                             [0. - 8.j, 0. - 8.j, 0. - 8.j]],
                                            [[0. - 4.j, 0. - 4.j, 0. - 4.j],
                                             [0. - 4.j, 0. - 4.j, 0. - 4.j],
                                             [0. - 4.j, 0. - 4.j, 0. - 4.j]],
                                            [[0.-16.j, 0.-16.j, 0.-16.j],
                                             [0.-16.j, 0.-16.j, 0.-16.j],
                                             [0.-16.j, 0.-16.j, 0.-16.j]],
                                            [[0. - 2.j, 0. - 2.j, 0. - 2.j],
                                             [0. - 2.j, 0. - 2.j, 0. - 2.j],
                                             [0. - 2.j, 0. - 2.j, 0. - 2.j]],
                                            [[0. - 8.j, 0. - 8.j, 0. - 8.j],
                                             [0. - 8.j, 0. - 8.j, 0. - 8.j],
                                             [0. - 8.j, 0. - 8.j, 0. - 8.j]],
                                            [[0. - 4.j, 0. - 4.j, 0. - 4.j],
                                             [0. - 4.j, 0. - 4.j, 0. - 4.j],
                                             [0. - 4.j, 0. - 4.j, 0. - 4.j]],
                                            [[0.-16.j, 0.-16.j, 0.-16.j],
                                             [0.-16.j, 0.-16.j, 0.-16.j],
                                             [0.-16.j, 0.-16.j, 0.-16.j]],
                                            [[0. - 2.j, 0. - 2.j, 0. - 2.j],
                                             [0. - 2.j, 0. - 2.j, 0. - 2.j],
                                             [0. - 2.j, 0. - 2.j, 0. - 2.j]],
                                            [[0. - 8.j, 0. - 8.j, 0. - 8.j],
                                             [0. - 8.j, 0. - 8.j, 0. - 8.j],
                                             [0. - 8.j, 0. - 8.j, 0. - 8.j]],
                                            [[0. - 4.j, 0. - 4.j, 0. - 4.j],
                                             [0. - 4.j, 0. - 4.j, 0. - 4.j],
                                             [0. - 4.j, 0. - 4.j, 0. - 4.j]],
                                            [[0.-16.j, 0.-16.j, 0.-16.j],
                                             [0.-16.j, 0.-16.j, 0.-16.j],
                                             [0.-16.j, 0.-16.j, 0.-16.j]]], dtype=COMPLEX_TYPE)

        np.testing.assert_array_equal(expected_auxiliary_wave, expected_auxiliary_wave,
                                      err_msg="The auxiliary_wave has not been updated as expected")

        expected_exit_wave = np.array([[[1. - 1.j,  1. - 1.j,  1. - 1.j],
                                        [1. - 1.j,  1. - 1.j,  1. - 1.j],
                                        [1. - 1.j,  1. - 1.j,  1. - 1.j]],
                                       [[2. - 6.j,  2. - 6.j,  2. - 6.j],
                                        [2. - 6.j,  2. - 6.j,  2. - 6.j],
                                        [2. - 6.j,  2. - 6.j,  2. - 6.j]],
                                       [[3. - 1.j,  3. - 1.j,  3. - 1.j],
                                        [3. - 1.j,  3. - 1.j,  3. - 1.j],
                                        [3. - 1.j,  3. - 1.j,  3. - 1.j]],
                                       [[4. - 12.j,  4. - 12.j,  4. - 12.j],
                                        [4. - 12.j,  4. - 12.j,  4. - 12.j],
                                        [4. - 12.j,  4. - 12.j,  4. - 12.j]],
                                       [[5. + 3.j,  5. + 3.j,  5. + 3.j],
                                        [5. + 3.j,  5. + 3.j,  5. + 3.j],
                                        [5. + 3.j,  5. + 3.j,  5. + 3.j]],
                                       [[6. - 2.j,  6. - 2.j,  6. - 2.j],
                                        [6. - 2.j,  6. - 2.j,  6. - 2.j],
                                        [6. - 2.j,  6. - 2.j,  6. - 2.j]],
                                       [[7. + 3.j,  7. + 3.j,  7. + 3.j],
                                        [7. + 3.j,  7. + 3.j,  7. + 3.j],
                                        [7. + 3.j,  7. + 3.j,  7. + 3.j]],
                                       [[8. - 8.j,  8. - 8.j,  8. - 8.j],
                                        [8. - 8.j,  8. - 8.j,  8. - 8.j],
                                        [8. - 8.j,  8. - 8.j,  8. - 8.j]],
                                       [[9. + 7.j,  9. + 7.j,  9. + 7.j],
                                        [9. + 7.j,  9. + 7.j,  9. + 7.j],
                                        [9. + 7.j,  9. + 7.j,  9. + 7.j]],
                                       [[10. + 2.j, 10. + 2.j, 10. + 2.j],
                                        [10. + 2.j, 10. + 2.j, 10. + 2.j],
                                        [10. + 2.j, 10. + 2.j, 10. + 2.j]],
                                       [[11. + 7.j, 11. + 7.j, 11. + 7.j],
                                        [11. + 7.j, 11. + 7.j, 11. + 7.j],
                                        [11. + 7.j, 11. + 7.j, 11. + 7.j]],
                                       [[12. - 4.j, 12. - 4.j, 12. - 4.j],
                                        [12. - 4.j, 12. - 4.j, 12. - 4.j],
                                        [12. - 4.j, 12. - 4.j, 12. - 4.j]],
                                       [[13. + 11.j, 13. + 11.j, 13. + 11.j],
                                        [13. + 11.j, 13. + 11.j, 13. + 11.j],
                                        [13. + 11.j, 13. + 11.j, 13. + 11.j]],
                                       [[14. + 6.j, 14. + 6.j, 14. + 6.j],
                                        [14. + 6.j, 14. + 6.j, 14. + 6.j],
                                        [14. + 6.j, 14. + 6.j, 14. + 6.j]],
                                       [[15. + 11.j, 15. + 11.j, 15. + 11.j],
                                        [15. + 11.j, 15. + 11.j, 15. + 11.j],
                                        [15. + 11.j, 15. + 11.j, 15. + 11.j]],
                                       [[16. + 0.j, 16. + 0.j, 16. + 0.j],
                                        [16. + 0.j, 16. + 0.j, 16. + 0.j],
                                        [16. + 0.j, 16. + 0.j, 16. + 0.j]]], dtype=COMPLEX_TYPE)

        np.testing.assert_array_equal(expected_exit_wave, expected_exit_wave,
                                      err_msg="The exit_wave has not been updated as expected")

if __name__ == '__main__':
    unittest.main()
