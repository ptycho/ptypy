'''


'''

import unittest
import numpy as np
from ptypy.accelerate.array_based.po_update_kernel import PoUpdateKernel
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
        attrs = ["ob_shape",
                 "pr_shape",
                 "nviews",
                 "nmodes",
                 "ncoords",
                 "num_pods"]

        POUK = PoUpdateKernel()
        for attr in attrs:
            self.assertTrue(hasattr(POUK, attr), msg="PoUpdateKernel does not have attribute: %s" % attr)

        np.testing.assert_equal(POUK.kernels,
                                ['pr_update', 'ob_update'],
                                err_msg='PoUpdateKernel does not have the correct functions registered.')


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
        POUK = PoUpdateKernel()

        POUK.configure(object_array, probe, addr)

        expected_ob_shape = tuple([INT_TYPE(G), INT_TYPE(H), INT_TYPE(I)])
        expected_pr_shape = tuple([INT_TYPE(D), INT_TYPE(E), INT_TYPE(F)])
        expected_nviews = INT_TYPE(total_number_scan_positions)
        expected_nmodes = INT_TYPE(total_number_modes)
        expected_ncoords = INT_TYPE(5)
        expected_naxes = INT_TYPE(3)
        expected_num_pods = INT_TYPE(A)

        np.testing.assert_equal(POUK.ob_shape, expected_ob_shape)
        np.testing.assert_equal(POUK.pr_shape, expected_pr_shape)
        np.testing.assert_equal(POUK.nviews, expected_nviews)
        np.testing.assert_equal(POUK.nmodes, expected_nmodes)
        np.testing.assert_equal(POUK.ncoords, expected_ncoords)
        np.testing.assert_equal(POUK.naxes, expected_naxes)
        np.testing.assert_equal(POUK.num_pods, expected_num_pods)

    def test_ob_update(self):
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
        object_array_denominator = np.empty_like(object_array)
        for idx in range(G):
            object_array_denominator[idx] = np.ones((H, I)) * (5 * idx + 2) + 1j * np.ones((H, I)) * (5 * idx + 2)


        POUK = PoUpdateKernel()

        POUK.configure(object_array, probe, addr)

        # print("object array denom before:")
        # print(object_array_denominator)

        POUK.ob_update(object_array, object_array_denominator, probe, exit_wave, addr)

        # print("object array denom after:")
        # print(repr(object_array_denominator))

        expected_object_array = np.array([[[15.+1.j, 53.+1.j, 53.+1.j, 53.+1.j, 53.+1.j, 39.+1.j, 1.+1.j],
                                           [77.+1.j, 201.+1.j, 201.+1.j, 201.+1.j, 201.+1.j, 125.+1.j, 1.+1.j],
                                           [77.+1.j, 201.+1.j, 201.+1.j, 201.+1.j, 201.+1.j, 125.+1.j, 1.+1.j],
                                           [77.+1.j, 201.+1.j, 201.+1.j, 201.+1.j, 201.+1.j, 125.+1.j, 1.+1.j],
                                           [77.+1.j, 201.+1.j, 201.+1.j, 201.+1.j, 201.+1.j, 125.+1.j, 1.+1.j],
                                           [63.+1.j, 149.+1.j, 149.+1.j, 149.+1.j, 149.+1.j, 87.+1.j, 1.+1.j],
                                           [1.+1.j, 1.+1.j, 1.+1.j, 1.+1.j, 1.+1.j, 1.+1.j, 1.+1.j]],
                                          [[24. + 4.j, 68. + 4.j, 68. + 4.j, 68. + 4.j, 68. + 4.j, 48. + 4.j, 4. + 4.j],
                                           [92. + 4.j, 228. + 4.j, 228. + 4.j, 228. + 4.j, 228. + 4.j, 140. + 4.j, 4. + 4.j],
                                           [92. + 4.j, 228. + 4.j, 228. + 4.j, 228. + 4.j, 228. + 4.j, 140. + 4.j, 4. + 4.j],
                                           [92. + 4.j, 228. + 4.j, 228. + 4.j, 228. + 4.j, 228. + 4.j, 140. + 4.j, 4. + 4.j],
                                           [92. + 4.j, 228. + 4.j, 228. + 4.j, 228. + 4.j, 228. + 4.j, 140. + 4.j, 4. + 4.j],
                                           [72. + 4.j, 164. + 4.j, 164. + 4.j, 164. + 4.j, 164. + 4.j,  96. + 4.j, 4. + 4.j],
                                           [4. + 4.j,  4. + 4.j,   4. + 4.j,   4. + 4.j,   4. + 4.j,   4. + 4.j,   4. + 4.j]]],
                                         dtype=COMPLEX_TYPE)

        np.testing.assert_array_equal(object_array, expected_object_array,
                                      err_msg="The object array has not been updated as expected")

        expected_object_array_denominator = np.array([[[12.+2.j, 22.+2.j, 22.+2.j, 22.+2.j, 22.+2.j, 12.+2.j,  2.+2.j],
                                                       [22.+2.j, 42.+2.j, 42.+2.j, 42.+2.j, 42.+2.j, 22.+2.j,  2.+2.j],
                                                       [22.+2.j, 42.+2.j, 42.+2.j, 42.+2.j, 42.+2.j, 22.+2.j,  2.+2.j],
                                                       [22.+2.j, 42.+2.j, 42.+2.j, 42.+2.j, 42.+2.j, 22.+2.j,  2.+2.j],
                                                       [22.+2.j, 42.+2.j, 42.+2.j, 42.+2.j, 42.+2.j, 22.+2.j,  2.+2.j],
                                                       [12.+2.j, 22.+2.j, 22.+2.j, 22.+2.j, 22.+2.j, 12.+2.j,  2.+2.j],
                                                       [ 2.+2.j,  2.+2.j,  2.+2.j,  2.+2.j,  2.+2.j,  2.+2.j,  2.+2.j]],

                                                      [[17.+7.j, 27.+7.j, 27.+7.j, 27.+7.j, 27.+7.j, 17.+7.j,  7.+7.j],
                                                       [27.+7.j, 47.+7.j, 47.+7.j, 47.+7.j, 47.+7.j, 27.+7.j,  7.+7.j],
                                                       [27.+7.j, 47.+7.j, 47.+7.j, 47.+7.j, 47.+7.j, 27.+7.j,  7.+7.j],
                                                       [27.+7.j, 47.+7.j, 47.+7.j, 47.+7.j, 47.+7.j, 27.+7.j,  7.+7.j],
                                                       [27.+7.j, 47.+7.j, 47.+7.j, 47.+7.j, 47.+7.j, 27.+7.j,  7.+7.j],
                                                       [17.+7.j, 27.+7.j, 27.+7.j, 27.+7.j, 27.+7.j, 17.+7.j,  7.+7.j],
                                                       [ 7.+7.j,  7.+7.j,  7.+7.j,  7.+7.j,  7.+7.j,  7.+7.j,  7.+7.j]]],
                                                     dtype=COMPLEX_TYPE)


        np.testing.assert_array_equal(object_array_denominator, expected_object_array_denominator,
                                      err_msg="The object array denominatorhas not been updated as expected")

    def test_ob_update(self):
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
        object_array_denominator = np.empty_like(object_array)
        for idx in range(G):
            object_array_denominator[idx] = np.ones((H, I)) * (5 * idx + 2) + 1j * np.ones((H, I)) * (5 * idx + 2)


        POUK = PoUpdateKernel()

        POUK.configure(object_array, probe, addr)

        # print("object array denom before:")
        # print(object_array_denominator)

        POUK.ob_update(object_array, object_array_denominator, probe, exit_wave, addr)

        # print("object array denom after:")
        # print(repr(object_array_denominator))

        expected_object_array = np.array([[[15.+1.j, 53.+1.j, 53.+1.j, 53.+1.j, 53.+1.j, 39.+1.j, 1.+1.j],
                                           [77.+1.j, 201.+1.j, 201.+1.j, 201.+1.j, 201.+1.j, 125.+1.j, 1.+1.j],
                                           [77.+1.j, 201.+1.j, 201.+1.j, 201.+1.j, 201.+1.j, 125.+1.j, 1.+1.j],
                                           [77.+1.j, 201.+1.j, 201.+1.j, 201.+1.j, 201.+1.j, 125.+1.j, 1.+1.j],
                                           [77.+1.j, 201.+1.j, 201.+1.j, 201.+1.j, 201.+1.j, 125.+1.j, 1.+1.j],
                                           [63.+1.j, 149.+1.j, 149.+1.j, 149.+1.j, 149.+1.j, 87.+1.j, 1.+1.j],
                                           [1.+1.j, 1.+1.j, 1.+1.j, 1.+1.j, 1.+1.j, 1.+1.j, 1.+1.j]],
                                          [[24. + 4.j, 68. + 4.j, 68. + 4.j, 68. + 4.j, 68. + 4.j, 48. + 4.j, 4. + 4.j],
                                           [92. + 4.j, 228. + 4.j, 228. + 4.j, 228. + 4.j, 228. + 4.j, 140. + 4.j, 4. + 4.j],
                                           [92. + 4.j, 228. + 4.j, 228. + 4.j, 228. + 4.j, 228. + 4.j, 140. + 4.j, 4. + 4.j],
                                           [92. + 4.j, 228. + 4.j, 228. + 4.j, 228. + 4.j, 228. + 4.j, 140. + 4.j, 4. + 4.j],
                                           [92. + 4.j, 228. + 4.j, 228. + 4.j, 228. + 4.j, 228. + 4.j, 140. + 4.j, 4. + 4.j],
                                           [72. + 4.j, 164. + 4.j, 164. + 4.j, 164. + 4.j, 164. + 4.j,  96. + 4.j, 4. + 4.j],
                                           [4. + 4.j,  4. + 4.j,   4. + 4.j,   4. + 4.j,   4. + 4.j,   4. + 4.j,   4. + 4.j]]],
                                         dtype=COMPLEX_TYPE)

        np.testing.assert_array_equal(object_array, expected_object_array,
                                      err_msg="The object array has not been updated as expected")

        expected_object_array_denominator = np.array([[[12.+2.j, 22.+2.j, 22.+2.j, 22.+2.j, 22.+2.j, 12.+2.j,  2.+2.j],
                                                       [22.+2.j, 42.+2.j, 42.+2.j, 42.+2.j, 42.+2.j, 22.+2.j,  2.+2.j],
                                                       [22.+2.j, 42.+2.j, 42.+2.j, 42.+2.j, 42.+2.j, 22.+2.j,  2.+2.j],
                                                       [22.+2.j, 42.+2.j, 42.+2.j, 42.+2.j, 42.+2.j, 22.+2.j,  2.+2.j],
                                                       [22.+2.j, 42.+2.j, 42.+2.j, 42.+2.j, 42.+2.j, 22.+2.j,  2.+2.j],
                                                       [12.+2.j, 22.+2.j, 22.+2.j, 22.+2.j, 22.+2.j, 12.+2.j,  2.+2.j],
                                                       [ 2.+2.j,  2.+2.j,  2.+2.j,  2.+2.j,  2.+2.j,  2.+2.j,  2.+2.j]],

                                                      [[17.+7.j, 27.+7.j, 27.+7.j, 27.+7.j, 27.+7.j, 17.+7.j,  7.+7.j],
                                                       [27.+7.j, 47.+7.j, 47.+7.j, 47.+7.j, 47.+7.j, 27.+7.j,  7.+7.j],
                                                       [27.+7.j, 47.+7.j, 47.+7.j, 47.+7.j, 47.+7.j, 27.+7.j,  7.+7.j],
                                                       [27.+7.j, 47.+7.j, 47.+7.j, 47.+7.j, 47.+7.j, 27.+7.j,  7.+7.j],
                                                       [27.+7.j, 47.+7.j, 47.+7.j, 47.+7.j, 47.+7.j, 27.+7.j,  7.+7.j],
                                                       [17.+7.j, 27.+7.j, 27.+7.j, 27.+7.j, 27.+7.j, 17.+7.j,  7.+7.j],
                                                       [ 7.+7.j,  7.+7.j,  7.+7.j,  7.+7.j,  7.+7.j,  7.+7.j,  7.+7.j]]],
                                                     dtype=COMPLEX_TYPE)


        np.testing.assert_array_equal(object_array_denominator, expected_object_array_denominator,
                                      err_msg="The object array denominatorhas not been updated as expected")

    def test_pr_update(self):
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

        '''
        test
        '''
        probe_denominator = np.empty_like(probe)
        for idx in range(D):
            probe_denominator[idx] = np.ones((E, F)) * (5 * idx + 2) + 1j * np.ones((E, F)) * (5 * idx + 2)

        POUK = PoUpdateKernel()

        POUK.configure(object_array, probe, addr)

        # print("probe array before:")
        # print(repr(probe))
        # print("probe denominator array before:")
        # print(repr(probe_denominator))

        POUK.pr_update(probe, probe_denominator, object_array, exit_wave, addr)

        # print("probe array after:")
        # print(repr(probe))
        # print("probe denominator array after:")
        # print(repr(probe_denominator))


        expected_probe = np.array([[[313.+1.j, 313.+1.j, 313.+1.j, 313.+1.j, 313.+1.j],
                                    [313.+1.j, 313.+1.j, 313.+1.j, 313.+1.j, 313.+1.j],
                                    [313.+1.j, 313.+1.j, 313.+1.j, 313.+1.j, 313.+1.j],
                                    [313.+1.j, 313.+1.j, 313.+1.j, 313.+1.j, 313.+1.j],
                                    [313.+1.j, 313.+1.j, 313.+1.j, 313.+1.j, 313.+1.j]],

                                   [[394.+2.j, 394.+2.j, 394.+2.j, 394.+2.j, 394.+2.j],
                                    [394.+2.j, 394.+2.j, 394.+2.j, 394.+2.j, 394.+2.j],
                                    [394.+2.j, 394.+2.j, 394.+2.j, 394.+2.j, 394.+2.j],
                                    [394.+2.j, 394.+2.j, 394.+2.j, 394.+2.j, 394.+2.j],
                                    [394.+2.j, 394.+2.j, 394.+2.j, 394.+2.j, 394.+2.j]]],
                                  dtype=COMPLEX_TYPE)


        np.testing.assert_array_equal(probe, expected_probe,
                                      err_msg="The probe has not been updated as expected")

        expected_probe_denominator = np.array([[[138.+2.j, 138.+2.j, 138.+2.j, 138.+2.j, 138.+2.j],
                                                [138.+2.j, 138.+2.j, 138.+2.j, 138.+2.j, 138.+2.j],
                                                [138.+2.j, 138.+2.j, 138.+2.j, 138.+2.j, 138.+2.j],
                                                [138.+2.j, 138.+2.j, 138.+2.j, 138.+2.j, 138.+2.j],
                                                [138.+2.j, 138.+2.j, 138.+2.j, 138.+2.j, 138.+2.j]],

                                               [[143.+7.j, 143.+7.j, 143.+7.j, 143.+7.j, 143.+7.j],
                                                [143.+7.j, 143.+7.j, 143.+7.j, 143.+7.j, 143.+7.j],
                                                [143.+7.j, 143.+7.j, 143.+7.j, 143.+7.j, 143.+7.j],
                                                [143.+7.j, 143.+7.j, 143.+7.j, 143.+7.j, 143.+7.j],
                                                [143.+7.j, 143.+7.j, 143.+7.j, 143.+7.j, 143.+7.j]]],
                                              dtype=COMPLEX_TYPE)

        np.testing.assert_array_equal(probe_denominator, expected_probe_denominator,
                                      err_msg="The probe denominatorhas not been updated as expected")


if __name__ == '__main__':
    unittest.main()
