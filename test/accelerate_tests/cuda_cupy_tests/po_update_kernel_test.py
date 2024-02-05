'''


'''

import unittest
import numpy as np
from . import CupyCudaTest, have_cupy
from ptypy.accelerate.base.array_utils import max_abs2

if have_cupy():
    import cupy as cp
    from ptypy.accelerate.cuda_cupy.kernels import PoUpdateKernel

COMPLEX_TYPE = np.complex64
FLOAT_TYPE = np.float32
INT_TYPE = np.int32


class PoUpdateKernelTest(CupyCudaTest):

    def prepare_arrays(self, scan_points=None):
        B = 5  # frame size y
        C = 5  # frame size x

        D = 2  # number of probe modes
        E = B  # probe size y
        F = C  # probe size x

        npts_greater_than = 2  # how many points bigger than the probe the object is.
        G = 2  # number of object modes
        H = B + npts_greater_than  # object size y
        I = C + npts_greater_than  # object size x

        if scan_points is None:
            scan_pts = 2  # one dimensional scan point number
        else:
            scan_pts = scan_points

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
            object_array_denominator[idx] = np.ones((H, I)) * (5 * idx + 2)

        probe_denominator = np.empty_like(probe, dtype=FLOAT_TYPE)
        for idx in range(D):
            probe_denominator[idx] = np.ones((E, F)) * (5 * idx + 2)

        return (cp.asarray(addr),
            cp.asarray(object_array),
            cp.asarray(object_array_denominator),
            cp.asarray(probe),
            cp.asarray(exit_wave),
            cp.asarray(probe_denominator))


    def test_init(self):
        POUK = PoUpdateKernel()
        np.testing.assert_equal(POUK.kernels, ['pr_update', 'ob_update'],
                                err_msg='PoUpdateKernel does not have the correct functions registered.')

    def ob_update_REGRESSION_tester(self, atomics=True):

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
        object_array_denominator = np.empty_like(object_array, dtype=FLOAT_TYPE)
        for idx in range(G):
            object_array_denominator[idx] = np.ones((H, I)) * (5 * idx + 2)


        POUK = PoUpdateKernel()
        from ptypy.accelerate.base.kernels import PoUpdateKernel as npPoUpdateKernel
        nPOUK = npPoUpdateKernel()
        # print("object array denom before:")
        # print(object_array_denominator)
        object_array_dev = cp.asarray(object_array)
        object_array_denominator_dev = cp.asarray(object_array_denominator)
        probe_dev = cp.asarray(probe)
        exit_wave_dev = cp.asarray(exit_wave)
        if not atomics:
            addr2 = np.ascontiguousarray(np.transpose(addr, (2, 3, 0, 1)))
            addr_dev = cp.asarray(addr2)
        else:
            addr_dev = cp.asarray(addr)

        print(object_array_denominator)
        POUK.ob_update(addr_dev, object_array_dev, object_array_denominator_dev, probe_dev, exit_wave_dev, atomics=atomics)
        print("\n\n cuda  version")
        print(object_array_denominator_dev.get())
        nPOUK.ob_update(addr, object_array, object_array_denominator, probe, exit_wave)
        print("\n\n numpy version")
        print(object_array_denominator)



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

        expected_object_array_denominator = np.array([[[12., 22., 22., 22., 22., 12.,  2.],
                                                       [22., 42., 42., 42., 42., 22.,  2.],
                                                       [22., 42., 42., 42., 42., 22.,  2.],
                                                       [22., 42., 42., 42., 42., 22.,  2.],
                                                       [22., 42., 42., 42., 42., 22.,  2.],
                                                       [12., 22., 22., 22., 22., 12.,  2.],
                                                       [ 2.,  2.,  2.,  2.,  2.,  2.,  2.]],

                                                      [[17., 27., 27., 27., 27., 17.,  7.],
                                                       [27., 47., 47., 47., 47., 27.,  7.],
                                                       [27., 47., 47., 47., 47., 27.,  7.],
                                                       [27., 47., 47., 47., 47., 27.,  7.],
                                                       [27., 47., 47., 47., 47., 27.,  7.],
                                                       [17., 27., 27., 27., 27., 17.,  7.],
                                                       [ 7.,  7.,  7.,  7.,  7.,  7.,  7.]]],
                                                     dtype=FLOAT_TYPE)


        np.testing.assert_array_equal(object_array_denominator_dev.get(), expected_object_array_denominator,
                                      err_msg="The object array denominatorhas not been updated as expected")


    def test_ob_update_atomics_REGRESSION(self):
        self.ob_update_REGRESSION_tester(atomics=True)

    def test_ob_update_tiled_REGRESSION(self):
        self.ob_update_REGRESSION_tester(atomics=False)

    def ob_update_UNITY_tester(self, atomics=True):
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
        object_array_denominator = np.empty_like(object_array, dtype=FLOAT_TYPE)
        for idx in range(G):
            object_array_denominator[idx] = np.ones((H, I)) * (5 * idx + 2)


        POUK = PoUpdateKernel()

        from ptypy.accelerate.base.kernels import PoUpdateKernel as npPoUpdateKernel
        nPOUK = npPoUpdateKernel()

        object_array_dev = cp.asarray(object_array)
        object_array_denominator_dev = cp.asarray(object_array_denominator)
        probe_dev = cp.asarray(probe)
        exit_wave_dev = cp.asarray(exit_wave)
        if not atomics:
            addr2 = np.ascontiguousarray(np.transpose(addr, (2, 3, 0, 1)))
            addr_dev = cp.asarray(addr2)
        else:
            addr_dev = cp.asarray(addr)

        # print(object_array_denominator)
        POUK.ob_update(addr_dev, object_array_dev, object_array_denominator_dev, probe_dev, exit_wave_dev, atomics=atomics)
        # print("\n\n cuda  version")
        # print(repr(object_array_dev.get()))
        # print(repr(object_array_denominator_dev.get()))
        nPOUK.ob_update(addr, object_array, object_array_denominator, probe, exit_wave)
        # print("\n\n numpy version")
        # print(repr(object_array_denominator))
        # print(repr(object_array))


        np.testing.assert_array_equal(object_array, object_array_dev.get(),
                                      err_msg="The object array has not been updated as expected")


        np.testing.assert_array_equal(object_array_denominator, object_array_denominator_dev.get(),
                                      err_msg="The object array denominatorhas not been updated as expected")


    def test_ob_update_atomics_UNITY(self):
        self.ob_update_UNITY_tester(atomics=True)

    def test_ob_update_tiled_UNITY(self):
        self.ob_update_UNITY_tester(atomics=False)

    def pr_update_REGRESSION_tester(self, atomics=True):
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
        probe_denominator = np.empty_like(probe, dtype=FLOAT_TYPE)
        for idx in range(D):
            probe_denominator[idx] = np.ones((E, F)) * (5 * idx + 2)

        POUK = PoUpdateKernel()

        # print("probe array before:")
        # print(repr(probe))
        # print("probe denominator array before:")
        # print(repr(probe_denominator))

        object_array_dev = cp.asarray(object_array)
        probe_denominator_dev = cp.asarray(probe_denominator)
        probe_dev = cp.asarray(probe)
        exit_wave_dev = cp.asarray(exit_wave)
        if not atomics:
            addr2 = np.ascontiguousarray(np.transpose(addr, (2, 3, 0, 1)))
            addr_dev = cp.asarray(addr2)
        else:
            addr_dev = cp.asarray(addr)


        POUK.pr_update(addr_dev, probe_dev, probe_denominator_dev, object_array_dev, exit_wave_dev, atomics=atomics)

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

        np.testing.assert_array_equal(probe_dev.get(), expected_probe,
                                      err_msg="The probe has not been updated as expected")

        expected_probe_denominator = np.array([[[138., 138., 138., 138., 138.],
                                                [138., 138., 138., 138., 138.],
                                                [138., 138., 138., 138., 138.],
                                                [138., 138., 138., 138., 138.],
                                                [138., 138., 138., 138., 138.]],

                                               [[143., 143., 143., 143., 143.],
                                                [143., 143., 143., 143., 143.],
                                                [143., 143., 143., 143., 143.],
                                                [143., 143., 143., 143., 143.],
                                                [143., 143., 143., 143., 143.]]],
                                              dtype=FLOAT_TYPE)

        np.testing.assert_array_equal(probe_denominator_dev.get(), expected_probe_denominator,
                                      err_msg="The probe denominatorhas not been updated as expected")


    def test_pr_update_atomics_REGRESSION(self):
        self.pr_update_REGRESSION_tester(atomics=True)

    def test_pr_update_tiled_REGRESSION(self):
        self.pr_update_REGRESSION_tester(atomics=False)

    def pr_update_UNITY_tester(self, atomics=True):
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
        probe_denominator = np.empty_like(probe, dtype=FLOAT_TYPE)
        for idx in range(D):
            probe_denominator[idx] = np.ones((E, F)) * (5 * idx + 2)

        POUK = PoUpdateKernel()
        from ptypy.accelerate.base.kernels import PoUpdateKernel as npPoUpdateKernel
        nPOUK = npPoUpdateKernel()

        # print("probe array before:")
        # print(repr(probe))
        # print("probe denominator array before:")
        # print(repr(probe_denominator))

        object_array_dev = cp.asarray(object_array)
        probe_denominator_dev = cp.asarray(probe_denominator)
        probe_dev = cp.asarray(probe)
        exit_wave_dev = cp.asarray(exit_wave)
        if not atomics:
            addr2 = np.ascontiguousarray(np.transpose(addr, (2, 3, 0, 1)))
            addr_dev = cp.asarray(addr2)
        else:
            addr_dev = cp.asarray(addr)


        POUK.pr_update(addr_dev, probe_dev, probe_denominator_dev, object_array_dev, exit_wave_dev, atomics=atomics)
        nPOUK.pr_update(addr, probe, probe_denominator, object_array, exit_wave)

        # print("probe array after:")
        # print(repr(probe))
        # print("probe denominator array after:")
        # print(repr(probe_denominator))

        np.testing.assert_array_equal(probe, probe_dev.get(),
                                      err_msg="The probe has not been updated as expected")

        np.testing.assert_array_equal(probe_denominator, probe_denominator_dev.get(),
                                      err_msg="The probe denominatorhas not been updated as expected")


    def test_pr_update_atomics_UNITY(self):
        self.pr_update_UNITY_tester(atomics=True)

    def test_pr_update_tiled_UNITY(self):
        self.pr_update_UNITY_tester(atomics=False)


    def pr_update_ML_tester(self, atomics=False):
        '''
        setup
        '''
        addr, object_array, object_array_denominator, probe, exit_wave, probe_denominator = self.prepare_arrays()
        '''
        test
        '''
        POUK = PoUpdateKernel()

        POUK.allocate()  # this doesn't do anything, but is the call pattern.

        if not atomics:
            addr2 = np.ascontiguousarray(np.transpose(addr.get(), (2, 3, 0, 1)))
            addr = cp.asarray(addr2)

        POUK.pr_update_ML(addr, probe, object_array, exit_wave, atomics=atomics)

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

        np.testing.assert_array_equal(probe.get(), expected_probe,
                                      err_msg="The probe has not been updated as expected")

    def test_pr_update_ML_atomics_REGRESSION(self):
        self.pr_update_ML_tester(True)

    def test_pr_update_ML_tiled_REGRESSION(self):
        self.pr_update_ML_tester(False)

    def ob_update_ML_tester(self, atomics=True):
        '''
        setup
        '''
        addr, object_array, object_array_denominator, probe, exit_wave, probe_denominator = self.prepare_arrays()
        '''
        test
        '''
        POUK = PoUpdateKernel()

        POUK.allocate()  # this doesn't do anything, but is the call pattern.

        if not atomics:
            addr2 = np.ascontiguousarray(np.transpose(addr.get(), (2, 3, 0, 1)))
            addr = cp.asarray(addr2)

        POUK.ob_update_ML(addr, object_array, probe, exit_wave, atomics=atomics)

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

        np.testing.assert_array_equal(object_array.get(), expected_object_array,
                                      err_msg="The object array has not been updated as expected")

    def test_ob_update_ML_atomics_REGRESSION(self):
        self.ob_update_ML_tester(True)

    def test_ob_update_ML_tiled_REGRESSION(self):
        self.ob_update_ML_tester(False)

    def test_ob_update_local_UNITY(self):
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

        scan_pts = 1  # one dimensional scan point number

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
        auxiliary_wave = exit_wave.copy() * 2

        probe_norm = np.empty(shape=(1,B,C), dtype=FLOAT_TYPE)

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
        from ptypy.accelerate.base.kernels import PoUpdateKernel as npPoUpdateKernel
        nPOUK = npPoUpdateKernel()
        POUK = PoUpdateKernel(queue_thread=self.stream)

        object_array_dev = cp.asarray(object_array)
        probe_dev = cp.asarray(probe)
        exit_wave_dev = cp.asarray(exit_wave)
        auxiliary_wave_dev = cp.asarray(auxiliary_wave)
        probe_norm_dev = cp.asarray(probe_norm)
        addr_dev = cp.asarray(addr)

        POUK.pr_norm_local(addr_dev, probe_dev, probe_norm_dev)
        POUK.ob_update_local(addr_dev, object_array_dev, probe_dev, exit_wave_dev, auxiliary_wave_dev, probe_norm_dev, a=0.5, b=0.5)
        nPOUK.pr_norm_local(addr, probe, probe_norm)
        nPOUK.ob_update_local(addr, object_array, probe, exit_wave, auxiliary_wave, probe_norm, a=0.5, b=0.5)

        np.testing.assert_allclose(object_array_dev.get(), object_array, rtol=1e-6, atol=1e-6,
                                      err_msg="The object array has not been updated as expected")

    def test_pr_update_local_UNITY(self):
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

        scan_pts = 1  # one dimensional scan point number

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
        auxiliary_wave = exit_wave.copy() * 1.5

        object_norm = np.empty(shape=(1,B,C), dtype=FLOAT_TYPE)

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
        from ptypy.accelerate.base.kernels import PoUpdateKernel as npPoUpdateKernel
        nPOUK = npPoUpdateKernel()
        POUK = PoUpdateKernel()

        object_array_dev = cp.asarray(object_array)
        probe_dev = cp.asarray(probe)
        exit_wave_dev = cp.asarray(exit_wave)
        auxiliary_wave_dev = cp.asarray(auxiliary_wave)
        object_norm_dev = cp.asarray(object_norm)
        addr_dev = cp.asarray(addr)

        POUK.ob_norm_local(addr_dev, object_array_dev, object_norm_dev)
        POUK.pr_update_local(addr_dev,  probe_dev, object_array_dev,exit_wave_dev, auxiliary_wave_dev, object_norm_dev, cp.max(object_norm_dev), a=0.5, b=0.5)
        nPOUK.ob_norm_local(addr, object_array, object_norm)
        nPOUK.pr_update_local(addr, probe, object_array, exit_wave, auxiliary_wave, object_norm, object_norm.max(), a=0.5, b=0.5)

        np.testing.assert_allclose(probe_dev.get(), probe, rtol=1e-6, atol=1e-6,
                                      err_msg="The probe has not been updated as expected")

    def test_ob_norm_local_UNITY(self):
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

        scan_pts = 1  # one dimensional scan point number

        total_number_scan_positions = scan_pts ** 2
        total_number_modes = G * D
        A = total_number_scan_positions * total_number_modes # this is a 16 point scan pattern (4x4 grid) over all the modes

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
        from ptypy.accelerate.base.kernels import PoUpdateKernel as npPoUpdateKernel
        nPOUK = npPoUpdateKernel()
        POUK = PoUpdateKernel(queue_thread=self.stream)

        object_array_dev = cp.asarray(object_array)
        object_norm_dev = cp.asarray(object_norm)
        addr_dev = cp.asarray(addr)

        POUK.ob_norm_local(addr_dev, object_array_dev, object_norm_dev)
        nPOUK.ob_norm_local(addr, object_array, object_norm)

        np.testing.assert_allclose(object_norm_dev.get(), object_norm, rtol=1e-6, atol=1e-6,
                                      err_msg="The object norm has not been updated as expected")

    def test_pr_norm_local_UNITY(self):
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

        scan_pts = 1  # one dimensional scan point number

        total_number_scan_positions = scan_pts ** 2
        total_number_modes = G * D
        A = total_number_scan_positions * total_number_modes # this is a 16 point scan pattern (4x4 grid) over all the modes

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
        from ptypy.accelerate.base.kernels import PoUpdateKernel as npPoUpdateKernel
        nPOUK = npPoUpdateKernel()
        POUK = PoUpdateKernel()

        probe_dev = cp.asarray(probe)
        probe_norm_dev = cp.asarray(probe_norm)
        addr_dev = cp.asarray(addr)

        POUK.pr_norm_local(addr_dev,  probe_dev, probe_norm_dev)
        nPOUK.pr_norm_local(addr, probe, probe_norm)

        np.testing.assert_allclose(probe_norm_dev.get(), probe_norm, rtol=1e-6, atol=1e-6,
                                      err_msg="The probe norm has not been updated as expected")

    def test_ob_update_wasp_UNITY(self):
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

        scan_pts = 1  # one dimensional scan point number

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
        auxiliary_wave = exit_wave.copy() * 1.5

        ob_sum_nmr = np.zeros(shape=(G,H,I), dtype=COMPLEX_TYPE)
        ob_sum_dnm = np.zeros(shape=(G,H,I), dtype=FLOAT_TYPE)

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
        from ptypy.accelerate.base.kernels import PoUpdateKernel as npPoUpdateKernel
        nPOUK = npPoUpdateKernel()
        POUK = PoUpdateKernel()

        object_array_dev = cp.asarray(object_array)
        probe_dev = cp.asarray(probe)
        exit_wave_dev = cp.asarray(exit_wave)
        auxiliary_wave_dev = cp.asarray(auxiliary_wave)
        ob_sum_nmr_dev = cp.asarray(ob_sum_nmr)
        ob_sum_dnm_dev = cp.asarray(ob_sum_dnm)
        addr_dev = cp.asarray(addr)

        POUK.ob_update_wasp(addr_dev, object_array_dev, probe_dev,
                            exit_wave_dev, auxiliary_wave_dev, ob_sum_nmr_dev,
                            ob_sum_dnm_dev)

        nPOUK.ob_update_wasp(addr, object_array, probe, exit_wave,
                             auxiliary_wave, ob_sum_nmr, ob_sum_dnm)

        np.testing.assert_allclose(object_array_dev.get(), object_array, rtol=1e-6, atol=1e-6,
                                   err_msg="The object has not been updated as expected")
        np.testing.assert_allclose(ob_sum_nmr_dev.get(), ob_sum_nmr, rtol=1e-6, atol=1e-6,
                                   err_msg="The object sum numerator has not been updated as expected")
        np.testing.assert_allclose(ob_sum_dnm_dev.get(), ob_sum_dnm, rtol=1e-6, atol=1e-6,
                                   err_msg="The object sum denominator has not been updated as expected")

    def test_pr_update_wasp_UNITY(self):
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

        scan_pts = 1  # one dimensional scan point number

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
        auxiliary_wave = exit_wave.copy() * 1.5

        pr_sum_nmr = np.zeros(shape=(D,E,F), dtype=COMPLEX_TYPE)
        pr_sum_dnm = np.zeros(shape=(D,E,F), dtype=FLOAT_TYPE)

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
        from ptypy.accelerate.base.kernels import PoUpdateKernel as npPoUpdateKernel
        nPOUK = npPoUpdateKernel()
        POUK = PoUpdateKernel()

        object_array_dev = cp.asarray(object_array)
        probe_dev = cp.asarray(probe)
        exit_wave_dev = cp.asarray(exit_wave)
        auxiliary_wave_dev = cp.asarray(auxiliary_wave)
        pr_sum_nmr_dev = cp.asarray(pr_sum_nmr)
        pr_sum_dnm_dev = cp.asarray(pr_sum_dnm)
        addr_dev = cp.asarray(addr)

        POUK.pr_update_wasp(addr_dev, probe_dev, object_array_dev,
                            exit_wave_dev, auxiliary_wave_dev, pr_sum_nmr_dev,
                            pr_sum_dnm_dev)

        nPOUK.pr_update_wasp(addr, probe, object_array, exit_wave,
                             auxiliary_wave, pr_sum_nmr, pr_sum_dnm)

        np.testing.assert_allclose(probe_dev.get(), probe, rtol=1e-6, atol=1e-6,
                                   err_msg="The probe has not been updated as expected")
        np.testing.assert_allclose(pr_sum_nmr_dev.get(), pr_sum_nmr, rtol=1e-6, atol=1e-6,
                                   err_msg="The probe sum numerator has not been updated as expected")
        np.testing.assert_allclose(pr_sum_dnm_dev.get(), pr_sum_dnm, rtol=1e-6, atol=1e-6,
                                   err_msg="The probe sum denominator has not been updated as expected")

    def test_avg_wasp_UNITY(self):
        '''
        setup
        '''
        B = 5  # frame size y
        C = 5  # frame size x

        npts_greater_than = 2  # how many points bigger than the probe the object is.
        G = 2  # number of object modes
        H = B + npts_greater_than  #  object size y
        I = C + npts_greater_than  #  object size x

        object_array = np.empty(shape=(G, H, I), dtype=COMPLEX_TYPE)
        for idx in range(G):
            object_array[idx] = np.ones((H, I)) * (3 * idx + 1) + 1j * np.ones((H, I)) * (3 * idx + 1)

        ob_sum_nmr = np.empty(shape=(G, H, I), dtype=COMPLEX_TYPE)
        for idx in range(G):
            ob_sum_nmr[idx] = np.arange(idx, idx+H*I).reshape(H,I) + 1j * np.arange(idx, idx+H*I).reshape(H,I)

        ob_sum_dnm = ob_sum_nmr[:, ::-1, ::-1].copy().real

        '''
        test
        '''
        from ptypy.accelerate.base.kernels import PoUpdateKernel as npPoUpdateKernel
        nPOUK = npPoUpdateKernel()
        POUK = PoUpdateKernel()

        object_array_dev = cp.asarray(object_array)
        ob_sum_nmr_dev = cp.asarray(ob_sum_nmr)
        ob_sum_dnm_dev = cp.asarray(ob_sum_dnm)

        POUK.avg_wasp(object_array_dev, ob_sum_nmr_dev, ob_sum_dnm_dev)
        nPOUK.avg_wasp(object_array, ob_sum_nmr, ob_sum_dnm)

        np.testing.assert_allclose(object_array_dev.get(), object_array, rtol=1e-6, atol=1e-6,
                                   err_msg="The object_array has not been updated as expected")
        np.testing.assert_allclose(ob_sum_nmr_dev.get(), ob_sum_nmr, rtol=1e-6, atol=1e-6,
                                   err_msg="The object_array sum numerator should be unchanged")
        np.testing.assert_allclose(ob_sum_dnm_dev.get(), ob_sum_dnm, rtol=1e-6, atol=1e-6,
                                   err_msg="The object_array sum denominator should be unchanged")

if __name__ == '__main__':
    unittest.main()
