'''


'''

import unittest
import numpy as np
from . import PyCudaTest, have_pycuda

if have_pycuda():
    from pycuda import gpuarray
    from ptypy.accelerate.cuda_pycuda.kernels import PoUpdateKernel

COMPLEX_TYPE = np.complex64
FLOAT_TYPE = np.float32
INT_TYPE = np.int32


class PoUpdateKernelTest(PyCudaTest):

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

        return (gpuarray.to_gpu(addr), 
            gpuarray.to_gpu(object_array), 
            gpuarray.to_gpu(object_array_denominator), 
            gpuarray.to_gpu(probe), 
            gpuarray.to_gpu(exit_wave), 
            gpuarray.to_gpu(probe_denominator))


    def test_init(self):

        POUK = PoUpdateKernel()

        np.testing.assert_equal(POUK.kernels,
                                ['pr_update', 'ob_update'],
                                err_msg='PoUpdateKernel does not have the correct functions registered.')

    def ob_update_REGRESSION_tester(self, atomics=True):
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
        from ptypy.accelerate.base.kernels import PoUpdateKernel as npPoUpdateKernel
        nPOUK = npPoUpdateKernel()
        # print("object array denom before:")
        # print(object_array_denominator)
        object_array_dev = gpuarray.to_gpu(object_array)
        object_array_denominator_dev = gpuarray.to_gpu(object_array_denominator)
        probe_dev = gpuarray.to_gpu(probe)
        exit_wave_dev = gpuarray.to_gpu(exit_wave)
        if not atomics:
            addr2 = np.ascontiguousarray(np.transpose(addr, (2, 3, 0, 1)))
            addr_dev = gpuarray.to_gpu(addr2)
        else:
            addr_dev = gpuarray.to_gpu(addr)

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
        object_array_denominator = np.empty_like(object_array)
        for idx in range(G):
            object_array_denominator[idx] = np.ones((H, I)) * (5 * idx + 2) + 1j * np.ones((H, I)) * (5 * idx + 2)


        POUK = PoUpdateKernel()

        from ptypy.accelerate.base.kernels import PoUpdateKernel as npPoUpdateKernel
        nPOUK = npPoUpdateKernel()

        object_array_dev = gpuarray.to_gpu(object_array)
        object_array_denominator_dev = gpuarray.to_gpu(object_array_denominator)
        probe_dev = gpuarray.to_gpu(probe)
        exit_wave_dev = gpuarray.to_gpu(exit_wave)
        if not atomics:
            addr2 = np.ascontiguousarray(np.transpose(addr, (2, 3, 0, 1)))
            addr_dev = gpuarray.to_gpu(addr2)
        else:
            addr_dev = gpuarray.to_gpu(addr)

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
        probe_denominator = np.empty_like(probe)
        for idx in range(D):
            probe_denominator[idx] = np.ones((E, F)) * (5 * idx + 2) + 1j * np.ones((E, F)) * (5 * idx + 2)

        POUK = PoUpdateKernel()

        # print("probe array before:")
        # print(repr(probe))
        # print("probe denominator array before:")
        # print(repr(probe_denominator))

        object_array_dev = gpuarray.to_gpu(object_array)
        probe_denominator_dev = gpuarray.to_gpu(probe_denominator)
        probe_dev = gpuarray.to_gpu(probe)
        exit_wave_dev = gpuarray.to_gpu(exit_wave)
        if not atomics:
            addr2 = np.ascontiguousarray(np.transpose(addr, (2, 3, 0, 1)))
            addr_dev = gpuarray.to_gpu(addr2)
        else:
            addr_dev = gpuarray.to_gpu(addr)


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
        probe_denominator = np.empty_like(probe)
        for idx in range(D):
            probe_denominator[idx] = np.ones((E, F)) * (5 * idx + 2) + 1j * np.ones((E, F)) * (5 * idx + 2)

        POUK = PoUpdateKernel()
        from ptypy.accelerate.base.kernels import PoUpdateKernel as npPoUpdateKernel
        nPOUK = npPoUpdateKernel()

        # print("probe array before:")
        # print(repr(probe))
        # print("probe denominator array before:")
        # print(repr(probe_denominator))

        object_array_dev = gpuarray.to_gpu(object_array)
        probe_denominator_dev = gpuarray.to_gpu(probe_denominator)
        probe_dev = gpuarray.to_gpu(probe)
        exit_wave_dev = gpuarray.to_gpu(exit_wave)
        if not atomics:
            addr2 = np.ascontiguousarray(np.transpose(addr, (2, 3, 0, 1)))
            addr_dev = gpuarray.to_gpu(addr2)
        else:
            addr_dev = gpuarray.to_gpu(addr)


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
            addr = gpuarray.to_gpu(addr2)

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
            addr = gpuarray.to_gpu(addr2)

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


if __name__ == '__main__':
    unittest.main()
