'''


'''

import unittest
import numpy as np
from . import perfrun

def have_pycuda():
    try:
        import pycuda.driver
        return True
    except:
        return False

if have_pycuda():
    import pycuda.driver as cuda
    from pycuda import gpuarray
    from pycuda.tools import make_default_context
    from ptypy.accelerate.py_cuda.kernels import AuxiliaryWaveKernel

COMPLEX_TYPE = np.complex64
FLOAT_TYPE = np.float32
INT_TYPE = np.int32


@unittest.skipIf(not have_pycuda(), "no PyCUDA or GPU drivers available")
class AuxiliaryWaveKernelTest(unittest.TestCase):

    def setUp(self):
        import sys
        np.set_printoptions(threshold=sys.maxsize, linewidth=np.inf)
        cuda.init()
        self.ctx = make_default_context()
        self.stream = cuda.Stream()

    def tearDown(self):
        np.set_printoptions()
        self.ctx.pop()
        self.ctx.detach()

    def test_init(self):
        attrs = ["_ob_shape",
                 "_ob_id"]

        AWK = AuxiliaryWaveKernel(self.stream)
        for attr in attrs:
            self.assertTrue(hasattr(AWK, attr), msg="AuxiliaryWaveKernel does not have attribute: %s" % attr)

        np.testing.assert_equal(AWK.kernels,
                                ['build_aux', 'build_exit'],
                                err_msg='AuxiliaryWaveKernel does not have the correct functions registered.')

    def test_build_aux_same_as_exit_REGRESSION(self):
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

        AWK = AuxiliaryWaveKernel(self.stream)
        alpha_set = FLOAT_TYPE(1.0)

        object_array_dev = gpuarray.to_gpu(object_array)
        probe_dev = gpuarray.to_gpu(probe)
        addr_dev = gpuarray.to_gpu(addr)
        auxiliary_wave_dev = gpuarray.to_gpu(auxiliary_wave)
        exit_wave_dev = gpuarray.to_gpu(exit_wave)

        AWK.build_aux(auxiliary_wave_dev, addr_dev, object_array_dev, probe_dev, exit_wave_dev, alpha=alpha_set)


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

        np.testing.assert_array_equal(expected_auxiliary_wave, auxiliary_wave_dev.get(),
                                      err_msg="The auxiliary_wave has not been updated as expected")

        object_array_dev.gpudata.free()
        auxiliary_wave_dev.gpudata.free()
        probe_dev.gpudata.free()
        exit_wave_dev.gpudata.free()
        addr_dev.gpudata.free()

    def test_build_aux_same_as_exit_UNITY(self):
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
        from ptypy.accelerate.array_based.kernels import AuxiliaryWaveKernel as npAuxiliaryWaveKernel
        nAWK = npAuxiliaryWaveKernel()
        AWK = AuxiliaryWaveKernel(self.stream)
        alpha_set = FLOAT_TYPE(1.0)

        object_array_dev = gpuarray.to_gpu(object_array)
        probe_dev = gpuarray.to_gpu(probe)
        addr_dev = gpuarray.to_gpu(addr)
        auxiliary_wave_dev = gpuarray.to_gpu(auxiliary_wave)
        exit_wave_dev = gpuarray.to_gpu(exit_wave)

        AWK.build_aux(auxiliary_wave_dev, addr_dev, object_array_dev, probe_dev, exit_wave_dev, alpha=alpha_set)
        nAWK.build_aux(auxiliary_wave, addr, object_array, probe, exit_wave, alpha=alpha_set)


        np.testing.assert_array_equal(auxiliary_wave, auxiliary_wave_dev.get(),
                                      err_msg="The gpu auxiliary_wave does not look the same as the numpy version")

        object_array_dev.gpudata.free()
        auxiliary_wave_dev.gpudata.free()
        probe_dev.gpudata.free()
        exit_wave_dev.gpudata.free()
        addr_dev.gpudata.free()

    def test_build_exit_aux_same_as_exit_REGRESSION(self):
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

        object_array_dev = gpuarray.to_gpu(object_array)
        probe_dev = gpuarray.to_gpu(probe)
        addr_dev = gpuarray.to_gpu(addr)
        auxiliary_wave_dev = gpuarray.to_gpu(auxiliary_wave)
        exit_wave_dev = gpuarray.to_gpu(exit_wave)
        AWK = AuxiliaryWaveKernel(self.stream)

        alpha_set = 1.0

        AWK.build_exit(auxiliary_wave_dev, addr_dev, object_array_dev, probe_dev, exit_wave_dev)
        #
        # print("auxiliary_wave after")
        # print(repr(auxiliary_wave_dev.get()))
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

        np.testing.assert_array_equal(expected_auxiliary_wave, auxiliary_wave_dev.get(),
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

        np.testing.assert_array_equal(expected_exit_wave, exit_wave_dev.get(),
                                      err_msg="The exit_wave has not been updated as expected")

        object_array_dev.gpudata.free()
        auxiliary_wave_dev.gpudata.free()
        probe_dev.gpudata.free()
        exit_wave_dev.gpudata.free()
        addr_dev.gpudata.free()

    def test_build_exit_aux_same_as_exit_UNITY(self):
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

        object_array_dev = gpuarray.to_gpu(object_array)
        probe_dev = gpuarray.to_gpu(probe)
        addr_dev = gpuarray.to_gpu(addr)
        auxiliary_wave_dev = gpuarray.to_gpu(auxiliary_wave)
        exit_wave_dev = gpuarray.to_gpu(exit_wave)

        from ptypy.accelerate.array_based.kernels import AuxiliaryWaveKernel as npAuxiliaryWaveKernel
        nAWK = npAuxiliaryWaveKernel()

        AWK = AuxiliaryWaveKernel(self.stream)

        AWK.build_exit(auxiliary_wave_dev, addr_dev, object_array_dev, probe_dev, exit_wave_dev)
        nAWK.build_exit(auxiliary_wave, addr, object_array, probe, exit_wave)

        np.testing.assert_array_equal(auxiliary_wave, auxiliary_wave_dev.get(),
                                      err_msg="The gpu auxiliary_wave does not look the same as the numpy version")

        np.testing.assert_array_equal(exit_wave, exit_wave_dev.get(),
                                      err_msg="The gpu exit_wave does not look the same as the numpy version")

        object_array_dev.gpudata.free()
        auxiliary_wave_dev.gpudata.free()
        probe_dev.gpudata.free()
        exit_wave_dev.gpudata.free()
        addr_dev.gpudata.free()

    def prepare_arrays(self, performance=False):
        if not performance:
            B = 3  # frame size y
            C = 3  # frame size x
            D = 2  # number of probe modes
            E = B  # probe size y
            F = C  # probe size x

            npts_greater_than = 2  # how many points bigger than the probe the object is.
            G = 2  # number of object modes

            scan_pts = 2  # one dimensional scan point number
        else:
            B = 256
            C = 256
            D = 5
            E = B
            F = C
            npts_greater_than = 500
            G = 4
            scan_pts = 10

        H = B + npts_greater_than  # object size y
        I = C + npts_greater_than  # object size x

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

        return (gpuarray.to_gpu(addr), 
            gpuarray.to_gpu(object_array), 
            gpuarray.to_gpu(probe), 
            gpuarray.to_gpu(exit_wave))


    def test_build_aux_no_ex_REGRESSION(self):
        '''
        setup
        '''
        addr, object_array, probe, exit_wave = self.prepare_arrays()

        '''
        test
        '''
        auxiliary_wave = gpuarray.zeros_like(exit_wave)

        AWK = AuxiliaryWaveKernel(self.stream)
        AWK.allocate()
        AWK.build_aux_no_ex(auxiliary_wave, addr, object_array, probe, 
            fac=1.0, add=False)
        expected_auxiliary_wave = np.array([[[0. + 2.j, 0. + 2.j, 0. + 2.j],
                                             [0. + 2.j, 0. + 2.j, 0. + 2.j],
                                             [0. + 2.j, 0. + 2.j, 0. + 2.j]],
                                            [[0. + 8.j, 0. + 8.j, 0. + 8.j],
                                             [0. + 8.j, 0. + 8.j, 0. + 8.j],
                                             [0. + 8.j, 0. + 8.j, 0. + 8.j]],
                                            [[0. + 4.j, 0. + 4.j, 0. + 4.j],
                                             [0. + 4.j, 0. + 4.j, 0. + 4.j],
                                             [0. + 4.j, 0. + 4.j, 0. + 4.j]],
                                            [[0. + 16.j, 0. + 16.j, 0. + 16.j],
                                             [0. + 16.j, 0. + 16.j, 0. + 16.j],
                                             [0. + 16.j, 0. + 16.j, 0. + 16.j]],
                                            [[0. + 2.j, 0. + 2.j, 0. + 2.j],
                                             [0. + 2.j, 0. + 2.j, 0. + 2.j],
                                             [0. + 2.j, 0. + 2.j, 0. + 2.j]],
                                            [[0. + 8.j, 0. + 8.j, 0. + 8.j],
                                             [0. + 8.j, 0. + 8.j, 0. + 8.j],
                                             [0. + 8.j, 0. + 8.j, 0. + 8.j]],
                                            [[0. + 4.j, 0. + 4.j, 0. + 4.j],
                                             [0. + 4.j, 0. + 4.j, 0. + 4.j],
                                             [0. + 4.j, 0. + 4.j, 0. + 4.j]],
                                            [[0. + 16.j, 0. + 16.j, 0. + 16.j],
                                             [0. + 16.j, 0. + 16.j, 0. + 16.j],
                                             [0. + 16.j, 0. + 16.j, 0. + 16.j]],
                                            [[0. + 2.j, 0. + 2.j, 0. + 2.j],
                                             [0. + 2.j, 0. + 2.j, 0. + 2.j],
                                             [0. + 2.j, 0. + 2.j, 0. + 2.j]],
                                            [[0. + 8.j, 0. + 8.j, 0. + 8.j],
                                             [0. + 8.j, 0. + 8.j, 0. + 8.j],
                                             [0. + 8.j, 0. + 8.j, 0. + 8.j]],
                                            [[0. + 4.j, 0. + 4.j, 0. + 4.j],
                                             [0. + 4.j, 0. + 4.j, 0. + 4.j],
                                             [0. + 4.j, 0. + 4.j, 0. + 4.j]],
                                            [[0. + 16.j, 0. + 16.j, 0. + 16.j],
                                             [0. + 16.j, 0. + 16.j, 0. + 16.j],
                                             [0. + 16.j, 0. + 16.j, 0. + 16.j]],
                                            [[0. + 2.j, 0. + 2.j, 0. + 2.j],
                                             [0. + 2.j, 0. + 2.j, 0. + 2.j],
                                             [0. + 2.j, 0. + 2.j, 0. + 2.j]],
                                            [[0. + 8.j, 0. + 8.j, 0. + 8.j],
                                             [0. + 8.j, 0. + 8.j, 0. + 8.j],
                                             [0. + 8.j, 0. + 8.j, 0. + 8.j]],
                                            [[0. + 4.j, 0. + 4.j, 0. + 4.j],
                                             [0. + 4.j, 0. + 4.j, 0. + 4.j],
                                             [0. + 4.j, 0. + 4.j, 0. + 4.j]],
                                            [[0. + 16.j, 0. + 16.j, 0. + 16.j],
                                             [0. + 16.j, 0. + 16.j, 0. + 16.j],
                                             [0. + 16.j, 0. + 16.j, 0. + 16.j]]], dtype=np.complex64)
        np.testing.assert_array_equal(auxiliary_wave.get(), expected_auxiliary_wave,
                                      err_msg="The auxiliary_wave has not been updated as expected")
        
        auxiliary_wave = exit_wave
        AWK.build_aux_no_ex(auxiliary_wave, addr, object_array, probe, fac=2.0, add=True)

        expected_auxiliary_wave = np.array([[[1. + 5.j, 1. + 5.j, 1. + 5.j],
                                             [1. + 5.j, 1. + 5.j, 1. + 5.j],
                                             [1. + 5.j, 1. + 5.j, 1. + 5.j]],
                                            [[2. + 18.j, 2. + 18.j, 2. + 18.j],
                                             [2. + 18.j, 2. + 18.j, 2. + 18.j],
                                             [2. + 18.j, 2. + 18.j, 2. + 18.j]],
                                            [[3. + 11.j, 3. + 11.j, 3. + 11.j],
                                             [3. + 11.j, 3. + 11.j, 3. + 11.j],
                                             [3. + 11.j, 3. + 11.j, 3. + 11.j]],
                                            [[4. + 36.j, 4. + 36.j, 4. + 36.j],
                                             [4. + 36.j, 4. + 36.j, 4. + 36.j],
                                             [4. + 36.j, 4. + 36.j, 4. + 36.j]],
                                            [[5. + 9.j, 5. + 9.j, 5. + 9.j],
                                             [5. + 9.j, 5. + 9.j, 5. + 9.j],
                                             [5. + 9.j, 5. + 9.j, 5. + 9.j]],
                                            [[6. + 22.j, 6. + 22.j, 6. + 22.j],
                                             [6. + 22.j, 6. + 22.j, 6. + 22.j],
                                             [6. + 22.j, 6. + 22.j, 6. + 22.j]],
                                            [[7. + 15.j, 7. + 15.j, 7. + 15.j],
                                             [7. + 15.j, 7. + 15.j, 7. + 15.j],
                                             [7. + 15.j, 7. + 15.j, 7. + 15.j]],
                                            [[8. + 40.j, 8. + 40.j, 8. + 40.j],
                                             [8. + 40.j, 8. + 40.j, 8. + 40.j],
                                             [8. + 40.j, 8. + 40.j, 8. + 40.j]],
                                            [[9. + 13.j, 9. + 13.j, 9. + 13.j],
                                             [9. + 13.j, 9. + 13.j, 9. + 13.j],
                                             [9. + 13.j, 9. + 13.j, 9. + 13.j]],
                                            [[10. + 26.j, 10. + 26.j, 10. + 26.j],
                                             [10. + 26.j, 10. + 26.j, 10. + 26.j],
                                             [10. + 26.j, 10. + 26.j, 10. + 26.j]],
                                            [[11. + 19.j, 11. + 19.j, 11. + 19.j],
                                             [11. + 19.j, 11. + 19.j, 11. + 19.j],
                                             [11. + 19.j, 11. + 19.j, 11. + 19.j]],
                                            [[12. + 44.j, 12. + 44.j, 12. + 44.j],
                                             [12. + 44.j, 12. + 44.j, 12. + 44.j],
                                             [12. + 44.j, 12. + 44.j, 12. + 44.j]],
                                            [[13. + 17.j, 13. + 17.j, 13. + 17.j],
                                             [13. + 17.j, 13. + 17.j, 13. + 17.j],
                                             [13. + 17.j, 13. + 17.j, 13. + 17.j]],
                                            [[14. + 30.j, 14. + 30.j, 14. + 30.j],
                                             [14. + 30.j, 14. + 30.j, 14. + 30.j],
                                             [14. + 30.j, 14. + 30.j, 14. + 30.j]],
                                            [[15. + 23.j, 15. + 23.j, 15. + 23.j],
                                             [15. + 23.j, 15. + 23.j, 15. + 23.j],
                                             [15. + 23.j, 15. + 23.j, 15. + 23.j]],
                                            [[16. + 48.j, 16. + 48.j, 16. + 48.j],
                                             [16. + 48.j, 16. + 48.j, 16. + 48.j],
                                             [16. + 48.j, 16. + 48.j, 16. + 48.j]]], dtype=np.complex64)
        np.testing.assert_array_equal(auxiliary_wave.get(), expected_auxiliary_wave,
                                      err_msg="The auxiliary_wave has not been updated as expected")

    @unittest.skipIf(not perfrun, "performance test")
    def test_build_aux_no_ex_performance(self):
        addr, object_array, probe, exit_wave = self.prepare_arrays(performance=True)
        auxiliary_wave = gpuarray.zeros_like(exit_wave)

        AWK = AuxiliaryWaveKernel(self.stream)
        AWK.allocate()
        AWK.build_aux_no_ex(auxiliary_wave, addr, object_array, probe, 
            fac=1.0, add=False)


if __name__ == '__main__':
    unittest.main()
