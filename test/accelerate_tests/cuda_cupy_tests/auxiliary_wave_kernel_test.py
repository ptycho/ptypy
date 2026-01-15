'''


'''

import unittest
import numpy as np
from . import perfrun, CupyCudaTest, have_cupy

if have_cupy():
    import cupy as cp
    from ptypy.accelerate.cuda_cupy.kernels import AuxiliaryWaveKernel

COMPLEX_TYPE = np.complex64
FLOAT_TYPE = np.float32
INT_TYPE = np.int32

class AuxiliaryWaveKernelTest(CupyCudaTest):

    def prepare_arrays(self, performance=False, scan_points=None):
        if not performance:
            B = 3  # frame size y
            C = 3  # frame size x
            D = 2  # number of probe modes
            E = B  # probe size y
            F = C  # probe size x

            npts_greater_than = 2  # how many points bigger than the probe the object is.
            G = 2  # number of object modes
            if scan_points is None:
                scan_pts = 2  # one dimensional scan point number
            else:
                scan_pts = scan_points
        else:
            B = 128
            C = 128
            D = 2
            E = B
            F = C
            npts_greater_than = 1215
            G = 4
            if scan_points is None:
                scan_pts = 14
            else:
                scan_pts = scan_points

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
        if performance:
            print('addr={}, obj={}, pr={}, ex={}'.format(addr.shape, object_array.shape, probe.shape, exit_wave.shape))
            # assert False

        return addr, object_array, probe, exit_wave

    def copy_to_gpu(self, addr, object_array, probe, exit_wave):
        return (cp.asarray(addr), 
            cp.asarray(object_array), 
            cp.asarray(probe), 
            cp.asarray(exit_wave))

    def test_init(self):
        # should we really test for private attributes? 
        # Only the public interface should be checked - what clients rely on
        attrs = ["_ob_shape",
                 "_ob_id"]

        AWK = AuxiliaryWaveKernel(self.stream)
        for attr in attrs:
            self.assertTrue(hasattr(AWK, attr), msg="AuxiliaryWaveKernel does not have attribute: %s" % attr)

        np.testing.assert_equal(AWK.kernels,
                                ['build_aux', 'build_exit'],
                                err_msg='AuxiliaryWaveKernel does not have the correct functions registered.')

    def test_build_aux_same_as_exit_REGRESSION(self):
        ## Arrange
        cpudata = self.prepare_arrays()
        addr, object_array, probe, exit_wave = self.copy_to_gpu(*cpudata)
        auxiliary_wave = cp.zeros_like(exit_wave)

        ## Act
        AWK = AuxiliaryWaveKernel(self.stream)
        alpha_set = FLOAT_TYPE(1.0)

        AWK.build_aux(auxiliary_wave, addr, object_array, probe, exit_wave, alpha=alpha_set)

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

        np.testing.assert_array_equal(expected_auxiliary_wave, auxiliary_wave.get(),
                                      err_msg="The auxiliary_wave has not been updated as expected")


    def test_build_aux_same_as_exit_UNITY(self):
        ## Arrange
        addr, object_array, probe, exit_wave = self.prepare_arrays()
        addr_dev, object_array_dev, probe_dev, exit_wave_dev = self.copy_to_gpu(addr, object_array, probe, exit_wave)
        auxiliary_wave = np.zeros_like(exit_wave)
        auxiliary_wave_dev = cp.zeros_like(exit_wave_dev)
        
        ## Act
        from ptypy.accelerate.base.kernels import AuxiliaryWaveKernel as npAuxiliaryWaveKernel
        nAWK = npAuxiliaryWaveKernel()
        AWK = AuxiliaryWaveKernel(self.stream)
        alpha_set = FLOAT_TYPE(.75)

        AWK.build_aux(auxiliary_wave_dev, addr_dev, object_array_dev, probe_dev, exit_wave_dev, alpha=alpha_set)
        nAWK.build_aux(auxiliary_wave, addr, object_array, probe, exit_wave, alpha=alpha_set)
        
        ## Assert
        np.testing.assert_array_equal(auxiliary_wave, auxiliary_wave_dev.get(),
                                      err_msg="The gpu auxiliary_wave does not look the same as the numpy version")

    def test_build_aux2_same_as_exit_UNITY(self):
        ## Arrange
        addr, object_array, probe, exit_wave = self.prepare_arrays()
        addr_dev, object_array_dev, probe_dev, exit_wave_dev = self.copy_to_gpu(addr, object_array, probe, exit_wave)
        auxiliary_wave = np.zeros_like(exit_wave)
        auxiliary_wave_dev = cp.zeros_like(exit_wave_dev)
        
        ## Act
        from ptypy.accelerate.base.kernels import AuxiliaryWaveKernel as npAuxiliaryWaveKernel
        nAWK = npAuxiliaryWaveKernel()
        AWK = AuxiliaryWaveKernel(self.stream)
        alpha_set = FLOAT_TYPE(.75)

        AWK.build_aux2(auxiliary_wave_dev, addr_dev, object_array_dev, probe_dev, exit_wave_dev, alpha=alpha_set)
        nAWK.build_aux(auxiliary_wave, addr, object_array, probe, exit_wave, alpha=alpha_set)
        
        ## Assert
        np.testing.assert_array_equal(auxiliary_wave, auxiliary_wave_dev.get(),
                                      err_msg="The gpu auxiliary_wave does not look the same as the numpy version")

    def test_build_exit_aux_same_as_exit_REGRESSION(self):
        ## Arrange
        addr, object_array, probe, exit_wave = self.prepare_arrays()
        addr_dev, object_array_dev, probe_dev, exit_wave_dev = self.copy_to_gpu(addr, object_array, probe, exit_wave)
        auxiliary_wave_dev = cp.zeros_like(exit_wave_dev)
        
        ## Act
        AWK = AuxiliaryWaveKernel(self.stream)
        alpha_set = 1.0
        AWK.build_exit(auxiliary_wave_dev, addr_dev, object_array_dev, probe_dev, exit_wave_dev)

        ## Assert
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

    def test_build_exit_aux_same_as_exit_UNITY(self):
        ## Arrange
        addr, object_array, probe, exit_wave = self.prepare_arrays()
        addr_dev, object_array_dev, probe_dev, exit_wave_dev = self.copy_to_gpu(addr, object_array, probe, exit_wave)
        auxiliary_wave = np.zeros_like(exit_wave)
        auxiliary_wave_dev = cp.zeros_like(exit_wave_dev)

        ## Act
        from ptypy.accelerate.base.kernels import AuxiliaryWaveKernel as npAuxiliaryWaveKernel
        nAWK = npAuxiliaryWaveKernel()
        AWK = AuxiliaryWaveKernel(self.stream)

        AWK.build_exit(auxiliary_wave_dev, addr_dev, object_array_dev, probe_dev, exit_wave_dev)
        nAWK.build_exit(auxiliary_wave, addr, object_array, probe, exit_wave)

        ## Assert
        np.testing.assert_array_equal(auxiliary_wave, auxiliary_wave_dev.get(),
                                      err_msg="The gpu auxiliary_wave does not look the same as the numpy version")

        np.testing.assert_array_equal(exit_wave, exit_wave_dev.get(),
                                      err_msg="The gpu exit_wave does not look the same as the numpy version")

    def test_build_aux_no_ex_noadd_REGRESSION(self):
        ## Arrange
        addr, object_array, probe, exit_wave = self.prepare_arrays()
        addr, object_array, probe, exit_wave = self.copy_to_gpu(addr, object_array, probe, exit_wave)
        auxiliary_wave = cp.zeros_like(exit_wave)

        ## Act
        AWK = AuxiliaryWaveKernel(self.stream)
        AWK.allocate()
        AWK.build_aux_no_ex(auxiliary_wave, addr, object_array, probe, 
            fac=1.0, add=False)

        ## Assert
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

    def test_build_aux_no_ex_noadd_UNITY(self):
        ## Arrange
        addr, object_array, probe, exit_wave = self.prepare_arrays()
        addr_dev, object_array_dev, probe_dev, exit_wave_dev = self.copy_to_gpu(addr, object_array, probe, exit_wave)
        auxiliary_wave_dev = cp.zeros_like(exit_wave_dev)
        auxiliary_wave = np.zeros_like(exit_wave)

        ## Act
        AWK = AuxiliaryWaveKernel(self.stream)
        AWK.allocate()
        AWK.build_aux_no_ex(auxiliary_wave_dev, addr_dev, object_array_dev, probe_dev, 
            fac=1.0, add=False)
        from ptypy.accelerate.base.kernels import AuxiliaryWaveKernel as npAuxiliaryWaveKernel
        nAWK = npAuxiliaryWaveKernel()
        nAWK.allocate()
        nAWK.build_aux_no_ex(auxiliary_wave, addr, object_array, probe, fac=1.0, add=False)

        ## Assert
        np.testing.assert_array_equal(auxiliary_wave_dev.get(), auxiliary_wave,
                                      err_msg="The auxiliary_wave does not match numpy")

    def test_build_aux2_no_ex_noadd_UNITY(self):
        ## Arrange
        addr, object_array, probe, exit_wave = self.prepare_arrays()
        addr_dev, object_array_dev, probe_dev, exit_wave_dev = self.copy_to_gpu(addr, object_array, probe, exit_wave)
        auxiliary_wave_dev = cp.zeros_like(exit_wave_dev)
        auxiliary_wave = np.zeros_like(exit_wave)

        ## Act
        AWK = AuxiliaryWaveKernel(self.stream)
        AWK.allocate()
        AWK.build_aux2_no_ex(auxiliary_wave_dev, addr_dev, object_array_dev, probe_dev, 
            fac=1.0, add=False)
        from ptypy.accelerate.base.kernels import AuxiliaryWaveKernel as npAuxiliaryWaveKernel
        nAWK = npAuxiliaryWaveKernel()
        nAWK.allocate()
        nAWK.build_aux_no_ex(auxiliary_wave, addr, object_array, probe, fac=1.0, add=False)

        ## Assert
        np.testing.assert_array_equal(auxiliary_wave_dev.get(), auxiliary_wave,
                                      err_msg="The auxiliary_wave does not match numpy")


    def test_build_aux_no_ex_add_REGRESSION(self):
        ## Arrange
        addr, object_array, probe, exit_wave = self.prepare_arrays()
        addr, object_array, probe, exit_wave = self.copy_to_gpu(addr, object_array, probe, exit_wave)
        auxiliary_wave = cp.ones_like(exit_wave)

        ## Act
        AWK = AuxiliaryWaveKernel(self.stream)
        AWK.allocate()
        fac = 2.0
        AWK.build_aux_no_ex(auxiliary_wave, addr, object_array, probe, fac=fac, add=True)

        ## Assert
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
        expected_auxiliary_wave = fac*expected_auxiliary_wave + 1
        np.testing.assert_array_equal(auxiliary_wave.get(), expected_auxiliary_wave,
                                      err_msg="The auxiliary_wave has not been updated as expected")

    def test_build_aux_no_ex_add_UNITY(self):
        ## Arrange
        addr, object_array, probe, exit_wave = self.prepare_arrays()
        addr_dev, object_array_dev, probe_dev, exit_wave_dev = self.copy_to_gpu(addr, object_array, probe, exit_wave)
        auxiliary_wave_dev = cp.ones_like(exit_wave_dev)
        auxiliary_wave = np.ones_like(exit_wave)

        ## Act
        AWK = AuxiliaryWaveKernel(self.stream)
        AWK.allocate()
        AWK.build_aux_no_ex(auxiliary_wave_dev, addr_dev, object_array_dev, probe_dev, 
            fac=2.0, add=True)
        from ptypy.accelerate.base.kernels import AuxiliaryWaveKernel as npAuxiliaryWaveKernel
        nAWK = npAuxiliaryWaveKernel()
        nAWK.allocate()
        nAWK.build_aux_no_ex(auxiliary_wave, addr, object_array, probe, fac=2.0, add=True)

        ## Assert
        np.testing.assert_array_equal(auxiliary_wave_dev.get(), auxiliary_wave,
                                      err_msg="The auxiliary_wave does not match numpy")

    def test_build_aux2_no_ex_add_UNITY(self):
        ## Arrange
        addr, object_array, probe, exit_wave = self.prepare_arrays()
        addr_dev, object_array_dev, probe_dev, exit_wave_dev = self.copy_to_gpu(addr, object_array, probe, exit_wave)
        auxiliary_wave_dev = cp.ones_like(exit_wave_dev)
        auxiliary_wave = np.ones_like(exit_wave)

        ## Act
        AWK = AuxiliaryWaveKernel(self.stream)
        AWK.allocate()
        AWK.build_aux2_no_ex(auxiliary_wave_dev, addr_dev, object_array_dev, probe_dev, 
            fac=2.0, add=True)
        from ptypy.accelerate.base.kernels import AuxiliaryWaveKernel as npAuxiliaryWaveKernel
        nAWK = npAuxiliaryWaveKernel()
        nAWK.allocate()
        nAWK.build_aux_no_ex(auxiliary_wave, addr, object_array, probe, fac=2.0, add=True)

        ## Assert
        np.testing.assert_array_equal(auxiliary_wave_dev.get(), auxiliary_wave,
                                      err_msg="The auxiliary_wave does not match numpy")


    @unittest.skipIf(not perfrun, "performance test")
    def test_build_aux_no_ex_performance(self):
        addr, object_array, probe, exit_wave = self.prepare_arrays(performance=True)
        addr, object_array, probe, exit_wave = self.copy_to_gpu(addr, object_array, probe, exit_wave)
        auxiliary_wave = cp.zeros_like(exit_wave)

        AWK = AuxiliaryWaveKernel(self.stream)
        AWK.allocate()
        AWK.build_aux_no_ex(auxiliary_wave, addr, object_array, probe, 
            fac=1.0, add=False)


    def test_build_exit_alpha_tau_REGRESSION(self):
        ## Arrange
        addr, object_array, probe, exit_wave = self.prepare_arrays(scan_points=1)
        addr, object_array, probe, exit_wave = self.copy_to_gpu(addr, object_array, probe, exit_wave)
        auxiliary_wave = cp.zeros_like(exit_wave)

        ## Act
        AWK = AuxiliaryWaveKernel(self.stream)
        AWK.allocate()
        AWK.build_exit_alpha_tau(auxiliary_wave, addr, object_array, probe, exit_wave)

        # Assert
        expected_auxiliary_wave = np.array(
                [[[0. -2.j, 0. -2.j, 0. -2.j],
                [0. -2.j, 0. -2.j, 0. -2.j],
                [0. -2.j, 0. -2.j, 0. -2.j]],

                [[0. -8.j, 0. -8.j, 0. -8.j],
                [0. -8.j, 0. -8.j, 0. -8.j],
                [0. -8.j, 0. -8.j, 0. -8.j]],

                [[0. -4.j, 0. -4.j, 0. -4.j],
                [0. -4.j, 0. -4.j, 0. -4.j],
                [0. -4.j, 0. -4.j, 0. -4.j]],

                [[0.-16.j, 0.-16.j, 0.-16.j],
                [0.-16.j, 0.-16.j, 0.-16.j],
                [0.-16.j, 0.-16.j, 0.-16.j]]], dtype=np.complex64)
        np.testing.assert_allclose(auxiliary_wave.get(), expected_auxiliary_wave, rtol=1e-6, atol=1e-6,
                                      err_msg="The auxiliary_wave has not been updated as expected")

        expected_exit_wave = np.array(
                [[[1. -1.j, 1. -1.j, 1. -1.j],
                [1. -1.j, 1. -1.j, 1. -1.j],
                [1. -1.j, 1. -1.j, 1. -1.j]],

                [[2. -6.j, 2. -6.j, 2. -6.j],
                [2. -6.j, 2. -6.j, 2. -6.j],
                [2. -6.j, 2. -6.j, 2. -6.j]],

                [[3. -1.j, 3. -1.j, 3. -1.j],
                [3. -1.j, 3. -1.j, 3. -1.j],
                [3. -1.j, 3. -1.j, 3. -1.j]],

                [[4.-12.j, 4.-12.j, 4.-12.j],
                [4.-12.j, 4.-12.j, 4.-12.j],
                [4.-12.j, 4.-12.j, 4.-12.j]]], dtype=np.complex64)
        np.testing.assert_allclose(exit_wave.get(), expected_exit_wave, rtol=1e-6, atol=1e-6,
                                      err_msg="The exit_wave has not been updated as expected")
                              
    def test_build_exit_alpha_tau_UNITY(self):
        ## Arrange
        addr, object_array, probe, exit_wave = self.prepare_arrays(scan_points=1)
        addr_dev, object_array_dev, probe_dev, exit_wave_dev = self.copy_to_gpu(addr, object_array, probe, exit_wave)
        auxiliary_wave_dev = cp.ones_like(exit_wave_dev)
        auxiliary_wave = np.ones_like(exit_wave)
        
        ## Act
        AWK = AuxiliaryWaveKernel(self.stream)
        AWK.allocate()
        AWK.build_exit_alpha_tau(auxiliary_wave_dev, addr_dev, object_array_dev, probe_dev, exit_wave_dev, alpha=0.8, tau=0.6)
        from ptypy.accelerate.base.kernels import AuxiliaryWaveKernel as npAuxiliaryWaveKernel
        nAWK = npAuxiliaryWaveKernel()
        nAWK.allocate()
        nAWK.build_exit_alpha_tau(auxiliary_wave, addr, object_array, probe, exit_wave, alpha=0.8, tau=0.6)

        ## Assert
        np.testing.assert_allclose(auxiliary_wave_dev.get(), auxiliary_wave, rtol=1e-6, atol=1e-6,
                                      err_msg="The auxiliary_wave does not match numpy")
        ## Assert
        np.testing.assert_allclose(exit_wave_dev.get(), exit_wave, rtol=1e-6, atol=1e-6,
                                      err_msg="The exit_wave does not match numpy")

    @unittest.skipIf(not perfrun, "performance test")
    def test_build_exit_alpha_tau_performance(self):
        addr, object_array, probe, exit_wave = self.prepare_arrays(performance=True, scan_points=1)
        addr, object_array, probe, exit_wave = self.copy_to_gpu(addr, object_array, probe, exit_wave)
        auxiliary_wave = cp.zeros_like(exit_wave)

        AWK = AuxiliaryWaveKernel(self.stream)
        AWK.allocate()
        AWK.build_exit_alpha_tau(auxiliary_wave, addr, object_array, probe, exit_wave, alpha=0.8, tau=0.6)

if __name__ == '__main__':
    unittest.main()
