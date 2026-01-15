'''


'''

import unittest
import numpy as np
from ptypy.accelerate.base.kernels import PositionCorrectionKernel
from ptypy import utils as u
COMPLEX_TYPE = np.complex64
FLOAT_TYPE = np.float32
INT_TYPE = np.int32


class PositionCorrectionKernelTest(unittest.TestCase):

    def setUp(self):
        import sys
        np.set_printoptions(threshold=sys.maxsize, linewidth=np.inf)
        self.params = u.Param()
        self.params.nshifts = 4
        self.params.method = "Annealing"
        self.params.amplitude = 2e-9
        self.params.start = 0
        self.params.stop = 10
        self.params.max_shift = 2e-9
        self.params.amplitude_decay = True
        self.resolution = [1e-9,1e-9]

    def tearDown(self):
        np.set_printoptions()

    def test_build_aux(self):
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
        A = total_number_scan_positions * total_number_modes  # this is a 16 point scan pattern (4x4 grid) over all the modes

        probe = np.empty(shape=(D, E, F), dtype=COMPLEX_TYPE)
        for idx in range(D):
            probe[idx] = np.ones((E, F)) * (idx + 1) + 1j * np.ones((E, F)) * (idx + 2)

        object_array = np.empty(shape=(G, H, I), dtype=COMPLEX_TYPE)
        for idx in range(G):
            object_array[idx] = np.ones((H, I)) * (3 * idx + 1) + 1j * np.ones((H, I)) * (3 * idx + 2)

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
        auxiliary_wave = np.zeros((A, B, C), dtype=COMPLEX_TYPE)

        PCK = PositionCorrectionKernel(auxiliary_wave, total_number_modes, self.params, self.resolution)
        PCK.allocate()  # doesn't actually do anything at the moment
        PCK.build_aux(auxiliary_wave, addr, object_array, probe)

        expected_auxiliary_wave = np.array([[[-3. +4.j, -3. +4.j, -3. +4.j],
                                             [-3. +4.j, -3. +4.j, -3. +4.j],
                                             [-3. +4.j, -3. +4.j, -3. +4.j]],

                                            [[-6.+13.j, -6.+13.j, -6.+13.j],
                                             [-6.+13.j, -6.+13.j, -6.+13.j],
                                             [-6.+13.j, -6.+13.j, -6.+13.j]],

                                            [[-4. +7.j, -4. +7.j, -4. +7.j],
                                             [-4. +7.j, -4. +7.j, -4. +7.j],
                                             [-4. +7.j, -4. +7.j, -4. +7.j]],

                                            [[-7.+22.j, -7.+22.j, -7.+22.j],
                                             [-7.+22.j, -7.+22.j, -7.+22.j],
                                             [-7.+22.j, -7.+22.j, -7.+22.j]],

                                            [[-3. +4.j, -3. +4.j, -3. +4.j],
                                             [-3. +4.j, -3. +4.j, -3. +4.j],
                                             [-3. +4.j, -3. +4.j, -3. +4.j]],

                                            [[-6.+13.j, -6.+13.j, -6.+13.j],
                                             [-6.+13.j, -6.+13.j, -6.+13.j],
                                             [-6.+13.j, -6.+13.j, -6.+13.j]],

                                            [[-4. +7.j, -4. +7.j, -4. +7.j],
                                             [-4. +7.j, -4. +7.j, -4. +7.j],
                                             [-4. +7.j, -4. +7.j, -4. +7.j]],

                                            [[-7.+22.j, -7.+22.j, -7.+22.j],
                                             [-7.+22.j, -7.+22.j, -7.+22.j],
                                             [-7.+22.j, -7.+22.j, -7.+22.j]],

                                            [[-3. +4.j, -3. +4.j, -3. +4.j],
                                             [-3. +4.j, -3. +4.j, -3. +4.j],
                                             [-3. +4.j, -3. +4.j, -3. +4.j]],

                                            [[-6.+13.j, -6.+13.j, -6.+13.j],
                                             [-6.+13.j, -6.+13.j, -6.+13.j],
                                             [-6.+13.j, -6.+13.j, -6.+13.j]],

                                            [[-4. +7.j, -4. +7.j, -4. +7.j],
                                             [-4. +7.j, -4. +7.j, -4. +7.j],
                                             [-4. +7.j, -4. +7.j, -4. +7.j]],

                                            [[-7.+22.j, -7.+22.j, -7.+22.j],
                                             [-7.+22.j, -7.+22.j, -7.+22.j],
                                             [-7.+22.j, -7.+22.j, -7.+22.j]],

                                            [[-3. +4.j, -3. +4.j, -3. +4.j],
                                             [-3. +4.j, -3. +4.j, -3. +4.j],
                                             [-3. +4.j, -3. +4.j, -3. +4.j]],

                                            [[-6.+13.j, -6.+13.j, -6.+13.j],
                                             [-6.+13.j, -6.+13.j, -6.+13.j],
                                             [-6.+13.j, -6.+13.j, -6.+13.j]],

                                            [[-4. +7.j, -4. +7.j, -4. +7.j],
                                             [-4. +7.j, -4. +7.j, -4. +7.j],
                                             [-4. +7.j, -4. +7.j, -4. +7.j]],

                                            [[-7.+22.j, -7.+22.j, -7.+22.j],
                                             [-7.+22.j, -7.+22.j, -7.+22.j],
                                             [-7.+22.j, -7.+22.j, -7.+22.j]]], dtype=COMPLEX_TYPE)

        np.testing.assert_array_equal(expected_auxiliary_wave, expected_auxiliary_wave,
                                      err_msg="The auxiliary_wave has not been updated as expected")

    def test_fourier_error(self):
        '''
        setup
        '''
        B = 5  # frame size y
        C = 5  # frame size x

        D = 2  # number of probe modes
        G = 2 # number og object modes

        E = B  # probe size y
        F = C  # probe size x

        scan_pts = 2  # one dimensional scan point number

        N = scan_pts ** 2
        total_number_modes = G * D
        A = N * total_number_modes  # this is a 16 point scan pattern (4x4 grid) over all the modes

        auxiliary_wave = np.empty(shape=(A, B, C), dtype=COMPLEX_TYPE)
        for idx in range(A):
            auxiliary_wave[idx] = np.ones((B, C)) * (idx + 1) + 1j * np.ones((B, C)) * (idx + 1)

        fmag = np.empty(shape=(N, B, C), dtype=FLOAT_TYPE)  # the measured magnitudes NxAxB
        fmag_fill = np.arange(np.prod(fmag.shape)).reshape(fmag.shape).astype(fmag.dtype)
        fmag[:] = fmag_fill

        mask = np.empty(shape=(N, B, C), dtype=FLOAT_TYPE)# the masks for the measured magnitudes either 1xAxB or NxAxB
        mask_fill = np.ones_like(mask)
        mask_fill[::2, ::2] = 0 # checkerboard for testing
        mask[:] = mask_fill

        X, Y = np.meshgrid(range(scan_pts), range(scan_pts))
        X = X.reshape((N,))
        Y = Y.reshape((N,))

        addr = np.zeros((N, total_number_modes, 5, 3))

        exit_idx = 0
        position_idx = 0
        for xpos, ypos in zip(X, Y):
            mode_idx = 0
            for pr_mode in range(D):
                for ob_mode in range(G):
                    addr[position_idx, mode_idx] = np.array([[pr_mode, 0, 0],
                                                             [ob_mode, ypos, xpos],
                                                             [exit_idx, 0, 0],
                                                             [position_idx, 0, 0],
                                                             [position_idx, 0, 0]])
                    mode_idx += 1
                    exit_idx += 1
            position_idx += 1


        mask_sum = mask.sum(-1).sum(-1)


        PCK = PositionCorrectionKernel(auxiliary_wave, total_number_modes, self.params, self.resolution)
        PCK.allocate()
        PCK.fourier_error(auxiliary_wave, addr, fmag, mask, mask_sum)


        expected_ferr = np.array([[[0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
                                   [7.54033208e-01, 3.04839879e-01, 5.56465909e-02, 6.45330548e-03, 1.57260016e-01],
                                   [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
                                   [5.26210022e+00, 6.81290817e+00, 8.56371498e+00, 1.05145216e+01, 1.26653280e+01],
                                   [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00]],

                                  [[1.61048353e+00, 2.15810299e+00, 2.78572226e+00, 3.49334168e+00, 4.28096104e+00],
                                   [5.14858055e+00, 6.09619951e+00, 7.12381887e+00, 8.23143768e+00, 9.41905785e+00],
                                   [1.06866770e+01, 1.20342960e+01, 1.34619150e+01, 1.49695349e+01, 1.65571537e+01],
                                   [1.82247734e+01, 1.99723930e+01, 2.18000126e+01, 2.37076321e+01, 2.56952515e+01],
                                   [2.77628708e+01, 2.99104881e+01, 3.21381073e+01, 3.44457283e+01, 3.68333473e+01]],

                                  [[0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
                                   [6.31699409e+01, 6.82966690e+01, 7.36233978e+01, 7.91501160e+01, 8.48768463e+01],
                                   [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
                                   [1.23437180e+02, 1.30563919e+02, 1.37890640e+02, 1.45417374e+02, 1.53144089e+02],
                                   [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00]],

                                  [[4.58764343e+01, 4.86257210e+01, 5.14550095e+01, 5.43642960e+01, 5.73535805e+01],
                                   [6.04228668e+01, 6.35721550e+01, 6.68014374e+01, 7.01107254e+01, 7.35000076e+01],
                                   [7.69692993e+01, 8.05185852e+01, 8.41478729e+01, 8.78571548e+01, 9.16464386e+01],
                                   [9.55157242e+01, 9.94650116e+01, 1.03494293e+02, 1.07603584e+02, 1.11792870e+02],
                                   [1.16062157e+02, 1.20411446e+02, 1.24840721e+02, 1.29350006e+02, 1.33939301e+02]]],
                                 dtype=FLOAT_TYPE)
        np.testing.assert_array_equal(PCK.npy.ferr, expected_ferr,
                                      err_msg="ferr does not give the expected error "
                                              "for the fourier_update_kernel.fourier_error emthods")

    def test_error_reduce(self):
        # array from the previous test
        ferr = np.array([[[0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
                          [7.54033208e-01, 3.04839879e-01, 5.56465909e-02, 6.45330548e-03, 1.57260016e-01],
                          [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
                          [5.26210022e+00, 6.81290817e+00, 8.56371498e+00, 1.05145216e+01, 1.26653280e+01],
                          [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00]],

                         [[1.61048353e+00, 2.15810299e+00, 2.78572226e+00, 3.49334168e+00, 4.28096104e+00],
                          [5.14858055e+00, 6.09619951e+00, 7.12381887e+00, 8.23143768e+00, 9.41905785e+00],
                          [1.06866770e+01, 1.20342960e+01, 1.34619150e+01, 1.49695349e+01, 1.65571537e+01],
                          [1.82247734e+01, 1.99723930e+01, 2.18000126e+01, 2.37076321e+01, 2.56952515e+01],
                          [2.77628708e+01, 2.99104881e+01, 3.21381073e+01, 3.44457283e+01, 3.68333473e+01]],

                         [[0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
                          [6.31699409e+01, 6.82966690e+01, 7.36233978e+01, 7.91501160e+01, 8.48768463e+01],
                          [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
                          [1.23437180e+02, 1.30563919e+02, 1.37890640e+02, 1.45417374e+02, 1.53144089e+02],
                          [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00]],

                         [[4.58764343e+01, 4.86257210e+01, 5.14550095e+01, 5.43642960e+01, 5.73535805e+01],
                          [6.04228668e+01, 6.35721550e+01, 6.68014374e+01, 7.01107254e+01, 7.35000076e+01],
                          [7.69692993e+01, 8.05185852e+01, 8.41478729e+01, 8.78571548e+01, 9.16464386e+01],
                          [9.55157242e+01, 9.94650116e+01, 1.03494293e+02, 1.07603584e+02, 1.11792870e+02],
                          [1.16062157e+02, 1.20411446e+02, 1.24840721e+02, 1.29350006e+02, 1.33939301e+02]]],
                        dtype=FLOAT_TYPE)


        # print(repr(ferr))
        # print(ferr.shape)
        # print(repr(ferr))
        auxiliary_shape = (4, 5, 5)
        fake_aux = np.zeros(auxiliary_shape, dtype=COMPLEX_TYPE)
        scan_pts = 2  # one dimensional scan point number
        N = scan_pts ** 2

        addr = np.zeros((N, 1, 5, 3))

        PCK = PositionCorrectionKernel(fake_aux, 1, self.params, self.resolution)
        PCK.allocate()
        err_fmag = np.zeros(N, dtype=FLOAT_TYPE)
        PCK.error_reduce(addr, err_fmag)



        expected_ferr = np.array([[[0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
                                   [7.54033208e-01, 3.04839879e-01, 5.56465909e-02, 6.45330548e-03, 1.57260016e-01],
                                   [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
                                   [5.26210022e+00, 6.81290817e+00, 8.56371498e+00, 1.05145216e+01, 1.26653280e+01],
                                   [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00]],

                                  [[1.61048353e+00, 2.15810299e+00, 2.78572226e+00, 3.49334168e+00, 4.28096104e+00],
                                   [5.14858055e+00, 6.09619951e+00, 7.12381887e+00, 8.23143768e+00, 9.41905785e+00],
                                   [1.06866770e+01, 1.20342960e+01, 1.34619150e+01, 1.49695349e+01, 1.65571537e+01],
                                   [1.82247734e+01, 1.99723930e+01, 2.18000126e+01, 2.37076321e+01, 2.56952515e+01],
                                   [2.77628708e+01, 2.99104881e+01, 3.21381073e+01, 3.44457283e+01, 3.68333473e+01]],

                                  [[0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
                                   [6.31699409e+01, 6.82966690e+01, 7.36233978e+01, 7.91501160e+01, 8.48768463e+01],
                                   [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
                                   [1.23437180e+02, 1.30563919e+02, 1.37890640e+02, 1.45417374e+02, 1.53144089e+02],
                                   [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00]],

                                  [[4.58764343e+01, 4.86257210e+01, 5.14550095e+01, 5.43642960e+01, 5.73535805e+01],
                                   [6.04228668e+01, 6.35721550e+01, 6.68014374e+01, 7.01107254e+01, 7.35000076e+01],
                                   [7.69692993e+01, 8.05185852e+01, 8.41478729e+01, 8.78571548e+01, 9.16464386e+01],
                                   [9.55157242e+01, 9.94650116e+01, 1.03494293e+02, 1.07603584e+02, 1.11792870e+02],
                                   [1.16062157e+02, 1.20411446e+02, 1.24840721e+02, 1.29350006e+02, 1.33939301e+02]]],
                                 dtype=FLOAT_TYPE)

        np.testing.assert_array_equal(expected_ferr, ferr, err_msg="The fourier_update_kernel.error_reduce"
                                                                   "is not behaving as expected.")


if __name__ == '__main__':
    unittest.main()
