'''


'''

import unittest
import numpy as np
import ptypy.utils as u
from ptypy.accelerate.base.kernels import FourierUpdateKernel, AuxiliaryWaveKernel


COMPLEX_TYPE = np.complex64
FLOAT_TYPE = np.float32
INT_TYPE = np.int32


class FourierUpdateKernelTest(unittest.TestCase):

    def setUp(self):
        import sys
        np.set_printoptions(threshold=sys.maxsize, linewidth=np.inf)

    def tearDown(self):
        np.set_printoptions()

    def test_init(self):
        attrs = ["denom",
                 "fshape",
                 "nmodes",
                 "npy"]

        fake_aux = np.zeros((10, 20, 30)) # not used except to initialise
        fake_nmodes = 5#  not used except to initialise
        FUK = FourierUpdateKernel(fake_aux, nmodes=fake_nmodes)
        for attr in attrs:
            self.assertTrue(hasattr(FUK, attr), msg="FourierUpdateKernel does not have attribute: %s" % attr)

        np.testing.assert_equal(FUK.kernels,
                                ['fourier_error', 'error_reduce', 'fmag_all_update'],
                                err_msg='FourierUpdateKernel does not have the correct functions registered.')

    def test_allocate(self):
        '''
        setup
        '''
        B = 5  # frame size y
        C = 5  # frame size x

        D = 2  # number of probe modes
        G = 2  # number og object modes

        scan_pts = 2  # one dimensional scan point number

        N = scan_pts ** 2
        total_number_modes = G * D
        A = N * total_number_modes  # this is a 16 point scan pattern (4x4 grid) over all the modes

        f = np.empty(shape=(A, B, C), dtype=COMPLEX_TYPE)
        for idx in range(A):
            f[idx] = np.ones((B, C)) * (idx + 1) + 1j * np.ones((B, C)) * (idx + 1)

        fmag = np.empty(shape=(N, B, C), dtype=FLOAT_TYPE)  # the measured magnitudes NxAxB
        fmag_fill = np.arange(np.prod(fmag.shape)).reshape(fmag.shape).astype(fmag.dtype)
        fmag[:] = fmag_fill

        FUK = FourierUpdateKernel(f, nmodes=total_number_modes)
        FUK.allocate()

        expected_fdev_shape = (f.shape[0] // total_number_modes, f.shape[1], f.shape[2])
        expected_fdev_type = FLOAT_TYPE

        expected_ferr_shape = (f.shape[0] // total_number_modes, f.shape[1], f.shape[2])
        expected_ferr_type = FLOAT_TYPE

        np.testing.assert_equal(FUK.npy.fdev.shape, expected_fdev_shape)
        np.testing.assert_equal(FUK.npy.fdev.dtype, expected_fdev_type)
        np.testing.assert_equal(FUK.npy.ferr.shape, expected_ferr_shape)
        np.testing.assert_equal(FUK.npy.ferr.dtype, expected_ferr_type)

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

        f = np.empty(shape=(A, B, C), dtype=COMPLEX_TYPE)
        for idx in range(A):
            f[idx] = np.ones((B, C)) * (idx + 1) + 1j * np.ones((B, C)) * (idx + 1)

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

        err_fmag = np.zeros(N, dtype=FLOAT_TYPE)
        pbound_set = 0.9
        FUK = FourierUpdateKernel(f, nmodes=total_number_modes)
        FUK.allocate()
        FUK.fourier_error(f, addr, fmag, mask, mask_sum)


        expected_fdev = np.array([[[7.7459664,   6.7459664,   5.7459664,   4.7459664,   3.7459664],
                                   [2.7459664,   1.7459664,   0.74596643,  -0.25403357,  -1.2540336],
                                   [-2.2540336,  -3.2540336,  -4.2540336,  -5.2540336,  -6.2540336],
                                   [-7.2540336,  -8.254034,  -9.254034, -10.254034, -11.254034],
                                   [-12.254034, -13.254034, -14.254034, -15.254034, -16.254034]],

                                  [[-6.3452415,  -7.3452415,  -8.345242,  -9.345242, -10.345242],
                                   [-11.345242, -12.345242, -13.345242, -14.345242, -15.345242],
                                   [-16.345242, -17.345242, -18.345242, -19.345242, -20.345242],
                                   [-21.345242, -22.345242, -23.345242, -24.345242, -25.345242],
                                   [-26.345242, -27.345242, -28.345242, -29.345242, -30.345242]],

                                  [[-20.13363, -21.13363, -22.13363, -23.13363, -24.13363],
                                   [-25.13363, -26.13363, -27.13363, -28.13363, -29.13363],
                                   [-30.13363, -31.13363, -32.13363, -33.13363, -34.13363],
                                   [-35.13363, -36.13363, -37.13363, -38.13363, -39.13363],
                                   [-40.13363, -41.13363, -42.13363, -43.13363, -44.13363]],

                                  [[-33.866074, -34.866074, -35.866074, -36.866074, -37.866074],
                                   [-38.866074, -39.866074, -40.866074, -41.866074, -42.866074],
                                   [-43.866074, -44.866074, -45.866074, -46.866074, -47.866074],
                                   [-48.866074, -49.866074, -50.866074, -51.866074, -52.866074],
                                   [-53.866074, -54.866074, -55.866074, -56.866074, -57.866074]]],
                                 dtype=FLOAT_TYPE)
        np.testing.assert_array_equal(FUK.npy.fdev, expected_fdev,
                                      err_msg="fdev does not give the expected error "
                                              "for the fourier_update_kernel.fourier_error emthods")

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
        np.testing.assert_array_equal(FUK.npy.ferr, expected_ferr,
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
        #print(ferr.shape)
        #print(repr(ferr))
        auxiliary_shape = (4, 5, 5)
        fake_aux = np.zeros(auxiliary_shape, dtype=COMPLEX_TYPE)
        scan_pts = 2  # one dimensional scan point number
        N = scan_pts ** 2

        addr = np.zeros((N, 1, 5, 3))

        FUK = FourierUpdateKernel(fake_aux, nmodes=1)
        FUK.allocate()
        err_fmag = np.zeros(N, dtype=FLOAT_TYPE)
        FUK.error_reduce(addr, err_fmag)



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

    def test_fmag_update(self):
        '''
        setup
        '''
        B = 5  # frame size y
        C = 5  # frame size x

        D = 2  # number of probe modes
        G = 2  # number og object modes

        scan_pts = 2  # one dimensional scan point number

        N = scan_pts ** 2
        total_number_modes = G * D
        A = N * total_number_modes  # this is a 16 point scan pattern (4x4 grid) over all the modes

        f = np.empty(shape=(A, B, C), dtype=COMPLEX_TYPE)
        for idx in range(A):
            f[idx] = np.ones((B, C)) * (idx + 1) + 1j * np.ones((B, C)) * (idx + 1)

        fmag = np.empty(shape=(N, B, C), dtype=FLOAT_TYPE)  # the measured magnitudes NxAxB
        fmag_fill = np.arange(np.prod(fmag.shape)).reshape(fmag.shape).astype(fmag.dtype)
        fmag[:] = fmag_fill

        mask = np.empty(shape=(N, B, C), dtype=FLOAT_TYPE) # the masks for the measured magnitudes either 1xAxB or NxAxB
        mask_fill = np.ones_like(mask)
        mask_fill[::2, ::2] = 0  # checkerboard for testing
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


        # print("address book is:")
        # print(repr(addr))

        '''
        test
        '''

        err_fmag = np.zeros(N, dtype=FLOAT_TYPE)
        pbound_set = 0.9
        mask_sum = mask.sum(-1).sum(-1)

        FUK = FourierUpdateKernel(f, nmodes=total_number_modes)
        FUK.allocate()
        FUK.fourier_error(f, addr, fmag, mask, mask_sum)
        FUK.error_reduce(addr, err_fmag)
        FUK.fmag_all_update(f, addr, fmag, mask, err_fmag, pbound=pbound_set)
        # print("f before:")
        # print(repr(f))
        # print(fm.shape)
        # print(f.shape)
        # print("f after:")
        # print(repr(f))
        # self.assertTrue(False)
        expected_f = np.array([[[ 1.         +1.j        ,  1.         +1.j        ,  1.         +1.j        ,  1.         +1.j        ,  1.         +1.j        ],
                                [ 0.6955777  +0.6955777j ,  0.8064393  +0.8064393j ,  0.91730094 +0.91730094j,  1.0281626  +1.0281626j ,  1.1390243  +1.1390243j ],
                                [ 1.         +1.j        ,  1.         +1.j        ,  1.         +1.j        ,  1.         +1.j        ,  1.         +1.j        ],
                                [ 1.804194   +1.804194j  ,  1.9150558  +1.9150558j ,  2.0259173  +2.0259173j ,  2.136779   +2.136779j  ,  2.2476408  +2.2476408j ],
                                [ 1.         +1.j        ,  1.         +1.j        ,  1.         +1.j        ,  1.         +1.j        ,  1.         +1.j        ]],

                               [[ 2.         +2.j        ,  2.         +2.j        ,  2.         +2.j        ,  2.         +2.j        ,  2.         +2.j        ],
                                [ 1.3911554  +1.3911554j ,  1.6128786  +1.6128786j ,  1.8346019  +1.8346019j ,  2.0563252  +2.0563252j ,  2.2780485  +2.2780485j ],
                                [ 2.         +2.j        ,  2.         +2.j        ,  2.         +2.j        ,  2.         +2.j        ,  2.         +2.j        ],
                                [ 3.608388   +3.608388j  ,  3.8301115  +3.8301115j ,  4.0518346  +4.0518346j ,  4.273558   +4.273558j  ,  4.4952817  +4.4952817j ],
                                [ 2.         +2.j        ,  2.         +2.j        ,  2.         +2.j        ,  2.         +2.j        ,  2.         +2.j        ]],

                               [[ 3.         +3.j        ,  3.         +3.j        ,  3.         +3.j        ,  3.         +3.j        ,  3.         +3.j        ],
                                [ 2.086733   +2.086733j  ,  2.4193177  +2.4193177j ,  2.7519028  +2.7519028j ,  3.084488   +3.084488j  ,  3.4170728  +3.4170728j ],
                                [ 3.         +3.j        ,  3.         +3.j        ,  3.         +3.j        ,  3.         +3.j        ,  3.         +3.j        ],
                                [ 5.412582   +5.412582j  ,  5.7451673  +5.7451673j ,  6.077752   +6.077752j  ,  6.4103374  +6.4103374j ,  6.742923   +6.742923j  ],
                                [ 3.         +3.j        ,  3.         +3.j        ,  3.         +3.j        ,  3.         +3.j        ,  3.         +3.j        ]],

                               [[ 4.         +4.j        ,  4.         +4.j        ,  4.         +4.j        ,  4.         +4.j        ,  4.         +4.j        ],
                                [ 2.7823107  +2.7823107j ,  3.2257571  +3.2257571j ,  3.6692038  +3.6692038j ,  4.1126504  +4.1126504j ,  4.556097   +4.556097j  ],
                                [ 4.         +4.j        ,  4.         +4.j        ,  4.         +4.j        ,  4.         +4.j        ,  4.         +4.j        ],
                                [ 7.216776   +7.216776j  ,  7.660223   +7.660223j  ,  8.103669   +8.103669j  ,  8.547116   +8.547116j  ,  8.990563   +8.990563j  ],
                                [ 4.         +4.j        ,  4.         +4.j        ,  4.         +4.j        ,  4.         +4.j        ,  4.         +4.j        ]],

                               [[ 6.618852   +6.618852j  ,  6.87398    +6.87398j   ,  7.129109   +7.129109j  ,  7.3842373  +7.3842373j ,  7.6393657  +7.6393657j ],
                                [ 7.894494   +7.894494j  ,  8.149623   +8.149623j  ,  8.404751   +8.404751j  ,  8.659879   +8.659879j  ,  8.915008   +8.915008j  ],
                                [ 9.1701355  +9.1701355j ,  9.425264   +9.425264j  ,  9.680393   +9.680393j  ,  9.935521   +9.935521j  , 10.190649  +10.190649j  ],
                                [10.445778  +10.445778j  , 10.700907  +10.700907j  , 10.956035  +10.956035j  , 11.211163  +11.211163j  , 11.466292  +11.466292j  ],
                                [11.72142   +11.72142j   , 11.976548  +11.976548j  , 12.231676  +12.231676j  , 12.486806  +12.486806j  , 12.741934  +12.741934j  ]],

                               [[ 7.942622   +7.942622j  ,  8.248776   +8.248776j  ,  8.554931   +8.554931j  ,  8.861084   +8.861084j  ,  9.167239   +9.167239j  ],
                                [ 9.4733925  +9.4733925j ,  9.779547   +9.779547j  , 10.085701  +10.085701j  , 10.391855  +10.391855j  , 10.6980095 +10.6980095j ],
                                [11.004163  +11.004163j  , 11.310318  +11.310318j  , 11.616471  +11.616471j  , 11.922626  +11.922626j  , 12.228779  +12.228779j  ],
                                [12.534934  +12.534934j  , 12.841087  +12.841087j  , 13.147242  +13.147242j  , 13.453396  +13.453396j  , 13.75955   +13.75955j   ],
                                [14.065704  +14.065704j  , 14.371859  +14.371859j  , 14.678012  +14.678012j  , 14.984167  +14.984167j  , 15.290321  +15.290321j  ]],

                               [[ 9.266393   +9.266393j  ,  9.623572   +9.623572j  ,  9.980753   +9.980753j  , 10.337932  +10.337932j  , 10.695112  +10.695112j  ],
                                [11.052292  +11.052292j  , 11.4094715 +11.4094715j , 11.766651  +11.766651j  , 12.123831  +12.123831j  , 12.48101   +12.48101j   ],
                                [12.83819   +12.83819j   , 13.195371  +13.195371j  , 13.552549  +13.552549j  , 13.90973   +13.90973j   , 14.266909  +14.266909j  ],
                                [14.62409   +14.62409j   , 14.981269  +14.981269j  , 15.338449  +15.338449j  , 15.695628  +15.695628j  , 16.052809  +16.052809j  ],
                                [16.409988  +16.409988j  , 16.767168  +16.767168j  , 17.124348  +17.124348j  , 17.48153   +17.48153j   , 17.838707  +17.838707j  ]],

                               [[10.590163  +10.590163j  , 10.998368  +10.998368j  , 11.406574  +11.406574j  , 11.814779  +11.814779j  , 12.222985  +12.222985j  ],
                                [12.63119   +12.63119j   , 13.039396  +13.039396j  , 13.447601  +13.447601j  , 13.855806  +13.855806j  , 14.264012  +14.264012j  ],
                                [14.672217  +14.672217j  , 15.080423  +15.080423j  , 15.488628  +15.488628j  , 15.896834  +15.896834j  , 16.305038  +16.305038j  ],
                                [16.713245  +16.713245j  , 17.12145   +17.12145j   , 17.529655  +17.529655j  , 17.93786   +17.93786j   , 18.346067  +18.346067j  ],
                                [18.754272  +18.754272j  , 19.162477  +19.162477j  , 19.570683  +19.570683j  , 19.97889   +19.97889j   , 20.387094  +20.387094j  ]],

                               [[ 9.         +9.j        ,  9.         +9.j        ,  9.         +9.j        ,  9.         +9.j        ,  9.         +9.j        ],
                                [16.35309   +16.35309j   , 16.64565   +16.64565j   , 16.938211  +16.938211j  , 17.23077   +17.23077j   , 17.52333   +17.52333j   ],
                                [ 9.         +9.j        ,  9.         +9.j        ,  9.         +9.j        ,  9.         +9.j        ,  9.         +9.j        ],
                                [19.278687  +19.278687j  , 19.571249  +19.571249j  , 19.86381   +19.86381j   , 20.156368  +20.156368j  , 20.448927  +20.448927j  ],
                                [ 9.         +9.j        ,  9.         +9.j        ,  9.         +9.j        ,  9.         +9.j        ,  9.         +9.j        ]],

                               [[10.        +10.j        , 10.        +10.j        , 10.        +10.j        , 10.        +10.j        , 10.        +10.j        ],
                                [18.170101  +18.170101j  , 18.495167  +18.495167j  , 18.820234  +18.820234j  , 19.1453    +19.1453j    , 19.470367  +19.470367j  ],
                                [10.        +10.j        , 10.        +10.j        , 10.        +10.j        , 10.        +10.j        , 10.        +10.j        ],
                                [21.420763  +21.420763j  , 21.745832  +21.745832j  , 22.0709    +22.0709j    , 22.395964  +22.395964j  , 22.721031  +22.721031j  ],
                                [10.        +10.j        , 10.        +10.j        , 10.        +10.j        , 10.        +10.j        , 10.        +10.j        ]],

                               [[11.        +11.j        , 11.        +11.j        , 11.        +11.j        , 11.        +11.j        , 11.        +11.j        ],
                                [19.98711   +19.98711j   , 20.344685  +20.344685j  , 20.702257  +20.702257j  , 21.05983   +21.05983j   , 21.417404  +21.417404j  ],
                                [11.        +11.j        , 11.        +11.j        , 11.        +11.j        , 11.        +11.j        , 11.        +11.j        ],
                                [23.56284   +23.56284j   , 23.920416  +23.920416j  , 24.277988  +24.277988j  , 24.63556   +24.63556j   , 24.993134  +24.993134j  ],
                                [11.        +11.j        , 11.        +11.j        , 11.        +11.j        , 11.        +11.j        , 11.        +11.j        ]],

                               [[12.        +12.j        , 12.        +12.j        , 12.        +12.j        , 12.        +12.j        , 12.        +12.j        ],
                                [21.804121  +21.804121j  , 22.1942    +22.1942j    , 22.584282  +22.584282j  , 22.974361  +22.974361j  , 23.36444   +23.36444j   ],
                                [12.        +12.j        , 12.        +12.j        , 12.        +12.j        , 12.        +12.j        , 12.        +12.j        ],
                                [25.704914  +25.704914j  , 26.094997  +26.094997j  , 26.485079  +26.485079j  , 26.875156  +26.875156j  , 27.265236  +27.265236j  ],
                                [12.        +12.j        , 12.        +12.j        , 12.        +12.j        , 12.        +12.j        , 12.        +12.j        ]],

                               [[23.484367  +23.484367j  , 23.793953  +23.793953j  , 24.103535  +24.103535j  , 24.41312   +24.41312j   , 24.7227    +24.7227j    ],
                                [25.032284  +25.032284j  , 25.341867  +25.341867j  , 25.651451  +25.651451j  , 25.961035  +25.961035j  , 26.270618  +26.270618j  ],
                                [26.5802    +26.5802j    , 26.889784  +26.889784j  , 27.199366  +27.199366j  , 27.508951  +27.508951j  , 27.818535  +27.818535j  ],
                                [28.128117  +28.128117j  , 28.437698  +28.437698j  , 28.747284  +28.747284j  , 29.056868  +29.056868j  , 29.36645   +29.36645j   ],
                                [29.676033  +29.676033j  , 29.985619  +29.985619j  , 30.2952    +30.2952j    , 30.604782  +30.604782j  , 30.914366  +30.914366j  ]],

                               [[25.290857  +25.290857j  , 25.624256  +25.624256j  , 25.957653  +25.957653j  , 26.291052  +26.291052j  , 26.624447  +26.624447j  ],
                                [26.957846  +26.957846j  , 27.291243  +27.291243j  , 27.62464   +27.62464j   , 27.958038  +27.958038j  , 28.291435  +28.291435j  ],
                                [28.62483   +28.62483j   , 28.95823   +28.95823j   , 29.291626  +29.291626j  , 29.625023  +29.625023j  , 29.958424  +29.958424j  ],
                                [30.291819  +30.291819j  , 30.625214  +30.625214j  , 30.958612  +30.958612j  , 31.292011  +31.292011j  , 31.625406  +31.625406j  ],
                                [31.958805  +31.958805j  , 32.292206  +32.292206j  , 32.6256    +32.6256j    , 32.958996  +32.958996j  , 33.292393  +33.292393j  ]],

                               [[27.097347  +27.097347j  , 27.454561  +27.454561j  , 27.811771  +27.811771j  , 28.168985  +28.168985j  , 28.526192  +28.526192j  ],
                                [28.883406  +28.883406j  , 29.240616  +29.240616j  , 29.597828  +29.597828j  , 29.95504   +29.95504j   , 30.312252  +30.312252j  ],
                                [30.669462  +30.669462j  , 31.026674  +31.026674j  , 31.383884  +31.383884j  , 31.741096  +31.741096j  , 32.09831   +32.09831j   ],
                                [32.45552   +32.45552j   , 32.81273   +32.81273j   , 33.16994   +33.16994j   , 33.527153  +33.527153j  , 33.884365  +33.884365j  ],
                                [34.241577  +34.241577j  , 34.59879   +34.59879j   , 34.956     +34.956j     , 35.31321   +35.31321j   , 35.67042   +35.67042j   ]],

                               [[28.903837  +28.903837j  , 29.284864  +29.284864j  , 29.66589   +29.66589j   , 30.046917  +30.046917j  , 30.427938  +30.427938j  ],
                                [30.808966  +30.808966j  , 31.189991  +31.189991j  , 31.571016  +31.571016j  , 31.952044  +31.952044j  , 32.33307   +32.33307j   ],
                                [32.714092  +32.714092j  , 33.09512   +33.09512j   , 33.476143  +33.476143j  , 33.85717   +33.85717j   , 34.238197  +34.238197j  ],
                                [34.61922   +34.61922j   , 35.000244  +35.000244j  , 35.38127   +35.38127j   , 35.7623    +35.7623j    , 36.143322  +36.143322j  ],
                                [36.52435   +36.52435j   , 36.905376  +36.905376j  , 37.2864    +37.2864j    , 37.667423  +37.667423j  , 38.04845   +38.04845j   ]]], dtype=COMPLEX_TYPE)

        np.testing.assert_array_equal(f, expected_f, err_msg="the f array from the fmag_all_update kernesl isnot behaving as expected.")

    # TODO: This test needs to be redesigne to NOT use components from the archive test
    @unittest.skip('This test needs to be redone')
    def test_log_likelihood(self):
        nmodes = 1
        PtychoInstance = tu.get_ptycho_instance('log_likelihood_test', nmodes) # noqa: F821
        ptypy_error_metric = self.get_ptypy_loglikelihood(PtychoInstance)
        LLerr_expected = np.array([LL for LL in ptypy_error_metric.values()]).astype(np.float32)

        vectorised_scan = du.pod_to_arrays(PtychoInstance, 'S0000') # noqa: F821
        addr = vectorised_scan['meta']['addr'].reshape((len(ptypy_error_metric)//nmodes, nmodes, 5, 3))
        probe = vectorised_scan['probe']
        obj = vectorised_scan['obj']
        mask = vectorised_scan['mask']
        exit_wave = vectorised_scan['exit wave']
        mag = np.sqrt(vectorised_scan['diffraction'])

        aux = np.zeros_like(exit_wave)
        AWK = AuxiliaryWaveKernel()
        AWK.allocate()
        AWK.build_aux_no_ex(aux, addr, obj, probe, fac=1.0, add=False)

        scan = list(PtychoInstance.model.scans.values())[0]
        geo = scan.geometries[0]
        aux[:] = geo.propagator.fw(aux)

        FUK = FourierUpdateKernel(aux, nmodes=1)
        FUK.allocate()
        LLerr = np.zeros_like(LLerr_expected, dtype=np.float32)
        FUK.log_likelihood(aux, addr, mag, mask, LLerr)

        np.testing.assert_allclose(LLerr, LLerr_expected, rtol=1e-6, err_msg="LLerr does not give the expected error "
                                                                             "for the fourier_update_kernel.log_likelihood method") 

    def get_ptypy_loglikelihood(self, a_ptycho_instance):
        error_dct = {}
        for dname, diff_view in a_ptycho_instance.diff.views.items():
            I = diff_view.data
            fmask = diff_view.pod.mask
            LL = np.zeros_like(diff_view.data)
            for name, pod in diff_view.pods.items():
                LL += u.abs2(pod.fw(pod.probe * pod.object))

            error_dct[dname] = (np.sum(fmask * (LL - I) ** 2 / (I + 1.))
                            / np.prod(LL.shape))
        return error_dct


    def test_exit_error(self):
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

        aux = np.empty(shape=(A, B, C), dtype=COMPLEX_TYPE)
        for idx in range(A):
            aux[idx] = np.ones((B, C)) * (idx + 1) + 1j * np.ones((B, C)) * (idx + 1)

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

        err_sum = np.zeros(N, dtype=FLOAT_TYPE)
        FUK = FourierUpdateKernel(aux, nmodes=total_number_modes)
        FUK.allocate()
        FUK.exit_error(aux, addr)

        expected_ferr = np.array([[[ 2.3999996, 2.3999996, 2.3999996, 2.3999996, 2.3999996],
                                   [ 2.3999996, 2.3999996, 2.3999996, 2.3999996, 2.3999996],
                                   [ 2.3999996, 2.3999996, 2.3999996, 2.3999996, 2.3999996],
                                   [ 2.3999996, 2.3999996, 2.3999996, 2.3999996, 2.3999996],
                                   [ 2.3999996, 2.3999996, 2.3999996, 2.3999996, 2.3999996]],

                                  [[13.92, 13.92, 13.92, 13.92, 13.92],
                                   [13.92, 13.92, 13.92, 13.92, 13.92],
                                   [13.92, 13.92, 13.92, 13.92, 13.92],
                                   [13.92, 13.92, 13.92, 13.92, 13.92],
                                   [13.92, 13.92, 13.92, 13.92, 13.92]],

                                  [[35.68, 35.68, 35.68, 35.68, 35.68 ],
                                   [35.68, 35.68, 35.68, 35.68, 35.68 ],
                                   [35.68, 35.68, 35.68, 35.68, 35.68 ],
                                   [35.68, 35.68, 35.68, 35.68, 35.68 ],
                                   [35.68, 35.68, 35.68, 35.68, 35.68 ]],

                                  [[67.68, 67.68, 67.68, 67.68, 67.68 ],
                                   [67.68, 67.68, 67.68, 67.68, 67.68 ],
                                   [67.68, 67.68, 67.68, 67.68, 67.68 ],
                                   [67.68, 67.68, 67.68, 67.68, 67.68 ],
                                   [67.68, 67.68, 67.68, 67.68, 67.68 ]]], dtype=FLOAT_TYPE)

        np.testing.assert_array_equal(FUK.npy.ferr, expected_ferr,
                                      err_msg="ferr does not give the expected error "
                                              "for the fourier_update_kernel.fourier_error emthods")


if __name__ == '__main__':
    unittest.main()
