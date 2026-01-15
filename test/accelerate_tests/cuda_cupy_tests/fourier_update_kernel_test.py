'''


'''

import unittest
import numpy as np
from . import CupyCudaTest, have_cupy


if have_cupy():
    import cupy as cp
    from ptypy.accelerate.cuda_cupy.kernels import FourierUpdateKernel

COMPLEX_TYPE = np.complex64
FLOAT_TYPE = np.float32
INT_TYPE = np.int32

class FourierUpdateKernelTest(CupyCudaTest):


    def test_fmag_all_update_UNITY(self):
        '''
        setup
        '''
        B = 5  # frame size y
        C = 5  # frame size x

        D = 2  # number of probe modes
        G = 2  # number og object modes

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

        addr = np.zeros((N, total_number_modes, 5, 3), dtype=INT_TYPE)

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
        mask_sum = mask.sum(-1).sum(-1)

        err_fmag = np.zeros(N, dtype=FLOAT_TYPE)
        from ptypy.accelerate.base.kernels import FourierUpdateKernel as npFourierUpdateKernel
        pbound_set = 0.9
        nFUK = npFourierUpdateKernel(f, nmodes=total_number_modes)
        FUK = FourierUpdateKernel(f, nmodes=total_number_modes)

        nFUK.allocate()
        FUK.allocate()

        nFUK.fourier_error(f, addr, fmag, mask, mask_sum)
        nFUK.error_reduce(addr, err_fmag)
        # print(np.sqrt(pbound_set/err_fmag))
        f_d = cp.asarray(f)
        fmag_d = cp.asarray(fmag)
        mask_d = cp.asarray(mask)
        err_fmag_d = cp.asarray(err_fmag)
        addr_d = cp.asarray(addr)

        # now set the state for both.

        FUK.gpu.fdev = cp.asarray(nFUK.npy.fdev)
        FUK.gpu.ferr = cp.asarray(nFUK.npy.ferr)

        FUK.fmag_all_update(f_d, addr_d, fmag_d, mask_d, err_fmag_d, pbound=pbound_set)


        nFUK.fmag_all_update(f, addr, fmag, mask, err_fmag, pbound=pbound_set)
        expected_f = f
        measured_f = f_d.get()
        np.testing.assert_allclose(expected_f, measured_f, rtol=1e-6, err_msg="Numpy f "
                                                                      "is \n%s, \nbut gpu f is \n %s, \n mask is:\n %s \n" %  (repr(expected_f),
                                                                                                                               repr(measured_f),
                                                                                                                               repr(mask)))

    def test_fmag_update_nopbound_UNITY(self):
        '''
        setup
        '''
        B = 5  # frame size y
        C = 5  # frame size x

        D = 2  # number of probe modes
        G = 2  # number og object modes

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

        addr = np.zeros((N, total_number_modes, 5, 3), dtype=INT_TYPE)

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
        mask_sum = mask.sum(-1).sum(-1)

        err_fmag = np.zeros(N, dtype=FLOAT_TYPE)
        from ptypy.accelerate.base.kernels import FourierUpdateKernel as npFourierUpdateKernel
        nFUK = npFourierUpdateKernel(f, nmodes=total_number_modes)
        FUK = FourierUpdateKernel(f, nmodes=total_number_modes)

        nFUK.allocate()
        FUK.allocate()

        nFUK.fourier_error(f, addr, fmag, mask, mask_sum)
        nFUK.error_reduce(addr, err_fmag)
        # print(np.sqrt(pbound_set/err_fmag))
        f_d = cp.asarray(f)
        fmag_d = cp.asarray(fmag)
        mask_d = cp.asarray(mask)
        addr_d = cp.asarray(addr)

        # now set the state for both.

        FUK.gpu.fdev = cp.asarray(nFUK.npy.fdev)
        FUK.gpu.ferr = cp.asarray(nFUK.npy.ferr)

        FUK.fmag_update_nopbound(f_d, addr_d, fmag_d, mask_d)
        nFUK.fmag_update_nopbound(f, addr, fmag, mask)

        expected_f = f
        measured_f = f_d.get()
        np.testing.assert_allclose(measured_f, expected_f, rtol=1e-6, err_msg="Numpy f "
                                                                      "is \n%s, \nbut gpu f is \n %s, \n mask is:\n %s \n" %  (repr(expected_f),
                                                                                                                               repr(measured_f),
                                                                                                                               repr(mask)))


    def test_fourier_error_UNITY(self):
        '''
        setup
        '''
        B = 5  # frame size y
        C = 5  # frame size x

        D = 2  # number of probe modes
        G = 2  # number of object modes

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

        mask = np.empty(shape=(N, B, C),
                        dtype=FLOAT_TYPE)  # the masks for the measured magnitudes either 1xAxB or NxAxB
        mask_fill = np.ones_like(mask)
        mask_fill[::2, ::2] = 0  # checkerboard for testing
        mask[:] = mask_fill

        X, Y = np.meshgrid(range(scan_pts), range(scan_pts))
        X = X.reshape((N,))
        Y = Y.reshape((N,))

        addr = np.zeros((N, total_number_modes, 5, 3), dtype=INT_TYPE)

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

        '''
        test
        '''
        mask_sum = mask.sum(-1).sum(-1)

        from ptypy.accelerate.base.kernels import FourierUpdateKernel as npFourierUpdateKernel
        f_d = cp.asarray(f)
        fmag_d = cp.asarray(fmag)
        mask_d = cp.asarray(mask)
        addr_d = cp.asarray(addr)
        mask_sum_d = cp.asarray(mask_sum)

        nFUK = npFourierUpdateKernel(f, nmodes=total_number_modes)
        FUK = FourierUpdateKernel(f, nmodes=total_number_modes)

        nFUK.allocate()
        FUK.allocate()

        nFUK.fourier_error(f, addr, fmag, mask, mask_sum)
        FUK.fourier_error(f_d, addr_d, fmag_d, mask_d, mask_sum_d)

        expected_fdev = nFUK.npy.fdev
        measured_fdev = FUK.gpu.fdev.get()
        np.testing.assert_allclose(expected_fdev, measured_fdev, rtol=1e-6, err_msg="Numpy fdev "
                                                                            "is \n%s, \nbut gpu fdev is \n %s, \n " % (
                                                                            repr(expected_fdev),
                                                                            repr(measured_fdev)))

        expected_ferr = nFUK.npy.ferr
        measured_ferr = FUK.gpu.ferr.get()

        np.testing.assert_array_equal(expected_ferr, measured_ferr, err_msg="Numpy ferr"
                                                                            "is \n%s, \nbut gpu ferr is \n %s, \n " % (
                                                                            repr(expected_ferr),
                                                                            repr(measured_ferr)))
    def test_fourier_deviation_UNITY(self):
        '''
        setup - using the fourier_error as reference, so we need mask, etc.
        '''
        B = 5  # frame size y
        C = 5  # frame size x

        D = 2  # number of probe modes
        G = 2  # number of object modes

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

        mask = np.empty(shape=(N, B, C),
                        dtype=FLOAT_TYPE)  # the masks for the measured magnitudes either 1xAxB or NxAxB
        mask_fill = np.ones_like(mask)
        mask_fill[::2, ::2] = 0  # checkerboard for testing
        mask[:] = mask_fill

        X, Y = np.meshgrid(range(scan_pts), range(scan_pts))
        X = X.reshape((N,))
        Y = Y.reshape((N,))

        addr = np.zeros((N, total_number_modes, 5, 3), dtype=INT_TYPE)

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

        '''
        test
        '''
        mask_sum = mask.sum(-1).sum(-1)

        from ptypy.accelerate.base.kernels import FourierUpdateKernel as npFourierUpdateKernel
        f_d = cp.asarray(f)
        fmag_d = cp.asarray(fmag)
        addr_d = cp.asarray(addr)

        nFUK = npFourierUpdateKernel(f, nmodes=total_number_modes)
        FUK = FourierUpdateKernel(f, nmodes=total_number_modes)

        nFUK.allocate()
        FUK.allocate()

        nFUK.fourier_deviation(f, addr, fmag)
        FUK.fourier_deviation(f_d, addr_d, fmag_d)

        expected_fdev = nFUK.npy.fdev
        measured_fdev = FUK.gpu.fdev.get()
        np.testing.assert_allclose(measured_fdev, expected_fdev,  rtol=1e-6, err_msg="Numpy fdev "
                                                                            "is \n%s, \nbut gpu fdev is \n %s, \n " % (
                                                                            repr(expected_fdev),
                                                                            repr(measured_fdev)))



    def test_error_reduce_UNITY(self):
        '''
        setup
        '''
        B = 5  # frame size y
        C = 5  # frame size x

        D = 2  # number of probe modes
        G = 2  # number og object modes

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
        fmag_fill = np.arange(np.prod(fmag.shape).item()).reshape(fmag.shape).astype(fmag.dtype)
        fmag[:] = fmag_fill

        mask = np.empty(shape=(N, B, C),
                        dtype=FLOAT_TYPE)  # the masks for the measured magnitudes either 1xAxB or NxAxB
        mask_fill = np.ones_like(mask)
        mask_fill[::2, ::2] = 0  # checkerboard for testing
        mask[:] = mask_fill

        X, Y = np.meshgrid(range(scan_pts), range(scan_pts))
        X = X.reshape((N,))
        Y = Y.reshape((N,))

        addr = np.zeros((N, total_number_modes, 5, 3), dtype=INT_TYPE)

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
        mask_sum = mask.sum(-1).sum(-1)

        from ptypy.accelerate.base.kernels import FourierUpdateKernel as npFourierUpdateKernel
        f_d = cp.asarray(f)
        fmag_d = cp.asarray(fmag)
        mask_d = cp.asarray(mask)
        addr_d = cp.asarray(addr)
        err_fmag_d = cp.asarray(err_fmag)
        mask_sum_d = cp.asarray(mask_sum)
        pbound_set = 0.9
        nFUK = npFourierUpdateKernel(f, nmodes=total_number_modes)
        FUK = FourierUpdateKernel(f, nmodes=total_number_modes, queue_thread=self.stream)

        nFUK.allocate()
        FUK.allocate()

        nFUK.fourier_error(f, addr, fmag, mask, mask_sum)
        nFUK.error_reduce(addr, err_fmag)
        

        FUK.fourier_error(f_d, addr_d, fmag_d, mask_d, mask_sum_d)
        FUK.error_reduce(addr_d, err_fmag_d)

        expected_err_fmag = err_fmag
        measured_err_fmag = err_fmag_d.get()

        np.testing.assert_allclose(expected_err_fmag, measured_err_fmag, rtol=1.15207385e-07,
                                                                        err_msg="Numpy err_fmag"
                                                                            "is \n%s, \nbut gpu err_fmag is \n %s, \n " % (
                                                                            repr(expected_err_fmag),
                                                                            repr(measured_err_fmag)))

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
        # print(ferr.shape)
        scan_pts = 2  # one dimensional scan point number
        N = scan_pts ** 2

        addr = np.zeros((N, 1, 5, 3))
        aux = np.zeros((4, 5, 5))
        FUK = FourierUpdateKernel(aux, nmodes=1)
        err_mag = np.zeros(N, dtype=FLOAT_TYPE)
        err_mag_d = cp.asarray(err_mag)
        FUK.gpu.ferr = cp.asarray(ferr)
        addr_d = cp.asarray(addr)

        FUK.error_reduce(addr_d, err_mag_d)

        # print(repr(ferr))
        measured_err_mag = err_mag_d.get()

        # print(repr(measured_err_mag))

        expected_err_mag = np.array([45.096806,  388.54788, 1059.5702, 2155.6968], dtype=FLOAT_TYPE)

        np.testing.assert_array_equal(expected_err_mag, measured_err_mag, err_msg="The fourier_update_kernel.error_reduce"
                                                                   "is not behaving as expected.")


    def log_likelihood_UNITY_tester(self, use_version2=False):
        '''
        setup
        '''
        B = 5  # frame size y
        C = 5  # frame size x

        D = 2  # number of probe modes
        G = 2  # number of object modes

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

        mask = np.empty(shape=(N, B, C),
                        dtype=FLOAT_TYPE)  # the masks for the measured magnitudes either 1xAxB or NxAxB
        mask_fill = np.ones_like(mask)
        mask_fill[::2, ::2] = 0  # checkerboard for testing
        mask[:] = mask_fill

        X, Y = np.meshgrid(range(scan_pts), range(scan_pts))
        X = X.reshape((N,))
        Y = Y.reshape((N,))

        addr = np.zeros((N, total_number_modes, 5, 3), dtype=INT_TYPE)

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

        '''
        test
        '''
        mask_sum = mask.sum(-1).sum(-1)
        LLerr = np.zeros_like(mask_sum, dtype=np.float32)
        f_d = cp.asarray(f)
        fmag_d = cp.asarray(fmag)
        mask_d = cp.asarray(mask)
        addr_d = cp.asarray(addr)
        LLerr_d = cp.asarray(LLerr)

        from ptypy.accelerate.base.kernels import FourierUpdateKernel as npFourierUpdateKernel
        nFUK = npFourierUpdateKernel(f, nmodes=total_number_modes)
        nFUK.allocate()
        nFUK.log_likelihood(f, addr, fmag, mask, LLerr)

        FUK = FourierUpdateKernel(f, nmodes=total_number_modes)
        FUK.allocate()
        if use_version2:
            FUK.log_likelihood2(f_d, addr_d, fmag_d, mask_d, LLerr_d)
        else:
            FUK.log_likelihood(f_d, addr_d, fmag_d, mask_d, LLerr_d)

        expected_err_phot = LLerr
        measured_err_phot = LLerr_d.get()

        np.testing.assert_allclose(expected_err_phot, measured_err_phot, err_msg="Numpy log-likelihood error "
                                                                                 "is \n%s, \nbut gpu log-likelihood error is \n%s, \n " % (
                                                                                 repr(expected_err_phot),
                                                                                 repr(measured_err_phot)), rtol=1e-5)
    def test_log_likelihood_UNITY(self):
        self.log_likelihood_UNITY_tester(False)

    def test_log_likelihood2_UNITY(self):
        self.log_likelihood_UNITY_tester(True)

    def test_exit_error_UNITY(self):
        '''
        setup
        '''
        B = 5  # frame size y
        C = 5  # frame size x

        D = 2  # number of probe modes
        G = 2  # number of object modes

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

        addr = np.zeros((N, total_number_modes, 5, 3), dtype=INT_TYPE)

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

        '''
        test
        '''
        from ptypy.accelerate.base.kernels import FourierUpdateKernel as npFourierUpdateKernel
        aux_d = cp.asarray(aux)
        addr_d = cp.asarray(addr)

        nFUK = npFourierUpdateKernel(aux, nmodes=total_number_modes)
        FUK = FourierUpdateKernel(aux, nmodes=total_number_modes)

        nFUK.allocate()
        FUK.allocate()

        nFUK.exit_error(aux, addr, )
        FUK.exit_error(aux_d, addr_d)

        expected_ferr = nFUK.npy.ferr
        measured_ferr = FUK.gpu.ferr.get()

        np.testing.assert_allclose(expected_ferr, measured_ferr, err_msg="Numpy ferr"
                                                                            "is \n%s, \nbut gpu ferr is \n %s, \n " % (
                                                                            repr(expected_ferr),
                                                                            repr(measured_ferr)), rtol=1e-7)


if __name__ == '__main__':
    unittest.main()
