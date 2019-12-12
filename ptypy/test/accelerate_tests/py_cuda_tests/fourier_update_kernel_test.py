'''


'''

import unittest
import numpy as np
import pycuda.driver as cuda
from pycuda import gpuarray

from ptypy.accelerate.py_cuda.fourier_update_kernel import FourierUpdateKernel

COMPLEX_TYPE = np.complex64
FLOAT_TYPE = np.float32
INT_TYPE = np.int32


class FourierUpdateKernelTest(unittest.TestCase):

    def setUp(self):
        import sys
        np.set_printoptions(threshold=sys.maxsize, linewidth=np.inf)
        cuda.init()
        current_dev = cuda.Device(0)
        self.ctx = current_dev.make_context()
        self.ctx.push()

    def tearDown(self):
        np.set_printoptions()
        self.ctx.detach()

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
        fdev = np.zeros_like(fmag)
        ferr = np.zeros_like(fmag)
        err_fmag = np.zeros(N, dtype=FLOAT_TYPE)
        from ptypy.accelerate.array_based.fourier_update_kernel import FourierUpdateKernel as npFourierUpdateKernel
        pbound_set = 0.9
        nFUK = npFourierUpdateKernel(nmodes=total_number_modes, pbound=pbound_set)
        FUK = FourierUpdateKernel(nmodes=total_number_modes, pbound=pbound_set)

        nFUK.configure(fmag, mask, f, addr)
        FUK.configure(fmag, mask, f, addr)

        nFUK.fourier_error(f, fmag, fdev, ferr, mask, mask_sum, addr)
        nFUK.error_reduce(ferr, err_fmag, addr)
        # print(np.sqrt(pbound_set/err_fmag))
        f_d = gpuarray.to_gpu(f)
        fmag_d = gpuarray.to_gpu(fmag)
        fdev_d = gpuarray.to_gpu(fdev)
        ferr_d = gpuarray.to_gpu(ferr)
        mask_d = gpuarray.to_gpu(mask)
        err_fmag_d = gpuarray.to_gpu(err_fmag)
        addr_d = gpuarray.to_gpu(addr)

        FUK.fmag_all_update(f_d, mask_d, fmag_d, fdev_d, err_fmag_d, addr_d)


        nFUK.fmag_all_update(f, mask, fmag, fdev, err_fmag, addr)

        expected_f = f
        measured_f = f_d.get()
        np.testing.assert_array_equal(expected_f, measured_f, err_msg="Numpy f "
                                                                      "is \n%s, \nbut gpu f is \n %s, \n mask is:\n %s \n" %  (repr(expected_f),
                                                                                                                               repr(measured_f),
                                                                                                                               repr(mask)))

        f_d.gpudata.free()
        fmag_d.gpudata.free()
        fdev_d.gpudata.free()
        ferr_d.gpudata.free()
        mask_d.gpudata.free()
        err_fmag_d.gpudata.free()
        addr_d.gpudata.free()

    def test_fourier_error_UNITY(self):
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

        fdev = np.zeros_like(fmag)
        ferr = np.zeros_like(fmag)
        from ptypy.accelerate.array_based.fourier_update_kernel import FourierUpdateKernel as npFourierUpdateKernel
        f_d = gpuarray.to_gpu(f)
        fmag_d = gpuarray.to_gpu(fmag)
        fdev_d = gpuarray.to_gpu(fdev)
        ferr_d = gpuarray.to_gpu(ferr)
        mask_d = gpuarray.to_gpu(mask)
        addr_d = gpuarray.to_gpu(addr)
        mask_sum_d = gpuarray.to_gpu(mask_sum)

        pbound_set = 0.9
        nFUK = npFourierUpdateKernel(nmodes=total_number_modes, pbound=pbound_set)
        FUK = FourierUpdateKernel(nmodes=total_number_modes, pbound=pbound_set)

        nFUK.configure(fmag, mask, f, addr)
        FUK.configure(fmag, mask, f, addr)

        nFUK.fourier_error(f, fmag, fdev, ferr, mask, mask_sum, addr)

        FUK.fourier_error(f_d, fmag_d, fdev_d, ferr_d, mask_d, mask_sum_d, addr_d)

        expected_fdev = fdev
        measured_fdev = fdev_d.get()
        np.testing.assert_array_equal(expected_fdev, measured_fdev, err_msg="Numpy fdev "
                                                                            "is \n%s, \nbut gpu fdev is \n %s, \n " %  (repr(expected_fdev),
                                                                                                                     repr(measured_fdev)))

        expected_ferr = ferr
        measured_ferr = ferr_d.get()
        np.testing.assert_array_equal(expected_ferr, measured_ferr, err_msg="Numpy ferr"
                                                                            "is \n%s, \nbut gpu ferr is \n %s, \n " %  (repr(expected_ferr),
                                                                                                                     repr(measured_ferr)))

        f_d.gpudata.free()
        fmag_d.gpudata.free()
        fdev_d.gpudata.free()
        ferr_d.gpudata.free()
        mask_d.gpudata.free()
        addr_d.gpudata.free()

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

        # print("address book is:")
        # print(repr(addr))

        '''
        test
        '''
        err_fmag = np.zeros(N, dtype=FLOAT_TYPE)
        mask_sum = mask.sum(-1).sum(-1)

        fdev = np.zeros_like(fmag)
        ferr = np.zeros_like(fmag)
        from ptypy.accelerate.array_based.fourier_update_kernel import FourierUpdateKernel as npFourierUpdateKernel
        f_d = gpuarray.to_gpu(f)
        fmag_d = gpuarray.to_gpu(fmag)
        fdev_d = gpuarray.to_gpu(fdev)
        ferr_d = gpuarray.to_gpu(ferr)
        mask_d = gpuarray.to_gpu(mask)
        addr_d = gpuarray.to_gpu(addr)
        err_fmag_d = gpuarray.to_gpu(err_fmag)
        mask_sum_d = gpuarray.to_gpu(mask_sum)
        pbound_set = 0.9
        nFUK = npFourierUpdateKernel(nmodes=total_number_modes, pbound=pbound_set)
        FUK = FourierUpdateKernel(nmodes=total_number_modes, pbound=pbound_set)

        nFUK.configure(fmag, mask, f, addr)
        FUK.configure(fmag, mask, f, addr)

        nFUK.fourier_error(f, fmag, fdev, ferr, mask, mask_sum, addr)
        nFUK.error_reduce(ferr, err_fmag, addr)

        FUK.fourier_error(f_d, fmag_d, fdev_d, ferr_d, mask_d, mask_sum_d, addr_d)
        FUK.error_reduce(ferr_d, err_fmag_d, addr_d)



        expected_err_fmag = err_fmag
        measured_err_fmag = err_fmag_d.get()
        np.testing.assert_array_equal(expected_err_fmag, measured_err_fmag, err_msg="Numpy err_fmag"
                                                                            "is \n%s, \nbut gpu err_fmag is \n %s, \n " % (
                                                                            repr(expected_err_fmag),
                                                                            repr(measured_err_fmag)))

        f_d.gpudata.free()
        fmag_d.gpudata.free()
        fdev_d.gpudata.free()
        ferr_d.gpudata.free()
        mask_d.gpudata.free()
        addr_d.gpudata.free()
        err_fmag_d.gpudata.free()

if __name__ == '__main__':
    unittest.main()
