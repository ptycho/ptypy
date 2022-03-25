'''


'''

import unittest
import numpy as np
from . import PyCudaTest, have_pycuda
from ptypy import utils as u

if have_pycuda():
    from pycuda import gpuarray
    from ptypy.accelerate.cuda_pycuda.kernels import PositionCorrectionKernel
    from ptypy.accelerate.base.kernels import PositionCorrectionKernel as abPositionCorrectionKernel

COMPLEX_TYPE = np.complex64
FLOAT_TYPE = np.float32
INT_TYPE = np.int32


class PositionCorrectionKernelTest(PyCudaTest):

    def setUp(self):
        PyCudaTest.setUp(self)
        self.params = u.Param()
        self.params.nshifts = 4
        self.params.method = "Annealing"
        self.params.amplitude = 2e-9
        self.params.start = 0
        self.params.stop = 10
        self.params.max_shift = 2e-9
        self.params.amplitude_decay = True
        self.resolution = [1e-9,1e-9]

    def update_addr_and_error_state_UNITY_helper(self, size, modes):
        ## Arrange
        addr = np.ones((size, modes, 5, 3), dtype=np.int32)
        mangled_addr = 2 * addr
        err_state = np.zeros((size,), dtype=np.float32)
        err_state[5:] = 2.
        err_sum = np.ones((size, ), dtype=np.float32)
        addr_gpu = gpuarray.to_gpu(addr)
        mangled_addr_gpu = gpuarray.to_gpu(mangled_addr)
        err_state_gpu = gpuarray.to_gpu(err_state)
        err_sum_gpu = gpuarray.to_gpu(err_sum)
        aux = np.ones((1,1,1), dtype=np.complex64)

        ## Act
        PCK = PositionCorrectionKernel(aux, modes, self.params, self.resolution, queue_thread=self.stream)
        PCK.update_addr_and_error_state(addr_gpu, err_state_gpu, mangled_addr_gpu, err_sum_gpu)
        abPCK = abPositionCorrectionKernel(aux, modes, self.params, self.resolution)
        abPCK.update_addr_and_error_state(addr, err_state, mangled_addr, err_sum)

        ## Assert
        np.testing.assert_array_equal(addr_gpu.get(), addr)
        np.testing.assert_array_equal(err_state_gpu.get(), err_state)

    def test_update_addr_and_error_state_UNITY_small_onemode(self):
        self.update_addr_and_error_state_UNITY_helper(4, 1)

    def test_update_addr_and_error_state_UNITY_large_onemode(self):
        self.update_addr_and_error_state_UNITY_helper(323, 1)
    
    def test_update_addr_and_error_state_UNITY_small_multimode(self):
        self.update_addr_and_error_state_UNITY_helper(4, 3)

    def test_update_addr_and_error_state_UNITY_large_multimode(self):
        self.update_addr_and_error_state_UNITY_helper(323, 3)

    def log_likelihood_ml_UNITY(self):
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
        I = fmag**2

        mask = np.empty(shape=(N, B, C),
                        dtype=FLOAT_TYPE)  # the masks for the measured magnitudes either 1xAxB or NxAxB
        mask_fill = np.ones_like(mask)
        mask_fill[::2, ::2] = 0  # checkerboard for testing
        mask[:] = mask_fill
        w = mask /(I+1.)

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
        f_d = gpuarray.to_gpu(f)
        w_d = gpuarray.to_gpu(w)
        I_d = gpuarray.to_gpu(I)
        addr_d = gpuarray.to_gpu(addr)
        LLerr_d = gpuarray.to_gpu(LLerr)

        ## Act
        PCK = PositionCorrectionKernel(f, total_number_modes, self.params, self.resolution, queue_thread=self.stream)
        abPCK = abPositionCorrectionKernel(f, total_number_modes, self.params, self.resolution)
        abPCK.log_likelihood_ml(f, addr, I, w, LLerr)
        PCK.log_likelihood_ml(f_d, addr_d, I_d, w_d, LLerr_d)

        expected_err_phot = LLerr
        measured_err_phot = LLerr_d.get()

        np.testing.assert_allclose(expected_err_phot, measured_err_phot, err_msg="Numpy log-likelihood error "
                                                                                 "is \n%s, \nbut gpu log-likelihood error is \n%s, \n " % (
                                                                                 repr(expected_err_phot),
                                                                                 repr(measured_err_phot)), rtol=1e-5)
