'''


'''

import unittest
import numpy as np
from . import perfrun, PyCudaTest, have_pycuda


if have_pycuda():
    from pycuda import gpuarray
    from ptypy.accelerate.cuda_pycuda.kernels import GradientDescentKernel


COMPLEX_TYPE = np.complex64
FLOAT_TYPE = np.float32
INT_TYPE = np.int32


class GradientDescentKernelTest(PyCudaTest):

    def prepare_arrays(self, performance=False):
        if not performance:
            nmodes = 2
            N_buf = 4 
            N = 3 
            A = 3 
        else:
            nmodes = 4
            N_buf = 100
            N = 80
            A =  512
        i_sh = (N, A, A)
        e_sh = (N*nmodes, A, A)
        f_sh = (N_buf, A, A)
        a_sh = (N_buf * nmodes, A, A)
        w = np.ones(i_sh, dtype=FLOAT_TYPE)
        for idx, sl in enumerate(w):
            sl[idx % A, idx % A] = 0.0
        X, Y, Z = np.indices(a_sh, dtype=COMPLEX_TYPE)
        b_f = X + 1j * Y
        b_a = Y + 1j * Z
        b_b = Z + 1j * X
        err_sum = np.zeros((N,), dtype=FLOAT_TYPE)
        fic = np.ones((N,), dtype=FLOAT_TYPE)
        addr = np.zeros((N, nmodes, 5, 3), dtype=INT_TYPE)
        I = np.empty(i_sh, dtype=FLOAT_TYPE)
        I[:] = np.round(np.abs(b_f[:N])**2 % 20)
        for pos_idx in range(N):
            for mode_idx in range(nmodes):
                exit_idx = pos_idx * nmodes + mode_idx
                addr[pos_idx, mode_idx] = np.array([[mode_idx, 0, 0],
                                                    [0, 0, 0],
                                                    [exit_idx, 0, 0],
                                                    [pos_idx, 0, 0],
                                                    [pos_idx, 0, 0]], dtype=INT_TYPE)
        return (gpuarray.to_gpu(b_f),
                gpuarray.to_gpu(b_a),
                gpuarray.to_gpu(b_b),
                gpuarray.to_gpu(I),
                gpuarray.to_gpu(w),
                gpuarray.to_gpu(err_sum),
                gpuarray.to_gpu(addr),
                gpuarray.to_gpu(fic))

    def test_allocate(self):
        b_f, b_a, b_b, I, w, err_sum, addr, fic = self.prepare_arrays()
        GDK = GradientDescentKernel(b_f, addr.shape[1])
        GDK.allocate()

    def test_make_model(self):
        b_f, b_a, b_b, I, w, err_sum, addr, fic = self.prepare_arrays()

        GDK = GradientDescentKernel(b_f, addr.shape[1])
        GDK.allocate()
        GDK.make_model(b_f, addr)

        exp_Imodel = np.array([[[1.,  1.,  1.],
                                [3.,  3.,  3.],
                                [9.,  9.,  9.]],

                               [[13., 13., 13.],
                                [15., 15., 15.],
                                [21., 21., 21.]],

                               [[41., 41., 41.],
                                [43., 43., 43.],
                                [49., 49., 49.]],

                               [[85., 85., 85.],
                                [87., 87., 87.],
                                [93., 93., 93.]]], dtype=FLOAT_TYPE)

        np.testing.assert_array_almost_equal(
            exp_Imodel, GDK.gpu.Imodel.get(),
            err_msg="`Imodel` buffer has not been updated as expected")

    @unittest.skipIf(not perfrun, "performance test")
    def test_make_model_performance(self):
        b_f, b_a, b_b, I, w, err_sum, addr, fic = self.prepare_arrays(performance=True)

        GDK = GradientDescentKernel(b_f, addr.shape[1])
        GDK.allocate()
        GDK.make_model(b_f, addr)

    def test_floating_intensity(self):
        b_f, b_a, b_b, I, w, err_sum, addr, fic = self.prepare_arrays()
        GDK=GradientDescentKernel(b_f, addr.shape[1])
        GDK.allocate()
        GDK.gpu.Imodel[0] = I[0] * 3.
        GDK.gpu.Imodel[1] = I[1] * 2.
        GDK.gpu.Imodel[2] = I[2]
        GDK.floating_intensity(addr, w, I, fic)
        #print('Imodel',repr(GDK.gpu.Imodel))
        #print('fic',repr(1./fic))
        exp_Imodel = np.array([[[0., 0., 0.],
                [1., 1., 1.],
                [4., 4., 4.]],

               [[1., 1., 1.],
                [2., 2., 2.],
                [5., 5., 5.]],

               [[4., 4., 4.],
                [5., 5., 5.],
                [8., 8., 8.]],

               [[0., 0., 0.],
                [0., 0., 0.],
                [0., 0., 0.]]], dtype=np.float32)
        exp_fic=1./np.array([3., 2., 1.], dtype=np.float32)
        np.testing.assert_array_almost_equal(exp_Imodel, GDK.gpu.Imodel.get(),
            err_msg="`Imodel` buffer has not been updated as expected")
        np.testing.assert_array_almost_equal(exp_fic, fic.get(),
            err_msg="floating intensity coeff (fic) has not been updated as expected")

    def test_floating_intensity_high_pass(self):
        b_f, b_a, b_b, I, w, err_sum, addr, fic = self.prepare_arrays()
        GDK=GradientDescentKernel(b_f, addr.shape[1])
        GDK.allocate()
        GDK.gpu.Imodel[0] = I[0] * 3.
        GDK.gpu.Imodel[1] = I[1] * 2.
        GDK.gpu.Imodel[2] = I[2]
        GDK.floating_intensity(addr, w, I, fic, high_pass=True)
        #print('Imodel',repr(GDK.gpu.Imodel))
        #print('fic',repr(1./fic))
        exp_Imodel = np.array([[[0.       , 0.       , 0.       ],
                                [2.3547409, 2.3547409, 2.3547409],
                                [9.418963 , 9.418963 , 9.418963 ]],
                                [[1.670289 , 1.670289 , 1.670289 ],
                                [3.340578 , 3.340578 , 3.340578 ],
                                [8.351445 , 8.351445 , 8.351445 ]],
                                [[4.9385195, 4.9385195, 4.9385195],
                                [6.173149 , 6.173149 , 6.173149 ],
                                [9.877039 , 9.877039 , 9.877039 ]],
                                [[0.       , 0.       , 0.       ],
                                [0.       , 0.       , 0.       ],
                                [0.       , 0.       , 0.       ]]], dtype=np.float32)
        exp_fic=1./np.array([1.2740256 , 1.1973976 , 0.80995935], dtype=np.float32)
        np.testing.assert_array_almost_equal(exp_Imodel, GDK.gpu.Imodel.get(),
            err_msg="`Imodel` buffer has not been updated as expected")
        np.testing.assert_array_almost_equal(exp_fic, fic.get(),
            err_msg="floating intensity coeff (fic) has not been updated as expected")

    def test_make_a012(self):
        b_f, b_a, b_b, I, w, err_sum, addr, fic = self.prepare_arrays()
        GDK = GradientDescentKernel(b_f, addr.shape[1])
        GDK.allocate()
        GDK.make_a012(b_f, b_a, b_b, addr, I, fic)

        exp_A0 = np.array([[[1.,  1.,  1.],
                            [2.,  2.,  2.],
                            [5.,  5.,  5.]],

                           [[12., 12., 12.],
                            [13., 13., 13.],
                            [16., 16., 16.]],

                           [[37., 37., 37.],
                            [38., 38., 38.],
                            [41., 41., 41.]],

                           [[0.,  0.,  0.],
                            [0.,  0.,  0.],
                            [0.,  0.,  0.]]], dtype=FLOAT_TYPE)
        np.testing.assert_array_almost_equal(
            exp_A0, GDK.gpu.Imodel.get(),
            err_msg="`Imodel` buffer (=A0) has not been updated as expected")

        exp_A1 = np.array([[[0., 0., 0.],
                            [2., 6., 10.],
                            [4., 12., 20.]],

                           [[0., 0., 0.],
                            [10., 14., 18.],
                            [20., 28., 36.]],

                           [[0., 0., 0.],
                            [18., 22., 26.],
                            [36., 44., 52.]],

                           [[0., 0., 0.],
                            [0., 0., 0.],
                            [0., 0., 0.]]], dtype=FLOAT_TYPE)
        np.testing.assert_array_almost_equal(
            exp_A1, GDK.gpu.LLerr.get(),
            err_msg="`LLerr` buffer (=A1) has not been updated as expected")

        exp_A2 = np.array([[[0., 4., 12.],
                            [4., 8., 16.],
                            [12., 16., 24.]],

                           [[0., 12., 28.],
                            [12., 24., 40.],
                            [28., 40., 56.]],

                           [[0., 20., 44.],
                            [20., 40., 64.],
                            [44., 64., 88.]],

                           [[0., 0., 0.],
                            [0., 0., 0.],
                            [0., 0., 0.]]], dtype=FLOAT_TYPE)
        np.testing.assert_array_almost_equal(
            exp_A2, GDK.gpu.LLden.get(),
            err_msg="`LLden` buffer (=A2) has not been updated as expected")

    @unittest.skipIf(not perfrun, "performance test")
    def test_make_a012_performance(self):
        b_f, b_a, b_b, I, w, err_sum, addr, fic = self.prepare_arrays(performance=True)

        GDK = GradientDescentKernel(b_f, addr.shape[1])
        GDK.allocate()
        GDK.make_a012(b_f, b_a, b_b, addr, I, fic)

    def test_fill_b(self):
        b_f, b_a, b_b, I, w, err_sum, addr, fic = self.prepare_arrays()
        Brenorm = 0.35
        B = np.zeros((3,), dtype=FLOAT_TYPE)
        B_dev = gpuarray.to_gpu(B)
        GDK = GradientDescentKernel(b_f, addr.shape[1])
        GDK.allocate()
        GDK.make_a012(b_f, b_a, b_b, addr, I, fic)
        GDK.fill_b(addr, Brenorm, w, B_dev)
        B[:] = B_dev.get()

        exp_B = np.array([ 4699.8,  5398.4, 13398.], dtype=FLOAT_TYPE)
        np.testing.assert_allclose(
            B, exp_B,
            rtol=1e-7,
            err_msg="`B` has not been updated as expected")

    @unittest.skipIf(not perfrun, "performance test")
    def test_fill_b_perf(self):
        b_f, b_a, b_b, I, w, err_sum, addr, fic = self.prepare_arrays(performance=True)
        Brenorm = 0.35
        B = np.zeros((3,), dtype=FLOAT_TYPE)
        B_dev = gpuarray.to_gpu(B)
        GDK = GradientDescentKernel(b_f, addr.shape[1])
        GDK.allocate()
        GDK.make_a012(b_f, b_a, b_b, addr, I, fic)
        GDK.fill_b(addr, Brenorm, w, B_dev)
    
    def test_error_reduce(self):
        b_f, b_a, b_b, I, w, err_sum, addr, fic = self.prepare_arrays()
        GDK = GradientDescentKernel(b_f, addr.shape[1])
        GDK.allocate()
        GDK.npy.LLerr = np.indices(GDK.gpu.LLerr.shape, dtype=FLOAT_TYPE)[0]
        GDK.gpu.LLerr = gpuarray.to_gpu(GDK.npy.LLerr)
        GDK.error_reduce(addr, err_sum)

        exp_err = np.array([0.,  9., 18.], dtype=FLOAT_TYPE)
        np.testing.assert_array_almost_equal(
            exp_err, err_sum.get(),
            err_msg="`err_sum` has not been updated as expected")
    
    @unittest.skipIf(not perfrun, "performance test")
    def test_error_reduce_perf(self):
        b_f, b_a, b_b, I, w, err_sum, addr, fic = self.prepare_arrays(performance=True)
        GDK = GradientDescentKernel(b_f, addr.shape[1])
        GDK.allocate()
        GDK.npy.LLerr = np.indices(GDK.gpu.LLerr.shape, dtype=FLOAT_TYPE)[0]
        GDK.gpu.LLerr = gpuarray.to_gpu(GDK.npy.LLerr)
        GDK.error_reduce(addr, err_sum)

    def test_main(self):
        b_f, b_a, b_b, I, w, err_sum, addr, fic = self.prepare_arrays()
        GDK = GradientDescentKernel(b_f, addr.shape[1])
        GDK.allocate()
        GDK.main(b_f, addr, w, I)

        exp_b_f = np.array([[[0. + 0.j,   0. + 0.j,   0. + 0.j],
                             [-0. - 1.j,  -0. - 1.j,  -0. - 1.j],
                             [-0. - 8.j,  -0. - 8.j,  -0. - 8.j]],

                            [[0. + 0.j,   0. + 0.j,   0. + 0.j],
                             [-1. - 1.j,  -1. - 1.j,  -1. - 1.j],
                             [-4. - 8.j,  -4. - 8.j,  -4. - 8.j]],

                            [[-2. + 0.j,  -2. + 0.j,  -2. + 0.j],
                             [-4. - 2.j,  -0. + 0.j,  -4. - 2.j],
                             [-10.-10.j, -10.-10.j, -10.-10.j]],

                            [[-3. + 0.j,  -3. + 0.j,  -3. + 0.j],
                             [-6. - 2.j,  -0. + 0.j,  -6. - 2.j],
                             [-15.-10.j, -15.-10.j, -15.-10.j]],

                            [[-16. + 0.j, -16. + 0.j, -16. + 0.j],
                             [-20. - 5.j, -20. - 5.j, -20. - 5.j],
                             [-32.-16.j, -32.-16.j,  -0. + 0.j]],

                            [[-20. + 0.j, -20. + 0.j, -20. + 0.j],
                             [-25. - 5.j, -25. - 5.j, -25. - 5.j],
                             [-40.-16.j, -40.-16.j,  -0. + 0.j]],

                            [[6. + 0.j,   6. + 0.j,   6. + 0.j],
                             [6. + 1.j,   6. + 1.j,   6. + 1.j],
                             [6. + 2.j,   6. + 2.j,   6. + 2.j]],

                            [[7. + 0.j,   7. + 0.j,   7. + 0.j],
                             [7. + 1.j,   7. + 1.j,   7. + 1.j],
                             [7. + 2.j,   7. + 2.j,   7. + 2.j]]], dtype=COMPLEX_TYPE)
        np.testing.assert_array_almost_equal(
            exp_b_f, b_f.get(),
            err_msg="Auxiliary has not been updated as expected")

        exp_LL = np.array([[[0.,  0.,  0.],
                            [1.,  1.,  1.],
                            [16., 16., 16.]],

                           [[1.,  1.,  1.],
                            [4.,  0.,  4.],
                            [25., 25., 25.]],

                           [[16., 16., 16.],
                            [25., 25., 25.],
                            [64., 64.,  0.]],

                           [[0.,  0.,  0.],
                            [0.,  0.,  0.],
                            [0.,  0.,  0.]]], dtype=FLOAT_TYPE)
        np.testing.assert_array_almost_equal(
            exp_LL, GDK.gpu.LLerr.get(),
            err_msg="LogLikelihood error has not been updated as expected")

    @unittest.skipIf(not perfrun, "performance test")
    def test_main_perf(self):
        b_f, b_a, b_b, I, w, err_sum, addr, fic = self.prepare_arrays(performance=True)
        GDK = GradientDescentKernel(b_f, addr.shape[1])
        GDK.allocate()
        GDK.main(b_f, addr, w, I)

if __name__ == '__main__':
    unittest.main()