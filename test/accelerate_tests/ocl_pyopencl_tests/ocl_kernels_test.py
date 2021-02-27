'''


'''

import unittest
import numpy as np

try:
    import pyopencl as pocl
    from pyopencl import array as cla
    from ptypy.accelerate.ocl.ocl_kernels import AuxiliaryWaveKernel, FourierUpdateKernel
    # from ptypy.accelerate.ocl.npy_kernels_for_block import AuxiliaryWaveKernel
    from ptypy.accelerate.ocl import get_ocl_queue
    have_ocl = True
except ImportError:
    have_ocl = False

COMPLEX_TYPE = np.complex64
FLOAT_TYPE = np.float32
INT_TYPE = np.int32

@unittest.skipIf(not have_ocl, "no PyOpenCL or GPU drivers available")
class AuxiliaryWaveKernelTest(unittest.TestCase):

    def setUp(self):
        import sys
        np.set_printoptions(threshold=sys.maxsize, linewidth=np.inf)
        self.queue = get_ocl_queue()

    def tearDown(self):
        np.set_printoptions()
        del self.queue

    def test_init(self):
        attrs = []

        AWK = AuxiliaryWaveKernel(self.queue)
        self.queue.finish()
        for attr in attrs:
            self.assertTrue(hasattr(AWK, attr), msg="AuxiliaryWaveKernel does not have attribute: %s" % attr)

        np.testing.assert_equal(AWK.kernels,
                                ['build_aux', 'build_exit'],
                                err_msg='AuxiliaryWaveKernel does not have the correct functions registered.')

    def _configure(self):
        B = 4  # frame size y
        C = 4  # frame size x

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

        pr_npy = np.empty(shape=(D, E, F), dtype=COMPLEX_TYPE)
        for idx in range(D):
            pr_npy[idx] = np.ones((E, F)) * (idx + 1) + 1j * np.ones((E, F)) * (idx + 1)

        ob_npy = np.empty(shape=(G, H, I), dtype=COMPLEX_TYPE)
        for idx in range(G):
            ob_npy[idx] = np.ones((H, I)) * (3 * idx + 1) + 1j * np.ones((H, I)) * (3 * idx + 1)

        ex_npy = np.empty(shape=(A, B, C), dtype=COMPLEX_TYPE)
        for idx in range(A):
            ex_npy[idx] = np.ones((B, C)) * (idx + 1) + 1j * np.ones((B, C)) * (idx + 1)

        X, Y = np.meshgrid(range(scan_pts), range(scan_pts))
        addr = np.zeros((total_number_scan_positions, total_number_modes, 5, 3), dtype=INT_TYPE)

        exit_idx = 0
        position_idx = 0
        for xpos, ypos in zip(Y.flat, X.flat):
            mode_idx = 0
            for pr_mode in range(D):
                for ob_mode in range(G):
                    addr[position_idx, mode_idx] = np.array([[pr_mode, 0, 0],
                                                             [ob_mode, ypos, xpos],
                                                             [exit_idx, 0, 0],
                                                             [0, 0, 0],
                                                             [0, 0, 0]], dtype=INT_TYPE)
                    exit_idx += 1
                    mode_idx += 1
            position_idx += 1

        return pr_npy, ob_npy, ex_npy, addr

    def test_build_aux_unity(self):
        '''
        test
        '''
        pr_npy, ob_npy, ex_npy, addr = self._configure()
        aux_npy = np.zeros_like(ex_npy)
        from ptypy.accelerate.ocl.npy_kernels_for_block import AuxiliaryWaveKernel as npAuxiliaryWaveKernel
        nAWK = npAuxiliaryWaveKernel()
        AWK = AuxiliaryWaveKernel(self.queue)
        AWK.ocl_wg_size = None  # (1, 3, 3) #None
        alpha = 1.0

        ob_dev = cla.to_device(self.queue, ob_npy)
        pr_dev = cla.to_device(self.queue, pr_npy)
        addr_dev = cla.to_device(self.queue, addr)
        aux_dev = cla.to_device(self.queue, aux_npy)
        ex_dev = cla.to_device(self.queue, ex_npy)
        self.queue.finish()
        AWK.build_aux(aux_dev, addr_dev, ob_dev, pr_dev, ex_dev, alpha)
        nAWK.build_aux(aux_npy, addr, ob_npy, pr_npy, ex_npy, alpha)
        d = aux_dev.get()
        np.testing.assert_array_equal(aux_npy, aux_dev.get(),
                                      err_msg="The gpu auxiliary_wave does not look the same as the numpy version")

    def test_build_aux_capped_unity(self):
        '''
        test
        '''
        pr_npy, ob_npy, ex_npy, addr = self._configure()
        aux_npy = np.zeros_like(ex_npy)
        # now use only a part of the stacks
        sh = addr.shape
        addr = addr[:sh[0] // 2, ...]
        ex_npy = ex_npy[:sh[0] // 2 * sh[1], ...]
        from ptypy.accelerate.ocl.npy_kernels_for_block import AuxiliaryWaveKernel as npAuxiliaryWaveKernel
        nAWK = npAuxiliaryWaveKernel()
        AWK = AuxiliaryWaveKernel(self.queue)
        AWK.ocl_wg_size = None  # (1, 3, 3) #None
        alpha = 1.0

        ob_dev = cla.to_device(self.queue, ob_npy)
        pr_dev = cla.to_device(self.queue, pr_npy)
        addr_dev = cla.to_device(self.queue, addr)
        aux_dev = cla.to_device(self.queue, aux_npy)
        ex_dev = cla.to_device(self.queue, ex_npy)
        self.queue.finish()
        AWK.build_aux(aux_dev, addr_dev, ob_dev, pr_dev, ex_dev, alpha)
        nAWK.build_aux(aux_npy, addr, ob_npy, pr_npy, ex_npy, alpha)
        d = aux_dev.get()
        np.testing.assert_array_equal(aux_npy, aux_dev.get(),
                                      err_msg="The gpu auxiliary_wave does not look the same as the numpy version")

    def test_build_exit_unity(self):
        '''
        test
        '''
        pr_npy, ob_npy, ex_npy, addr = self._configure()
        aux_npy = np.zeros_like(ex_npy)
        from ptypy.accelerate.ocl.npy_kernels_for_block import AuxiliaryWaveKernel as npAuxiliaryWaveKernel
        nAWK = npAuxiliaryWaveKernel()
        AWK = AuxiliaryWaveKernel(self.queue)
        AWK.ocl_wg_size = None  # (1, 3, 3) #None

        ob_dev = cla.to_device(self.queue, ob_npy)
        pr_dev = cla.to_device(self.queue, pr_npy)
        addr_dev = cla.to_device(self.queue, addr)
        aux_dev = cla.to_device(self.queue, aux_npy)
        ex_dev = cla.to_device(self.queue, ex_npy)
        self.queue.finish()
        AWK.build_exit(aux_dev, addr_dev, ob_dev, pr_dev, ex_dev)
        nAWK.build_exit(aux_npy, addr, ob_npy, pr_npy, ex_npy)
        np.testing.assert_array_equal(aux_npy, aux_dev.get(),
                                      err_msg="The gpu auxiliary_wave does not look the same as the numpy version")

    def test_build_exit_capped_unity(self):
        '''
        test
        '''
        pr_npy, ob_npy, ex_npy, addr = self._configure()
        aux_npy = np.zeros_like(ex_npy)
        # now use only a part of the stacks
        sh = addr.shape
        addr = addr[:sh[0] // 2, ...]
        ex_npy = ex_npy[:sh[0] // 2 * sh[1], ...]
        from ptypy.accelerate.ocl.npy_kernels_for_block import AuxiliaryWaveKernel as npAuxiliaryWaveKernel
        nAWK = npAuxiliaryWaveKernel()
        AWK = AuxiliaryWaveKernel(self.queue)
        AWK.ocl_wg_size = None  # (1, 3, 3) #None

        ob_dev = cla.to_device(self.queue, ob_npy)
        pr_dev = cla.to_device(self.queue, pr_npy)
        addr_dev = cla.to_device(self.queue, addr)
        aux_dev = cla.to_device(self.queue, aux_npy)
        ex_dev = cla.to_device(self.queue, ex_npy)
        self.queue.finish()
        AWK.build_exit(aux_dev, addr_dev, ob_dev, pr_dev, ex_dev)
        nAWK.build_exit(aux_npy, addr, ob_npy, pr_npy, ex_npy)
        np.testing.assert_array_equal(aux_npy, aux_dev.get(),
                                      err_msg="The gpu auxiliary_wave does not look the same as the numpy version")

@unittest.skipIf(not have_ocl, "no PyOpenCL or GPU drivers available")
class FourierUpdateKernelTest(unittest.TestCase):

    def setUp(self):
        import sys
        np.set_printoptions(threshold=sys.maxsize, linewidth=np.inf)
        self.queue = get_ocl_queue()

    def tearDown(self):
        np.set_printoptions()
        del self.queue

    def test_init(self):
        attrs = []
        aux = np.zeros((16, 3, 3), dtype=COMPLEX_TYPE)
        nmodes = 4
        FUK = FourierUpdateKernel(aux, nmodes, self.queue)
        self.queue.finish()
        for attr in attrs:
            self.assertTrue(hasattr(FUK, attr), msg="AuxiliaryWaveKernel does not have attribute: %s" % attr)

    def test_all_capped_unity(self):
        '''
        test
        '''
        nmodes = 2
        # pr_npy, ob_npy, ex_npy, addr_npy = self._configure()
        addr_npy = np.zeros((4, nmodes, 5, 3), dtype=INT_TYPE)
        shape = (4, 3, 3)
        L, M, N = shape
        fshape = shape
        shape = (nmodes * L, M, N)

        X, Y, Z = np.indices(shape)
        aux_npy = (1j*Z+X+Z).astype(COMPLEX_TYPE) * 200
        mag_npy = np.indices(fshape).sum(0).astype(FLOAT_TYPE) * 200 ** 2 * nmodes
        ma_npy = (mag_npy > 10).astype(FLOAT_TYPE)
        err_fourier_npy = np.zeros((L,), dtype=FLOAT_TYPE)
        mask_sum_npy = ma_npy.sum(-1).sum(-1)

        from ptypy.accelerate.ocl.npy_kernels_for_block import FourierUpdateKernel as nFourierUpdateKernel
        nFUK = nFourierUpdateKernel(aux_npy, nmodes)
        FUK = FourierUpdateKernel(aux_npy, nmodes, self.queue)
        FUK.ocl_wg_size = None  # (1, 3, 3) #None

        FUK.allocate()
        nFUK.allocate()
        self.queue.finish()

        # now use only a part of the stacks
        sh = addr_npy.shape
        mag_npy = mag_npy[:sh[0] // 2, ...]
        ma_npy = ma_npy[:sh[0] // 2, ...]
        addr_npy = addr_npy[:sh[0] // 2, ...]
        mask_sum_npy = mask_sum_npy[:sh[0] // 2, ...]
        err_fourier_npy = err_fourier_npy[:sh[0] // 2, ...]

        # copy
        mag_dev = cla.to_device(self.queue, mag_npy)
        ma_dev = cla.to_device(self.queue, ma_npy)
        aux_dev = cla.to_device(self.queue, aux_npy)
        addr_dev = cla.to_device(self.queue, addr_npy)
        mask_sum_dev = cla.to_device(self.queue, mask_sum_npy)
        err_fourier_dev = cla.to_device(self.queue, err_fourier_npy)

        self.queue.finish()
        FUK.fourier_error(aux_dev, addr_dev, mag_dev, ma_dev, mask_sum_dev)
        FUK.error_reduce(addr_dev, err_fourier_dev)
        FUK.fmag_all_update(aux_dev, addr_dev, mag_dev, ma_dev, err_fourier_dev, pbound=0.5)
        nFUK.fourier_error(aux_npy, addr_npy, mag_npy, ma_npy, mask_sum_npy)
        nFUK.error_reduce(addr_npy, err_fourier_npy)
        nFUK.fmag_all_update(aux_npy, addr_npy, mag_npy, ma_npy, err_fourier_npy, pbound=0.5)
        np.testing.assert_array_almost_equal_nulp(nFUK.npy.fdev, FUK.npy.fdev.get())
              #  err_msg="The gpu fdev differs more than single precision allows.")
        np.testing.assert_array_almost_equal_nulp(nFUK.npy.ferr, FUK.npy.ferr.get())
              #  err_msg="The gpu ferr differs more than single precision allows.")
        np.testing.assert_array_almost_equal_nulp(aux_npy, aux_dev.get(), 80) #,
              # err_msg="The gpu auxiliary_wave differs more than single precision allows.")

    """
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

        ex_npy = np.empty(shape=(A, B, C), dtype=COMPLEX_TYPE)
        for idx in range(A):
            ex_npy[idx] = np.ones((B, C)) * (idx + 1) + 1j * np.ones((B, C)) * (idx + 1)

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
        auxiliary_wave = np.zeros_like(ex_npy)

        AWK = AuxiliaryWaveKernel()
        alpha_set = 1.0
        AWK.configure(object_array, addr, alpha=alpha_set)

        object_array_dev = gpuarray.to_gpu(object_array)
        probe_dev = gpuarray.to_gpu(probe)
        addr_dev = gpuarray.to_gpu(addr)
        auxiliary_wave_dev = gpuarray.to_gpu(auxiliary_wave)
        ex_npy_dev = gpuarray.to_gpu(ex_npy)

        AWK.build_aux(auxiliary_wave_dev, object_array_dev, probe_dev, ex_npy_dev, addr_dev)


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
        ex_npy_dev.gpudata.free()
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

        ex_npy = np.empty(shape=(A, B, C), dtype=COMPLEX_TYPE)
        for idx in range(A):
            ex_npy[idx] = np.ones((B, C)) * (idx + 1) + 1j * np.ones((B, C)) * (idx + 1)

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
        auxiliary_wave = np.zeros_like(ex_npy)
        from ptypy.accelerate.base.auxiliary_wave_kernel import AuxiliaryWaveKernel as npAuxiliaryWaveKernel
        nAWK = npAuxiliaryWaveKernel()
        AWK = AuxiliaryWaveKernel()
        alpha_set = 1.0
        AWK.configure(object_array, addr, alpha=alpha_set)
        nAWK.configure(object_array,  addr, alpha=alpha_set)

        object_array_dev = gpuarray.to_gpu(object_array)
        probe_dev = gpuarray.to_gpu(probe)
        addr_dev = gpuarray.to_gpu(addr)
        auxiliary_wave_dev = gpuarray.to_gpu(auxiliary_wave)
        ex_npy_dev = gpuarray.to_gpu(ex_npy)

        AWK.build_aux(auxiliary_wave_dev, object_array_dev, probe_dev, ex_npy_dev, addr_dev)
        nAWK.build_aux(auxiliary_wave, object_array, probe, ex_npy, addr)


        np.testing.assert_array_equal(auxiliary_wave, auxiliary_wave_dev.get(),
                                      err_msg="The gpu auxiliary_wave does not look the same as the numpy version")

        object_array_dev.gpudata.free()
        auxiliary_wave_dev.gpudata.free()
        probe_dev.gpudata.free()
        ex_npy_dev.gpudata.free()
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

        ex_npy = np.empty(shape=(A, B, C), dtype=COMPLEX_TYPE)
        for idx in range(A):
            ex_npy[idx] = np.ones((B, C)) * (idx + 1) + 1j * np.ones((B, C)) * (idx + 1)

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
        auxiliary_wave = np.zeros_like(ex_npy)

        object_array_dev = gpuarray.to_gpu(object_array)
        probe_dev = gpuarray.to_gpu(probe)
        addr_dev = gpuarray.to_gpu(addr)
        auxiliary_wave_dev = gpuarray.to_gpu(auxiliary_wave)
        ex_npy_dev = gpuarray.to_gpu(ex_npy)
        AWK = AuxiliaryWaveKernel()

        alpha_set = 1.0
        AWK.configure(object_array, addr, alpha=alpha_set)

        AWK.build_exit(auxiliary_wave_dev, object_array_dev, probe_dev, ex_npy_dev, addr_dev)
        #
        # print("auxiliary_wave after")
        # print(repr(auxiliary_wave_dev.get()))
        #
        # print("ex_npy after")
        # print(repr(ex_npy))

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

        expected_ex_npy = np.array([[[1. - 1.j,  1. - 1.j,  1. - 1.j],
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

        np.testing.assert_array_equal(expected_ex_npy, ex_npy_dev.get(),
                                      err_msg="The ex_npy has not been updated as expected")

        object_array_dev.gpudata.free()
        auxiliary_wave_dev.gpudata.free()
        probe_dev.gpudata.free()
        ex_npy_dev.gpudata.free()
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

        ex_npy = np.empty(shape=(A, B, C), dtype=COMPLEX_TYPE)
        for idx in range(A):
            ex_npy[idx] = np.ones((B, C)) * (idx + 1) + 1j * np.ones((B, C)) * (idx + 1)

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
        auxiliary_wave = np.zeros_like(ex_npy)

        object_array_dev = gpuarray.to_gpu(object_array)
        probe_dev = gpuarray.to_gpu(probe)
        addr_dev = gpuarray.to_gpu(addr)
        auxiliary_wave_dev = gpuarray.to_gpu(auxiliary_wave)
        ex_npy_dev = gpuarray.to_gpu(ex_npy)

        from ptypy.accelerate.base.auxiliary_wave_kernel import AuxiliaryWaveKernel as npAuxiliaryWaveKernel
        nAWK = npAuxiliaryWaveKernel()

        AWK = AuxiliaryWaveKernel()

        alpha_set = 1.0

        AWK.configure(object_array, addr, alpha=alpha_set)
        nAWK.configure(object_array, addr, alpha=alpha_set)

        AWK.build_exit(auxiliary_wave_dev, object_array_dev, probe_dev, ex_npy_dev, addr_dev)
        nAWK.build_exit(auxiliary_wave, object_array, probe, ex_npy, addr)

        np.testing.assert_array_equal(auxiliary_wave, auxiliary_wave_dev.get(),
                                      err_msg="The gpu auxiliary_wave does not look the same as the numpy version")

        np.testing.assert_array_equal(ex_npy, ex_npy_dev.get(),
                                      err_msg="The gpu ex_npy does not look the same as the numpy version")

        object_array_dev.gpudata.free()
        auxiliary_wave_dev.gpudata.free()
        probe_dev.gpudata.free()
        ex_npy_dev.gpudata.free()
        addr_dev.gpudata.free()
    """


if __name__ == '__main__':
    unittest.main()
