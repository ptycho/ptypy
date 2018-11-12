'''
These tests compare the results from accelerate.array_based.bjoerns_kernels with
accelerate.ocl.npy_kernels

'''

import unittest
import numpy as np
from ptypy.accelerate.ocl import npy_kernels as bjoerns
from ptypy.accelerate.array_based import bjoerns_kernels as mine


class FourierUpdateKernelTest(unittest.TestCase):
    def test_npy_fourier_error(self):
            A = 20  # number of diffraction points
            B = 10  # frame size
            C = 11  # frame size
            D = 1  # number of modes
            pbound = 0.0
            diffraction = np.arange(A*B*C).reshape(A, B, C)

            aaron_kernel = mine.Fourier_update_kernel(pbound=pbound)
            bjoern_kernel = bjoerns.Fourier_update_kernel(pbound=pbound)

            aaron_kernel.allocate(diffraction.shape, nmodes=D)
            bjoern_kernel.allocate(diffraction.shape, nmodes=D)

            aaron_kernel.npy_fourier_error(f, fmag, fdev, ferr, fmask, mask_sum)
            bjoern_kernel.npy_fourier_error(f, fmag, fdev, ferr, fmask, mask_sum)




class AuxiliaryWaveKernelTest(unittest.TestCase):
    def test_build_aux(self):
        return

    def test_build_exit(self):
        return


if __name__ == '__main__':
    unittest.main()

