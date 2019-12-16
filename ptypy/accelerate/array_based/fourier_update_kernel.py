import numpy as np
from .base import BaseKernel
from inspect import getfullargspec


class FourierUpdateKernel(BaseKernel):

    def __init__(self, aux, nmodes=1):

        super(FourierUpdateKernel, self).__init__()
        self.denom = 1e-7
        self.nmodes = np.int32(nmodes)
        ash = aux.shape
        self.fshape = (ash[0] // nmodes, ash[1], ash[2])

        # temporary buffer arrays
        self.npy.fdev = None
        self.npy.ferr = None

        self.kernels = [
            'fourier_error',
            'error_reduce',
            'fmag_all_update'
        ]

    def allocate(self):
        """
        Allocate memory according to the number of modes and
        shape of the diffraction stack.
        """
        # temporary buffer arrays
        self.npy.fdev = np.zeros(self.fshape, dtype=np.float32)
        self.npy.ferr = np.zeros(self.fshape, dtype=np.float32)

    def fourier_error(self, b_aux, addr, mag, mask, mask_sum):
        # reference shape (write-to shape)
        sh = self.fshape
        # stopper
        maxz = mag.shape[0]

        # batch buffers
        fdev = self.npy.fdev[:maxz]
        ferr = self.npy.ferr[:maxz]
        aux = b_aux[:maxz * self.nmodes]

        ## Actual math ##

        # build model from complex fourier magnitudes, summing up
        # all modes incoherently
        tf = aux.reshape(maxz, self.nmodes, sh[1], sh[2])
        af = np.sqrt((np.abs(tf) ** 2).sum(1))

        # calculate difference to real data (g_mag)
        fdev[:] = af - mag

        # Calculate error on fourier magnitudes on a per-pixel basis
        ferr[:] = mask * np.abs(fdev) ** 2 / mask_sum.reshape((maxz, 1, 1))

    def error_reduce(self, addr, err_sum):
        # reference shape (write-to shape)
        sh = self.fshape

        # stopper
        maxz = err_sum.shape[0]

        # batch buffers
        ferr = self.npy.ferr[:maxz]

        ## Actual math ##

        # Reduceses the Fourier error along the last 2 dimensions.fd
        #err_sum[:] = ferr.astype(np.double).sum(-1).sum(-1).astype(np.float)
        err_sum[:] = ferr.sum(-1).sum(-1)

    def fmag_all_update(self, b_aux, addr, mag, mask, err_sum, pbound=0.0):

        sh = self.fshape
        nmodes = self.nmodes

        # stopper
        maxz = mag.shape[0]

        # batch buffers
        fdev = self.npy.fdev[:maxz]
        aux = b_aux[:maxz * nmodes]

        # write-to shape
        ish = aux.shape

        ## Actual math ##

        # local values
        fm = np.ones((maxz, sh[1], sh[2]), np.float32)
        renorm = np.ones((maxz,), np.float32)

        ## As opposed to DM we use renorm to differentiate the cases.

        # pbound >= g_err_sum
        # fm = 1.0 (as renorm = 1, i.e. renorm[~ind])
        # pbound < g_err_sum :
        # fm = (1 - g_mask) + g_mask * (g_mag + fdev * renorm) / (af + 1e-10)
        # (as renorm in [0,1])
        # pbound == 0.0
        # fm = (1 - g_mask) + g_mask * g_mag / (af + 1e-10) (as renorm=0)

        ind = err_sum > pbound
        renorm[ind] = np.sqrt(pbound / err_sum[ind])
        renorm = renorm.reshape((renorm.shape[0], 1, 1))

        af = fdev + mag
        fm[:] = (1 - mask) + mask * (mag + fdev * renorm) / (af + self.denom)

        #fm[:] = mag / (af + 1e-6)
        # upcasting
        aux[:] = (aux.reshape(ish[0] // nmodes, nmodes, ish[1], ish[2]) * fm[:, np.newaxis, :, :]).reshape(ish)
