import numpy as np
from collections import OrderedDict


class Adict(object):

    def __init__(self):
        pass


class BaseKernel(object):

    def __init__(self):
        self.verbose = False
        self.npy = Adict()
        self.benchmark = OrderedDict()

    def log(self, x):
        if self.verbose:
            print(x)


class Fourier_update_kernel(BaseKernel):

    def __init__(self):

        super(Fourier_update_kernel, self).__init__()

    def test(self, I, mask, f):
        """
        Test arrays for shape and data type
        """
        assert I.dtype == np.float32
        assert I.shape == self.fshape
        assert mask.dtype == np.float32
        assert mask.shape == self.fshape
        assert f.dtype == np.complex64
        assert f.shape == self.ishape

    def allocate(self, aux, nmodes=1):
        """
        Allocate memory according to the number of modes and
        shape of the diffraction stack.
        """
        self.nmodes = np.int32(nmodes)
        ash = aux.shape
        self.fshape = (ash[0] // nmodes, ash[1], ash[2])

        # temporary buffer arrays
        self.npy.fdev = np.zeros(self.fshape, dtype=np.float32)
        self.npy.ferr = np.zeros(self.fshape, dtype=np.float32)
        self.npy.aux = aux

        self.kernels = [
            'fourier_error',
            'error_reduce',
            'fmag_all_update'
        ]

    def fourier_error(self, g_mag, g_mask, g_mask_sum, offset=0):
        # reference shape (write-to shape)
        sh = self.fshape
        # stopper
        maxz = min(g_mag.shape[0] - offset, sh[0])

        # batch buffers
        fdev = self.npy.fdev[:maxz]
        ferr = self.npy.ferr[:maxz]
        aux = self.npy.aux[:maxz * self.nmodes]

        # slice global arrays for local references
        mag = g_mag[offset:offset + maxz]
        mask_sum = g_mask_sum[offset:offset + maxz]
        mask = g_mask[offset:offset + maxz]

        ## Actual math ##

        # build model from complex fourier magnitudes, summing up 
        # all modes incoherently
        tf = aux.reshape(maxz, self.nmodes, sh[1], sh[2])
        af = np.sqrt((np.abs(tf) ** 2).sum(1))

        # calculate difference to real data (g_mag)
        fdev[:] = af - mag

        # Calculate error on fourier magnitudes on a per-pixel basis
        ferr[:] = mask * np.abs(fdev) ** 2 / mask_sum.reshape((maxz, 1, 1))

    def error_reduce(self, g_err_sum, offset=0):
        # reference shape (write-to shape)
        sh = self.fshape

        # stopper
        maxz = min(g_err_sum.shape[0] - offset, sh[0])

        # batch buffers
        ferr = self.npy.ferr[:maxz]

        # read from slice for global arrays for local references
        error_sum = g_err_sum[offset:offset + maxz]

        ## Actual math ##

        # Reduceses the Fourier error along the last 2 dimensions.fd
        error_sum[:] = ferr.astype(np.double).sum(-1).sum(-1).astype(float)

    def fmag_all_update(self, pbound, g_mag, g_mask, g_err_sum, offset=0):

        sh = self.fshape
        nmodes = self.nmodes

        # stopper
        maxz = min(g_mag.shape[0] - offset, sh[0])

        # batch buffers
        fdev = self.npy.fdev[:maxz]
        aux = self.npy.aux[:maxz * nmodes]

        # slice global arrays for local references
        mag = g_mag[offset:offset + maxz]
        err_sum = g_err_sum[offset:offset + maxz]
        mask = g_mask[offset:offset + maxz]

        ## Actual math ##

        # reference shape (write-to shape)
        ish = aux.shape

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
        fm[:] = (1 - mask) + mask * (mag + fdev * renorm) / (af + 1e-7)

        #fm[:] = mag / (af + 1e-6)
        # upcasting
        aux[:] = (aux.reshape(ish[0] // nmodes, nmodes, ish[1], ish[2]) * fm[:, np.newaxis, :, :]).reshape(ish)

    def build_aux(self, alpha, ob, pr, ex, addr, offset=0):

        sh = addr.shape
        nmodes = sh[1]

        # stopper
        maxz = min(sh[0] - offset, self.fshape[0])

        # batch buffers
        addr = addr[:maxz * nmodes]
        aux = self.npy.aux[:maxz * nmodes]

        flat_addr = addr.reshape(maxz * nmodes, sh[2], sh[3])
        rows, cols = ex.shape[-2:]

        for ind, (prc, obc, exc, mac, dic) in enumerate(flat_addr):
            tmp = ob[obc[0], obc[1]:obc[1] + rows, obc[2]:obc[2] + cols] * \
                  pr[prc[0], :, :] * \
                  (1. + alpha) - \
                  ex[exc[0], exc[1]:exc[1] + rows, exc[2]:exc[2] + cols] * \
                  alpha
            aux[ind, :, :] = tmp

    def build_exit(self, ob, pr, ex, addr, offset=0):

        sh = addr.shape
        nmodes = sh[1]

        # stopper
        maxz = min(sh[0] - offset, self.fshape[0])

        # batch buffers
        addr = addr[:maxz * nmodes]
        aux = self.npy.aux[:maxz * nmodes]

        flat_addr = addr.reshape(maxz * nmodes, sh[2], sh[3])
        rows, cols = ex.shape[-2:]

        for ind, (prc, obc, exc, mac, dic) in enumerate(flat_addr):
            dex = aux[ind, :, :] - \
                  ob[obc[0], obc[1]:obc[1] + rows, obc[2]:obc[2] + cols] * \
                  pr[prc[0], prc[1]:prc[1] + rows, prc[2]:prc[2] + cols]

            ex[exc[0], exc[1]:exc[1] + rows, exc[2]:exc[2] + cols] += dex
            aux[ind, :, :] = dex


class PO_update_kernel(BaseKernel):

    def __init__(self):

        super(PO_update_kernel, self).__init__()

    def allocate(self):
        pass

    def ob_update(self, ob, obn, pr, ex, addr):

        sh = addr.shape
        flat_addr = addr.reshape(sh[0] * sh[1], sh[2], sh[3])
        rows, cols = ex.shape[-2:]
        for ind, (prc, obc, exc, mac, dic) in enumerate(flat_addr):
            ob[obc[0], obc[1]:obc[1] + rows, obc[2]:obc[2] + cols] += \
                pr[prc[0], prc[1]:prc[1] + rows, prc[2]:prc[2] + cols].conj() * \
                ex[exc[0], exc[1]:exc[1] + rows, exc[2]:exc[2] + cols]
            obn[obc[0], obc[1]:obc[1] + rows, obc[2]:obc[2] + cols] += \
                pr[prc[0], prc[1]:prc[1] + rows, prc[2]:prc[2] + cols].conj() * \
                pr[prc[0], prc[1]:prc[1] + rows, prc[2]:prc[2] + cols]
        return

    def pr_update(self, pr, prn, ob, ex, addr):

        sh = addr.shape
        flat_addr = addr.reshape(sh[0] * sh[1], sh[2], sh[3])
        rows, cols = ex.shape[-2:]
        for ind, (prc, obc, exc, mac, dic) in enumerate(flat_addr):
            pr[prc[0], prc[1]:prc[1] + rows, prc[2]:prc[2] + cols] += \
                ob[obc[0], obc[1]:obc[1] + rows, obc[2]:obc[2] + cols].conj() * \
                ex[exc[0], exc[1]:exc[1] + rows, exc[2]:exc[2] + cols]
            prn[prc[0], prc[1]:prc[1] + rows, prc[2]:prc[2] + cols] += \
                ob[obc[0], obc[1]:obc[1] + rows, obc[2]:obc[2] + cols].conj() * \
                ob[obc[0], obc[1]:obc[1] + rows, obc[2]:obc[2] + cols]
        return
