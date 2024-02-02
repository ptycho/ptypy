import numpy as np
from ptypy.utils.verbose import logger, log
from .array_utils import max_abs2, abs2

class Adict(object):

    def __init__(self):
        pass


class BaseKernel(object):

    def __init__(self):
        self.verbose = False
        self.npy = Adict()
        self.benchmark = {}

    def log(self, x):
        if self.verbose:
            print(x)


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
        return

    def fourier_deviation(self, b_aux, addr, mag):
        # reference shape (write-to shape)
        sh = self.fshape
        # stopper
        maxz = mag.shape[0]

        # batch buffers
        fdev = self.npy.fdev[:maxz]
        aux = b_aux[:maxz * self.nmodes]

        ## Actual math ##

        # build model from complex fourier magnitudes, summing up
        # all modes incoherently
        tf = aux.reshape(maxz, self.nmodes, sh[1], sh[2])
        af = np.sqrt((np.abs(tf) ** 2).sum(1))

        # calculate difference to real data (g_mag)
        fdev[:] = af - mag

        return

    def error_reduce(self, addr, err_sum):
        # reference shape (write-to shape)
        sh = self.fshape

        # stopper
        maxz = err_sum.shape[0]

        # batch buffers
        ferr = self.npy.ferr[:maxz]

        ## Actual math ##

        # Reduces the Fourier error along the last 2 dimensions.fd
        #err_sum[:] = ferr.astype(np.double).sum(-1).sum(-1).astype(float)
        err_sum[:] = ferr.sum(-1).sum(-1)
        return

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
        return

    def fmag_update_nopbound(self, b_aux, addr, mag, mask):

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

        af = fdev + mag
        fm[:] = (1 - mask) + mask * mag / (af + self.denom)

        # upcasting
        aux[:] = (aux.reshape(ish[0] // nmodes, nmodes, ish[1], ish[2]) * fm[:, np.newaxis, :, :]).reshape(ish)
        return

    def log_likelihood(self, b_aux, addr, mag, mask, err_phot):
        # reference shape (write-to shape)
        sh = self.fshape
        # stopper
        maxz = mag.shape[0]

        # batch buffers
        aux = b_aux[:maxz * self.nmodes]

        # build model from complex fourier magnitudes, summing up
        # all modes incoherently
        tf = aux.reshape(maxz, self.nmodes, sh[1], sh[2])
        LL = (np.abs(tf) ** 2).sum(1)

        # Intensity data
        I = mag**2

        # Calculate log likelihood error
        err_phot[:] = ((mask * (LL - I)**2 / (I + 1.)).sum(-1).sum(-1) /  np.prod(LL.shape[-2:]))
        return

    def exit_error(self, aux, addr):
        sh = addr.shape
        maxz = sh[0]

        # batch buffers
        ferr = self.npy.ferr[:maxz]
        dex = aux[:maxz * self.nmodes]
        fsh = dex.shape[-2:]
        ferr[:] = (np.abs(dex.reshape((maxz,self.nmodes,fsh[0], fsh[1])))**2).sum(axis=1) / np.prod(fsh)


class GradientDescentKernel(BaseKernel):

    def __init__(self, aux, nmodes=1):

        super(GradientDescentKernel, self).__init__()
        self.denom = 1e-7
        self.nmodes = np.int32(nmodes)
        ash = aux.shape
        self.bshape = ash
        self.fshape = (ash[0] // nmodes, ash[1], ash[2])
        self.ctype = aux.dtype
        self.ftype = np.float32 if self.ctype == np.complex64 else np.float64

        self.npy.LLden  = None
        self.npy.LLerr = None
        self.npy.Imodel = None

        self.npy.float_err1 = None
        self.npy.float_err2 = None

        self.kernels = [
            'make_model',
            'error_reduce',
            'make_a012',
            'fill_b',
            'main',
            'floating_intensity'
        ]

    def allocate(self):
        """
        Allocate memory according to the number of modes and
        shape of the diffraction stack.
        """
        # temporary buffer arrays
        self.npy.LLden = np.zeros(self.fshape, dtype=self.ftype)
        self.npy.LLerr = np.zeros(self.fshape, dtype=self.ftype)
        self.npy.Imodel = np.zeros(self.fshape, dtype=self.ftype)

        self.npy.fic_tmp = np.ones((self.fshape[0],), dtype=self.ftype)

    def make_model(self, b_aux, addr):

        # reference shape (= GPU global dims)
        sh = self.fshape

        # batch buffers
        Imodel = self.npy.Imodel
        aux = b_aux

        ## Actual math ## (subset of FUK.fourier_error)
        tf = aux.reshape(sh[0], self.nmodes, sh[1], sh[2])
        Imodel[:] = ((tf * tf.conj()).real).sum(1)

    def make_a012(self, b_f, b_a, b_b, addr, I, fic):

        # reference shape (= GPU global dims)
        sh = I.shape

        # stopper
        maxz = I.shape[0]

        A0 = self.npy.Imodel
        A1 = self.npy.LLerr
        A2 = self.npy.LLden

        # batch buffers
        f = b_f[:maxz * self.nmodes]
        a = b_a[:maxz * self.nmodes]
        b = b_b[:maxz * self.nmodes]

        ## Actual math ## (subset of FUK.fourier_error)
        fc = fic.reshape((maxz,1,1))
        A0.fill(0.)
        tf = np.real(f * f.conj()).astype(self.ftype)
        A0[:maxz] = np.double(tf.reshape(maxz, self.nmodes, sh[1], sh[2]).sum(1) * fc) - I

        A1.fill(0.)
        tf = 2. * np.real(f * a.conj())
        A1[:maxz] = tf.reshape(maxz, self.nmodes, sh[1], sh[2]).sum(1) * fc

        A2.fill(0.)
        tf = 2. * np.real(f * b.conj()) + np.real(a * a.conj())
        A2[:maxz] = tf.reshape(maxz, self.nmodes, sh[1], sh[2]).sum(1) * fc
        return

    def fill_b(self, addr, Brenorm, w, B):

        # don't know the best dims but this element wise anyway

        # stopper
        maxz = w.shape[0]

        A0 = self.npy.Imodel[:maxz]
        A1 = self.npy.LLerr[:maxz]
        A2 = self.npy.LLden[:maxz]

        ## Actual math ##

        # maybe two kernel calls?

        B[0] += np.dot(w.flat, (A0 ** 2).flat) * Brenorm
        B[1] += np.dot(w.flat, (2 * A0 * A1).flat) * Brenorm
        B[2] += np.dot(w.flat, (A1 ** 2 + 2 * A0 * A2).flat) * Brenorm
        return

    def error_reduce(self, addr, err_sum):

        # reference shape  (= GPU global dims)
        sh = err_sum.shape

        # stopper
        maxz = err_sum.shape[0]

        # batch buffers
        ferr = self.npy.LLerr[:maxz]

        ## Actual math ##

        # Reduces the LL error along the last 2 dimensions.fd
        err_sum[:] = ferr.sum(-1).sum(-1)
        return

    def floating_intensity(self, addr, w, I, fic):

        # reference shape  (= GPU global dims)
        sh = fic.shape

        # stopper
        maxz = fic.shape[0]

        # internal buffers
        num = self.npy.LLerr[:maxz]
        den = self.npy.LLden[:maxz]
        Imodel = self.npy.Imodel[:maxz]
        fic_tmp = self.npy.fic_tmp[:maxz]

        ## math ##
        num[:] = w * Imodel * I
        den[:] = w * Imodel ** 2
        fic[:] = num.sum(-1).sum(-1)
        fic_tmp[:]= den.sum(-1).sum(-1)
        fic/=fic_tmp
        Imodel *= fic.reshape(Imodel.shape[0], 1, 1)

    def main(self, b_aux, addr, w, I):

        nmodes = self.nmodes
        # stopper
        maxz = I.shape[0]

        # batch buffers
        err = self.npy.LLerr[:maxz]
        Imodel = self.npy.Imodel[:maxz]
        aux = b_aux[:maxz*nmodes]

        # write-to shape  (= GPU global dims)
        ish = aux.shape

        ## math ##
        DI = np.double(Imodel) - I
        tmp = w * DI
        err[:] = tmp * DI

        aux[:] = (aux.reshape(ish[0] // nmodes, nmodes, ish[1], ish[2]) * tmp[:, np.newaxis, :, :]).reshape(ish)
        return


class AuxiliaryWaveKernel(BaseKernel):

    def __init__(self):
        super(AuxiliaryWaveKernel, self).__init__()
        self.kernels = [
            'build_aux',
            'build_exit',
        ]

    def allocate(self):
        pass

    def build_aux(self, b_aux, addr, ob, pr, ex, alpha=1.0):
        # DM only, legacy
        self.make_aux(b_aux, addr, ob, pr, ex, 1.+alpha, -alpha)

    def _build_aux(self, b_aux, addr, ob, pr, ex, alpha=1.0):

        sh = addr.shape

        nmodes = sh[1]

        # stopper
        maxz = sh[0]

        # batch buffers
        aux = b_aux[:maxz * nmodes]
        flat_addr = addr.reshape(maxz * nmodes, sh[2], sh[3])
        rows, cols = ex.shape[-2:]
        for ind, (prc, obc, exc, mac, dic) in enumerate(flat_addr):
            tmp = ob[obc[0], obc[1]:obc[1] + rows, obc[2]:obc[2] + cols] * \
                  pr[prc[0], :, :] * \
                  (1. + alpha) - \
                  ex[exc[0], exc[1]:exc[1] + rows, exc[2]:exc[2] + cols] * \
                  alpha
            aux[ind, :, :] = tmp
        return

    def make_aux(self, b_aux, addr, ob, pr, ex, c_po=1.0, c_e=0.0):

        sh = addr.shape

        nmodes = sh[1]

        # stopper
        maxz = sh[0]

        # batch buffers
        aux = b_aux[:maxz * nmodes]
        flat_addr = addr.reshape(maxz * nmodes, sh[2], sh[3])
        rows, cols = ex.shape[-2:]
        for ind, (prc, obc, exc, mac, dic) in enumerate(flat_addr):
            tmp = ob[obc[0], obc[1]:obc[1] + rows, obc[2]:obc[2] + cols] * \
                  pr[prc[0], :, :] * c_po + \
                  ex[exc[0], exc[1]:exc[1] + rows, exc[2]:exc[2] + cols] * c_e
            aux[ind, :, :] = tmp
        return

    def build_exit(self, b_aux, addr, ob, pr, ex, alpha=1):
        self.make_exit(b_aux, addr, ob, pr, ex, 1.0, -alpha, alpha-1)

    def build_exit_alpha_tau(self, b_aux, addr, ob, pr, ex, alpha=1, tau=1):
        self.make_exit(b_aux, addr, ob, pr, ex, tau, 1 - tau * (1 + alpha), tau * alpha - 1)

    def make_exit(self, b_aux, addr, ob, pr, ex, c_a=1.0, c_po=0.0, c_e=-1.0):

        sh = addr.shape

        nmodes = sh[1]

        # stopper
        maxz = sh[0]

        # batch buffers
        aux = b_aux[:maxz * nmodes]

        flat_addr = addr.reshape(maxz * nmodes, sh[2], sh[3])
        rows, cols = ex.shape[-2:]

        for ind, (prc, obc, exc, mac, dic) in enumerate(flat_addr):
            dex = c_a * aux[ind, :, :] + c_po * \
                  ob[obc[0], obc[1]:obc[1] + rows, obc[2]:obc[2] + cols] * \
                  pr[prc[0], prc[1]:prc[1] + rows, prc[2]:prc[2] + cols] + c_e * \
                  ex[exc[0], exc[1]:exc[1] + rows, exc[2]:exc[2] + cols]

            ex[exc[0], exc[1]:exc[1] + rows, exc[2]:exc[2] + cols] += dex
            aux[ind, :, :] = dex
        return

    def _build_exit(self, b_aux, addr, ob, pr, ex, alpha=1):

        sh = addr.shape

        nmodes = sh[1]

        # stopper
        maxz = sh[0]

        # batch buffers
        aux = b_aux[:maxz * nmodes]

        flat_addr = addr.reshape(maxz * nmodes, sh[2], sh[3])
        rows, cols = ex.shape[-2:]

        for ind, (prc, obc, exc, mac, dic) in enumerate(flat_addr):
            dex = aux[ind, :, :] - alpha * \
                  ob[obc[0], obc[1]:obc[1] + rows, obc[2]:obc[2] + cols] * \
                  pr[prc[0], prc[1]:prc[1] + rows, prc[2]:prc[2] + cols] + (alpha - 1) * \
                  ex[exc[0], exc[1]:exc[1] + rows, exc[2]:exc[2] + cols]

            ex[exc[0], exc[1]:exc[1] + rows, exc[2]:exc[2] + cols] += dex
            aux[ind, :, :] = dex
        return

    def _build_exit_alpha_tau(self, b_aux, addr, ob, pr, ex, alpha=1, tau=1):
        sh = addr.shape

        nmodes = sh[1]

        # stopper
        maxz = sh[0]

        # batch buffers
        aux = b_aux[:maxz * nmodes]

        flat_addr = addr.reshape(maxz * nmodes, sh[2], sh[3])
        rows, cols = ex.shape[-2:]

        for ind, (prc, obc, exc, mac, dic) in enumerate(flat_addr):
            dex = tau * aux[ind, :, :] + (tau * alpha - 1) * \
                  ex[exc[0], exc[1]:exc[1] + rows, exc[2]:exc[2] + cols] + \
                  (1 - tau * (1 + alpha)) * \
                  ob[obc[0], obc[1]:obc[1] + rows, obc[2]:obc[2] + cols] * \
                  pr[prc[0], prc[1]:prc[1] + rows, prc[2]:prc[2] + cols]

            ex[exc[0], exc[1]:exc[1] + rows, exc[2]:exc[2] + cols] += dex
            aux[ind, :, :] = dex
        return

    def build_aux_no_ex(self, b_aux, addr, ob, pr, fac=1.0, add=False):

        sh = addr.shape

        nmodes = sh[1]

        # stopper
        maxz = sh[0]

        # batch buffers
        aux = b_aux[:maxz * nmodes]
        flat_addr = addr.reshape(maxz * nmodes, sh[2], sh[3])
        rows, cols = b_aux.shape[-2:]

        for ind, (prc, obc, exc, mac, dic) in enumerate(flat_addr):
            tmp = ob[obc[0], obc[1]:obc[1] + rows, obc[2]:obc[2] + cols] * \
                  pr[prc[0], prc[1]:prc[1] + rows, prc[2]:prc[2] + cols] * fac
            if add:
                aux[ind, :, :] += tmp
            else:
                aux[ind, :, :] = tmp
        return

class PoUpdateKernel(BaseKernel):

    def __init__(self):

        super(PoUpdateKernel, self).__init__()
        self.kernels = [
            'pr_update',
            'ob_update',
        ]

    def allocate(self):
        pass

    def ob_update(self, addr, ob, obn, pr, ex):

        sh = addr.shape
        flat_addr = addr.reshape(sh[0] * sh[1], sh[2], sh[3])
        rows, cols = ex.shape[-2:]
        for ind, (prc, obc, exc, mac, dic) in enumerate(flat_addr):
            ob[obc[0], obc[1]:obc[1] + rows, obc[2]:obc[2] + cols] += \
                pr[prc[0], prc[1]:prc[1] + rows, prc[2]:prc[2] + cols].conj() * \
                ex[exc[0], exc[1]:exc[1] + rows, exc[2]:exc[2] + cols]
            obn[obc[0], obc[1]:obc[1] + rows, obc[2]:obc[2] + cols] += \
                (pr[prc[0], prc[1]:prc[1] + rows, prc[2]:prc[2] + cols].conj() * \
                pr[prc[0], prc[1]:prc[1] + rows, prc[2]:prc[2] + cols]).real
        return

    def pr_update(self, addr, pr, prn, ob, ex):

        sh = addr.shape
        flat_addr = addr.reshape(sh[0] * sh[1], sh[2], sh[3])
        rows, cols = ex.shape[-2:]
        for ind, (prc, obc, exc, mac, dic) in enumerate(flat_addr):
            pr[prc[0], prc[1]:prc[1] + rows, prc[2]:prc[2] + cols] += \
                ob[obc[0], obc[1]:obc[1] + rows, obc[2]:obc[2] + cols].conj() * \
                ex[exc[0], exc[1]:exc[1] + rows, exc[2]:exc[2] + cols]
            prn[prc[0], prc[1]:prc[1] + rows, prc[2]:prc[2] + cols] += \
                (ob[obc[0], obc[1]:obc[1] + rows, obc[2]:obc[2] + cols].conj() * \
                ob[obc[0], obc[1]:obc[1] + rows, obc[2]:obc[2] + cols]).real
        return

    def ob_update_ML(self, addr, ob, pr, ex, fac=2.0):

        sh = addr.shape
        flat_addr = addr.reshape(sh[0] * sh[1], sh[2], sh[3])
        rows, cols = ex.shape[-2:]
        for ind, (prc, obc, exc, mac, dic) in enumerate(flat_addr):
            ob[obc[0], obc[1]:obc[1] + rows, obc[2]:obc[2] + cols] += \
                pr[prc[0], prc[1]:prc[1] + rows, prc[2]:prc[2] + cols].conj() * \
                ex[exc[0], exc[1]:exc[1] + rows, exc[2]:exc[2] + cols] * fac
        return

    def pr_update_ML(self, addr, pr, ob, ex, fac=2.0):

        sh = addr.shape
        flat_addr = addr.reshape(sh[0] * sh[1], sh[2], sh[3])
        rows, cols = ex.shape[-2:]
        for ind, (prc, obc, exc, mac, dic) in enumerate(flat_addr):
            pr[prc[0], prc[1]:prc[1] + rows, prc[2]:prc[2] + cols] += \
                ob[obc[0], obc[1]:obc[1] + rows, obc[2]:obc[2] + cols].conj() * \
                ex[exc[0], exc[1]:exc[1] + rows, exc[2]:exc[2] + cols] * fac
        return

    def ob_update_local(self, addr, ob, pr, ex, aux, prn, a=0., b=1.):
        sh = addr.shape
        flat_addr = addr.reshape(sh[0] * sh[1], sh[2], sh[3])
        rows, cols = ex.shape[-2:]
        pr_norm = (1 - a) * prn.max() + a * prn
        for ind, (prc, obc, exc, mac, dic) in enumerate(flat_addr):
            ob[obc[0], obc[1]:obc[1] + rows, obc[2]:obc[2] + cols] += \
                (a + b) * pr[prc[0], prc[1]:prc[1] + rows, prc[2]:prc[2] + cols].conj() * \
                (ex[exc[0], exc[1]:exc[1] + rows, exc[2]:exc[2] + cols] - aux[ind,:,:]) / \
                pr_norm[dic[0], dic[1]:dic[1] + rows, dic[2]:dic[2] + cols]
        return

    def pr_update_local(self, addr, pr, ob, ex, aux, obn, obn_max, a=0., b=1.):
        sh = addr.shape
        flat_addr = addr.reshape(sh[0] * sh[1], sh[2], sh[3])
        rows, cols = ex.shape[-2:]
        ob_norm = (1 - a) * obn_max + a * obn
        for ind, (prc, obc, exc, mac, dic) in enumerate(flat_addr):
            pr[prc[0], prc[1]:prc[1] + rows, prc[2]:prc[2] + cols] += \
                (a + b) * ob[obc[0], obc[1]:obc[1] + rows, obc[2]:obc[2] + cols].conj() * \
                (ex[exc[0], exc[1]:exc[1] + rows, exc[2]:exc[2] + cols] - aux[ind,:,:]) / \
                ob_norm[dic[0], dic[1]:dic[1] + rows, dic[2]:dic[2] + cols]
        return

    def ob_norm_local(self, addr, ob, obn):
        sh = addr.shape
        flat_addr = addr.reshape(sh[0] * sh[1], sh[2], sh[3])
        rows, cols = obn.shape[-2:]
        obn[:] = 0.
        for ind, (prc, obc, exc, mac, dic) in enumerate(flat_addr):
            # each object mode should only be counted once
            if prc[0] > 0:
                continue
            obn[dic[0],dic[1]:dic[1] + rows, dic[2]:dic[2] + cols] += \
            (ob[obc[0], obc[1]:obc[1] + rows, obc[2]:obc[2] + cols].conj() * \
            ob[obc[0], obc[1]:obc[1] + rows, obc[2]:obc[2] + cols]).real
        return

    def pr_norm_local(self, addr, pr, prn):
        sh = addr.shape
        flat_addr = addr.reshape(sh[0] * sh[1], sh[2], sh[3])
        rows, cols = prn.shape[-2:]
        prn[:] = 0.
        for ind, (prc, obc, exc, mac, dic) in enumerate(flat_addr):
            # each probe mode should only be counted once
            if obc[0] > 0:
                continue
            prn[dic[0],dic[1]:dic[1] + rows, dic[2]:dic[2] + cols] += \
            (pr[prc[0], prc[1]:prc[1] + rows, prc[2]:prc[2] + cols].conj() * \
            pr[prc[0], prc[1]:prc[1] + rows, prc[2]:prc[2] + cols]).real
        return

    def ob_update_wasp(self, addr, ob, pr, ex, aux, ob_sum_nmr, ob_sum_dnm, alpha=1):
        sh = addr.shape
        flat_addr = addr.reshape(sh[0] * sh[1], sh[2], sh[3])
        rows, cols = ex.shape[-2:]

        for ind, (prc, obc, exc, mac, dic) in enumerate(flat_addr):
            pr_conj = pr[prc[0], prc[1]:prc[1] + rows, prc[2]:prc[2] + cols].conj()
            pr_abs2 = abs2(pr[prc[0], prc[1]:prc[1] + rows, prc[2]:prc[2] + cols])
            deltaEW = ex[exc[0], exc[1]:exc[1] + rows, exc[2]:exc[2] + cols] - aux[ind, :, :]

            ob[obc[0], obc[1]:obc[1] + rows, obc[2]:obc[2] + cols] += 0.5 * pr_conj * deltaEW / (pr_abs2.mean() * alpha + pr_abs2)

            ob_sum_nmr[obc[0], obc[1]:obc[1] + rows, obc[2]:obc[2] + cols] += pr_conj * ex[exc[0], exc[1]:exc[1] + rows, exc[2]:exc[2] + cols]
            ob_sum_dnm[obc[0], obc[1]:obc[1] + rows, obc[2]:obc[2] + cols] += pr_abs2

    def pr_update_wasp(self, addr, pr, ob, ex, aux, pr_sum_nmr, pr_sum_dnm, beta=1):
        sh = addr.shape
        flat_addr = addr.reshape(sh[0] * sh[1], sh[2], sh[3])
        rows, cols = ex.shape[-2:]

        for ind, (prc, obc, exc, mac, dic) in enumerate(flat_addr):
            ob_conj = ob[obc[0], obc[1]:obc[1] + rows, obc[2]:obc[2] + cols].conj()
            ob_abs2 = abs2(ob[obc[0], obc[1]:obc[1] + rows, obc[2]:obc[2] + cols])
            deltaEW = ex[exc[0], exc[1]:exc[1] + rows, exc[2]:exc[2] + cols] - aux[ind, :, :]

            pr[prc[0], prc[1]:prc[1] + rows, prc[2]:prc[2] + cols] += ob_conj * deltaEW / (beta + ob_abs2)

            pr_sum_nmr[prc[0], prc[1]:prc[1] + rows, prc[2]:prc[2] + cols] += ob_conj * ex[exc[0], exc[1]:exc[1] + rows, exc[2]:exc[2] + cols]
            pr_sum_dnm[prc[0], prc[1]:prc[1] + rows, prc[2]:prc[2] + cols] += ob_abs2

    def avg_wasp(self, arr, nmr, dnm):
        is_zero = np.isclose(dnm, 0)
        arr[:] = np.where(is_zero, nmr, nmr / dnm)


class PositionCorrectionKernel(BaseKernel):
    from ptypy.accelerate.base import address_manglers

    MANGLERS = {
        'Annealing': address_manglers.RandomIntMangler,
        'GridSearch': address_manglers.GridSearchMangler
    }

    def __init__(self, aux, nmodes, parameters, resolution):
        super(PositionCorrectionKernel, self).__init__()
        ash = aux.shape
        self.fshape = (ash[0] // nmodes, ash[1], ash[2])
        self.npy.ferr = None
        self.npy.fdev = None
        self.addr = None
        self.nmodes = nmodes
        self.param = parameters
        self.nshifts = parameters.nshifts
        self.resolution = resolution
        self.kernels = ['build_aux',
                        'fourier_error',
                        'error_reduce',
                        'update_addr']
        self.setup()

    def setup(self):
        Mangler = self.MANGLERS[self.param.method]
        amplitude = int(np.ceil(self.param.amplitude / self.resolution[0]))
        max_shift = int(np.ceil(self.param.max_shift / self.resolution[0]))
        self.mangler = Mangler(amplitude, self.param.start, self.param.stop,
                               self.param.nshifts, decay=self.param.amplitude_decay,
                               max_bound=max_shift, randomseed=0)

    def allocate(self):
        self.npy.fdev = np.zeros(self.fshape, dtype=np.float32) # we won't use this again but preallocate for speed
        self.npy.ferr = np.zeros(self.fshape, dtype=np.float32)

    def build_aux(self, b_aux, addr, ob, pr):
        """
        different to the AWK, no alpha subtraction. It would be the same, but with alpha permanentaly set to 0.
        """
        sh = addr.shape

        nmodes = sh[1]

        # stopper
        maxz = sh[0]

        # batch buffers
        aux = b_aux[:maxz * nmodes]
        flat_addr = addr.reshape(maxz * nmodes, sh[2], sh[3])
        rows, cols = aux.shape[-2:]

        for ind, (prc, obc, exc, mac, dic) in enumerate(flat_addr):
            dex = ob[obc[0], obc[1]:obc[1] + rows, obc[2]:obc[2] + cols] * pr[prc[0], :, :]
            aux[ind, :, :] = dex

    def fourier_error(self, b_aux, addr, mag, mask, mask_sum):
        """
        Should be identical to that of the FUK, but we don't need fdev out.
        """
        # reference shape (write-to shape)
        sh = self.fshape
        # stopper
        maxz = mag.shape[0]

        # batch buffers
        ferr = self.npy.ferr[:maxz]
        fdev = self.npy.fdev[:maxz]
        aux = b_aux[:maxz * self.nmodes]

        ## Actual math ##

        # build model from complex fourier magnitudes, summing up
        # all modes incoherently
        tf = aux.reshape(maxz, self.nmodes, sh[1], sh[2])
        af = np.sqrt((np.abs(tf) ** 2).sum(1))

        # calculate difference to real data (g_mag)
        fdev[:] = af - mag # we won't reuse this so don't need to keep a persistent buffer

        # Calculate error on fourier magnitudes on a per-pixel basis
        ferr[:] = mask * np.abs(fdev) ** 2 / mask_sum.reshape((maxz, 1, 1))
        return

    def error_reduce(self, addr, err_sum):
        """
        This should the exact same tree reduction as the FUK.
        """
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
        return

    def log_likelihood(self, b_aux, addr, mag, mask, err_sum):
        # reference shape (write-to shape)
        sh = self.fshape
        # stopper
        maxz = mag.shape[0]

        # batch buffers
        aux = b_aux[:maxz * self.nmodes]

        # build model from complex fourier magnitudes, summing up
        # all modes incoherently
        tf = aux.reshape(maxz, self.nmodes, sh[1], sh[2])
        LL = (np.abs(tf) ** 2).sum(1)

        # Intensity data
        I = mag**2

        # Calculate log likelihood error
        err_sum[:] = ((mask * (LL - I)**2 / (I + 1.)).sum(-1).sum(-1) /  np.prod(LL.shape[-2:]))
        return

    def log_likelihood_ml(self, b_aux, addr, I, weights, err_sum):
        # reference shape (write-to shape)
        sh = self.fshape
        # stopper
        maxz = I.shape[0]

        # batch buffers
        aux = b_aux[:maxz * self.nmodes]

        # build model from complex fourier magnitudes, summing up
        # all modes incoherently
        tf = aux.reshape(maxz, self.nmodes, sh[1], sh[2])
        LL = (np.abs(tf) ** 2).sum(1)

        # Calculate log likelihood error
        err_sum[:] = ((weights * (LL - I)**2).sum(-1).sum(-1) /  np.prod(LL.shape[-2:]))
        return

    def update_addr_and_error_state(self, addr, error_state, mangled_addr, err_sum):
        """
        updates the addresses and err state vector corresponding to the smallest error. I think this can be done on the cpu
        """
        update_indices = err_sum < error_state
        #log(4, "Position correction: updating %s indices" % np.sum(update_indices))
        addr[update_indices] = mangled_addr[update_indices]
        error_state[update_indices] = err_sum[update_indices]
