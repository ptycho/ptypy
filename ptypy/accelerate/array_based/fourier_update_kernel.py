import numpy as np
from .base import BaseKernel
from inspect import getfullargspec


class FourierUpdateKernel(BaseKernel):

    def __init__(self, queue_thread=None, nmodes=1, pbound=0.0):

        super(FourierUpdateKernel, self).__init__(queue_thread)
        self.fshape = None
        self.pbound = np.float32(pbound)
        self.nmodes = np.int32(nmodes)
        self.framesize = None
        self.shape = None
        self.kernels = [
            'fourier_error',
            'error_reduce',
            'fmag_all_update'
        ]

    def configure(self, I, mask, f, addr):
        self.fshape = I.shape
        self.framesize = np.int32(np.prod(I.shape[-2:]))
        print(f.shape)
        assert I.dtype == np.float32
        assert mask.dtype == np.float32
        assert f.dtype == np.complex64

        self.npy.f = f
        self.npy.addr = addr
        self.npy.fmask = mask
        self.npy.mask_sum = mask.sum(-1).sum(-1)
        d = I.copy()
        d[d < 0.] = 0.0  # just in case
        d[np.isnan(d)] = 0.0
        self.npy.fmag = np.sqrt(d)
        self.npy.err_fmag = np.zeros((self.fshape[0],), dtype=np.float32)
        # temporary buffer arrays
        self.npy.fdev = np.zeros_like(self.npy.fmag)
        self.npy.ferr = np.zeros_like(self.npy.fmag)

    def execute(self, kernel_name=None):

        if kernel_name is None:
            for kernel in self.kernels:
                self.execute(kernel)
        else:
            self.log("KERNEL " + kernel_name)
            m_npy = getattr(self, kernel_name)
            npy_kernel_args = getfullargspec(m_npy).args[1:]
            args = [getattr(self.npy, a) for a in npy_kernel_args]
            m_npy(*args)

        return self.npy.err_fmag

    def fourier_error(self, f, fmag, fdev, ferr, fmask, mask_sum, addr):
        sh = f.shape
        tf = f.reshape(sh[0] // self.nmodes, self.nmodes, sh[1], sh[2])

        af = np.sqrt((np.abs(tf) ** 2).sum(1))

        fdev[:] = af - fmag
        ferr[:] = fmask * np.abs(fdev) ** 2 / mask_sum.reshape((mask_sum.shape[0], 1, 1))

    def error_reduce(self, ferr, err_fmag, addr):
        err_fmag[:] = ferr.astype(np.double).sum(-1).sum(-1).astype(np.float)

    def _calc_fm(self, fm, fmask, fmag, fdev, err_fmag, addr):

        renorm = np.ones_like(err_fmag)
        ind = err_fmag > self.pbound
        renorm[ind] = np.sqrt(self.pbound / err_fmag[ind])
        renorm = renorm.reshape((renorm.shape[0], 1, 1))
        af = fdev + fmag
        fm[:] = (1 - fmask) + fmask * (fmag + fdev * renorm) / (af + 1e-7)
        """
        # C Amplitude correction           
        if err_fmag > self.pbound:
            # Power bound is applied
            renorm = np.sqrt(pbound / err_fmag)
            fm = (1 - fmask) + fmask * (fmag + fdev * renorm) / (af + 1e-10)
        else:
            fm = 1.0
        """

    def _fmag_update(self, f, fm, addr):
        sh = f.shape
        tf = f.reshape(sh[0] // self.nmodes, self.nmodes, sh[1], sh[2])
        sh = fm.shape
        tf *= fm.reshape(sh[0], 1, sh[1], sh[2])

    def fmag_all_update(self, f, fmask, fmag, fdev, err_fmag, addr):
        fm = np.ones_like(fmask)
        self._calc_fm(fm, fmask, fmag, fdev, err_fmag, addr)
        self._fmag_update(f, fm, addr)

