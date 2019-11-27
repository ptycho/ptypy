import numpy as np

from .base import BaseKernel


class FourierUpdateKernel(BaseKernel):

    def __init__(self, queue_thread=None):

        super(FourierUpdateKernel, self).__init__(queue_thread)

        self.f_shape = None
        self.nmodes = None
        self.nviews = None
        self.nmodes = None
        self.ncoords = None
        self.naxes = None
        self.num_pods = None
        self.mask_sum = None
        self.pbound = None
        self.kernels = [
            'fourier_error',
            'error_reduce',
            'fmag_all_update'
        ]

    def configure(self, f, mask, addr, pbound=0.0):
        '''
        [f]
        default = fourier space update array
        type = complex64
        [mask]
        default = mask for the data
        type = float32
        [addr]
        default = the address book for operations
        type = int32
        '''
        self.pbound = np.float32(pbound)
        self.f_shape = tuple([np.int32(ax) for ax in f.shape])
        self.mask_sum = mask.sum(-1).sum(-1)
        self.nviews, self.nmodes, self.ncoords, self.naxes = addr.shape
        self.num_pods = np.int32(self.nviews * self.nmodes)

    def fourier_error(self, f, fmag, fdev, ferr, fmask):
        tf = f.reshape(self.nviews, self.nmodes, self.f_shape[1], self.f_shape[2])
        af = np.sqrt((np.abs(tf) ** 2).sum(1))  # sum down the mode axis?
        fdev[:] = af - fmag
        ferr[:] = fmask * np.abs(fdev) ** 2 / self.mask_sum.reshape((self.nviews, 1, 1))

    def error_reduce(self, ferr, err_fmag):
        err_fmag[:] = ferr.astype(np.double).sum(-1).sum(-1).astype(np.float)

    def calc_fm(self, fm, fmask, fmag, fdev, err_fmag):
        renorm = np.ones_like(err_fmag)
        ind = err_fmag > self.pbound
        renorm[ind] = np.sqrt(self.pbound / err_fmag[ind])
        renorm = renorm.reshape((renorm.shape[0], 1, 1))
        af = fdev + fmag
        fm[:] = (1 - fmask) + fmask * (fmag + fdev * renorm) / (af + 1e-10)

    def fmag_update(self, f, fm):
        tf = f.reshape(self.nviews, self.nmodes, self.f_shape[1], self.f_shape[2])
        tf *= fm.reshape(self.nviews, 1, self.f_shape[1], self.f_shape[2])

    def fmag_all_update(self, f, fmask, fmag, fdev, err_fmag):
        fm = np.ones_like(fmask)
        self.calc_fm(fm, fmask, fmag, fdev, err_fmag)
        self.fmag_update(f, fm)



