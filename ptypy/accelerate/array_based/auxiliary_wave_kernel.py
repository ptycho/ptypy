
import numpy as np
from .base import BaseKernel
from inspect import getfullargspec

class AuxiliaryWaveKernel(BaseKernel):

    def __init__(self, queue_thread=None):

        super(AuxiliaryWaveKernel, self).__init__(queue_thread)
        self.alpha = None
        self.ob_shape = None
        self.nviews = None
        self.nmodes = None
        self.ncoords = None
        self.naxes = None
        self.queue = queue_thread
        self.kernels = [
            'build_aux',
            'build_exit',
        ]

    def configure(self, ob, addr, alpha=1.0):

        self.alpha = np.float32(alpha)
        self.ob_shape = (np.int32(ob.shape[-2]), np.int32(ob.shape[-1]))

        self.nviews, self.nmodes, self.ncoords, self.naxes = [np.int32(ix) for ix in addr.shape]
        self.ocl_wg_size = (1, 1, 32)

    def load(self, aux, ob, pr, ex, addr):

        assert pr.dtype == np.complex64
        assert ex.dtype == np.complex64
        assert aux.dtype == np.complex64
        assert ob.dtype == np.complex64
        assert addr.dtype == np.int32

        self.npy.aux = aux
        self.npy.pr = pr
        self.npy.ob = ob
        self.npy.ex = ex
        self.npy.addr = addr

    def execute(self, kernel_name=None):

        if kernel_name is None:
            for kernel in self.kernels:
                self.execute_npy(kernel)
        else:
            self.log("KERNEL " + kernel_name)
            m_npy = getattr(self, '_npy_' + kernel_name)
            npy_kernel_args = getfullargspec(m_npy).args[1:]
            args = [getattr(self.npy, a) for a in npy_kernel_args]
            m_npy(*args)

        return

    def build_aux(self, aux, ob, pr, ex, addr):

        sh = addr.shape
        flat_addr = addr.reshape(sh[0] * sh[1], sh[2], sh[3])
        rows, cols = ex.shape[-2:]

        for ind, (prc, obc, exc, mac, dic) in enumerate(flat_addr):
            tmp = ob[obc[0], obc[1]:obc[1] + rows, obc[2]:obc[2] + cols] * \
                  pr[prc[0], :, :] * \
                  (1. + self.alpha) - \
                  ex[exc[0], exc[1]:exc[1] + rows, exc[2]:exc[2] + cols] * \
                  self.alpha
            aux[ind, :, :] = tmp

    def build_exit(self, aux, ob, pr, ex, addr):

        sh = addr.shape
        flat_addr = addr.reshape(sh[0] * sh[1], sh[2], sh[3])
        rows, cols = ex.shape[-2:]
        for ind, (prc, obc, exc, mac, dic) in enumerate(flat_addr):
            dex = aux[ind, :, :] - \
                  ob[obc[0], obc[1]:obc[1] + rows, obc[2]:obc[2] + cols] * \
                  pr[prc[0], prc[1]:prc[1] + rows, prc[2]:prc[2] + cols]

            ex[exc[0], exc[1]:exc[1] + rows, exc[2]:exc[2] + cols] += dex
            aux[ind, :, :] = dex


