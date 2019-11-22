import numpy as np

from .base import BaseKernel


class AuxiliaryWaveKernel(BaseKernel):

    def __init__(self, queue_thread=None):

        super(AuxiliaryWaveKernel, self).__init__(queue_thread)
        self._offset = None
        self.ob_shape = None
        self.pr_shape = None
        self.nviews = None
        self.nmodes = None
        self.ncoords = None
        self.naxes = None
        self.num_pods = None
        self.ocl_wg_size = None

        self.kernels = [
            'build_aux',
            'build_exit',
        ]

    def configure(self, ob, pr, addr, alpha=1.0):
        # changed to be consistent with PoUpdateKernel
        self.alpha = np.float32(alpha)
        self.batch_offset = 0
        self.ob_shape = tuple([np.int32(ax) for ax in ob.shape]) # in Bjoerns version this is only the last two dimensions.
        self.pr_shape = tuple([np.int32(ax) for ax in pr.shape])
        self.nviews, self.nmodes, self.ncoords, self.naxes = addr.shape
        self.num_pods = np.int32(self.nviews * self.nmodes)
        self.nviews, self.nmodes, self.ncoords, self.naxes = addr.shape


    @property
    def batch_offset(self):
        return self._offset

    @batch_offset.setter
    def batch_offset(self, x):
        self._offset = np.int32(x)


    def build_aux(self, aux, ob, pr, ex, addr):

        sh = addr.shape
        flat_addr = addr.reshape(sh[0] * sh[1], sh[2], sh[3])
        off = self.batch_offset
        flat_addr = flat_addr[off:off + aux.shape[0]]
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
        off = self.batch_offset
        flat_addr = flat_addr[off:off + aux.shape[0]]
        rows, cols = ex.shape[-2:]
        for ind, (prc, obc, exc, mac, dic) in enumerate(flat_addr):
            dex = aux[ind, :, :] - \
                  ob[obc[0], obc[1]:obc[1] + rows, obc[2]:obc[2] + cols] * \
                  pr[prc[0], prc[1]:prc[1] + rows, prc[2]:prc[2] + cols]

            ex[exc[0], exc[1]:exc[1] + rows, exc[2]:exc[2] + cols] += dex
            aux[ind, :, :] = dex


