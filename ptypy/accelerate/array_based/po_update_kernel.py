import numpy as np

from .base import BaseKernel


class PoUpdateKernel(BaseKernel):

    def __init__(self, queue_thread=None):

        super(PoUpdateKernel, self).__init__(queue_thread)
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
            'pr_update',
            'ob_update',
        ]

    def configure(self, ob, pr, addr):
        '''
        Method works out all the relevant shapes that are required for the calculation
        '''
        self.batch_offset = 0
        self.ob_shape = tuple([np.int32(ax) for ax in ob.shape])
        self.pr_shape = tuple([np.int32(ax) for ax in pr.shape])
        self.nviews, self.nmodes, self.ncoords, self.naxes = addr.shape
        self.num_pods = np.int32(self.nviews * self.nmodes)
        self.ocl_wg_size = (16, 16)

    @property
    def batch_offset(self):
        return self._offset

    @batch_offset.setter
    def batch_offset(self, x):
        self._offset = np.int32(x)

    def ob_update(self, ob, obn, pr, ex, addr):
        '''
        In-place update of object and object denominator
        '''
        sh = addr.shape
        flat_addr = addr.reshape(sh[0] * sh[1], sh[2], sh[3])
        rows, cols = ex.shape[-2:]
        print(rows)
        print(cols)
        for ind, (prc, obc, exc, mac, dic) in enumerate(flat_addr):
            print(obc)
            ob[obc[0], obc[1]:obc[1] + rows, obc[2]:obc[2] + cols] += \
                pr[prc[0], prc[1]:prc[1] + rows, prc[2]:prc[2] + cols].conj() * \
                ex[exc[0], exc[1]:exc[1] + rows, exc[2]:exc[2] + cols]
            obn[obc[0], obc[1]:obc[1] + rows, obc[2]:obc[2] + cols] += \
                pr[prc[0], prc[1]:prc[1] + rows, prc[2]:prc[2] + cols].conj() * \
                pr[prc[0], prc[1]:prc[1] + rows, prc[2]:prc[2] + cols]

    def pr_update(self, pr, prn, ob, ex, addr):
        '''
        In-place update of probe and probe denominator
        '''
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
