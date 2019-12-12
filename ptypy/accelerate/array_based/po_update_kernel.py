import numpy as np
from .base import BaseKernel
from inspect import getfullargspec


class PoUpdateKernel(BaseKernel):

    def __init__(self, queue_thread=None):

        super(PoUpdateKernel, self).__init__(queue_thread)
        self.ob_shape = None
        self.pr_shape = None
        self.nviews = None
        self.nmodes = None
        self.ncoords = None
        self.nmodes = None
        self.num_pods = None

        self.kernels = [
            'pr_update',
            'ob_update',
        ]

    def configure(self, ob, pr, addr):

        self.ob_shape = tuple([np.int32(ax) for ax in ob.shape])
        self.pr_shape = tuple([np.int32(ax) for ax in pr.shape])

        self.nviews, self.nmodes, self.ncoords, self.naxes = addr.shape
        self.num_pods = np.int32(self.nviews * self.nmodes)


    def load(self, obn, prn, ob, pr, ex, addr):
        assert pr.dtype == np.complex64
        assert ex.dtype == np.complex64
        assert ob.dtype == np.complex64
        assert addr.dtype == np.int32

        self.npy.pr = pr
        self.npy.prn = prn
        self.npy.ob = ob
        self.npy.obn = obn
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

    def ob_update(self, ob, obn, pr, ex, addr):
        obsh = self.ob_shape
        prsh = self.pr_shape
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

    def pr_update(self, pr, prn, ob, ex, addr):
        obsh = self.ob_shape
        prsh = self.pr_shape
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


