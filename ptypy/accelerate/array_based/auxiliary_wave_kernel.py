
import numpy as np
from .base import BaseKernel
from inspect import getfullargspec


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

    def build_exit(self, b_aux, addr, ob, pr, ex):

        sh = addr.shape

        nmodes = sh[1]

        # stopper
        maxz = sh[0]

        # batch buffers
        aux = b_aux[:maxz * nmodes]

        flat_addr = addr.reshape(maxz * nmodes, sh[2], sh[3])
        rows, cols = ex.shape[-2:]

        for ind, (prc, obc, exc, mac, dic) in enumerate(flat_addr):
            dex = aux[ind, :, :] - \
                  ob[obc[0], obc[1]:obc[1] + rows, obc[2]:obc[2] + cols] * \
                  pr[prc[0], prc[1]:prc[1] + rows, prc[2]:prc[2] + cols]

            ex[exc[0], exc[1]:exc[1] + rows, exc[2]:exc[2] + cols] += dex
            aux[ind, :, :] = dex
