"""
An implementation of the ptycho-tomography using the DM engine

Authors: Benedikt Daurer
"""
import time

import numpy as np

from ..engines import base, projectional, register
from ..engines.utils import projection_update_generalized, log_likelihood
from ..utils.tomo import AstraTomoWrapperViewBased
from ..core import geometry
from ..core.manager import Full, Vanilla, Bragg3dModel, BlockVanilla, BlockFull
from ..utils import Param
from ..utils.verbose import logger, log
from ..utils import parallel
from .. import io
from .. import utils as u

__all__ = ['DMPtychoTomo']


@register()
class DMPtychoTomo(projectional.DM):
    """
    Ptycho-tomography with DM

    Defaults:

    [name]
    default = DMPtychoTomo
    type = str
    help =
    doc =

    """
    def __init__(self, ptycho_parent, pars=None):
        super().__init__(ptycho_parent, pars)
        # self.Nx = 64
        # self.Na = 5
        # self.Nf = 32
        # self.Ns = 76

        # # # Build volume
        # self.vol = np.zeros((self.Nx, self.Nx, self.Nx), dtype=complex)

        # v0 = list(self.ptycho.obj.views.values())[0]

        # angles = np.linspace(0, np.pi, 5, endpoint=False)
        # angles_dict = {}
        # for i,k in enumerate(self.ptycho.obj.S):
        #     angles_dict[k] = angles[i]

        # # # # Build ptycho-tomo projector
        # self.projector = AstraTomoWrapperViewBased(self.ptycho.obj, vol=self.vol, angles=angles_dict)

    
    def tomo_update(self, it):
        
        # # tomography (backward and forward)
        # xx,yy,zz = np.meshgrid(np.arange(self.Nx)-self.Nx//2, 
        #                        np.arange(self.Nx)-self.Nx//2, 
        #                        np.arange(self.Nx)-self.Nx//2)
        # M = np.sqrt(xx**2 + yy**2 + zz**2) < 10
        # m = M.reshape((self.Nx, self.Nx, self.Nx))

        # self.projector.backward()
        # self.vol[~m] = 0
        # self.projector.forward()
        return


    def engine_iterate(self, num=1):
        """
        Compute `num` iterations.
        """
        to = 0.
        tf = 0.
        tp = 0.
        for it in range(num):
            t1 = time.time()

            # Fourier update
            error_dct = self.fourier_update()

            t2 = time.time()
            tf += t2 - t1

            # Overlap update
            self.overlap_update()

            # Recenter the probe
            self.center_probe()

            t3 = time.time()
            to += t3 - t2

            # Position update
            self.position_update()

            t4 = time.time()
            tp += t4 - t3

            # Tomography update
            # TODO
            self.tomo_update(it)

            # count up
            self.curiter +=1

        logger.info('Time spent in Fourier update: %.2f' % tf)
        logger.info('Time spent in Overlap update: %.2f' % to)
        logger.info('Time spent in Position update: %.2f' % tp)
        return error_dct