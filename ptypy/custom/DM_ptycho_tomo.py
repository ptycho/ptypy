"""
An implementation of the ptycho-tomography using the DM engine

Authors: Benedikt Daurer
"""
import time

import numpy as np

from ..engines import base, projectional, register
from ..engines.utils import projection_update_generalized, log_likelihood
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
        print("here")


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

            # count up
            self.curiter +=1

        logger.info('Time spent in Fourier update: %.2f' % tf)
        logger.info('Time spent in Overlap update: %.2f' % to)
        logger.info('Time spent in Position update: %.2f' % tp)
        return error_dct
