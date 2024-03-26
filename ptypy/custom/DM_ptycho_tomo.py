"""
An implementation of the ptycho-tomography using the DM engine

Authors: Benedikt Daurer
"""
import time

import numpy as np

from ..engines import base, projectional, register
from ptypy.utils.tomo import forward_projector_matrix_ptychotomo, forward_projector_matrix_tomo, sirt_projectors2
from ptypy.utils.simulation import refractive_index_map, Ptycho2DSimulation
from ptypy.utils.utils import  scan_view_limits
from ..engines.utils import projection_update_generalized, log_likelihood
from ..core import geometry
from ..core.manager import Full, Vanilla, Bragg3dModel, BlockVanilla, BlockFull
from ..utils import Param
from ..utils.verbose import logger, log
from ..utils import parallel
from .. import io
from .. import utils as u

__all__ = ['WASP']


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

        # Parameters
        self.Nx = 64
        self.Na = 5
        self.Nf = 32
        self.Ns = 76

        self.num_projections = 5
        self.num_positions = 76

        self.pos = self.find_pos()
        # posx_pix, posy_pix = self.pos[1, :], self.pos[0, :]

        # # # Build ptycho-tomo projector
        self.APT = forward_projector_matrix_ptychotomo(self.Nx, self.Na, self.Nf, self.pos)
        #CATR, CATRA = sirt_projectors2(self.APT)
        #self.sirt_update = lambda x,b: (CATR@b - CATRA@x)

        # # Build volume
        self.stacked_views = self.stack_data()
        self.r = np.zeros((self.Nx, self.Nx, self.Nx), dtype=complex).ravel()

    def find_pos(self):
        all_pos = []
        for _, S in self.ptycho.obj.storages.items():
            pos = np.array([v.coord for v in S.views])
            all_pos.append(pos)
        pos = all_pos[0]
        all_pos = []
        return pos.T.astype(np.int32)


    def stack_data(self):
        # stack = np.ones((5, 76, 32, 32), dtype=complex)
        all_views_splitted = np.array_split(
            np.array([v.data for v in self.ptycho.obj.views.values()]),
            5
        )
        return all_views_splitted

    def tomo_update(self, it):

        Nsirt = 1

        #print(self.ptycho.obj.storages.__dir__())
        # print(len(self.ptycho.obj.storages.items()))
            #self.stacked_views = self.stack_data()
        # for v in self.ptycho.obj.views.values():
        #     print(np.shape(v.data))

        # for _, S in self.ptycho.obj.storages.items():
        #     print(np.shape([v.data for v in S.views]))
        #pos = np.array([v.coord for v in self.ptycho.obj.views])
        #    print(len(self.ptycho.obj.views.values()), np.shape(v.data))
        #     print(v.data)

        ##########################################################
        n = np.angle(self.stacked_views) - 1j*np.log(np.abs(self.stacked_views))
        CATR, CATRA = sirt_projectors2(self.APT)
        for k in range(Nsirt):

            print(np.shape(CATRA@self.r))
            print(np.shape(CATR@n.ravel()))
            #r += self.sirt_update(self.r, n.ravel())

        # self.r[~self.m] = 0
        # self.stacked_views = np.exp(1j * (self.APT@self.r).reshape(self.Na, self.Ns, self.Nf, self.Nf))

        ###############################################################
        # initialise exit waves
        # if it == 0:
        #     for j in range(self.num_projections):
        #         self.ptycho.exit[j] = self.ptycho.probe[j] * self.ptycho.obj[j]
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
