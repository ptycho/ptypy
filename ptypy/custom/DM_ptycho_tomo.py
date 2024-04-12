"""
An implementation of the ptycho-tomography using the DM engine

Authors: Benedikt Daurer
"""
import time

import numpy as np

from ..engines import base, projectional, register
from ..engines.utils import projection_update_generalized, log_likelihood
from ..utils.tomo import AstraTomoWrapperViewBased, AstraTomoWrapperProjectionBased
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
        pshape = list(self.ptycho.obj.S.values())[0].data.shape[-1]
        n_angles = len(self.ptycho.obj.S)

        # # # Build volume
        self.vol = np.zeros((pshape, pshape, pshape), dtype=np.complex64)

        angles = np.linspace(0, np.pi, n_angles, endpoint=True)
        self.angles_dict = {}
        for i,k in enumerate(self.ptycho.obj.S):
            self.angles_dict[k] = angles[i]

    def engine_prepare(self):
        super().engine_prepare()
        
        # Can be changed into ProjectionBased
        self.projector = AstraTomoWrapperViewBased (    
            obj=self.ptycho.obj, 
            vol=self.vol, 
            angles=self.angles_dict, 
            obj_is_refractive_index=False, 
            mask_threshold=35
            )

    def tomo_update(self):
    
        self.projector.apply_phase_shift_to_obj()
        self.projector.do_conversion_and_set_proj_array()

        self.projector.apply_mask_to_proj_array()
        self.projector._create_astra_proj_array()

        # Backward 
        self.projector.backward(type='SIRT3D_CUDA', iter=100) 

        #Forward
        self.projector.forward(iter=100)

        return self.projector._vol


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

            # Tomography update (only starts after 30it of ptycho)
            iter = None
            start_tomo_update = False
            try:
                iter = self.ptycho.runtime.iter_info[-1]['iteration']
                if iter > 30:
                    start_tomo_update = True
            except:
                pass

            if start_tomo_update:
                self.ptycho._vol = self.tomo_update()


            # count up
            self.curiter +=1

        logger.info('Time spent in Fourier update: %.2f' % tf)
        logger.info('Time spent in Overlap update: %.2f' % to)
        logger.info('Time spent in Position update: %.2f' % tp)
        return error_dct