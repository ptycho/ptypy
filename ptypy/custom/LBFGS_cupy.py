# -*- coding: utf-8 -*-
"""
Limited-memory BFGS reconstruction engine.

TODO.

  * Implement other regularizers

This file is part of the PTYPY package.

    :copyright: Copyright 2014 by the PTYPY team, see AUTHORS.
    :license: GPLv2, see LICENSE for details.
"""
import numpy as np
# from pycuda import gpuarray
# import pycuda.driver as cuda
# import pycuda.cumath
# from pycuda.tools import DeviceMemoryPool

from ptypy.engines import register
from ptypy.custom.LBFGS_serial import LBFGS_serial
from ptypy.accelerate.base.engines.ML_serial import ML_serial, BaseModelSerial
from ptypy.accelerate.cuda_cupy.engines.ML_cupy import ML_cupy, GaussianModel
from ptypy import utils as u
from ptypy.utils.verbose import logger, log
from ptypy.utils import parallel
from ptypy.accelerate.cuda_cupy import get_context
from ptypy.accelerate.cuda_cupy.kernels import PropagationKernel, RealSupportKernel, FourierSupportKernel
from ptypy.accelerate.cuda_cupy.kernels import GradientDescentKernel, AuxiliaryWaveKernel, PoUpdateKernel, PositionCorrectionKernel
from ptypy.accelerate.cuda_cupy.array_utils import ArrayUtilsKernel, DerivativesKernel, FFTGaussianSmoothingKernel, TransposeKernel
from ptypy.accelerate.cuda_cupy.mem_utils import GpuDataManager
from ptypy.accelerate.base import address_manglers

__all__ = ['LBFGS_cupy']

MAX_BLOCKS = 99999  # can be used to limit the number of blocks, simulating that they don't fit
#MAX_BLOCKS = 3  # can be used to limit the number of blocks, simulating that they don't fit

@register()
class LBFGS_cupy(LBFGS_serial, ML_cupy):

    """
    Defaults:

    [probe_update_cuda_atomics]
    default = False
    type = bool
    help = For GPU, use the atomics version for probe update kernel

    [object_update_cuda_atomics]
    default = True
    type = bool
    help = For GPU, use the atomics version for object update kernel

    [use_cuda_device_memory_pool]
    default = True
    type = bool
    help = For GPU, use a device memory pool

    [fft_lib]
    default = reikna
    type = str
    help = Choose the pycuda-compatible FFT module.
    doc = One of:
      - ``'reikna'`` : the reikna packaga (fast load, competitive compute for streaming)
      - ``'cuda'`` : ptypy's cuda wrapper (delayed load, but fastest compute if all data is on GPU)
      - ``'skcuda'`` : scikit-cuda (fast load, slowest compute due to additional store/load stages)
    choices = 'reikna','cuda','skcuda'
    userlevel = 2
    """

    def __init__(self, ptycho_parent, pars=None):
        """
        Limited-memory BFGS reconstruction engine.
        """
        super().__init__(ptycho_parent, pars)

    def engine_initialize(self):
        """
        Prepare for LBFGS reconstruction.
        """
        super().engine_initialize()

    def engine_prepare(self):
        super().engine_prepare()

    def _initialize_model(self):

        # Create noise model
        if self.p.ML_type.lower() == "gaussian":
            self.ML_model = GaussianModel(self)
        elif self.p.ML_type.lower() == "poisson":
            raise NotImplementedError('Poisson norm model not yet implemented')
        elif self.p.ML_type.lower() == "euclid":
            raise NotImplementedError('Euclid norm model not yet implemented')
        else:
            raise RuntimeError("Unsupported ML_type: '%s'" % self.p.ML_type)

    def _get_smooth_gradient(self, data, sigma):
        if self.p.smooth_gradient_method == "convolution":
            if self.GSK.tmp is None:
                self.GSK.tmp = cp.empty(data.shape, dtype=np.complex64)
            try:
                self.GSK.convolution(data, [sigma, sigma], tmp=self.GSK.tmp)
            except MemoryError:
                raise RuntimeError("Convolution kernel too large for direct convolution on GPU",
                                   "Please reduce parameter smooth_gradient or set smooth_gradient_method='fft'.")
        elif self.p.smooth_gradient_method == "fft":
            self.FGSK.filter(data, sigma)
        else:
            raise NotImplementedError("smooth_gradient_method should be ```convolution``` or ```fft```.")
        return data

    def _smooth_gradient(self):
        new_ob_grad = self.ob_grad_new
        # Smoothing preconditioner decay (once per iteration)
        if self.smooth_gradient:
            self.smooth_gradient.sigma *= (1. - self.p.smooth_gradient_decay)
            for name, s in new_ob_grad.storages.items():
                s.gpu = self._get_smooth_gradient(s.gpu, self.smooth_gradient.sigma)

    def _get_ob_norm(self):
        norm = self._get_norm(self.ob_grad_new)
        return norm

    def _get_pr_norm(self):
        norm = self._get_norm(self.pr_grad_new)
        return norm

    def _get_dot(self, c1, c2):
        dot = np.double(0.)
        for name, s2 in c2.storages.items():
            s1 = c1.storages[name]
            dot += self._dot_kernel(s2.gpu, s1.gpu).get()[0]
        return dot

    def _get_norm(self, c):
        norm = np.double(0.)
        for name, s in c.storages.items():
            norm += self._dot_kernel(s.gpu, s.gpu).get()[0]
        return norm

    def _replace_ob_pr_ys(self, mi):
        self.cdotr_ob_ys[mi-1] = self._get_dot(self.ob_y[mi-1],
                self.ob_s[mi-1])
        self.cdotr_pr_ys[mi-1] = self._get_dot(self.pr_y[mi-1],
                self.pr_s[mi-1])
        self.cn2_ob_y[mi-1] = self._get_norm(self.ob_y[mi-1])
        self.cn2_pr_y[mi-1] = self._get_norm(self.pr_y[mi-1])

    def _get_ob_pr_sh(self, k):
        ob_sh = self._get_dot(self.ob_s[k], self.ob_h)
        pr_sh = self._get_dot(self.pr_s[k], self.pr_h)
        return ob_sh, pr_sh

    def _get_ob_pr_yh(self, k):
        ob_yh = self._get_dot(self.ob_y[k], self.ob_h)
        pr_yh = self._get_dot(self.pr_y[k], self.pr_h)
        return ob_yh, pr_yh

    def engine_iterate(self, num=1):
        """
        Compute `num` iterations.
        """
        error_dct = super().engine_iterate(num)
        # copy all data back to cpu
        self._set_pr_ob_ref_for_data(dev='cpu', container=None, sync_copy=True)
        return error_dct

    def engine_finalize(self):
        """
        Clear all GPU data, pinned memory, etc
        """
        super().engine_finalize()

