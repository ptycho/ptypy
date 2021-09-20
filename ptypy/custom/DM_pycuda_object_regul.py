# -*- coding: utf-8 -*-
"""
An extension plugin of the accelerated (pycuda) Difference Map engine
with object regularisation for air/vacuum regions.

authors: Benedikt J. Daurer
"""
from ptypy.accelerate.cuda_pycuda.engines import projectional_pycuda
from ptypy.engines import register
from pycuda import gpuarray
import numpy as np

@register()
class DM_pycuda_object_regul(projectional_pycuda.DM_pycuda):
    """
    An extension of DM_pycuda with the following additional parameters

    Defaults:

    [object_regul_mask]
    default = None
    type = ndarray
    help = A mask used for regularisation of the object
    doc = Numpy.ndarray with same shape as the object that will be casted to a complex-valued mask

    [object_regul_fill]
    default = 0.0 + 0.0j
    type = float, complex
    help = Fill value for regularisation of the object
    doc = Providing a complex number, e.g. 1.0 + 0.1j will replace both real and imaginary parts\
          Providing a floating number, e.g. 0.5 will replace only the phase

    [object_regul_start]
    default = None
    type = int
    help = Number of iterations until object regularisation starts
    doc = If None, object regularisation starts at first iteration

    [object_regul_stop]
    default = None
    type = int
    help = Number of iterations after which object regularisation stops
    doc = If None, object regularisation stops after last iteration

    """

    def __init__(self, ptycho_parent, pars=None):
        super(DM_pycuda_object_regul, self).__init__(ptycho_parent, pars)
    
    def engine_prepare(self):
        super().engine_prepare()
        if self.p.object_regul_mask is not None:
            self.object_mask_gpu = gpuarray.to_gpu(self.p.object_regul_mask.astype(np.complex64))

    def _setup_kernels(self):
        super()._setup_kernels()
        from pycuda.elementwise import ElementwiseKernel
        self.obj_regul_complex = ElementwiseKernel(
            "pycuda::complex<float> *in, pycuda::complex<float> *mask, pycuda::complex<float> fill",
            "in[i] = fill*mask[i] + in[i]*(pycuda::complex<float>(1) - mask[i])",
            "obj_regulariser_complex")
        self.obj_regul_phase = ElementwiseKernel(
            "pycuda::complex<float> *in, pycuda::complex<float> *mask, float fill",
            "in[i] = pycuda::abs(in[i])*mask[i]*pycuda::exp(fill*pycuda::complex<float>(1)) + in[i]*(pycuda::complex<float>(1) - mask[i])",
            "obj_regulariser_phase")

    def object_update(self,*args, **kwargs):
        """
        Replace values inside mask with given fill value.
        """
        super().object_update(*args,**kwargs)
        do_regul = True
        if (self.p.object_regul_start is not None): 
            do_regul &= (self.curiter >= self.p.object_regul_start)
        if (self.p.object_regul_stop is not None):
            do_regul &= (self.curiter < self.p.object_regul_stop)
        if (self.p.object_regul_mask is not None) and do_regul:
            for oID, ob in self.ob.storages.items():
                assert ob.shape == self.object_mask_gpu.shape, "Object regulariser mask needs to have same shape as object = {}".format(ob.shape)
                if isinstance(self.p.object_regul_fill, complex):
                    self.obj_regul_complex(ob.gpu, self.object_mask_gpu, self.p.object_regul_fill)
                elif isinstance(self.p.object_regul_fill, float):
                    self.obj_regul_phase(ob.gpu, self.object_mask_gpu, self.p.object_regul_fill)
