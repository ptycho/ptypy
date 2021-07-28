# -*- coding: utf-8 -*-
"""
An extension plugin of the accelerated (pycuda) Difference Map engine
with object regularisation for air/vacuum regions.

authors: Benedikt J. Daurer
"""
from ptypy.accelerate.cuda_pycuda.engines import DM_pycuda
from ptypy.engines import register
from pycuda import gpuarray
import numpy as np

@register()
class DM_pycuda_object_regul(DM_pycuda.DM_pycuda):
    """
    An extension of DM_pycuda with the following additional parameters

    Defaults:

    [object_regul_mask]
    default = None
    type = ndarray
    help = A mask used for regularisation of the object
    doc = Numpy.ndarray with same shape as the object that will be casted to a complex-valued mask

    [object_regul_fill]
    default = 0. + 0.j
    type = float, complex
    help = Fill value for regularisation of the object
    doc = Provide a complex number with both real and imaginary part, e.g. 1.2 + 0.1j

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
        self.obj_regul = ElementwiseKernel(
            "pycuda::complex<float> *in, pycuda::complex<float> *mask, pycuda::complex<float> fill",
            "in[i] = fill*mask[i] + in[i]*(pycuda::complex<float>(1) - mask[i])",
            "obj_regulariser")

    def object_update(self,*args, **kwargs):
        """
        Replace values inside mask with given fill value.
        """
        super().object_update(*args,**kwargs)
        if self.p.object_regul_mask is not None:
            for oID, ob in self.ob.storages.items():
                assert ob.shape == self.object_mask_gpu.shape, "Object regulariser mask needs to have same shape as object = {}".format(ob.shape)
                self.obj_regul(ob.gpu, self.object_mask_gpu, self.p.object_regul_fill)
