# -*- coding: utf-8 -*-
"""
An extension plugin of the Difference Map engine
with object regularisation for air/vacuum regions.

authors: Benedikt J. Daurer
"""
from ptypy.engines import projectional
from ptypy.engines import register
import numpy as np

@register()
class DM_object_regul(projectional.DM):
    """
    An extension of DM with the following additional parameters

    Defaults:

    [object_regul_mask]
    default = None
    type = ndarray
    help = A mask used for regularisation of the object
    doc = Numpy.ndarray with same shape as the object that will be casted to a boolean mask

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
        super(DM_object_regul, self).__init__(ptycho_parent, pars)
    
    def object_update(self):
        """
        Replace values inside mask with given fill value.
        """
        super().object_update()
        do_regul = True
        if (self.p.object_regul_start is not None): 
            do_regul &= (self.curiter >= self.p.object_regul_start)
        if (self.p.object_regul_stop is not None):
            do_regul &= (self.curiter < self.p.object_regul_stop)

        if (self.p.object_regul_mask is not None) and do_regul:
            for name, s in self.ob.storages.items():
                assert s.shape == self.p.object_regul_mask.shape, "Object regulariser mask needs to have same shape as object = {}".format(s.shape)
                if isinstance(self.p.object_regul_fill, complex):
                    s.data[self.p.object_regul_mask.astype(bool)] = self.p.object_regul_fill
                elif isinstance(self.p.object_regul_fill, float):
                    s.data[self.p.object_regul_mask.astype(bool)] = np.abs(s.data[self.p.object_regul_mask.astype(bool)]) * np.exp(1j*self.p.object_regul_fill)
