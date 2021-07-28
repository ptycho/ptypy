# -*- coding: utf-8 -*-
"""
An extension plugin of the Difference Map engine
with object regularisation for air/vacuum regions.

authors: Benedikt J. Daurer
"""
from ptypy.engines import DM
from ptypy.engines import register

@register()
class DM_object_regul(DM.DM):
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
    doc = Provide a complex number with both real and imaginary part, e.g. 1.0 + 0.1j

    """

    def __init__(self, ptycho_parent, pars=None):
        super(DM_object_regul, self).__init__(ptycho_parent, pars)
    
    def object_update(self):
        """
        Replace values inside mask with given fill value.
        """
        super().object_update()
        if self.p.object_regul_mask is not None:
            for name, s in self.ob.storages.items():
                assert s.shape == self.p.object_regul_mask.shape, "Object regulariser mask needs to have same shape as object = {}".format(s.shape)
                s.data[self.p.object_regul_mask.astype(bool)] = self.p.object_regul_fill
