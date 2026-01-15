# -*- coding: utf-8 -*-
"""\
Scan loading recipe for the savu data processing pipeline.

This file is part of the PTYPY package.

    :copyright: Copyright 2014 by the PTYPY team, see AUTHORS.
    :license: see LICENSE for details.
"""

from .. import utils as u
from ..core.data import PtyScan
from ..utils.verbose import log
from . import register

logger = u.verbose.logger


@register()
class Savu(PtyScan):
    """
    Defaults:

    [name]
    default = 'Savu'
    type = str
    help =

    [mask]
    default = None
    type = array
    help = 

    [data]
    default = None
    type = array
    help = 

    [positions]
    default = None
    type = array
    help = 

    """

    def __init__(self, pars=None, **kwargs):
        """
        savu data preparation class.
        """
        # Initialise parent class
        p = self.DEFAULT.copy(99)
        p.update(pars)
        super(Savu, self).__init__(p, **kwargs)
        log(4, u.verbose.report(self.info))

    def load_weight(self):
        if self.info.mask is not None:
            return self.info.mask.astype(float)
        else:
            log(2,'The mask was a None')

    def load_positions(self):
        if self.info.positions is not None:
            return self.info.positions
        else:
            log(2,'The positions were None')

    def load(self, indices):
        """
        Load frames given by the indices.

        :param indices:
        :return:
        """
        raw = {}
        pos = {}
        weights = {}
        if self.info.data is not None:
            for j in indices:
                raw[j] = self.info.data[j]
        else:
            log(2,'The data had None')
        return raw, pos, weights

