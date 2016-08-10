# -*- coding: utf-8 -*-
"""\
Scan loading recipe for the savu data processing pipeline.

This file is part of the PTYPY package.

    :copyright: Copyright 2014 by the PTYPY team, see AUTHORS.
    :license: GPLv2, see LICENSE for details.
"""

import numpy as np
import os
from .. import utils as u
from .. import io
from ..utils import parallel
from ..core.data import PtyScan
from ..utils.verbose import log
from ..core.paths import Paths
from ..core import DEFAULT_io as IO_par
import h5py as h5

logger = u.verbose.logger

# Recipe defaults

SAVU = PtyScan.DEFAULT.copy()
SAVU.mask = None
SAVU.data = None
SAVU.positions = None


class Savu(PtyScan):
    DEFAULT = SAVU

    def __init__(self, pars=None, **kwargs):
        """
        savu data preparation class.
        """
        # Initialise parent class
        recipe_default = SAVU.copy()
        recipe_default.update(pars.recipe, in_place_depth=99)
        pars.recipe.update(recipe_default)
        super(Savu, self).__init__(pars, **kwargs)
        log(4, u.verbose.report(self.info))

    def load_weight(self):
        if self.info.mask is not None:
            return self.info.recipe.mask.astype(float)
        else:
            print('The mask was a None')

    def load_positions(self):
        if self.info.recipe.positions is not None:
            return self.info.recipe.positions
        else:
            print('The positions were None')

    def load(self, indices):
        """
        Load frames given by the indices.

        :param indices:
        :return:
        """
        raw = {}
        pos = {}
        weights = {}
        if self.info.recipe.data is not None:
            for j in indices:
                raw[j] = self.info.recipe.data[j]
        else:
            print('The data had None')
        return raw, pos, weights

