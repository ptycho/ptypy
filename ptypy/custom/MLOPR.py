# -*- coding: utf-8 -*-
"""
Maximum Likelihood reconstruction engine.

TODO:
 * Implement other regularizers

This file is part of the PTYPY package.

    :copyright: Copyright 2014 by the PTYPY team, see AUTHORS.
    :license: see LICENSE for details.
"""
import numpy as np
import time
from ptypy import utils as u
from ptypy.utils.verbose import logger
from ptypy.utils import parallel
from ptypy.engines.utils import reduce_dimension
from ptypy.engines import register
from ptypy.engines.ML import ML
from ptypy.core.manager import OPRModel

__all__ = ['MLOPR']

@register()
class MLOPR(ML):
    """
    Subclass of Maximum likelihood reconstruction engine for independent probes (OPR).

    Defaults:

    [name]
    default = MLOPR
    type = str
    help =
    doc =

    [subspace_dim]
    default = 1
    type = int
    help = The dimension of the subspace spanned by the probe ensemble

    [subspace_start]
    default = 0
    type = int
    help = Number of iterations before starting to span the probe ensemble subspace

    """

    SUPPORTED_MODELS = [OPRModel]

    def __init__(self, ptycho_parent, pars=None):
        """
        Maximum likelihood reconstruction engine for independent probes.
        """
        p = self.DEFAULT.copy()
        if pars is not None:
            p.update(pars)
        self.p = p
        super(MLOPR, self).__init__(ptycho_parent, pars)

    def engine_initialize(self):
        """
        Prepare for ML reconstruction.
        """
        super(MLOPR, self).engine_initialize()
        self.model  = self.pods[list(self.pods.keys())[0]].model

        # Clear OPR probes from runtime
        if "OPR_allprobes" in self.ptycho.runtime:
            self.ptycho.runtime['OPR_allprobes'] = None

    def engine_finalize(self):
        """
        Try deleting every helper container.
        Gather independent probes for saving.
        """
        super(MLOPR, self).engine_finalize()
        for name, s in self.pr.storages.items():
            N = parallel.allreduce(len(s.layermap))
            pr = np.array(parallel.gather_list(list(s.data), N, s.layermap))
            if parallel.master:
                self.model.OPR_allprobes[name] = pr

        # Add OPR data to runtime
        if parallel.master:
            self.ptycho.runtime['OPR_modes'] = self.model.OPR_modes
            self.ptycho.runtime['OPR_coeffs'] = self.model.OPR_coeffs
            self.ptycho.runtime['OPR_allprobes'] = self.model.OPR_allprobes

    def _post_iterate_update(self):
        """
        Orthogonal Probe Relaxation (OPR) update.
        """
        for name, s in self.pr.storages.items():
            if self.curiter < self.p.subspace_start:
                subdim = 1
            else:
                subdim = self.p.subspace_dim

            pr_input = np.array([s[l] for l in s.layermap])
            new_pr, modes, coeffs = reduce_dimension(a=pr_input,
                                                     dim=subdim, 
                                                     local_indices=s.layermap)

            self.model.OPR_modes[name] = modes
            self.model.OPR_coeffs[name] = coeffs

            # Update probes
            for k, l in enumerate(s.layermap):
                s[l] = new_pr[k]
