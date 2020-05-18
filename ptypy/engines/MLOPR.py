# -*- coding: utf-8 -*-
"""
Maximum Likelihood reconstruction engine.

TODO:
 * Implement other regularizers

This file is part of the PTYPY package.

    :copyright: Copyright 2014 by the PTYPY team, see AUTHORS.
    :license: GPLv2, see LICENSE for details.
"""
import numpy as np
import time
import numpy as np
import h5py
from .. import utils as u
from ..utils.verbose import logger
from ..utils import parallel
from .utils import reduce_dimension
from . import register
from .ML import ML
from ..core.manager import OPRModel

__all__ = ['MLOPR']

@register()
class MLOPR(ML):
    """
    Subclass of Maximum likelihood reconstruction engine for independent probes

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

    [initmodes_stop]
    default = 0
    type = int
    help = Number of iterations before stopping to use initial probe modes (HACK)

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
        self.model  = self.pods[self.pods.keys()[0]].model

        # Make sure that probe storage only contains local probes
        # BD: This is a consequence of gathering all probes for saving to file
        #     at some point before this engine is initialized
        for name, s in self.pr.storages.iteritems():
            ind = self.model.local_indices[name]
            if (s.data.shape != s.shape) & (len(ind) == s.shape[0]):
                s.data = s.data[ind]
                
    def engine_finalize(self):
        """
        Try deleting every helper container.
        Gather independent probes for saving.
        """
        super(MLOPR, self).engine_finalize()
        for name, s in self.pr.storages.iteritems():
            ind = self.model.local_indices[name]
            N = parallel.allreduce(len(ind))
            pr = parallel.gather_list(list(s.data), N, ind)
            if parallel.master:
                s.data = np.array(pr)
        if parallel.master:
            self.ptycho.runtime['OPR_modes'] = self.model.OPR_modes
            self.ptycho.runtime['OPR_coeffs'] = self.model.OPR_coeffs
            
    def hook_post_iterate_update(self):
        """
        Orthogonal Probe Relaxation (OPR) update.
        """
        # Outer loop, go through diffraction views
        for dname, diff_view in self.di.views.iteritems():
            if not diff_view.active:
                continue
            
            # Gather all probes
            probes = {}
            for name, pod in diff_view.pods.iteritems():
                if not pod.active:
                    continue
                probes[name] = pod.probe

            # Orthogonalise prob

        
        for name, s in self.pr.storages.iteritems():
            if self.curiter < self.p.subspace_start:
                return
                
            #if self.curiter < self.p.initmodes_stop:
            #    fname = '/scratch/loh/benedikt/LCLS/amok3415/paper/single_r2_ML_4000.ptyr'
            #    with h5py.File(fname, "r") as f:
            #        e,initmodes = u.ortho(f['content/probe/Sscan_00G00/data'][:])
            #else:
            initmodes = None
                
            ind = self.model.local_indices[name]
            pr_input = np.array([s[l] for i,l in self.model.local_layers[name]])
            new_pr, modes, coeffs = reduce_dimension(a=pr_input, dim=self.p.subspace_dim, local_indices=ind, initmodes=initmodes)
            
            self.model.OPR_modes[name] = modes
            self.model.OPR_coeffs[name] = coeffs

            if parallel.master & (self.curiter%5 == 0):
                # HACK: Saving modes to file
                fname = '/scratch/loh/benedikt/LCLS/amok3415/tmp/opr_modes_%04d.h5' %self.curiter
                with h5py.File(fname, "w") as f:
                    #for name, s in self.pr.storages.iteritems():
                    f['modes/%s' %name] = self.model.OPR_modes[name]
                    f['coeffs/%s' %name] = self.model.OPR_coeffs[name]

            
            # Update probes
            for k, il in enumerate(self.model.local_layers[name]):
                s[il[1]] = new_pr[k]
