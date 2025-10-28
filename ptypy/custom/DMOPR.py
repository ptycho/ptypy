# -*- coding: utf-8 -*-
"""
Difference Map reconstruction engine.
Independent-Probe flavour.

This file is part of the PTYPY package.

    :copyright: Copyright 2014 by the PTYPY team, see AUTHORS.
    :license: see LICENSE for details.
"""
import numpy as np
from ptypy import utils as u
from ptypy.utils import parallel
from ptypy.engines.utils import reduce_dimension
from ptypy.engines import register
from ptypy.engines.projectional import DM
from ptypy.core.manager import OPRModel

__all__ = ['DMOPR']

@register()
class DMOPR(DM):
    """
    Subclass of Difference Map engine for independent probes (OPR).

    Defaults:

    [name]
    default = DMOPR
    type = str
    help =
    doc =

    [IP_metric]
    default = 1.
    type = float
    help = The metric factor in the exit + probe augmented space

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
        Difference map reconstruction engine for independent probes
        """
        p = self.DEFAULT.copy()
        if pars is not None:
            p.update(pars)
        self.p = p
        super(DMOPR, self).__init__(ptycho_parent, pars)

    def engine_initialize(self):
        """
        Prepare for reconstruction.
        Obtain a reference to the OPR scan model
        """
        super(DMOPR, self).engine_initialize()
        self.model  = self.pods[list(self.pods.keys())[0]].model
        # Clear OPR probes from runtime
        if "OPR_allprobes" in self.ptycho.runtime:
            self.ptycho.runtime['OPR_allprobes'] = None

    def engine_finalize(self):
        """
        Try deleting every helper container.
        Gather independent probes for saving.
        """
        super(DMOPR, self).engine_finalize()
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

    def probe_update(self):
        """
        DM probe update for independent probes.
        """
        pr = self.pr
        pr_buf = self.pr_buf

        # DM update per node
        for name, pod in self.pods.items():
            if not pod.active:
                continue
            pod.probe += pod.object.conj() * pod.exit * pod.probe_weight * self.p.IP_metric
            pod.probe /= (u.cabs2(pod.object) * pod.probe_weight + self.p.IP_metric)

        change, nrm = 0., 0.

        # Distribute result with MPI
        for name, s in pr.storages.items():

            # Orthogonal probe relaxation (OPR) update step
            self._probe_consistency_update(s,name)

            # Apply probe support if requested
            self.support_constraint(s)
        
            # Compute relative change in probe
            buf = pr_buf.storages[name].data
            change += u.norm2(s.data - buf) 
            nrm += u.norm2(s.data)

            # Fill buffer with new probe
            buf[:] = s.data
        
        # Normalized MPI reduction of probe change
        change = parallel.allreduce(change) / parallel.allreduce(nrm)

        return np.sqrt(change / len(pr.storages))


    def _probe_consistency_update(self, s, name):
        """
        Probe consistency update for orthogonal probe relaxation.
        """ 
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
