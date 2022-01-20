# -*- coding: utf-8 -*-
"""
Simplest possible Difference Map reconstruction engine.

This file is part of the PTYPY package.

    :copyright: Copyright 2014 by the PTYPY team, see AUTHORS.
    :license: GPLv2, see LICENSE for details.
"""
import time
import numpy as np

from .. import utils as u
from .utils import basic_fourier_update
from . import BaseEngine, register
from ..utils.verbose import logger
from ..utils import parallel
from ..core.manager import Full, Vanilla

__all__ = ['DM_simple']

@register()
class DM_simple(BaseEngine):
    """
    Bare-bones DM reconstruction engine.


    Defaults:

    [name]
    default = DM_simple
    type = str
    help =
    doc =

    [numiter]
    default = 123
    type = int

    [alpha]
    default = 1
    type = float
    lowlim = 0.0
    help = Difference map parameter

    [overlap_converge_factor]
    default = 0.05
    type = float
    lowlim = 0.0
    help = Threshold for interruption of the inner overlap loop
    doc = The inner overlap loop refines the probe and the object simultaneously. This loop is escaped as soon as the overall change in probe, relative to the first iteration, is less than this value.

    [overlap_max_iterations]
    default = 10
    type = int
    lowlim = 1
    help = Maximum of iterations for the overlap constraint inner loop

    """

    SUPPORTED_MODELS = [Full, Vanilla]

    def __init__(self, ptycho, pars=None):
        """
        Simplest possible Difference map reconstruction engine.
        """
        super(DM_simple, self).__init__(ptycho, pars)

        p = self.DEFAULT.copy()
        if pars is not None:
            p.update(pars)
        self.p = p

        self.ptycho = ptycho
        self.ob_nrm = None
        self.pr_nrm = None
        self.pr_buf = None

    def engine_initialize(self):
        """
        Prepare for reconstruction.
        """

        # generate container copies for normalization
        self.ob_nrm = self.ob.copy(self.ob.ID + 'nrm', fill=0.)
        self.pr_nrm = self.pr.copy(self.pr.ID + 'nrm', fill=0.)

        # we also need a buffer copy of the probe, to check for overlap
        # convergence.
        self.pr_buf = self.pr.copy(self.pr.ID + 'buf', fill=0.)

    def engine_prepare(self):
        """
        Last-minute preparation before iterating.
        """
        pass

    def engine_iterate(self, num):
        """
        Compute one iteration.
        """
        tf = 0.
        to = 0.
        for it in range(num):
            t0 = time.time()

            # fourier update
            error_dct = {}
            for name, di_view in self.di.V.items():
                if not di_view.active:
                    continue
                error_dct[name] = basic_fourier_update(
                    di_view, alpha=self.p.alpha, pbound=None, LL_error=False)

            t1 = time.time()
            tf = t1 - t0

            # iterative overlap update
            for overlap_iter in range(self.p.overlap_max_iterations):

                # Update object
                self.object_update()

                # Update probe
                change = self.probe_update()

                # Stop iteration if probe change is small
                if change < self.p.overlap_converge_factor:
                    break

            # needed for BaseEngine
            self.curiter += 1

            t2 = time.time()
            to = t2 - t1

        logger.info('Time spent in Fourier update: %.2f' % tf)
        logger.info('Time spent in Overlap update: %.2f' % to)
        return error_dct

    def engine_finalize(self):
        """
        Try deleting ever helper container.
        """
        containers = [
            self.ob_nrm,
            self.pr_buf,
            self.pr_nrm]

        for c in containers:
            del self.ptycho.containers[c.ID]
            del c

        del containers

    def object_update(self):
        """
        DM object update.
        """

        # Fill containers with zeros for the sums
        self.ob.fill(0.0)
        self.ob_nrm.fill(0.)

        # DM update per node: sum over all the positions
        for name, pod in self.pods.items():
            if not pod.active:
                continue
            pod.object += pod.probe.conj() * pod.exit
            self.ob_nrm[pod.ob_view] += u.cabs2(pod.probe)

        # Distribute result with MPI
        for name, s in self.ob.S.items():
            nrm = self.ob_nrm.S[name].data
            parallel.allreduce(s.data)
            parallel.allreduce(nrm)
            s.data /= (nrm + 1e-10)

    def probe_update(self):
        """
        DM probe update.
        """

        # Fill containers with zeros for the sums
        self.pr.fill(0.0)
        self.pr_nrm.fill(0.0)

        # DM update per node: sum over all the positions
        for name, pod in self.pods.items():
            if not pod.active:
                continue
            pod.probe += pod.object.conj() * pod.exit
            self.pr_nrm[pod.pr_view] += u.cabs2(pod.object)

        # Distribute result with MPI and keep track of the overlap convergence.
        change = 0.
        for name, s in self.pr.S.items():
            # MPI reduction of results
            nrm = self.pr_nrm.S[name].data
            parallel.allreduce(s.data)
            parallel.allreduce(nrm)
            s.data /= (nrm + 1e-10)

            # Compute relative change in probe
            buf = self.pr_buf.S[name].data
            change += u.norm2(s.data - buf) / u.norm2(s.data)

            # Fill buffer with new probe
            buf[:] = s.data

        return np.sqrt(change / len(self.pr.S))
