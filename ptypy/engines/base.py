# -*- coding: utf-8 -*-
"""
Base engine. Used to define reconstruction parameters that are shared
by all engines.

This file is part of the PTYPY package.

    :copyright: Copyright 2014 by the PTYPY team, see AUTHORS.
    :license: GPLv2, see LICENSE for details.
"""
import numpy as np
import time
from .. import utils as u
from ..utils import parallel
from ..utils.verbose import logger, headerline

__all__ = ['BaseEngine', 'DEFAULT_iter_info']

DEFAULT = u.Param(
    # Total number of iterations
    numiter=20,
    # Number of iterations without interruption
    numiter_contiguous=1,
    # Fraction of valid probe area (circular) in probe frame
    probe_support=0.7,
    # Note: if probe support becomes of more complex nature,
    # consider putting it in illumination.py
    # Clip object amplitude into this interval
    clip_object=None,
    # Number of iterations before starting subpixel interpolation
    subpix_start=0,
    # Subpixel interpolation; 'fourier','linear' or None for no interpolation
    subpix='linear',
    # Number of iterations before probe update starts
    probe_update_start=2,
    # Weight of the current probe estimate in the update
    probe_inertia=0.001,
    # Weight of the current object in the update
    object_inertia=0.1,
    # Gaussian smoothing (pixel) of the current object prior to update
    obj_smooth_std=20,
    # Pixel radius around optical axes that the probe mass center must reside in
    # None or float
    probe_center_tol=None,
)

DEFAULT_iter_info = u.Param(
    iteration=0,
    iterations=0,
    engine='None',
    duration=0.,
    error=np.zeros((3,))
)


class BaseEngine(object):
    """
    Base reconstruction engine.

    In child classes, overwrite the following methods for custom behavior :

    engine_initialize
    engine_prepare
    engine_iterate
    engine_finalize
    """

    DEFAULT = DEFAULT.copy()

    def __init__(self, ptycho, pars=None):
        """
        Base reconstruction engine.

        Parameters
        ----------
        ptycho : Ptycho
            The parent :any:`Ptycho` object.

        pars: Param or dict
            Initialization parameters
        """
        self.ptycho = ptycho
        p = u.Param(self.DEFAULT)

        if pars is not None:
            p.update(pars)
        self.p = p

        # self.itermeta = []
        # self.meta = u.Param()
        self.finished = False
        self.numiter = self.p.numiter
        # self.initialize()

        # Instance attributes
        self.curiter = None
        self.alliter = None

        self.di = None
        self.ob = None
        self.pr = None
        self.ma = None
        self.ex = None
        self.pods = None

        self.probe_support = None
        self.t = None
        self.error = None

    def initialize(self):
        """
        Prepare for reconstruction.
        """
        logger.info('\n' +
                    headerline('Starting %s-algorithm.'
                               % str(type(self).__name__), 'l', '=') + '\n')
        logger.info('Parameter set:')
        logger.info(u.verbose.report(self.p, noheader=True).strip())
        logger.info(headerline('', 'l', '='))

        self.curiter = 0
        if self.ptycho.runtime.iter_info:
            self.alliter = self.ptycho.runtime.iter_info[-1]['iterations']
        else:
            self.alliter = 0

        # Common attributes for all reconstructions
        self.di = self.ptycho.diff
        self.ob = self.ptycho.obj
        self.pr = self.ptycho.probe
        self.ma = self.ptycho.mask
        self.ex = self.ptycho.exit
        self.pods = self.ptycho.pods

        self.probe_support = {}
        # Call engine specific initialization
        self.engine_initialize()

    def prepare(self):
        """
        Last-minute preparation before iterating.
        """
        self.finished = False
        # Calculate probe support
        # an individual support for each storage is calculated in saved
        # in the dict self.probe_support
        supp = self.p.probe_support
        if supp is not None:
            for name, s in self.pr.storages.iteritems():
                sh = s.data.shape
                ll, xx, yy = u.grids(sh, FFTlike=False)
                support = (np.pi * (xx**2 + yy**2) < supp * sh[1] * sh[2])
                self.probe_support[name] = support

        # Call engine specific preparation
        self.engine_prepare()

    def iterate(self, num=None):
        """
        Compute one or several iterations.

        num : None, int number of iterations.
            If None or num<1, a single iteration is performed.
        """
        # Several iterations
        if self.p.numiter_contiguous is not None:
            niter_contiguous = self.p.numiter_contiguous
        else:
            niter_contiguous = 1

        # Overwrite default parameter
        if num is not None:
            niter_contiguous = num

        # Support numiter == 0 for debugging purposes
        if self.numiter == 0:
            self.finished = True

        if self.finished:
            return

        # For benchmarking
        self.t = time.time()

        it = self.curiter

        # Call engine specific iteration routine
        # and collect the per-view error.
        self.error = self.engine_iterate(niter_contiguous)

        # Check if engine did things right.
        if it >= self.curiter:

            logger.warn("""Engine %s did not increase iteration counter
            `self.curiter` internally. Accessing this attribute in that
            engine is inaccurate""" % self.__class__.__name__)

            self.curiter += niter_contiguous

        elif self.curiter != (niter_contiguous + it):

            logger.error("""Engine %s increased iteration counter
            `self.curiter` by %d instead of %d. This may lead to
            unexpected behaviour""" % (self.__class__.__name__,
            self.curiter-it, niter_contiguous))

        else:
            pass

        self.alliter += niter_contiguous

        if self.curiter >= self.numiter:
            self.finished = True

        # Prepare runtime
        self._fill_runtime()

        parallel.barrier()

    def _fill_runtime(self):
        local_error = u.parallel.gather_dict(self.error)
        if local_error:
            error = np.array(local_error.values()).mean(0)
        else:
            error = np.zeros((1,))
        info = dict(
            iteration=self.curiter,
            iterations=self.alliter,
            engine=type(self).__name__,
            duration=time.time() - self.t,
            error=error
        )

        self.ptycho.runtime.iter_info.append(info)
        self.ptycho.runtime.error_local = local_error

    def finalize(self):
        """
        Clean up after iterations are done.
        """
        self.engine_finalize()
        pass

    def engine_initialize(self):
        """
        Engine-specific initialization.

        Called at the end of self.initialize().
        """
        raise NotImplementedError()

    def engine_prepare(self):
        """
        Engine-specific preparation.

        Last-minute initialization providing up-to-date information for
        reconstruction. Called at the end of self.prepare()
        """
        raise NotImplementedError()

    def engine_iterate(self, num):
        """
        Engine single-step iteration.

        All book-keeping is done in self.iterate(), so this routine only needs
        to implement the "core" actions.
        """
        raise NotImplementedError()

    def engine_finalize(self):
        """
        Engine-specific finalization.

        Used to wrap-up engine-specific stuff. Called at the end of
        self.finalize()
        """
        raise NotImplementedError()
