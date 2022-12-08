# -*- coding: utf-8 -*-
"""
Base engine. Used to define reconstruction parameters that are shared
by all engines.

This file is part of the PTYPY package.

    :copyright: Copyright 2014 by the PTYPY team, see AUTHORS.
    :license: see LICENSE for details.
"""
import numpy as np
import time
from .. import utils as u
from ..utils import parallel
from ..utils.verbose import logger, headerline, log
from .posref import AnnealingRefine, GridSearchRefine

__all__ = ['BaseEngine', 'Base3dBraggEngine', 'DEFAULT_iter_info', 'PositionCorrectionEngine']

DEFAULT_iter_info = u.Param(
    iteration=0,
    iterations=0,
    numiter=0,
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

    Defaults:

    [numiter]
    default = 20
    type = int
    lowlim = 1
    help = Total number of iterations

    [numiter_contiguous]
    default = 1
    type = int
    lowlim = 1
    help = Number of iterations without interruption
    doc = The engine will not return control to the caller until this number of iterations is completed (not processing server requests, I/O operations, ...).

    [probe_support]
    default = 0.7
    type = float, None
    lowlim = 0.0
    help = Valid probe area as fraction of the probe frame
    doc = Defines a circular area centered on the probe frame, in which the probe is allowed to be nonzero.

    [probe_fourier_support]
    default = None
    type = float, None
    lowlim = 0.0
    help = Valid probe area in frequency domain as fraction of the probe frame
    doc = Defines a circular area centered on the probe frame (in frequency domain), in which the probe is allowed to be nonzero.

    [record_local_error]
    default = False
    type = bool
    help = If True, save the local map of errors into the runtime dictionary.
    userlevel = 2

    """

    # Define with which models this engine can work.
    COMPATIBLE_MODELS = []

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

        p = self.DEFAULT.copy()
        if pars is not None:
            p.update(pars)
        self.p = p

        self.finished = False
        self.numiter = self.p.numiter

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
                               % (self.p.name), 'l', '=') + '\n')
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

        self._probe_support = {}
        self._probe_fourier_support = {}
        # Call engine specific initialization
        # TODO: Maybe child classes should be calling this?

        # # Make sure all the pods are supported
        # for label_, pod_ in self.pods.items():
        #     if not pod_.model.__class__ in self.SUPPORTED_MODELS:
        #         raise Exception('Model %s not supported by engine' % pod_.model.__class__)

        # Make sure all scan models are supported
        for model in self.ptycho.model.scans.values():
            if not model.__class__ in self.SUPPORTED_MODELS:
                raise Exception('Model %s not supported by engine %s' % (model.__class__,self.p.name))

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
            for s in self.pr.storages.values():
                sh = s.data.shape
                ll, xx, yy = u.grids(sh, FFTlike=False)
                support = (np.pi * (xx**2 + yy**2) < supp * sh[1] * sh[2])
                self._probe_support[s.ID] = support

        supp = self.p.probe_fourier_support
        if supp is not None:
            for s in self.pr.storages.values():
                sh = s.data.shape
                ll, xx, yy = u.grids(sh, center='fft',FFTlike=True)
                support = (np.pi * (xx**2 + yy**2) < supp * sh[1] * sh[2])
                self._probe_fourier_support[s.ID] = support

        # Call engine specific preparation
        self.engine_prepare()

    def support_constraint(self, storage=None):
        """
        Enforces 2D support contraint on probe.
        """
        if storage is None:
            for s in self.pr.storages.values():
                self.support_contraint(s)

        # Fourier space
        support = self._probe_fourier_support.get(storage.ID)
        if support is not None:
            storage.data[:] = np.fft.ifft2(support * np.fft.fft2(storage.data))

        # Real space
        support = self._probe_support.get(storage.ID)
        if support is not None:
            storage.data *= support

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

            logger.warning("""Engine %s did not increase iteration counter
            `self.curiter` internally. Accessing this attribute in that
            engine is inaccurate""" % self.p.name)

            self.curiter += niter_contiguous

        elif self.curiter != (niter_contiguous + it):

            logger.error("""Engine %s increased iteration counter
            `self.curiter` by %d instead of %d. This may lead to
            unexpected behaviour""" % (self.p.name,
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
            error = np.array(list(local_error.values())).mean(0)
        else:
            error = np.zeros((1,))
        info = dict(
            iteration=self.curiter,
            iterations=self.alliter,
            numiter=self.numiter,
            engine=self.p.name,
            duration=time.time() - self.t,
            error=error
        )

        self.ptycho.runtime.iter_info.append(info)
        if self.p.record_local_error:
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


class PositionCorrectionEngine(BaseEngine):
    """
    A sub class engine that supports position correction

    Defaults:

    [position_refinement]
    default = False
    type = Param, bool
    help = If True refine scan positions

    [position_refinement.method]
    default = Annealing
    type = str
    help = Annealing or GridSearch

    [position_refinement.start]
    default = None
    type = int
    help = Number of iterations until position refinement starts
    doc = If None, position refinement starts at first iteration

    [position_refinement.stop]
    default = None
    type = int
    help = Number of iterations after which positon refinement stops
    doc = If None, position refinement stops after last iteration

    [position_refinement.interval]
    default = 1
    type = int
    help = Frequency of position refinement

    [position_refinement.nshifts]
    default = 4
    type = int
    help = Number of random shifts calculated in each position refinement step (has to be multiple of 4)

    [position_refinement.amplitude]
    default = 0.000001
    type = float
    help = Distance from original position per random shift [m]

    [position_refinement.amplitude_decay]
    default = True
    type = bool
    help = After each interation, multiply amplitude by factor (stop - iteration) / (stop - start)

    [position_refinement.max_shift]
    default = 0.000002
    type = float
    help = Maximum distance from original position [m]

    [position_refinement.metric]
    default = "fourier"
    type = str
    help = Error metric, can choose between "fourier" and "photon"
    
    [position_refinement.record]
    default = False
    type = bool
    help = record movement of positions
    """

    POSREF_ENGINES = {
        "Annealing": AnnealingRefine,
        "GridSearch": GridSearchRefine
    }

    def __init__(self, ptycho_parent, pars):
        """
        Position Correction engine.
        """
        super(PositionCorrectionEngine, self).__init__(ptycho_parent, pars)

        # TODO: this just a workaround fix, see issue #256
        # Make a copy of position refinenment defaults
        p = self.DEFAULT.position_refinement.copy()
        # If position correction is turned on, use defaults and start from beginning
        if self.p.position_refinement is True:
            p.start = 0
        # If new position correction params are provided, update defaults
        elif isinstance(self.p.position_refinement,u.Param):
            p.update(self.p.position_refinement)
        # Overwrite position refinement parameters
        self.p.position_refinement = p


    def engine_initialize(self):
        """
        Prepare the position refinement object for use further down the line.
        """

        # Switch for position refinement
        if (self.p.position_refinement.start is None) and (self.p.position_refinement.stop is None):
            self.do_position_refinement = False
        else:
            for label, scan in self.ptycho.model.scans.items():
                if self.p.position_refinement.amplitude < scan.geometries[0].resolution[0]:
                    self.do_position_refinement = False
                    log(3,"Failed to initialise position refinement, search amplitude is smaller than the resolution")
                    return
            self.do_position_refinement = True
            log(3, "Initialising position refinement (%s)" %self.p.position_refinement.method)
            
            # Enlarge object arrays, 
            # This can be skipped though if the boundary is less important
            for name, s in self.ob.storages.items():
               s.padding = int(self.p.position_refinement.max_shift // np.max(s.psize))
               s.reformat()

            # Choose position refinement engine from dictionary
            PosrefEngine = self.POSREF_ENGINES[self.p.position_refinement.method]
            self.position_refinement = PosrefEngine(self.p.position_refinement, self.ob, metric=self.p.position_refinement.metric)
            self.ptycho.citations.add_article(**self.position_refinement.citation_dictionary)
            if self.p.position_refinement.stop is None:
                self.p.position_refinement.stop = self.p.numiter
            if self.p.position_refinement.start is None:
                self.p.position_refinement.start = 0

    def position_update(self):
        """
        Position refinement update.
        """
        if not self.do_position_refinement:
            return
        do_update_pos = (self.p.position_refinement.stop > self.curiter >= self.p.position_refinement.start)
        do_update_pos &= (self.curiter % self.p.position_refinement.interval) == 0

        # Update positions
        if do_update_pos:
            """
            Iterates through all positions and refines them by a given algorithm. 
            """
            log(4, "----------- START POS REF -------------")
            self.position_refinement.update_constraints(self.curiter) # this stays here

            # Iterate through all diffraction views
            for dname, di_view in self.di.views.items():
                # Check for new coordinates
                if di_view.active:
                    self.position_refinement.update_view_position(di_view)

            # We may not need this
            #parallel.barrier()
            #self.ob.reformat(True)

    def engine_finalize(self):
        """
        Synchronize positions
        """
        if self.do_position_refinement is False:
            return
        if self.p.position_refinement.record is False:
            return

        # Gather all new positions from each node
        coords = {}
        for ID, v in self.di.views.items():
            if v.active:
                coords[v.pod.ob_view.ID] = v.pod.ob_view.coord
        coords = parallel.gather_dict(coords)

        # Update storage
        if parallel.master:
            for ID, S in self.ob.storages.items():
                for v in S.views:
                    if v.pod.pr_view.layer == 0:
                        v.coord = coords[v.ID]

        self.ptycho.record_positions = True


class Base3dBraggEngine(BaseEngine):
    """
    3d Bragg engines need a slightly different prepare() method, because
    a 2d probe support makes no sense (at least not yet...)

    Defaults:

    [probe_support]
    default = None
    """

    def prepare(self):
        """
        Last-minute preparation before iterating.
        """
        self.finished = False
        # Simple 2d probe support isn't applicable to the 3d case.
        supp = self.p.probe_support
        if supp is not None:
            raise NotImplementedError

        # Make sure all the pods are supported
        for label_, pod_ in self.pods.items():
            if not pod_.model.__class__ in self.SUPPORTED_MODELS:
                raise Exception('Model %s not supported by engine' % pod_.model.__class__)

        # Call engine specific preparation
        self.engine_prepare()
