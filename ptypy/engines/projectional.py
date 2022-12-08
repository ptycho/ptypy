# -*- coding: utf-8 -*-
"""
Difference Map reconstruction engine.

This file is part of the PTYPY package.

    :copyright: Copyright 2014 by the PTYPY team, see AUTHORS.
    :license: see LICENSE for details.
"""
import numpy as np
import time
from .. import utils as u
from ..utils.verbose import logger, log
from ..utils import parallel
from .utils import projection_update_generalized, log_likelihood
from . import register
from .base import PositionCorrectionEngine
from ..core.manager import Full, Vanilla, Bragg3dModel, BlockVanilla, BlockFull

__all__ = ['DM', 'RAAR']

class _ProjectionEngine(PositionCorrectionEngine):
    """
    Defaults:

    [probe_update_start]
    default = 2
    type = int
    lowlim = 0
    help = Number of iterations before probe update starts

    [subpix_start]
    default = 0
    type = int
    lowlim = 0
    help = Number of iterations before starting subpixel interpolation

    [subpix]
    default = 'linear'
    type = str
    help = Subpixel interpolation; 'fourier','linear' or None for no interpolation

    [update_object_first]
    default = True
    type = bool
    help = If True update object before probe

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

    [probe_inertia]
    default = 1e-9
    type = float
    lowlim = 0.0
    help = Weight of the current probe estimate in the update

    [object_inertia]
    default = 1e-4
    type = float
    lowlim = 0.0
    help = Weight of the current object in the update

    [fourier_power_bound]
    default = None
    type = float
    help = If rms error of model vs diffraction data is smaller than this value, Fourier constraint is met
    doc = For Poisson-sampled data, the theoretical value for this parameter is 1/4. Set this value higher for noisy data. By default, power bound is calculated using fourier_relax_factor

    [fourier_relax_factor]
    default = 0.05
    type = float
    lowlim = 0.0
    help = A factor used to calculate the Fourier power bound as 0.25 * fourier_relax_factor**2 * maximum power in diffraction data
    doc = Set this value higher for noisy data.

    [obj_smooth_std]
    default = None
    type = float
    lowlim = 0
    help = Gaussian smoothing (pixel) of the current object prior to update
    doc = If None, smoothing is deactivated. This smoothing can be used to reduce the amplitude of spurious pixels in the outer, least constrained areas of the object.

    [clip_object]
    default = None
    type = tuple
    help = Clip object amplitude into this interval

    [probe_center_tol]
    default = None
    type = float
    lowlim = 0.0
    help = Pixel radius around optical axes that the probe mass center must reside in

    [compute_log_likelihood]
    default = True
    type = bool
    help = A switch for computing the log-likelihood error (this can impact the performance of the engine)

    """

    SUPPORTED_MODELS = [Full, Vanilla, Bragg3dModel, BlockVanilla, BlockFull]

    def __init__(self, ptycho_parent, pars=None):
        """
        Difference map reconstruction engine.
        """
        super().__init__(ptycho_parent, pars)

        self._a = 0.
        self._b = 1.
        self._c = 1.

        self.error = None

        self.ob_buf = None
        self.ob_nrm = None
        self.ob_viewcover = None

        self.pr_buf = None
        self.pr_nrm = None

        self.pbound = None

        # Required to get proper normalization of object inertia
        # The actual value is computed in engine_prepare
        # Another possibility would be to use the maximum value of all probe storages.
        self.mean_power = None


    def engine_initialize(self):
        """
        Prepare for reconstruction.
        """
        super().engine_initialize()

        self.error = []

        # Generate container copies
        self.ob_buf = self.ob.copy(self.ob.ID + '_alt', fill=0.)
        self.ob_nrm = self.ob.copy(self.ob.ID + '_nrm', fill=0., dtype='real')
        self.ob_viewcover = self.ob.copy(self.ob.ID + '_vcover', fill=0.)

        self.pr_buf = self.pr.copy(self.pr.ID + '_alt', fill=0.)
        self.pr_nrm = self.pr.copy(self.pr.ID + '_nrm', fill=0., dtype='real')

    def engine_prepare(self):

        """
        Last minute initialization.

        Everything that needs to be recalculated when new data arrives.
        """
        if self.ptycho.new_data:

            # recalculate everything
            mean_power = 0.
            self.pbound_scan = {}
            for s in self.di.storages.values():
                if self.p.fourier_power_bound is None:
                    pb = .25 * self.p.fourier_relax_factor**2 * s.pbound_stub
                else:
                    pb = self.p.fourier_power_bound
                log(4, "power bound for scan %s = %f" %(s.label, pb))
                if not self.pbound_scan.get(s.label):
                    self.pbound_scan[s.label] = pb
                else:
                    self.pbound_scan[s.label] = max(pb, self.pbound_scan[s.label])
                mean_power += s.mean_power
            self.mean_power = mean_power / len(self.di.storages)

        # Fill object with coverage of views
        for name, s in self.ob_viewcover.storages.items():
            s.fill(s.get_view_coverage())

    def engine_iterate(self, num=1):
        """
        Compute `num` iterations.
        """
        to = 0.
        tf = 0.
        tp = 0.
        for it in range(num):
            t1 = time.time()

            # Fourier update
            error_dct = self.fourier_update()

            t2 = time.time()
            tf += t2 - t1

            # Overlap update
            self.overlap_update()

            # Recenter the probe
            self.center_probe()

            t3 = time.time()
            to += t3 - t2

            # Position update
            self.position_update()

            t4 = time.time()
            tp += t4 - t3

            # count up
            self.curiter +=1

        logger.info('Time spent in Fourier update: %.2f' % tf)
        logger.info('Time spent in Overlap update: %.2f' % to)
        logger.info('Time spent in Position update: %.2f' % tp)
        return error_dct

    def engine_finalize(self):
        """
        Try deleting ever helper container.
        """
        super().engine_finalize()

        containers = [
            self.ob_buf,
            self.ob_nrm,
            self.ob_viewcover,
            self.pr_buf,
            self.pr_nrm]

        for c in containers:
            logger.debug('Attempt to remove container %s' % c.ID)
            del self.ptycho.containers[c.ID]
        #    IDM.used.remove(c.ID)

        del self.ob_buf
        del self.ob_nrm
        del self.ob_viewcover
        del self.pr_buf
        del self.pr_nrm

        del containers

    def fourier_update(self):
        """
        DM Fourier constraint update (including DM step).
        """
        error_dct = {}
        for name, di_view in self.di.views.items():
            if not di_view.active:
                continue
            #pbound = self.pbound[di_view.storage.ID]
            pbound = self.pbound_scan[di_view.storage.label]
            """
            error_dct[name] = basic_fourier_update(di_view,
                                                   pbound=pbound,
                                                   alpha=self.p.alpha,
                                                   LL_error=self.p.compute_log_likelihood)
            """
            err_fmag, err_exit = projection_update_generalized(di_view, self._a, self._b, self._c, pbound)
            if self.p.compute_log_likelihood:
                err_phot = log_likelihood(di_view)
            else:
                err_phot = 0.
            error_dct[name] = np.array([err_fmag, err_phot, err_exit])

        return error_dct

    def clip_object(self, ob):
        # Clip object (This call takes like one ms. Not time critical)
        if self.p.clip_object is not None:
            clip_min, clip_max = self.p.clip_object
            ampl_obj = np.abs(ob.data)
            phase_obj = np.exp(1j * np.angle(ob.data))
            too_high = (ampl_obj > clip_max)
            too_low = (ampl_obj < clip_min)
            ob.data[too_high] = clip_max * phase_obj[too_high]
            ob.data[too_low] = clip_min * phase_obj[too_low]

    def overlap_update(self):
        """
        DM overlap constraint update.
        """
        # Condition to update probe
        do_update_probe = (self.p.probe_update_start <= self.curiter)

        for inner in range(self.p.overlap_max_iterations):
            pre_str = 'Iteration (Overlap) #%02d:  ' % inner

            # Update object first
            if self.p.update_object_first or (inner > 0) or not do_update_probe:
                # Update object
                log(4, pre_str + '----- object update -----')
                self.object_update()

            # Exit if probe should not be updated yet
            if not do_update_probe:
                break

            # Update probe
            log(4, pre_str + '----- probe update -----')
            change = self.probe_update()
            log(4, pre_str + 'change in probe is %.3f' % change)

            # Stop iteration if probe change is small
            if change < self.p.overlap_converge_factor:
                break

    def center_probe(self):
        if self.p.probe_center_tol is not None:
            for name, pr_s in self.pr.storages.items():
                c1 = u.mass_center(u.abs2(pr_s.data).sum(0))
                c2 = np.asarray(pr_s.shape[-2:]) // 2
                # fft convention should however use geometry instead
                if u.norm(c1 - c2) < self.p.probe_center_tol:
                    break
                # SC: possible BUG here, wrong input parameter
                pr_s.data[:] = u.shift_zoom(pr_s.data, (1.,)*3,
                        (0, c1[0], c1[1]), (0, c2[0], c2[1]))

                # shift the object
                ob_s = pr_s.views[0].pod.ob_view.storage
                ob_s.data[:] = u.shift_zoom(ob_s.data, (1.,)*3,
                        (0, c1[0], c1[1]), (0, c2[0], c2[1]))

                # shift the exit waves, loop through different exit wave views
                for pv in pr_s.views:
                    pv.pod.exit = u.shift_zoom(pv.pod.exit, (1.,)*2,
                            (c1[0], c1[1]), (c2[0], c2[1]))

                log(4,'Probe recentered from %s to %s'
                            % (str(tuple(c1)), str(tuple(c2))))

    def object_update(self):
        """
        DM object update.
        """
        ob = self.ob
        ob_nrm = self.ob_nrm

        # Fill container
        if not parallel.master:
            ob.fill(0.0)
            ob_nrm.fill(0.)
        else:
            for name, s in self.ob.storages.items():
                # The amplitude of the regularization term has to be scaled with the
                # power of the probe (which is estimated from the power in diffraction patterns).
                # This estimate assumes that the probe power is uniformly distributed through the
                # array and therefore underestimate the strength of the probe terms.
                cfact = self.p.object_inertia * self.mean_power
                if self.p.obj_smooth_std is not None:
                    log(4, 'Smoothing object, average cfact is %.2f'
                        % np.mean(cfact).real)
                    smooth_mfs = [0,
                                  self.p.obj_smooth_std,
                                  self.p.obj_smooth_std]
                    s.data[:] = cfact * u.c_gf(s.data, smooth_mfs)
                else:
                    s.data[:] = s.data * cfact

                ob_nrm.storages[name].fill(cfact)

        # DM update per node
        for name, pod in self.pods.items():
            if not pod.active:
                continue
            pod.object += pod.probe.conj() * pod.exit * pod.object_weight
            ob_nrm[pod.ob_view] += u.abs2(pod.probe) * pod.object_weight

        # Distribute result with MPI
        for name, s in self.ob.storages.items():
            # Get the np arrays
            nrm = ob_nrm.storages[name].data
            parallel.allreduce(s.data)
            parallel.allreduce(nrm)
            s.data /= nrm

            # A possible (but costly) sanity check would be as follows:
            # if all((np.abs(nrm)-np.abs(cfact))/np.abs(cfact) < 1.):
            #    logger.warning('object_inertia seem too high!')
            self.clip_object(s)

    def probe_update(self):
        """
        DM probe update.
        """
        pr = self.pr
        pr_nrm = self.pr_nrm
        pr_buf = self.pr_buf

        # Fill container
        # "cfact" fill
        # BE: was this asymmetric in original code
        # only because of the number of MPI nodes ?
        if parallel.master:
            for name, s in pr.storages.items():
                # Instead of Npts_scan, the number of views should be considered
                # Please note that a call to s.views may be
                # slow for many views in the probe.
                cfact = self.p.probe_inertia * len(s.views) / s.data.shape[0]
                s.data[:] = cfact * s.data
                pr_nrm.storages[name].fill(cfact)
        else:
            pr.fill(0.0)
            pr_nrm.fill(0.0)

        # DM update per node
        for name, pod in self.pods.items():
            if not pod.active:
                continue
            pod.probe += pod.object.conj() * pod.exit * pod.probe_weight
            pr_nrm[pod.pr_view] += u.abs2(pod.object) * pod.probe_weight

        change = 0.

        # Distribute result with MPI
        for name, s in pr.storages.items():
            # MPI reduction of results
            nrm = pr_nrm.storages[name].data
            parallel.allreduce(s.data)
            parallel.allreduce(nrm)
            s.data /= nrm

            # Apply probe support if requested
            self.support_constraint(s)

            # Compute relative change in probe
            buf = pr_buf.storages[name].data
            change += u.norm2(s.data - buf) / u.norm2(s.data)

            # Fill buffer with new probe
            buf[:] = s.data

        return np.sqrt(change / len(pr.storages))


class DMMixin:

    """
    Defaults:

    [alpha]
    default = 1.
    type = float
    lowlim = 0.0
    help = Mix parameter between Difference Map (alpha=1.) and Alternating Projections (alpha=0.)
    """

    def __init__(self, alpha):
        self._alpha = 1.
        self._a = -alpha
        self._b = 1
        self._c = 1.+alpha
        self.alpha = alpha
        self.article = dict(
            title='Probe retrieval in ptychographic coherent diffractive imaging',
            author='Thibault et al.',
            journal='Ultramicroscopy',
            volume=109,
            year=2009,
            page=338,
            doi='10.1016/j.ultramic.2008.12.011',
            comment='The difference map reconstruction algorithm',
        )

    @property
    def alpha(self):
        return self._alpha

    @alpha.setter
    def alpha(self, alpha):
        self._alpha = alpha
        self._a = -alpha
        self._b = 1
        self._c = 1.+alpha

class RAARMixin:
    """
    Defaults:

    [beta]
    default = 0.75
    type = float
    lowlim = 0.0
    help = Beta parameter for RAAR algorithm
    """

    def __init__(self, beta):
        self._beta = 1.
        self._a = 1. - 2. * beta
        self._b = beta
        self._c = 2
        self.beta = beta

    @property
    def beta(self):
        return self._beta

    @beta.setter
    def beta(self, beta):
        self._beta = beta
        self._a = 1. - 2. * beta
        self._b = beta
        self._c = 2


@register()
class DM(_ProjectionEngine, DMMixin):
    """
    A full-fledged Difference Map engine.

    Defaults:

    [name]
    default = DM
    type = str
    help =
    doc =

    """

    def __init__(self, ptycho_parent, pars=None):
        _ProjectionEngine.__init__(self, ptycho_parent, pars)
        DMMixin.__init__(self, self.p.alpha)
        ptycho_parent.citations.add_article(**self.article)


@register()
class RAAR(_ProjectionEngine, RAARMixin):
    """
    A RAAR engine.

    Defaults:

    [name]
    default = RAAR
    type = str
    help =
    doc =

    """

    def __init__(self, ptycho_parent, pars=None):

        _ProjectionEngine.__init__(self, ptycho_parent, pars)
        RAARMixin.__init__(self, self.p.beta)
