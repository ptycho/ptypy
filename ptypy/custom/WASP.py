"""
An implementation of the Weighted Average of Sequential Projections (WASP)
ptychographic algorithm

Authors: Andy Maiden
"""
import time

import numpy as np

from ..engines import base, projectional, register
from ..engines.utils import projection_update_generalized, log_likelihood
from ..core import geometry
from ..core.manager import Full, Vanilla, Bragg3dModel, BlockVanilla, BlockFull
from ..utils import Param
from ..utils.verbose import logger, log
from ..utils import parallel
from .. import io
from .. import utils as u

__all__ = ['WASP']


@register()
class WASP(base.PositionCorrectionEngine):
    """
    Weighted Average of Sequential Projections

    Defaults:

    [name]
    default = WASP
    type = str
    help =
    doc =

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
    choices = ['fourier','linear',None]

    [update_object_first]
    default = True
    type = bool
    help = If True update object before probe

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

    [alpha]
    default = 1.
    type = float
    lowlim = 0.0
    help = object step parameter

    [beta]
    default = 1.
    type = float
    lowlim = 0.0
    help = probe step parameter

    [random_seed]
    default = None
    type = int
    lowlim = 0
    help = the seed to the random number generator for shuffling views
    """

    SUPPORTED_MODELS = [Full, Vanilla, Bragg3dModel, BlockVanilla, BlockFull]

    def __init__(self, ptycho_parent, pars=None):
        super().__init__(ptycho_parent, pars)

        self._a = 0.
        self._b = 1.
        self._c = 1.

        self.article = dict(
                title='WASP: Weighted Average of Sequential Projections for ptychographic phase retrieval',
                author='A. M. Maiden, W. Mei and P. Li',
                journal='Optica',
                volume=42,
                year=2024,
                page=42,
                doi='doi',
                comment='Weighted Average of Sequential Projections',
                )
        self.ptycho.citations.add_article(**self.article)

    def engine_initialize(self):
        super().engine_initialize()

        self.error = []

        # these are the sum for averaging the global object/probe
        # they are added for each 'successive projection'
        # nmr and dnm stand for numerator and denominator respectively
        self.ob_sum_nmr = self.ob.copy(self.ob.ID + '_ob_sum_nmr', fill=0.)
        self.ob_sum_dnm = self.ob.copy(self.ob.ID + '_ob_sum_dnm', fill=0., dtype='real')
        self.pr_sum_nmr = self.pr.copy(self.pr.ID + '_pr_sum_nmr', fill=0.)
        self.pr_sum_dnm = self.pr.copy(self.pr.ID + '_pr_sum_dnm', fill=0., dtype='real')

    def engine_prepare(self):
        """Copied from _ProjectionEngine (a large part of it)
        """

        # create RNG only once
        self.rng = np.random.default_rng(self.p.random_seed)

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

    def engine_iterate(self, num=1):
        """
        Compute `num` iterations.
        """
        to = 0.
        tp = 0.
        for it in range(num):
            t1 = time.time()

            # Overlap update
            error_dct = self.overlap_update()

            # Recenter the probe
            self.center_probe()

            t2 = time.time()
            to += t2 - t1

            # Position update
            self.position_update()

            t3 = time.time()
            tp += t3 - t2

            # count up
            self.curiter +=1

        logger.info('Time spent in Overlap update: %.2f' % to)
        logger.info('Time spent in Position update: %.2f' % tp)

        return error_dct

    def overlap_update(self):

        vieworder = list(self.di.views.keys())
        # the sorting is important to ensure they are the same input to RNG in
        # every iteration
        vieworder.sort()

        # reset the accumulated sum of object/probe before going through all
        # the diffraction view for this iteration
        self.ob_sum_nmr.fill(0.)
        self.ob_sum_dnm.fill(0.)
        self.pr_sum_nmr.fill(0.)
        self.pr_sum_dnm.fill(0.)

        self.rng.shuffle(vieworder)

        error_dct = {}
        for name in vieworder:
            view = self.di.views[name]
            if not view.active:
                continue

            # A copy of the old exit wave and object
            ex_old = {}
            ob_old = {}
            for name, pod in view.pods.items():
                ex_old[name] = pod.object * pod.probe
                ob_old[name] = pod.object.copy()

            error_dct[name] = self.fourier_update(view)

            # update object first, then probe, and accumulate their sum for
            # averaging after going through all the views
            self.object_update(view, ex_old)
            self.probe_update(view, ex_old, ob_old)

        # WASP
        self.wasp_averaging()

        return error_dct

    def engine_finalize(self):
        super().engine_finalize()

        # remove helper containers
        containers = [
            self.ob_sum_nmr,
            self.ob_sum_dnm,
            self.pr_sum_nmr,
            self.pr_sum_dnm]

        for c in containers:
            logger.debug('Attempt to remove container %s' % c.ID)
            del self.ptycho.containers[c.ID]

        del self.ob_sum_nmr
        del self.ob_sum_dnm
        del self.pr_sum_nmr
        del self.pr_sum_dnm

    def fourier_update(self, view):
        """
        General implementation of Fourier update (copied from stochastic)

        Parameters
        ----------
        view : View
        View to diffraction data
        """

        err_fmag, err_exit = projection_update_generalized(view, a=self._a,
                                                           b=self._b, c=self._c)

        if self.p.compute_log_likelihood:
            err_phot = log_likelihood(view)
        else:
            err_phot = 0.

        return np.array([err_fmag, err_phot, err_exit])

    def object_update(self, view, ex_old):

        for name, pod in view.pods.items():
            pr_conj = np.conj(pod.probe)
            pr_abs2 = u.abs2(pod.probe)

            self.ob_sum_nmr[pod.ob_view] += pr_conj * pod.exit
            self.ob_sum_dnm[pod.ob_view] += pr_abs2

            probe_norm = np.mean(pr_abs2)*self.p.alpha + pr_abs2
            pod.object += 0.5*pr_conj*(pod.exit - ex_old[name]) / probe_norm

    def probe_update(self, view, ex_old, ob_old):

        for name, pod in view.pods.items():
            # it is important to use ob_old, but not the updated pod.object
            ob_conj = np.conj(ob_old[name])
            ob_abs2 = u.abs2(ob_old[name])

            self.pr_sum_nmr[pod.pr_view] += ob_conj * pod.exit
            self.pr_sum_dnm[pod.pr_view] += ob_abs2

            object_norm = self.p.beta + ob_abs2
            pod.probe += ob_conj*(pod.exit - ex_old[name]) / object_norm

    def wasp_averaging(self):

        for name, s in self.ob.storages.items():
            ob_sum_nmr = self.ob_sum_nmr.storages[name].data
            ob_sum_dnm = self.ob_sum_dnm.storages[name].data

            parallel.allreduce(ob_sum_nmr)
            parallel.allreduce(ob_sum_dnm)

            # avoid division by zero
            is_zero = np.isclose(ob_sum_dnm, 0)
            s.data = np.where(is_zero, ob_sum_nmr, ob_sum_nmr / ob_sum_dnm)

            self.clip_object(s)

        for name, p in self.pr.storages.items():
            pr_sum_nmr = self.pr_sum_nmr.storages[name].data
            pr_sum_dnm = self.pr_sum_dnm.storages[name].data

            parallel.allreduce(pr_sum_nmr)
            parallel.allreduce(pr_sum_dnm)

            # avoid division by zero
            is_zero = np.isclose(pr_sum_dnm, 0)
            p.data = np.where(is_zero, pr_sum_nmr, pr_sum_nmr / pr_sum_dnm)

    def clip_object(self, ob):
        """Copied from _ProjectionEngine
        """

        # Clip object (This call takes like one ms. Not time critical)
        if self.p.clip_object is not None:
            clip_min, clip_max = self.p.clip_object
            ampl_obj = np.abs(ob.data)
            phase_obj = np.exp(1j * np.angle(ob.data))
            too_high = (ampl_obj > clip_max)
            too_low = (ampl_obj < clip_min)
            ob.data[too_high] = clip_max * phase_obj[too_high]
            ob.data[too_low] = clip_min * phase_obj[too_low]

    def center_probe(self):
        """Copied from _ProjectionEngine
        """

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
