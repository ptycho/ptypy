"""
An implementation of the Regularised Average Successive Projections (RASP)
ptychographic algorithm

Authors: Andy Maiden
"""
import time

import numpy as np

from ..engines import projectional, register
from ..engines.utils import projection_update_generalized, log_likelihood
from ..core import geometry
from ..utils import Param
from ..utils.verbose import logger, log
from ..utils import parallel
from .. import io
from .. import utils as u


@register()
class RASP(projectional._ProjectionEngine):
    """
    Regularised Average Successive Projections

    Defaults:

    [name]
    default = RASP
    type = str
    help =
    doc =

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

    [probe_power_correction]
    default = True
    type = bool
    help = A switch to correct probe power
    """

    def __init__(self, ptycho_parent, pars=None):
        super().__init__(ptycho_parent, pars)

        self.article = dict(
                title='Regularised Average Successive Projections: A Brilliant Ptychographic Algorithm',
                author='A. M. Maiden et al.',
                journal='Journal',
                volume=42,
                year=2023,
                page=42,
                doi='doi',
                comment='Regularised Average Successive Projections',
                )
        self.ptycho.citations.add_article(**self.article)

    def engine_initialize(self):
        super().engine_initialize()

        # these are the sum for averaging the global object/probe
        # they are added for each 'successive projection'
        # nmr and dnm stand for numerator and denominator respectively
        self.ob_sum_nmr = self.ob.copy(self.ob.ID + '_ob_sum_nmr')
        self.ob_sum_dnm = self.ob.copy(self.ob.ID + '_ob_sum_dnm')
        self.pr_sum_nmr = self.pr.copy(self.pr.ID + '_pr_sum_nmr')
        self.pr_sum_dnm = self.pr.copy(self.pr.ID + '_pr_sum_dnm')

        if self.p.probe_power_correction:
            self.probe_power_correction()

    def probe_power_correction(self):
        # find probe power from brightest diffraction pattern
        probe_power = 0
        for d in self.di.storages.values():
            max_ind = np.argmax(np.sum(d.data, axis=(1,2)))
            current_pp = np.sum(d.data[max_ind, :, :])
            if current_pp > probe_power:
                probe_power = current_pp

        # correct the initial probe's power
        for name, pod in self.pods.items():
            pod.probe *= np.sqrt(probe_power / (pod.probe.size * np.sum(u.abs2(pod.probe))))

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
        rng = np.random.default_rng()

        # reset the accumulated sum of object/probe before going through all
        # the diffraction view for this iteration
        self.ob_sum_nmr.fill(0.)
        self.ob_sum_dnm.fill(0.)
        self.pr_sum_nmr.fill(0.)
        self.pr_sum_dnm.fill(0.)

        rng.shuffle(vieworder)

        error_dct = {}
        for name in vieworder:
            view = self.di.views[name]
            if not view.active:
                continue

            # A copy of the old exit wave, object and probe
            ex_old = {}
            ob_old = {}
            pr_old = {}
            for name, pod in view.pods.items():
                ex_old[name] = pod.object * pod.probe
                ob_old[name] = pod.object.copy()
                pr_old[name] = pod.probe.copy()

            error_dct[name] = self.fourier_update(view)

            # update object/probe and accumulate their sum for averaging next
            self.object_update(view, ex_old, ob_old, pr_old)
            self.probe_update(view, ex_old, ob_old, pr_old)

        # RASP
        self.rasp_averaging()

        return error_dct

    def fourier_update(self, view):
        """
        General implementation of Fourier update (copied from stochastic)

        Parameters
        ----------
        view : View
        View to diffraction data
        """

        err_fmag, err_exit = projection_update_generalized(view, self._a, self._b, self._c)

        if self.p.compute_log_likelihood:
            err_phot = log_likelihood(view)
        else:
            err_phot = 0.

        return np.array([err_fmag, err_phot, err_exit])

    def object_update(self, view, ex_old, ob_old, pr_old):

        for name, pod in view.pods.items():
            pr_conj = np.conj(pr_old[name])
            pr_abs2 = u.abs2(pr_old[name])

            self.ob_sum_nmr[pod.ob_view] += pr_conj * pod.exit
            self.ob_sum_dnm[pod.ob_view] += pr_abs2

            probe_norm = np.mean(pr_abs2)*self.p.alpha + pr_abs2
            pod.object = ob_old[name] + 0.5*pr_conj*(pod.exit - ex_old[name]) / probe_norm

    def probe_update(self, view, ex_old, ob_old, pr_old):

        for name, pod in view.pods.items():
            ob_conj = np.conj(ob_old[name])
            ob_abs2 = u.abs2(ob_old[name])

            self.pr_sum_nmr[pod.pr_view] += ob_conj * pod.exit
            self.pr_sum_dnm[pod.pr_view] += ob_abs2

            object_norm = self.p.beta + ob_abs2
            pod.probe += ob_conj*(pod.exit - ex_old[name]) / object_norm

    def rasp_averaging(self):

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
