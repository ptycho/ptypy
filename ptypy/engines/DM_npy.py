# -*- coding: utf-8 -*-
"""
Difference Map reconstruction engine that uses numpy arrays instead of iteration.

This file is part of the PTYPY package.

    :copyright: Copyright 2014 by the PTYPY team, see AUTHORS.
    :license: GPLv2, see LICENSE for details.
"""
import numpy as np
import time

from .. import utils as u
from ..utils.verbose import logger, log
from ..utils import parallel
from .utils import basic_fourier_update
from DM import DM
from ..utils.descriptor import defaults_tree
from ..core.manager import Full, Vanilla
from ..gpu import data_utils as du
from ..gpu import constraints as con


__all__ = ['DMNpy']


@defaults_tree.parse_doc('engine.DMNpy')
class DMNpy(DM):
    """
    A full-fledged Difference Map engine that uses numpy arrays instead of iteration.


    Defaults:

    [name]
    default = DMNpy
    type = str
    help =
    doc =

    [alpha]
    default = 1
    type = float
    lowlim = 0.0
    help = Difference map parameter

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

    [fourier_relax_factor]
    default = 0.05
    type = float
    lowlim = 0.0
    help = If rms error of model vs diffraction data is smaller than this fraction, Fourier constraint is met
    doc = Set this value higher for noisy data.

    [obj_smooth_std]
    default = None
    type = int
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

    """

    SUPPORTED_MODELS = [Vanilla, Full]

    def __init__(self, ptycho_parent, pars=None):
        """
        Difference map reconstruction engine.
        """
        super(DMNpy, self).__init__(ptycho_parent, pars)

    def engine_prepare(self):
        """
        Last minute initialization.

        Everything that needs to be recalculated when new data arrives.
        """
        super(DMNpy, self).engine_prepare()
        # and then something to convert the arrays to numpy
        self.vectorised_scan = {}
        self.propagator = {}
        for dID, _diffs in self.di.S.iteritems():
            self.vectorised_scan[dID] = du.pod_to_arrays(self, dID)
            first_view_id = self.vectorised_scan[dID]['meta']['view_IDs'][0]
            self.propagator[dID] = self.di.V[first_view_id].pod.geometry.propagator


    def engine_iterate(self, num=1):
        """
        Compute `num` iterations.
        """
        to = 0.
        tf = 0.
        # run the data for `num` iterations on the cards, then pull the relevant off to sync
        for dID, _diffs in self.di.S.iteritems():

            for it in range(num):
                t1 = time.time()

                # Fourier update
                # error_dct = self.fourier_update()
                exit_wave, error_dct = self.numpy_fourier_update(self.vectorised_scan[dID]['mask'],
                                                                 self.vectorised_scan[dID]['diffraction'],
                                                                 self.vectorised_scan[dID]['obj'],
                                                                 self.vectorised_scan[dID]['probe'],
                                                                 self.vectorised_scan[dID]['exit wave'],
                                                                 self.vectorised_scan[dID]['meta']['addr'],
                                                                 self.propagator[dID],
                                                                 pbound=self.pbound[dID])



                t2 = time.time()
                tf += t2 - t1

                # Overlap update
                self.overlap_update()

                t3 = time.time()
                to += t3 - t2

                # count up
                self.curiter +=1




        logger.info('Time spent in Fourier update: %.2f' % tf)
        logger.info('Time spent in Overlap update: %.2f' % to)
        error = parallel.gather_dict(error_dct)
        return error

    def engine_finalize(self):
        """
        Try deleting ever helper container.
        """
        super(DMNpy, self).engine_finalize(self)
        # and then something to clean up the gpu

    def fourier_update(self):
        """
        DM Fourier constraint update (including DM step).
        """
        error_dct = {}
        for name, di_view in self.di.views.iteritems():

            pbound = self.pbound[di_view.storage.ID]
            error_dct[name] = basic_fourier_update(di_view,
                                                   pbound=pbound,
                                                   alpha=self.p.alpha)
        return error_dct

    def numpy_fourier_update(self, mask, Idata, obj, probe, exit_wave, addr, propagator, pbound):
        error_dct = {}

        exit_wave, errors = con.difference_map_fourier_constraint(mask,
                                                                  Idata,
                                                                  obj,
                                                                  probe,
                                                                  exit_wave,
                                                                  addr,
                                                                  prefilter=propagator.pre_fft,
                                                                  postfilter=propagator.post_fft,
                                                                  pbound=pbound,
                                                                  alpha=self.p.alpha,
                                                                  LL_error=False)

        k=0
        for idx, name in self.di.views.iteritems():
            error_dct[idx] = errors[:, k]
            k+=1

        return exit_wave, error_dct

    def overlap_update(self):
        """
        DM overlap constraint update.
        """
        # Condition to update probe
        do_update_probe = (self.p.probe_update_start <= self.curiter)

        for inner in range(self.p.overlap_max_iterations):
            pre_str = 'Iteration (Overlap) #%02d:  ' % inner

            # Update object first
            if self.p.update_object_first or (inner > 0):
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

            # Recenter the probe
            self.center_probe()

            # Stop iteration if probe change is small
            if change < self.p.overlap_converge_factor:
                break

    def center_probe(self):
        if self.p.probe_center_tol is not None:
            for name, s in self.pr.storages.iteritems():
                c1 = u.mass_center(u.abs2(s.data).sum(0))
                c2 = np.asarray(s.shape[-2:]) // 2
                # fft convention should however use geometry instead
                if u.norm(c1 - c2) < self.p.probe_center_tol:
                    break
                # SC: possible BUG here, wrong input parameter
                s.data[:] = u.shift_zoom(s.data,
                                         (1.,) * 3,
                                         (0, c1[0], c1[1]),
                                         (0, c2[0], c2[1]))

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
            for name, s in self.ob.storages.iteritems():
                # The amplitude of the regularization term has to be scaled with the
                # power of the probe (which is estimated from the power in diffraction patterns).
                # This estimate assumes that the probe power is uniformly distributed through the
                # array and therefore underestimate the strength of the probe terms.
                cfact = self.p.object_inertia * self.mean_power *\
                    (self.ob_viewcover.storages[name].data + 1.)

                if self.p.obj_smooth_std is not None:
                    logger.info(
                        'Smoothing object, average cfact is %.2f'
                        % np.mean(cfact).real)
                    smooth_mfs = [0,
                                  self.p.obj_smooth_std,
                                  self.p.obj_smooth_std]
                    s.data[:] = cfact * u.c_gf(s.data, smooth_mfs)
                else:
                    s.data[:] = s.data * cfact

                ob_nrm.storages[name].fill(cfact)

        # DM update per node
        for name, pod in self.pods.iteritems():
            if not pod.active:
                continue
            pod.object += pod.probe.conj() * pod.exit * pod.object_weight
            ob_nrm[pod.ob_view] += u.cabs2(pod.probe) * pod.object_weight

        # Distribute result with MPI
        for name, s in self.ob.storages.iteritems():
            # Get the np arrays
            nrm = ob_nrm.storages[name].data
            parallel.allreduce(s.data)
            parallel.allreduce(nrm)
            s.data /= nrm

            # A possible (but costly) sanity check would be as follows:
            # if all((np.abs(nrm)-np.abs(cfact))/np.abs(cfact) < 1.):
            #    logger.warning('object_inertia seem too high!')

            # Clip object
            if self.p.clip_object is not None:
                clip_min, clip_max = self.p.clip_object
                ampl_obj = np.abs(s.data)
                phase_obj = np.exp(1j * np.angle(s.data))
                too_high = (ampl_obj > clip_max)
                too_low = (ampl_obj < clip_min)
                s.data[too_high] = clip_max * phase_obj[too_high]
                s.data[too_low] = clip_min * phase_obj[too_low]

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
            for name, s in pr.storages.iteritems():
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
        for name, pod in self.pods.iteritems():
            if not pod.active:
                continue
            pod.probe += pod.object.conj() * pod.exit * pod.probe_weight
            pr_nrm[pod.pr_view] += u.cabs2(pod.object) * pod.probe_weight

        change = 0.

        # Distribute result with MPI
        for name, s in pr.storages.iteritems():
            # MPI reduction of results
            nrm = pr_nrm.storages[name].data
            parallel.allreduce(s.data)
            parallel.allreduce(nrm)
            s.data /= nrm

            # Apply probe support if requested
            support = self.probe_support.get(name)
            if support is not None:
                s.data *= self.probe_support[name]

            # Compute relative change in probe
            buf = pr_buf.storages[name].data
            change += u.norm2(s.data - buf) / u.norm2(s.data)

            # Fill buffer with new probe
            buf[:] = s.data

        return np.sqrt(change / len(pr.storages))
