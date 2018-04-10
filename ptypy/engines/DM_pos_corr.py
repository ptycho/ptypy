# -*- coding: utf-8 -*-
"""
Difference Map reconstruction engine.

This file is part of the PTYPY package.

    :copyright: Copyright 2014 by the PTYPY team, see AUTHORS.
    :license: GPLv2, see LICENSE for details.
"""
import numpy as np
import os
import sys
import time
from .. import utils as u
# This is needed for parallesiation and position correction
from ..core import View
from ..core.classes import DEFAULT_ACCESSRULE
import utils_pos_corr as pos_corr
# new imports end
from ..utils.verbose import logger, log
from ..utils import parallel
from utils import basic_fourier_update
from . import BaseEngine

# debugging
import matplotlib.pyplot as plt
plt.switch_backend("Qt5Agg")
# end

__all__ = ['DM']

DEFAULT = u.Param(
    # Difference map parameter
    alpha=1,
    # Number of iterations before probe update starts
    probe_update_start=2,
    # If True update object before probe
    update_object_first=True,
    # Threshold for interruption of the inner overlap loop
    overlap_converge_factor=0.05,
    # Maximum of iterations for the overlap constraint inner loop
    overlap_max_iterations=10,
    # Weight of the current probe estimate in the update, formally cfact
    probe_inertia=1e-9,
    # Weight of the current object in the update, formally DM_smooth_amplitude
    object_inertia=1e-4,
    # If rms error of model vs diffraction data is smaller than this fraction,
    # Fourier constraint is met
    fourier_relax_factor=0.05,
    # Gaussian smoothing (pixel) of the current object prior to update
    obj_smooth_std=None,
    # None or tuple(min, max) of desired limits of the object modulus,
    # currently in under common in documentation
    clip_object=None,
)


class DM_pos_corr(BaseEngine):

    DEFAULT = DEFAULT

    def __init__(self, ptycho_parent, pars=None):
        """
        Difference map reconstruction engine.
        """
        if pars is None:
            pars = DEFAULT.copy()

        super(DM_pos_corr, self).__init__(ptycho_parent, pars)

        # Instance attributes
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
        self.error = []

        # Generate container copies
        self.ob_buf = self.ob.copy(self.ob.ID + '_alt', fill=0.)
        self.ob_nrm = self.ob.copy(self.ob.ID + '_nrm', fill=0.)
        self.ob_viewcover = self.ob.copy(self.ob.ID + '_vcover', fill=0.)

        self.pr_buf = self.pr.copy(self.pr.ID + '_alt', fill=0.)
        self.pr_nrm = self.pr.copy(self.pr.ID + '_nrm', fill=0.)

        # Generate storages for every node, will be saved in a dict
        # self.ob.new_storage()

    def engine_prepare(self):
        """
        Last minute initialization.

        Everything that needs to be recalculated when new data arrives.
        """

        self.pbound = {}
        mean_power = 0.
        for name, s in self.di.storages.iteritems():
            self.pbound[name] = (
                .25 * self.p.fourier_relax_factor**2 * s.pbound_stub)
            mean_power += s.tot_power/np.prod(s.shape)
        self.mean_power = mean_power / len(self.di.storages)

        # Fill object with coverage of views
        # this is only testing
        self.ob_viewcover = self.ob.copy(self.ob.ID + '_vcover', fill=0.)
        # creepy fix??
        self.ob_nrm = self.ob.copy(self.ob.ID + '_nrm', fill=0.)
        # testing end

        for name, s in self.ob_viewcover.storages.iteritems():
            s.fill(s.get_view_coverage())

    def engine_iterate(self, num=1):
        """
        Compute `num` iterations.
        """
        to = 0.
        tf = 0.

        for it in range(num):
            t1 = time.time()

            # Fourier update
            error_dct = self.fourier_update()

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

        # Start position refinement
        if self.curiter == 1 and self.p.pos_ref:
            length = len(self.di.views)
            self.initial_pos = np.zeros((length, 2))
            pos_corr.save_pos(self)

            # Save the initial position
            di_view_order = self.di.views.keys()
            di_view_order.sort()

            self.shape = self.di.views[di_view_order[0]].shape

            for i, name in enumerate(di_view_order):
                di_view = self.di.views[name]
                self.initial_pos[i, 0] = di_view.pod.ob_view.coord[0]
                self.initial_pos[i, 1] = di_view.pod.ob_view.coord[1]

        if self.p.pos_ref_stop > self.curiter >= self.p.pos_ref_start and self.curiter % self.p.pos_ref_cycle == 0 \
                and self.p.pos_ref:
            pos_corr.pos_ref(self)
            pos_corr.save_pos(self)

        # End position refinement

        return error

    def engine_finalize(self):
        """
        Try deleting ever helper container.
        """
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
        for name, di_view in self.di.views.iteritems():
            if not di_view.active:
                continue
            pbound = self.pbound[di_view.storage.ID]
            error_dct[name] = basic_fourier_update(di_view,
                                                   pbound=pbound,
                                                   alpha=self.p.alpha)
        return error_dct

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
                log(4,pre_str + '----- object update -----')
                self.object_update()

            # Exit if probe should not be updated yet
            if not do_update_probe:
                break

            # Update probe
            log(4,pre_str + '----- probe update -----')
            change = self.probe_update()
            log(4,pre_str + 'change in probe is %.3f' % change)

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
                # print(self.ob_viewcover.storages[name].data.shape)

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

    def save_pos(self):
        """
        Debugging purpose, to see if the reconstructed coordinates match the real coordinates.
        """
        pod_names = self.pods.keys()
        pod_names.sort()
        coords = []

        for name in pod_names:
            # print(name)
            pod = self.pods[name]
            coords.append(pod.ob_view.coord)

        coords = np.asarray(coords)
        coords = coords

        directory = "positions_" + sys.argv[0][:-3] + "\\"

        if not os.path.exists(directory):
            os.makedirs(directory)

        np.savetxt(directory + "pos_" + str(self.p.name) + "_" + str(self.curiter).zfill(4) + ".txt", coords)

    # def single_pos_ref(self, pod_name):
    #     pod = self.pods[pod_name]
    #     # Monte Carlo parameter
    #     number_rand_shifts = self.p.number_rand_shifts  # should be a multiple of 4
    #     pxl_size_obj = self.ob.S.values()[0].psize[0]  # Pixel size in the object plane
    #     num_pixel = 15.
    #     end = self.p.pos_ref_stop
    #     start = self.p.pos_ref_start
    #     it = self.curiter
    #     max_shift_dist = pxl_size_obj * num_pixel * (end - it) / (end - start)
    #
    #     if max_shift_dist < pxl_size_obj * 3.:
    #         # smallest distance is 3 pixel in every direction
    #         max_shift_dist = pxl_size_obj * 3.
    #
    #     max_shift_allowed = self.p.max_shift_allowed
    #     delta = np.zeros((number_rand_shifts, 2))   # coordinate shift
    #     errors = np.zeros(number_rand_shifts)       # calculated error for the shifted position
    #     coord = np.copy(pod.ob_view.coord)
    #     self.ar.coord = coord
    #     self.ar.storageID = pod.ob_view.storageID
    #     # Create temporal object view that can be shifted without reformatting
    #
    #     ob_view_temp = View(self.temp_ob, accessrule=self.ar)
    #
    #     # This can be optimized by saving existing iteration fourier error...
    #     error_inital = self.get_fourier_error_view(pod_name, ob_view_temp)
    #
    #     for i in range(number_rand_shifts):
    #
    #         delta_y = np.random.uniform(0, max_shift_dist)
    #         delta_x = np.random.uniform(0, max_shift_dist)
    #
    #         if i % 4 == 1:
    #             delta_y *= -1
    #             delta_x *= -1
    #         elif i % 4 == 2:
    #             delta_x *= -1
    #         elif i % 4 == 3:
    #             delta_y *= -1
    #
    #         delta[i, 0] = delta_y
    #         delta[i, 1] = delta_x
    #
    #         rand_coord = [coord[0] + delta_y, coord[1] + delta_x]
    #         norm = np.linalg.norm(rand_coord - self.initial_pos[int(pod_name[1:]), :])
    #
    #         if norm > max_shift_allowed:
    #             # log(4, "Position drifted too far!")
    #             errors[i] = error_inital + 1.
    #             continue
    #
    #         self.ar.coord = rand_coord
    #         ob_view_temp = View(self.temp_ob, accessrule=self.ar)
    #
    #         if ob_view_temp.dlow[0] < 0:
    #             ob_view_temp.dlow[0] = 0
    #
    #         if ob_view_temp.dlow[1] < 0:
    #             ob_view_temp.dlow[1] = 0
    #
    #         shape = (256, 256)  # still has to be changed
    #         new_obj = np.zeros(shape)
    #
    #         if ob_view_temp.data.shape != (256, 256):
    #             # if the data of the view has the wrong shape, zero-pad the data
    #             # new data for calculating the fourier transform
    #             # calculate limits of the grid
    #             ymin = self.ob.storages["S00G00"].grids()[0][0, 0, 0]
    #             ymax = self.ob.storages["S00G00"].grids()[0][0, -1, -1]
    #             xmin = self.ob.storages["S00G00"].grids()[1][0, 0, 0]
    #             xmax = self.ob.storages["S00G00"].grids()[1][0, -1, -1]
    #
    #             # check if the new array would be bigger
    #             new_xmin = rand_coord[1] - (pxl_size_obj * shape[1] / 2.)
    #             new_xmax = rand_coord[1] + pxl_size_obj * shape[1] / 2.
    #             new_ymin = rand_coord[0] - (pxl_size_obj * shape[0] / 2.)
    #             new_ymax = rand_coord[0] + pxl_size_obj * shape[0] / 2.
    #
    #             # probably not needed
    #             from copy import copy
    #
    #             idx_x_low = 0
    #             idx_x_high = shape[1]
    #             idx_y_low = 0
    #             idx_y_high = shape[0]
    #
    #             if new_ymin < ymin:
    #                 idx_y_low = shape[0] - ob_view_temp.data.shape[0]
    #             elif new_ymax > ymax:
    #                 idx_y_high = ob_view_temp.data.shape[0]
    #
    #             if new_xmin < xmin:
    #                 idx_x_low = shape[1] - ob_view_temp.data.shape[1]
    #             elif new_xmax > xmax:
    #                 idx_x_high = ob_view_temp.data.shape[1]
    #
    #             new_obj[idx_y_low: idx_y_high, idx_x_low: idx_x_high] = copy(ob_view_temp.data)
    #         else:
    #             new_obj = ob_view_temp.data
    #
    #         # debugging stuff
    #         # errors[i] = self.get_fourier_error_view(pod_name, ob_view_temp)
    #         errors[i] = self.get_fourier_error_obj(pod_name, new_obj)
    #         # log(4, "Error pos " + str(rand_coord) + ": " + str(errors[i]))
    #
    #     if np.min(errors) < error_inital:
    #         # if a better coordinate is found
    #         arg = np.argmin(errors)
    #         new_coordinate = np.array([coord[0] + delta[arg, 0], coord[1] + delta[arg, 1]])
    #         # log(4, "New position found for pos: " + str(new_coordinate))
    #
    #     else:
    #         new_coordinate = (0, 0)
    #
    #     return new_coordinate
    #
    # def pos_ref(self):
    #     log(4, "----------- START POS REF -------------")
    #     pod_names = self.pods.keys()
    #     pod_names.sort()
    #     t_pos_s = time.time()
    #     # List of refined coordinates which will be used to reformat the object
    #     # has to be initialized only once
    #     new_coords = np.zeros((len(pod_names), 2))
    #
    #     # Only used for calculating the shifted pos
    #     self.temp_ob = self.ob.copy()
    #
    #     for i, pod_name in enumerate(pod_names):
    #
    #         pod = self.pods[pod_name]
    #         if i == 0:
    #             # create accessrule
    #             # actually this has to be created only once in the first iteration
    #             self.ar = DEFAULT_ACCESSRULE.copy()
    #             self.ar.psize = pod.ob_view.psize
    #             self.ar.shape = pod.ob_view.shape
    #
    #         if pod.active:
    #             new_coords[i, :] = self.single_pos_ref(pod_name)
    #
    #     new_coords = parallel.allreduce(new_coords)
    #
    #     for i, pod_name in enumerate(pod_names):
    #         # change this for the case that the actual new coordinate is (0,0)
    #         pod = self.pods[pod_name]
    #
    #         if new_coords[i, 0] != 0 and new_coords[i, 1] != 0:
    #             log(4, "Old coordinate: " + str(pod.ob_view.coord), parallel=True)
    #             log(4, "New coordinate: " + str(new_coords[i, :]), parallel=True)
    #             pod.ob_view.coord = new_coords[i, :]
    #
    #     # Change the coordinates of the object
    #     self.ob.reformat()
    #
    #     t_pos_f = time.time()
    #     log(4, "Pos ref time: " + str(t_pos_f - t_pos_s))
    #
    # def get_fourier_error_view(self, pod_name, ob_view):
    #     pod = self.pods[pod_name]
    #     probe = np.copy(pod.probe)
    #     probe[np.abs(probe) < np.mean(np.abs(probe)) * .1] = 0
    #
    #     object = ob_view.data
    #     error = np.sum((np.abs(pod.fw(object * probe)) - np.sqrt(pod.diff)) ** 2)
    #
    #     return error
    #
    # def get_fourier_error_obj(self, pod_name, object):
    #     pod = self.pods[pod_name]
    #     probe = np.copy(pod.probe)
    #     probe[np.abs(probe) < np.mean(np.abs(probe)) * .1] = 0
    #
    #     error = np.sum((np.abs(pod.fw(object * probe)) - np.sqrt(pod.diff)) ** 2)
    #
    #     return error
