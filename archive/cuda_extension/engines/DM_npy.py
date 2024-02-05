# -*- coding: utf-8 -*-
"""
Difference Map reconstruction engine that uses numpy arrays instead of iteration.

This file is part of the PTYPY package.

    :copyright: Copyright 2014 by the PTYPY team, see AUTHORS.
    :license: GPLv2, see LICENSE for details.
"""

from ..utils import parallel
from .DM import DM
from ..core.manager import Full, Vanilla
from ptypy.accelerate.array_based import constraints as con, data_utils as du
import numpy as np
from . import register

#from memory_profiler import profile
__all__ = ['DMNpy']


@register()
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
    choices = ['fourier','linear',None]

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

    def engine_initialize(self):
        self.error = []
        self.ob_viewcover = self.ob.copy(self.ob.ID + '_vcover', fill=0.)

    def engine_prepare(self):
        """
        Last minute initialization.

        Everything that needs to be recalculated when new data arrives.
        """
        super(DMNpy, self).engine_prepare()
        # and then something to convert the arrays to numpy
        self.vectorised_scan = {}
        self.propagator = {}
        for dID, _diffs in self.di.S.items():
            self.vectorised_scan[dID] = du.pod_to_arrays(self, dID)
            first_view_id = self.vectorised_scan[dID]['meta']['view_IDs'][0]
            self.propagator[dID] = self.di.V[first_view_id].pod.geometry.propagator



    def engine_iterate(self, num=1):
        """
        Compute `num` iterations.
        """
        to = 0.
        tf = 0.
        # num=5
        for dID, _diffs in self.di.S.items():






            cfact_probe = (self.p.probe_inertia * len(self.vectorised_scan[dID]['meta']['addr']) /
                           self.vectorised_scan[dID]['probe'].shape[0]) * np.ones_like(
                self.vectorised_scan[dID]['probe'])




            cfact_object = self.p.object_inertia * self.mean_power * (self.vectorised_scan[dID]['object viewcover'] + 1.)

            # path_to_numpy_file = "/dls/mx-scratch/aaron/gpu_real_world_test_cases/i08_iterate_inputs_p100_2modes.npy"
            # path_to_hdf_file = "/dls/mx-scratch/aaron/gpu_real_world_test_cases/i08_iterate_inputs_p100_2modes.h5"

            # import h5py as h5

            # print "saving out the data"
            # f = h5.File(path_to_hdf_file, 'w')
            # f['diffraction'] = self.vectorised_scan[dID]['diffraction']
            # f['exit_wave'] = self.vectorised_scan[dID]['exit wave']
            # f['mask'] = self.vectorised_scan[dID]['mask']
            # f['probe'] = self.vectorised_scan[dID]['probe']
            # f['obj'] = self.vectorised_scan[dID]['obj']
            # f.close()
            # print "done"

            probe_support = self.probe_support[self.vectorised_scan[dID]['meta']['poe_IDs'][0]]
            mean_power = self.mean_power
            object_weights = self.vectorised_scan[dID]['object weights']
            probe_weights = self.vectorised_scan[dID]['probe weights']
            pbound = self.pbound
            object_viewcover = self.vectorised_scan[dID]['object viewcover']
            # overlap_max_iterations = self.vectorised_scan[dID]['overlap_max_iterations']
            # update_object_first = self.vectorised_scan[dID]['update_object_first']
            # obj_smooth_std = self.vectorised_scan[dID]['obj_smooth_std']
            # overlap_converge_factor = self.vectorised_scan[dID]['overlap_converge_factor']
            # probe_center_tol = self.vectorised_scan[dID]['probe_center_tol']
            # update_probe_after = self.vectorised_scan[dID]['update_probe_after']
            # alpha = self.vectorised_scan[dID]['alpha']
            # clip_object = self.vectorised_scan[dID]['clip_object']
            # LL_error = self.vectorised_scan[dID]['LL_error']
            # cfact_object = self.vectorised_scan[dID]['cfact_object']
            # cfact_probe = self.vectorised_scan[dID]['cfact_probe']
            pre_fft = self.propagator[dID].pre_fft
            post_fft = self.propagator[dID].post_fft
            #
            # out_dict = {'pre_fft': pre_fft,
            #             'post_fft' :post_fft,
            #             'probe_support': self.probe_support[self.vectorised_scan[dID]['meta']['poe_IDs'][0]],
            #             'mean_power': mean_power,
            #             'object weights': object_weights,
            #             'probe weights': probe_weights,
            #             'pbound': pbound,
            #             'object_viewcover': object_viewcover,
            #             'overlap_max_iterations': self.p.overlap_max_iterations,
            #             'update_object_first': self.p.update_object_first,
            #             'obj_smooth_std' : self.p.obj_smooth_std,
            #             'overlap_converge_factor': self.p.overlap_converge_factor,
            #             'probe_center_tol': self.p.probe_center_tol,
            #             'update_probe_after' : 0,
            #             'alpha' : self.p.alpha,
            #             'clip_object' : self.p.clip_object,
            #             'LL_error' : False,
            #             'cfact_object': cfact_object,
            #             'cfact_probe': cfact_probe,
            #             'pre_fft' : pre_fft,
            #             'post_fft': post_fft,
            #             'addr' : self.vectorised_scan[dID]['meta']['addr']
            # }
            # print "saving out the metadata"
            # np.save(path_to_numpy_file, out_dict)
            # print "done"
            errors = con.difference_map_iterator(diffraction=self.vectorised_scan[dID]['diffraction'],
                                        obj=self.vectorised_scan[dID]['obj'],
                                        object_weights=self.vectorised_scan[dID]['object weights'],
                                        cfact_object=cfact_object,
                                        mask=self.vectorised_scan[dID]['mask'],
                                        probe=self.vectorised_scan[dID]['probe'],
                                        cfact_probe=cfact_probe,
                                        probe_support=self.probe_support[self.vectorised_scan[dID]['meta']['poe_IDs'][0]],
                                        probe_weights=self.vectorised_scan[dID]['probe weights'],
                                        exit_wave=self.vectorised_scan[dID]['exit wave'],
                                        addr=self.vectorised_scan[dID]['meta']['addr'],
                                        pre_fft=pre_fft,
                                        post_fft=post_fft,
                                        pbound=self.pbound[dID],
                                        overlap_max_iterations=self.p.overlap_max_iterations,
                                        update_object_first=self.p.update_object_first,
                                        obj_smooth_std=self.p.obj_smooth_std,
                                        overlap_converge_factor=self.p.overlap_converge_factor,
                                        probe_center_tol=self.p.probe_center_tol,
                                        probe_update_start=0,
                                        alpha=self.p.alpha,
                                        clip_object=self.p.clip_object,
                                        LL_error=True,
                                        num_iterations=num)




            #yuk yuk yuk
            error_dct = {}
            print(errors.shape)
            jx =0
            for jx in range(num):
                k = 0
                for idx, name in self.di.views.items():
                    error_dct[idx] = errors[jx, :, k]
                    k += 1
                jx +=1
                error = parallel.gather_dict(error_dct)
            # t3 = time.time()
            # to += t3 - t2

            # count up
            self.curiter += num

        # self.mpi_numpy_overlap_update()
        # logger.info('Time spent in Fourier update: %.2f' % tf)
        # logger.info('Time spent in Overlap update: %.2f' % to)
        # error = parallel.gather_dict(error_dct)
        return error

    def engine_finalize(self):
        """
        Try deleting ever helper container.
        """

        del self.ptycho.containers[self.ob_viewcover.ID]
        del self.ob_viewcover
