# -*- coding: utf-8 -*-
"""
Difference Map reconstruction engine that uses numpy arrays instead of iteration.

This file is part of the PTYPY package.

    :copyright: Copyright 2014 by the PTYPY team, see AUTHORS.
    :license: GPLv2, see LICENSE for details.
"""

import time

from ..utils import parallel
from DM_npy import DMNpy
from ..utils.descriptor import defaults_tree
from ..core.manager import Full, Vanilla
from ptypy.gpu.constraints import difference_map_iterator
import numpy as np
import time

__all__ = ['DMGpu']


@defaults_tree.parse_doc('engine.DMGpu')
class DMGpu(DMNpy):
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
        super(DMGpu, self).__init__(ptycho_parent, pars)

    def engine_iterate(self, num=1):
        """
        Compute `num` iterations.
        """
        t1 = time.time()

        for dID, _diffs in self.di.S.iteritems():

            cfact_probe = (self.p.probe_inertia * len(self.vectorised_scan[dID]['meta']['addr']) /
                           self.vectorised_scan[dID]['probe'].shape[0]) * np.ones_like(
                self.vectorised_scan[dID]['probe'])
            pre_fft = self.propagator[dID].pre_fft
            post_fft = self.propagator[dID].post_fft
            psupp = self.probe_support[self.vectorised_scan[dID]['meta']['poe_IDs'][0]].astype(np.complex64)

            cfact_object = self.p.object_inertia * self.mean_power * (self.vectorised_scan[dID]['object viewcover'] + 1.)

            error_dct = {}
            chunked_vectorised_scan, cfact_obj_sec = give_me_a_chunk(self.vectorised_scan[dID], cfact_object, parallel.rank, parallel.size)




            #print("{}, {}".format(psupp.shape, psupp.dtype))

            errors =difference_map_iterator(diffraction=chunked_vectorised_scan['diffraction'],
                                        obj=chunked_vectorised_scan['obj'],
                                        object_weights=chunked_vectorised_scan['object weights'],
                                        cfact_object=cfact_object,
                                        mask=chunked_vectorised_scan['mask'],
                                        probe=chunked_vectorised_scan['probe'],
                                        cfact_probe=cfact_probe,
                                        probe_support=psupp, #self.probe_support[chunked_vectorised_scan['meta']['poe_IDs'][0]],
                                        probe_weights=chunked_vectorised_scan['probe weights'],
                                        exit_wave=chunked_vectorised_scan['exit wave'],
                                        addr=chunked_vectorised_scan['meta']['addr'],
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

            for jx in range(num):
                k = 0
                for idx, name in self.di.views.iteritems():
                    error_dct[idx] = errors[jx, :, k]
                    k += 1
                jx +=1
                error = parallel.gather_dict(error_dct)

            # count up
            self.curiter += num
            t2 = time.time()

            print("I did {} iterations on the gpu in {}s".format(num, (t2-t1)))
            print("gathering probes...")
            probe_dct = parallel.gather_dict(chunked_vectorised_scan['probe_dct'])
            obj_dct = parallel.gather_dict(chunked_vectorised_scan['obj_dict'])
            object_mask_dict = parallel.gather_dict(chunked_vectorised_scan['object_mask_dict'])

            obj_array = self.vectorised_scan['dID']['obj'] * 0.0 # wipe it out for whats coming
            obj_mask_array = np.zeros((obj_array.shape))
            for k in range(len(obj_dct.keys())):
                obj_array[object_mask_dict[k]] += obj_dct[k]
                obj_mask_array[object_mask_dict[k]] += object_mask_dict[k]

            self.vectorised_scan[dID]['obj'] = obj_array / obj_mask_array


            av_probe = np.mean(probe_dct.values(), axis=(-2,-1)) # could the OPR step go in here?
            self.vectorised_scan[dID]['probe'] = av_probe




            print("done")

        return error


def get_num_xy_blocks(num_threads):
    if not num_threads > 0:
        return 1, 1
    elif int(np.sqrt(num_threads))**2 is not num_threads:
        raise NotImplementedError("Only square numbers are allowed for now")
    else:
        x_blocks = int(np.sqrt(num_threads))
        y_blocks = x_blocks
    return x_blocks, y_blocks


def get_labelled_subregions(addr, num_threads):
    _pa, oa, _ea, _da, _ma = zip(*addr)
    # all I care about is the object displacement here

    obj_addr = np.array(oa)


    x_blocks, y_blocks = get_num_xy_blocks(num_threads)


    x = obj_addr[:,1]
    y = obj_addr[:,2]

    x_min, x_max = np.min(x), np.max(x)
    y_min, y_max = np.min(y), np.max(y)
    dx = np.max(np.diff(x)) # assume that this could be sub pixel
    dy = np.max(np.diff(y))

    x_width = (x_max-x_min + dx)/x_blocks# the width of each section
    y_width = (y_max-y_min + dy)/y_blocks# the width of each y section

    # border_x = 1e-8
    # border_y = 1e-8
    border_x = 2* dx
    border_y = 2* dy

    x_intervals = [(x_min + idx*x_width -border_x, x_min + (idx+1)*x_width+border_x) for idx in range(x_blocks)]
    y_intervals = [(y_min + idy*y_width -border_y, y_min + (idy+1)*y_width+border_y) for idy in range(y_blocks)]

    intervals = []
    for xint in x_intervals:
        for yint in y_intervals:
            intervals.append([xint, yint])


    # 1. take a point, and compare it against rois or 3.take a roi and see which points go in it?
    # try 1 for now

    labels = dict.fromkeys(range(len(intervals)))

    for ix in range(len(intervals)):
        labels[ix] = []

    k=0
    for x_pt, y_pt in zip(x, y):
        for ix, roi in enumerate(intervals):
            if ((roi[0][0]<=x_pt< roi[0][1]) and (roi[1][0]<=y_pt<roi[1][1])):
                labels[ix].append(k)
        k+=1
    return labels


def give_me_a_chunk(vectorised_scan, cfact_obj, rank, num_threads):
    addr = vectorised_scan['meta']['addr']
    labels = get_labelled_subregions(addr, num_threads)
    addr_sectioned = addr[labels[rank]]
    from copy import deepcopy
    pa_sec, oa_sec, ea_sec, da_sec, ma_sec = zip(*deepcopy(addr_sectioned)) # this relationship should always hold
    da_sec = np.array(da_sec)
    oa_sec = np.array(oa_sec)
    ma_sec = np.array(ma_sec)
    ea_sec = np.array(ea_sec)

    # object_first THIS MUST BE A COPY
    # we will need to renormalise to the new bottom left hand corner in the address book.
    oa_sec = np.array(oa_sec)
    obj = vectorised_scan['obj']
    x_min, y_min = np.min(oa_sec[:, 1]), np.min(oa_sec[:, 2])
    x_max, y_max = np.max(oa_sec[:, 1]), np.max(oa_sec[:, 2])
    obj_section = obj[:, x_min:x_max, y_min:y_max]
    cfact_obj_sec = cfact_obj[:, x_min:x_max, y_min:y_max]
    obj_mask = np.zeros(obj.shape)
    obj_mask[:, x_min:x_max, y_min:y_max] =1.0 # the amount of weight to give this in the final reconstruction
    # update address book:
    oa_sec[:, 1] -= x_min
    oa_sec[:, 2] -= y_min


    # ALL THIS CAN BE REFERENCES
    # exit_waves now - we won't always want to do this so should provide a switch
    # there will gaps in the ones we need here. We can't crop it from min to max
    exit_wave = vectorised_scan['exit wave']
    exit_wave_sec = exit_wave[ea_sec[:,0]] # this relationship should always hold
    # and so the address book changes
    ea_sec[:,0] = range(len(ea_sec)) # this will always be true

    # now the diffraction patterns
    diffraction = vectorised_scan['diffraction']
    diffraction_sec = diffraction[da_sec[:, 0]]
    # and update the address book
    da_sec[:,0] = range(len(da_sec)) # this nice indexing has been done by cropping the data instead

    # and the mask
    mask = vectorised_scan['mask']
    mask_sec = mask[ma_sec[:, 0]]
    ma_sec[:,0] = range(len(ma_sec))


    addr_sectioned[:,0] = pa_sec
    addr_sectioned[:,1] = oa_sec
    addr_sectioned[:,2] = ea_sec
    addr_sectioned[:,3] = da_sec
    addr_sectioned[:,4] = ma_sec
    chunked_vectorised_scan = {}
    chunked_vectorised_scan['meta']['addr'] = addr_sectioned
    chunked_vectorised_scan['obj_dict'] = {rank: obj_section}
    chunked_vectorised_scan['diffraction'] = diffraction_sec
    chunked_vectorised_scan['mask'] = mask_sec
    chunked_vectorised_scan['exit wave'] = exit_wave_sec
    chunked_vectorised_scan['probe_dict'] = {rank : deepcopy(vectorised_scan['probe'])}
    chunked_vectorised_scan['object_mask_dict'] = {rank : obj_mask}
    chunked_vectorised_scan['probe weights'] = vectorised_scan['probe weights']
    chunked_vectorised_scan['object weights'] = vectorised_scan['object weights']
    return chunked_vectorised_scan, cfact_obj_sec