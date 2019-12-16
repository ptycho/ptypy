# -*- coding: utf-8 -*-
"""
Difference Map reconstruction engine.

This file is part of the PTYPY package.

    :copyright: Copyright 2014 by the PTYPY team, see AUTHORS.
    :license: GPLv2, see LICENSE for details.
"""

# from .. import core
from __future__ import division

import numpy as np
import time
from ptypy.accelerate.ocl.npy_kernels import Fourier_update_kernel
from ptypy.accelerate.ocl.npy_kernels import PO_update_kernel

from .. import utils as u
from ..utils.verbose import logger, log
from ..utils import parallel
from . import BaseEngine, register, DM
from .. import defaults_tree
from ..accelerate.ocl.npy_kernels_for_block import FourierUpdateKernel
from ..accelerate.ocl.npy_kernels_for_block import PoUpdateKernel
from ..accelerate.ocl.npy_kernels_for_block import AuxiliaryWaveKernel

### TODOS 
# 
# - The Propagator needs to be made somewhere else
# - Get it running faster with MPI (partial sync)
# - implement "batching" when processing frames to lower the pressure on memory
# - Be smarter about the engine.prepare() part
# - Propagator needs to be reconfigurable for a certain batch size, gpyfft hates that.
# - Fourier_update_kernel needs to allow batched execution

## for debugging
from matplotlib import pyplot as plt

__all__ = ['DM_serial']

parallel = u.parallel
MPI = (parallel.size > 1)


def gaussian_kernel(sigma, size=None, sigma_y=None, size_y=None):
    size = int(size)
    sigma = np.float(sigma)
    if not size_y:
        size_y = size
    if not sigma_y:
        sigma_y = sigma

    x, y = np.mgrid[-size:size + 1, -size_y:size_y + 1]

    g = np.exp(-(x ** 2 / (2 * sigma ** 2) + y ** 2 / (2 * sigma_y ** 2)))
    return g / g.sum()


def serialize_array_access(diff_storage):
    # Sort views according to layer in diffraction stack 
    views = diff_storage.views
    dlayers = [view.dlayer for view in views]
    views = [views[i] for i in np.argsort(dlayers)]
    view_IDs = [view.ID for view in views]

    # Master pod
    mpod = views[0].pod

    # Determine linked storages for probe, object and exit waves
    pr = mpod.pr_view.storage
    ob = mpod.ob_view.storage
    ex = mpod.ex_view.storage

    poe_ID = (pr.ID, ob.ID, ex.ID)

    addr = []
    for view in views:
        address = []

        for pname, pod in view.pods.items():
            ## store them for each pod
            # create addresses
            a = np.array(
                [(pod.pr_view.dlayer, pod.pr_view.dlow[0], pod.pr_view.dlow[1]),
                 (pod.ob_view.dlayer, pod.ob_view.dlow[0], pod.ob_view.dlow[1]),
                 (pod.ex_view.dlayer, pod.ex_view.dlow[0], pod.ex_view.dlow[1]),
                 (pod.di_view.dlayer, pod.di_view.dlow[0], pod.di_view.dlow[1]),
                 (pod.ma_view.dlayer, pod.ma_view.dlow[0], pod.ma_view.dlow[1])])

            address.append(a)

            if pod.pr_view.storage.ID != pr.ID:
                log(1, "Splitting probes for one diffraction stack is not supported in " + __name__)
            if pod.ob_view.storage.ID != ob.ID:
                log(1, "Splitting objects for one diffraction stack is not supported in " + __name__)
            if pod.ex_view.storage.ID != ex.ID:
                log(1, "Splitting exit stacks for one diffraction stack is not supported in " + __name__)

        ## store data for each view
        # adresses
        addr.append(address)

    # store them for each storage
    return view_IDs, poe_ID, np.array(addr).astype(np.int32)


@register()
class DM_serial(DM.DM):
    """
    A full-fledged Difference Map engine that uses numpy arrays instead of iteration.

    """

    def __init__(self, ptycho_parent, pars=None):
        """
        Difference map reconstruction engine.
        """

        super(DM_serial, self).__init__(ptycho_parent, pars)

        # allocator for READ only buffers
        # self.const_allocator = cl.tools.ImmediateAllocator(queue, cl.mem_flags.READ_ONLY)
        ## gaussian filter
        # dummy kernel
        """
        if not self.p.obj_smooth_std: 
            gauss_kernel = gaussian_kernel(1,1).astype(np.float32)
        else:
            gauss_kernel = gaussian_kernel(self.p.obj_smooth_std,self.p.obj_smooth_std).astype(np.float32)

        kernel_pars = {'kernel_sh_x' : gauss_kernel.shape[0], 'kernel_sh_y': gauss_kernel.shape[1]}
        """

        self.benchmark = u.Param()

        # Stores all information needed with respect to the diffraction storages.
        self.diff_info = {}
        self.ob_cfact = {}
        self.pr_cfact = {}
        self.kernels = {}

    def engine_initialize(self):
        """
        Prepare for reconstruction.
        """

        super(DM_serial, self).engine_initialize()
        self._reset_benchmarks()
        self._setup_kernels()

    def _reset_benchmarks(self):
        self.benchmark.A_Build_aux = 0.
        self.benchmark.B_Prop = 0.
        self.benchmark.C_Fourier_update = 0.
        self.benchmark.D_iProp = 0.
        self.benchmark.E_Build_exit = 0.
        self.benchmark.probe_update = 0.
        self.benchmark.object_update = 0.
        self.benchmark.calls_fourier = 0
        self.benchmark.calls_object = 0
        self.benchmark.calls_probe = 0

    def _setup_kernels(self):
        """
        Setup kernels, one for each scan. Derive scans from ptycho class
        """
        # get the scans
        for label, scan in self.ptycho.model.scans.items():

            kern = u.Param()
            self.kernels[label] = kern

            # TODO: needs to be adapted for broad bandwidth
            geo = scan.geometries[0]

            # Get info to shape buffer arrays
            # TODO: make this part of the engine rather than scan
            fpc = self.ptycho.frames_per_block

            # TODO : make this more foolproof
            try:
                nmodes = scan.p.coherence.num_probe_modes
            except:
                nmodes = 1

            # create buffer arrays
            ash = (fpc * nmodes,) + tuple(geo.shape)
            aux = np.zeros(ash, dtype=np.complex64)
            kern.aux = aux

            # setup kernels, one for each SCAN.
            kern.FUK = FourierUpdateKernel(aux, nmodes)
            kern.FUK.allocate()

            kern.POK = PoUpdateKernel()
            kern.POK.allocate()

            kern.AWK = AuxiliaryWaveKernel()
            kern.AWK.allocate()

            kern.FW = geo.propagator.fw
            kern.BW = geo.propagator.bw

    def engine_prepare(self):

        super(DM_serial, self).engine_prepare()

        ## Serialize new data ##

        for label, d in self.ptycho.new_data:
            prep = u.Param()

            prep.label = label
            self.diff_info[d.ID] = prep

            prep.mag = np.sqrt(d.data)
            prep.mask_sum = self.ma.S[d.ID].data.sum(-1).sum(-1)
            prep.err_fourier = np.zeros_like(prep.mask_sum)

        # Unfortunately this needs to be done for all pods, since
        # the shape of the probe / object was modified.
        # TODO: possible scaling issue
        for label, d in self.di.storages.items():
            prep = self.diff_info[d.ID]
            prep.view_IDs, prep.poe_IDs, prep.addr = serialize_array_access(d)
            pID, oID, eID = prep.poe_IDs

            ob = self.ob.S[oID]
            obn = self.ob_nrm.S[oID]
            obb = self.ob_buf.S[oID]
            misfit = np.asarray(ob.shape[-2:]) % 32
            if (misfit != 0).any():
                pad = 32 - np.asarray(ob.shape[-2:]) % 32
                ob.data = u.crop_pad(ob.data, [[0, pad[0]], [0, pad[1]]], axes=[-2, -1], filltype='project')
                obb.data = u.crop_pad(obb.data, [[0, pad[0]], [0, pad[1]]], axes=[-2, -1], filltype='project')
                obn.data = u.crop_pad(obn.data, [[0, pad[0]], [0, pad[1]]], axes=[-2, -1], filltype='project')
                ob.shape = ob.data.shape
                obb.shape = obb.data.shape
                obn.shape = obn.data.shape

            # calculate c_facts
            cfact = self.p.object_inertia * self.mean_power
            self.ob_cfact[oID] = cfact / u.parallel.size

            pr = self.pr.S[pID]
            cfact = self.p.probe_inertia * len(pr.views) / pr.data.shape[0]
            self.pr_cfact[pID] = cfact / u.parallel.size

    def engine_iterate(self, num=1):
        """
        Compute one iteration.
        """

        for it in range(num):

            error = {}

            for inner in range(self.p.overlap_max_iterations):
                #if inner > 0 : break
                # storage for-loop
                change = 0

                do_update_probe = (self.curiter >= self.p.probe_update_start)
                do_update_object = (self.p.update_object_first or (inner > 0) or not do_update_probe)
                do_update_fourier = (inner == 0)

                # initialize probe and object buffer to receive an update
                if do_update_object:
                    for oID, ob in self.ob_buf.storages.items():
                        cfact = self.ob_cfact[oID]
                        obn = self.ob_nrm.S[oID]
                        """
                        if self.p.obj_smooth_std is not None:
                            logger.info('Smoothing object, cfact is %.2f' % cfact)
                            t2 = time.time()
                            self.prg.gaussian_filter(queue, (info[3],info[4]), None, obj_gpu.data, self.gauss_kernel_gpu.data)
                            queue.finish()
                            obj_gpu *= cfact
                            print 'gauss: '  + str(time.time()-t2)
                        else:
                            obj_gpu *= cfact
                        """
                        ob.data *= cfact
                        obn.data[:] = cfact

                if do_update_probe:
                    for pID, pr in self.pr_buf.storages.items():
                        prn = self.pr_nrm.S[pID]
                        cfact = self.pr_cfact[pID]
                        pr.data *= cfact
                        prn.data.fill(cfact)

                for dID in self.di.S.keys():
                    t1 = time.time()

                    prep = self.diff_info[dID]
                    # find probe, object in exit ID in dependence of dID
                    pID, oID, eID = prep.poe_IDs

                    # references for kernels
                    kern = self.kernels[prep.label]
                    FUK = kern.FUK
                    AWK = kern.AWK
                    POK = kern.POK

                    pbound = self.pbound_scan[prep.label]
                    aux = kern.aux
                    FW = kern.FW
                    BW = kern.BW

                    # get addresses and auxilliary array
                    addr = prep.addr
                    mag = prep.mag
                    mask_sum = prep.mask_sum
                    err_fourier = prep.err_fourier

                    # local references
                    ma = self.ma.S[dID].data
                    ob = self.ob.S[oID].data
                    obn = self.ob_nrm.S[oID].data
                    obb = self.ob_buf.S[oID].data
                    pr = self.pr.S[pID].data
                    prn = self.pr_nrm.S[pID].data
                    prb = self.pr_buf.S[pID].data
                    ex = self.ex.S[eID].data

                    # Fourier update.
                    if do_update_fourier:
                        log(4, '----- Fourier update -----', True)
                        t1 = time.time()
                        AWK.build_aux(aux, addr, ob, pr, ex, alpha=self.p.alpha)
                        self.benchmark.A_Build_aux += time.time() - t1

                        ## FFT
                        t1 = time.time()
                        aux[:] = FW(aux)
                        self.benchmark.B_Prop += time.time() - t1

                        ## Deviation from measured data
                        t1 = time.time()
                        FUK.fourier_error(aux, addr, mag, ma, mask_sum)
                        FUK.error_reduce(addr, err_fourier)
                        FUK.fmag_all_update(aux, addr, mag, ma, err_fourier, pbound)
                        self.benchmark.C_Fourier_update += time.time() - t1

                        t1 = time.time()
                        aux[:] = BW(aux)
                        self.benchmark.D_iProp += time.time() - t1

                        ## apply changes #2
                        t1 = time.time()
                        AWK.build_exit(aux, addr, ob, pr, ex)
                        self.benchmark.E_Build_exit += time.time() - t1

                        err_phot = np.zeros_like(err_fourier)
                        err_exit = np.zeros_like(err_fourier)
                        errs = np.ascontiguousarray(np.vstack([err_fourier, err_phot, err_exit]).T)
                        error.update(zip(prep.view_IDs, errs))

                        self.benchmark.calls_fourier += 1

                    parallel.barrier()

                    prestr = '%d Iteration (Overlap) #%02d:  ' % (parallel.rank, inner)

                    # Update object
                    if do_update_object:
                        # Update object
                        log(4, prestr + '----- object update -----', True)
                        t1 = time.time()

                        # scan for loop
                        ev = POK.ob_update(addr, obb, obn, pr, ex)

                        # print 'object update: ' + str(time.time()-t1)
                        self.benchmark.object_update += time.time() - t1
                        self.benchmark.calls_object += 1

                    # Exit if probe should not yet be updated
                    if do_update_probe:
                        # Update probe
                        log(4, prestr + '----- probe update -----', True)
                        t1 = time.time()

                        # scan for-loop
                        ev = POK.pr_update(addr, prb, prn, ob, ex)

                        # print 'probe update: ' + str(time.time()-t1)
                        self.benchmark.probe_update += time.time() - t1
                        self.benchmark.calls_probe += 1

                if do_update_object:
                    for oID, ob in self.ob.storages.items():
                        obn = self.ob_nrm.S[oID]
                        obb = self.ob_buf.S[oID]
                        # MPI test
                        if MPI:
                            parallel.allreduce(obb.data)
                            parallel.allreduce(obn.data)
                            obb.data /= obn.data
                        else:
                            obb.data /= obn.data

                        self.clip_object(obb)
                        ob.data[:] = obb.data

                if do_update_probe:
                    for pID, pr in self.pr.storages.items():

                        prb = self.pr_buf.S[pID]
                        prn = self.pr_nrm.S[pID]

                        # MPI test
                        if MPI:
                            # if False:
                            parallel.allreduce(prb.data)
                            parallel.allreduce(prn.data)
                            prb.data /= prn.data
                        else:
                            prb.data /= prn.data

                        self.support_constraint(prb)

                        change += u.norm2(pr.data - prb.data) / u.norm2(prb.data)
                        pr.data[:] = prb.data
                        if MPI:
                            change = parallel.allreduce(change) / parallel.size



                change = np.sqrt(change)

                log(4, prestr + 'change in probe is %.3f' % change, True)

                # stop iteration if probe change is small
                if change < self.p.overlap_converge_factor: break

            parallel.barrier()
            self.curiter += 1

        self.error = error
        return error

    def engine_finalize(self):
        """
        try deleting ever helper contianer
        """
        if parallel.master:
            print("----- BENCHMARKS ----")
            acc = 0.
            for name in sorted(self.benchmark.keys()):
                t = self.benchmark[name]
                if name[0] in 'ABCDEFGHI':
                    print('%20s : %1.3f ms per iteration' % (name, t / self.benchmark.calls_fourier * 1000))
                    acc += t
                elif str(name) == 'probe_update':
                    print('%20s : %1.3f ms per call. %d calls' % (
                        name, t / self.benchmark.calls_probe * 1000, self.benchmark.calls_probe))
                elif str(name) == 'object_update':
                    print('%20s : %1.3f ms per call. %d calls' % (
                        name, t / self.benchmark.calls_object * 1000, self.benchmark.calls_object))

            print('%20s : %1.3f ms per iteration. %d calls' % (
                'Fourier_total', acc / self.benchmark.calls_fourier * 1000, self.benchmark.calls_fourier))

        self._reset_benchmarks()

        for original in [self.pr, self.ob, self.ex, self.di, self.ma]:
            original.delete_copy()

        # delete local references to container buffer copies
