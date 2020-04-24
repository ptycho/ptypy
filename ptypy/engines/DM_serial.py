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

from .. import utils as u
from ..utils.verbose import logger, log
from ..utils import parallel
from . import BaseEngine, register, DM
from .. import defaults_tree
from ..accelerate.array_based.kernels import FourierUpdateKernel, AuxiliaryWaveKernel, PoUpdateKernel, PositionCorrectionKernel
from ..accelerate.array_based import address_manglers


### TODOS 
# 
# - The Propagator needs to be made somewhere else
# - Get it running faster with MPI (partial sync)
# - implement "batching" when processing frames to lower the pressure on memory
# - Be smarter about the engine.prepare() part
# - Propagator needs to be reconfigurable for a certain batch size, gpyfft hates that.
# - Fourier_update_kernel needs to allow batched execution

## for debugging
#from matplotlib import pyplot as plt

__all__ = ['DM_serial']

parallel = u.parallel


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
                nmodes = scan.p.coherence.num_probe_modes * \
                         scan.p.coherence.num_object_modes
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

            if self.do_position_refinement:
                addr_mangler = address_manglers.RandomIntMangle(int(self.p.position_refinement.amplitude // geo.resolution[0]),
                                                                self.p.position_refinement.start,
                                                                self.p.position_refinement.stop,
                                                                max_bound=int(self.p.position_refinement.max_shift // geo.resolution[0]),
                                                                randomseed=0)
                logger.warning("amplitude is %s " % (self.p.position_refinement.amplitude // geo.resolution[0]))
                logger.warning("max bound is %s " % (self.p.position_refinement.max_shift // geo.resolution[0]))

                kern.PCK = PositionCorrectionKernel(aux, nmodes)
                kern.PCK.allocate()
                kern.PCK.address_mangler = addr_mangler

    def engine_prepare(self):

        super(DM_serial, self).engine_prepare()

        ## Serialize new data ##

        for label, d in self.ptycho.new_data:
            prep = u.Param()

            prep.label = label
            self.diff_info[d.ID] = prep

            prep.mag = np.sqrt(d.data)
            prep.ma = self.ma.S[d.ID].data.astype(np.float32)
            # self.ma.S[d.ID].data = prep.ma
            prep.ma_sum = prep.ma.sum(-1).sum(-1)
            prep.err_fourier = np.zeros_like(prep.ma_sum)

        # Unfortunately this needs to be done for all pods, since
        # the shape of the probe / object was modified.
        # TODO: possible scaling issue, remove the need for padding
        for label, d in self.di.storages.items():
            prep = self.diff_info[d.ID]
            prep.view_IDs, prep.poe_IDs, prep.addr = serialize_array_access(d)
            if self.do_position_refinement:
                prep.original_addr = np.zeros_like(prep.addr)
                prep.original_addr[:] = prep.addr
            pID, oID, eID = prep.poe_IDs

            ob = self.ob.S[oID]
            obn = self.ob_nrm.S[oID]
            obv = self.ob_buf.S[oID]
            misfit = np.asarray(ob.shape[-2:]) % 32
            if (misfit != 0).any():
                pad = 32 - np.asarray(ob.shape[-2:]) % 32
                ob.data = u.crop_pad(ob.data, [[0, pad[0]], [0, pad[1]]], axes=[-2, -1], filltype='project')
                obv.data = u.crop_pad(obv.data, [[0, pad[0]], [0, pad[1]]], axes=[-2, -1], filltype='project')
                obn.data = u.crop_pad(obn.data, [[0, pad[0]], [0, pad[1]]], axes=[-2, -1], filltype='project')
                ob.shape = ob.data.shape
                obv.shape = obv.data.shape
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

            for dID in self.di.S.keys():
                t1 = time.time()

                prep = self.diff_info[dID]
                # find probe, object in exit ID in dependence of dID
                pID, oID, eID = prep.poe_IDs

                # references for kernels
                kern = self.kernels[prep.label]
                FUK = kern.FUK
                AWK = kern.AWK

                pbound = self.pbound_scan[prep.label]
                aux = kern.aux
                FW = kern.FW
                BW = kern.BW

                # get addresses and auxilliary array
                addr = prep.addr
                mag = prep.mag
                ma_sum = prep.ma_sum
                err_fourier = prep.err_fourier

                # local references
                ma = prep.ma
                ob = self.ob.S[oID].data
                pr = self.pr.S[pID].data
                ex = self.ex.S[eID].data

                t1 = time.time()
                AWK.build_aux(aux, addr, ob, pr, ex, alpha=self.p.alpha)
                self.benchmark.A_Build_aux += time.time() - t1

                ## FFT
                t1 = time.time()
                aux[:] = FW(aux)
                self.benchmark.B_Prop += time.time() - t1

                ## Deviation from measured data
                t1 = time.time()
                FUK.fourier_error(aux, addr, mag, ma, ma_sum)
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

            sync = (self.curiter % 1 == 0)
            self.overlap_update(MPI=True)
            parallel.barrier()

            if self.do_position_refinement and (self.curiter):
                do_update_pos = (self.p.position_refinement.stop > self.curiter >= self.p.position_refinement.start)
                do_update_pos &= (self.curiter % self.p.position_refinement.interval) == 0

                # Update positions
                if do_update_pos:
                    """
                    Iterates through all positions and refines them by a given algorithm. 
                    """
                    log(4, "----------- START POS REF -------------")
                    for dID in self.di.S.keys():

                        prep = self.diff_info[dID]
                        pID, oID, eID = prep.poe_IDs
                        ma = self.ma.S[dID].data
                        ob = self.ob.S[oID].data
                        pr = self.pr.S[pID].data
                        kern = self.kernels[prep.label]
                        aux = kern.aux
                        addr = prep.addr
                        original_addr = prep.original_addr # use this instead of the one in the address mangler.
                        mag = prep.mag
                        ma_sum = prep.ma_sum
                        err_fourier = prep.err_fourier

                        PCK = kern.PCK
                        FW = kern.FW

                        error_state = np.zeros_like(err_fourier)
                        error_state[:] = err_fourier
                        log(4, 'Position refinement trial: iteration %s' % (self.curiter))
                        for i in range(self.p.position_refinement.nshifts):
                            mangled_addr = PCK.address_mangler.mangle_address(addr, original_addr, self.curiter)
                            PCK.build_aux(aux, mangled_addr, ob, pr)
                            aux[:] = FW(aux)
                            PCK.fourier_error(aux, mangled_addr, mag, ma, ma_sum)
                            PCK.error_reduce(mangled_addr, err_fourier)
                            PCK.update_addr_and_error_state(addr, error_state, mangled_addr, err_fourier)
                        prep.err_fourier = error_state
                        prep.addr = addr

            self.curiter += 1

        self.error = error
        return error

    def overlap_update(self, MPI=True):
        """
        DM overlap constraint update.
        """
        change = 1.
        # Condition to update probe
        do_update_probe = (self.p.probe_update_start <= self.curiter)

        for inner in range(self.p.overlap_max_iterations):
            prestr = '%d Iteration (Overlap) #%02d:  ' % (parallel.rank, inner)
            # Update object first
            if self.p.update_object_first or (inner > 0):
                # Update object
                log(4, prestr + '----- object update -----', True)
                self.object_update(MPI=(parallel.size > 1 and MPI))

            # Exit if probe should not yet be updated
            if not do_update_probe: break

            # Update probe
            log(4, prestr + '----- probe update -----', True)
            change = self.probe_update(MPI=(parallel.size > 1 and MPI))
            # change = self.probe_update(MPI=(parallel.size>1 and MPI))

            log(4, prestr + 'change in probe is %.3f' % change, True)

            # stop iteration if probe change is small
            if change < self.p.overlap_converge_factor: break

    ## object update
    def object_update(self, MPI=False):
        t1 = time.time()

        for oID, ob in self.ob.storages.items():
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
            cfact = self.p.object_inertia * self.mean_power
            ob.data *= cfact
            obn.data[:] = cfact

        # storage for-loop
        for dID in self.di.S.keys():
            prep = self.diff_info[dID]

            POK = self.kernels[prep.label].POK
            # find probe, object in exit ID in dependence of dID
            pID, oID, eID = prep.poe_IDs

            # scan for loop
            ev = POK.ob_update(prep.addr,
                               self.ob.S[oID].data,
                               self.ob_nrm.S[oID].data,
                               self.pr.S[pID].data,
                               self.ex.S[eID].data)

        for oID, ob in self.ob.storages.items():
            obn = self.ob_nrm.S[oID]
            # MPI test
            if MPI:
                parallel.allreduce(ob.data)
                parallel.allreduce(obn.data)
                ob.data /= obn.data

                # Clip object (This call takes like one ms. Not time critical)
                if self.p.clip_object is not None:
                    clip_min, clip_max = self.p.clip_object
                    ampl_obj = np.abs(ob.data)
                    phase_obj = np.exp(1j * np.angle(ob.data))
                    too_high = (ampl_obj > clip_max)
                    too_low = (ampl_obj < clip_min)
                    ob.data[too_high] = clip_max * phase_obj[too_high]
                    ob.data[too_low] = clip_min * phase_obj[too_low]
            else:
                ob.data /= obn.data

        self.benchmark.object_update += time.time() - t1
        self.benchmark.calls_object += 1

    ## probe update
    def probe_update(self, MPI=False):
        t1 = time.time()

        # storage for-loop
        change = 0

        for pID, pr in self.pr.storages.items():
            prn = self.pr_nrm.S[pID]
            cfact = self.pr_cfact[pID]
            pr.data *= cfact
            prn.data.fill(cfact)

        for dID in self.di.S.keys():
            prep = self.diff_info[dID]

            POK = self.kernels[prep.label].POK
            # find probe, object in exit ID in dependence of dID
            pID, oID, eID = prep.poe_IDs

            # scan for-loop
            ev = POK.pr_update(prep.addr,
                               self.pr.S[pID].data,
                               self.pr_nrm.S[pID].data,
                               self.ob.S[oID].data,
                               self.ex.S[eID].data)

            self.benchmark.probe_update += time.time() - t1
            self.benchmark.calls_probe += 1

        for pID, pr in self.pr.storages.items():

            buf = self.pr_buf.S[pID]
            prn = self.pr_nrm.S[pID]

            # MPI test
            if MPI:
                # if False:
                parallel.allreduce(pr.data)
                parallel.allreduce(prn.data)
                pr.data /= prn.data
            else:
                pr.data /= prn.data

            self.support_constraint(pr)

            change += u.norm2(pr.data - buf.data) / u.norm2(pr.data)
            buf.data[:] = pr.data
            if MPI:
                change = parallel.allreduce(change) / parallel.size

        return np.sqrt(change)

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
