# -*- coding: utf-8 -*-
"""
Difference Map reconstruction engine.

This file is part of the PTYPY package.

    :copyright: Copyright 2014 by the PTYPY team, see AUTHORS.
    :license: see LICENSE for details.
"""
import numpy as np
import time

from ptypy import utils as u
from ptypy.utils.verbose import logger, log
from ptypy.utils import parallel
from ptypy.engines import register
from ptypy.engines.projectional import _ProjectionEngine, DMMixin, RAARMixin
from ptypy.accelerate.base.kernels import FourierUpdateKernel, AuxiliaryWaveKernel, PoUpdateKernel, PositionCorrectionKernel
from ptypy.accelerate.base import array_utils as au


### TODOS
#
# - The Propagator needs to be made somewhere else
# - Get it running faster with MPI (partial sync)
# - implement "batching" when processing frames to lower the pressure on memory
# - Be smarter about the engine.prepare() part
# - Propagator needs to be reconfigurable for a certain batch size, gpyfft hates that.
# - Fourier_update_kernel needs to allow batched execution


__all__ = ['DM_serial', 'RAAR_serial']

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


class _ProjectionEngine_serial(_ProjectionEngine):
    """
    A full-fledged Difference Map engine that uses numpy arrays instead of iteration.

    """

    def __init__(self, ptycho_parent, pars=None):
        """
        Difference map reconstruction engine.
        """

        super().__init__(ptycho_parent, pars)

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

        super().engine_initialize()
        self._reset_benchmarks()
        self._setup_kernels()

    def _reset_benchmarks(self):
        self.benchmark.A_Build_aux = 0.
        self.benchmark.B_Prop = 0.
        self.benchmark.C_Fourier_update = 0.
        self.benchmark.D_iProp = 0.
        self.benchmark.E_Build_exit = 0.
        self.benchmark.F_LLerror = 0.
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
            kern.scanmodel = type(scan).__name__
            self.kernels[label] = kern

            # TODO: needs to be adapted for broad bandwidth
            geo = scan.geometries[0]

            # Get info to shape buffer arrays
            fpc = scan.max_frames_per_block

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
            kern.resolution = geo.resolution[0]

            if self.do_position_refinement:
                kern.PCK = PositionCorrectionKernel(aux, nmodes, self.p.position_refinement, geo.resolution)
                kern.PCK.allocate()

    def engine_prepare(self):

        super().engine_prepare()

        ## Serialize new data ##

        for label, d in self.ptycho.new_data:
            prep = u.Param()

            prep.label = label
            self.diff_info[d.ID] = prep
            prep.mag = np.sqrt(np.abs(d.data))
            prep.ma = self.ma.S[d.ID].data.astype(np.float32)
            # self.ma.S[d.ID].data = prep.ma
            prep.ma_sum = prep.ma.sum(-1).sum(-1)
            prep.err_phot = np.zeros_like(prep.ma_sum)
            prep.err_fourier = np.zeros_like(prep.ma_sum)
            prep.err_exit = np.zeros_like(prep.ma_sum)

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

                # find probe, object and exit ID in dependence of dID
                prep = self.diff_info[dID]
                pID, oID, eID = prep.poe_IDs

                # references for kernels
                kern = self.kernels[prep.label]
                FUK = kern.FUK
                AWK = kern.AWK
                FW = kern.FW
                BW = kern.BW

                # get addresses and buffers
                addr = prep.addr
                mag = prep.mag
                ma_sum = prep.ma_sum
                err_phot = prep.err_phot
                err_fourier = prep.err_fourier
                err_exit = prep.err_exit
                pbound = self.pbound_scan[prep.label]
                aux = kern.aux

                # local references
                ma = prep.ma
                ob = self.ob.S[oID].data
                pr = self.pr.S[pID].data
                ex = self.ex.S[eID].data

                ## compute log-likelihood
                if self.p.compute_log_likelihood:
                    t1 = time.time()
                    AWK.build_aux_no_ex(aux, addr, ob, pr)
                    aux[:] = FW(aux)
                    FUK.log_likelihood(aux, addr, mag, ma, err_phot)
                    self.benchmark.F_LLerror += time.time() - t1

                ## build auxilliary wave
                t1 = time.time()
                AWK.make_aux(aux, addr, ob, pr, ex, c_po=self._c, c_e=1-self._c)
                self.benchmark.A_Build_aux += time.time() - t1

                ## forward FFT
                t1 = time.time()
                aux[:] = FW(aux)
                self.benchmark.B_Prop += time.time() - t1

                ## Deviation from measured data
                t1 = time.time()
                FUK.fourier_error(aux, addr, mag, ma, ma_sum)
                FUK.error_reduce(addr, err_fourier)
                FUK.fmag_all_update(aux, addr, mag, ma, err_fourier, pbound)
                self.benchmark.C_Fourier_update += time.time() - t1

                ## backward FFT
                t1 = time.time()
                aux[:] = BW(aux)
                self.benchmark.D_iProp += time.time() - t1

                ## build exit wave
                t1 = time.time()
                AWK.make_exit(aux, addr, ob, pr, ex, c_a=self._b, c_po=self._a, c_e=-(self._a+self._b))
                FUK.exit_error(aux,addr)
                FUK.error_reduce(addr, err_exit)
                self.benchmark.E_Build_exit += time.time() - t1

                # update errors
                errs = np.ascontiguousarray(np.vstack([err_fourier, err_phot, err_exit]).T)
                error.update(zip(prep.view_IDs, errs))

                self.benchmark.calls_fourier += 1

            parallel.barrier()

            sync = (self.curiter % 1 == 0)
            self.overlap_update(MPI=True)

            # Recenter the probe
            self.center_probe()

            parallel.barrier()

            self.position_update()

            self.curiter += 1

        self.error = error
        return error

    def position_update(self):
        """
        Position refinement
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
            for dID in self.di.S.keys():

                prep = self.diff_info[dID]
                pID, oID, eID = prep.poe_IDs
                ma = self.ma.S[dID].data
                ob = self.ob.S[oID].data
                pr = self.pr.S[pID].data
                kern = self.kernels[prep.label]
                aux = kern.aux
                addr = prep.addr
                original_addr = prep.original_addr
                mangled_addr = addr.copy()
                mag = prep.mag
                ma_sum = prep.ma_sum
                err_fourier = prep.err_fourier

                PCK = kern.PCK
                FW = kern.FW

                # Keep track of object boundaries
                max_oby = ob.shape[-2] - aux.shape[-2] - 1
                max_obx = ob.shape[-1] - aux.shape[-1] - 1

                # We need to re-calculate the current error
                PCK.build_aux(aux, addr, ob, pr)
                aux[:] = FW(aux)
                if self.p.position_refinement.metric == "fourier":
                    PCK.fourier_error(aux, addr, mag, ma, ma_sum)
                    PCK.error_reduce(addr, err_fourier)
                if self.p.position_refinement.metric == "photon":
                    PCK.log_likelihood(aux, addr, mag, ma, err_fourier)
                error_state = np.zeros_like(err_fourier)
                error_state[:] = err_fourier
                PCK.mangler.setup_shifts(self.curiter, nframes=addr.shape[0])

                log(4, 'Position refinement trial: iteration %s' % (self.curiter))
                for i in range(PCK.mangler.nshifts):
                    PCK.mangler.get_address(i, addr, mangled_addr, max_oby, max_obx)
                    PCK.build_aux(aux, mangled_addr, ob, pr)
                    aux[:] = FW(aux)
                    if self.p.position_refinement.metric == "fourier":
                        PCK.fourier_error(aux, mangled_addr, mag, ma, ma_sum)
                        PCK.error_reduce(mangled_addr, err_fourier)
                    if self.p.position_refinement.metric == "photon":
                        PCK.log_likelihood(aux, mangled_addr, mag, ma, err_fourier)
                    PCK.update_addr_and_error_state(addr, error_state, mangled_addr, err_fourier)

                prep.err_fourier = error_state
                prep.addr = addr


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
            log(4, prestr + 'change in probe is %.3f' % change, True)

            # stop iteration if probe change is small
            if change < self.p.overlap_converge_factor: break


    ## object update
    def object_update(self, MPI=False):
        t1 = time.time()

        for oID, ob in self.ob.storages.items():
            obn = self.ob_nrm.S[oID]
            cfact = self.p.object_inertia * self.mean_power

            if self.p.obj_smooth_std is not None:
                log(4, 'Smoothing object, cfact is %.2f' % cfact)
                smooth_mfs = [self.p.obj_smooth_std, self.p.obj_smooth_std]
                ob.data = cfact * au.complex_gaussian_filter(ob.data, smooth_mfs)
            else:
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
            else:
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

    def engine_finalize(self, benchmark=True):
        """
        try deleting ever helper contianer
        """
        if parallel.master and benchmark:
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

        if self.do_position_refinement and self.p.position_refinement.record:
            for label, d in self.di.storages.items():
                prep = self.diff_info[d.ID]
                res = self.kernels[prep.label].resolution
                for i,view in enumerate(d.views):
                    for j,(pname, pod) in enumerate(view.pods.items()):
                        delta = (prep.addr[i][j][1][1:] - prep.original_addr[i][j][1][1:]) * res
                        pod.ob_view.coord += delta
                        pod.ob_view.storage.update_views(pod.ob_view)
            self.ptycho.record_positions = True

        super().engine_finalize()


@register()
class DM_serial(_ProjectionEngine_serial, DMMixin):
    """
    A full-fledged Difference Map engine serialized.

    Defaults:

    [name]
    default = DM_serial
    type = str
    help =
    doc =

    """

    def __init__(self, ptycho_parent, pars=None):
        _ProjectionEngine_serial.__init__(self, ptycho_parent, pars)
        DMMixin.__init__(self, self.p.alpha)
        ptycho_parent.citations.add_article(**self.article)


@register()
class RAAR_serial(_ProjectionEngine_serial, RAARMixin):
    """
    A RAAR engine.

    Defaults:

    [name]
    default = RAAR_serial
    type = str
    help =
    doc =

    """

    def __init__(self, ptycho_parent, pars=None):

        _ProjectionEngine_serial.__init__(self, ptycho_parent, pars)
        RAARMixin.__init__(self, self.p.beta)
