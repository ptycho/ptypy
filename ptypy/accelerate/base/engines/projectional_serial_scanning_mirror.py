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
from ptypy.accelerate.base.engines.projectional_serial import _ProjectionEngine_serial
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


__all__ = ['DM_scanning_mirror_serial']

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


class _ProjectionEngine_scanning_mirror_serial(_ProjectionEngine_serial):
    """
    A full-fledged Difference Map engine that uses numpy arrays instead of iteration.

    """

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
            kern.aux_pre_fft = np.zeros(aux.shape, dtype=np.complex64)
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

            for dID, block in self.di.S.items():

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

                # setup pre_fft
                for i, view in enumerate(block.views):
                    kern.aux_pre_fft[i] = view.pod.geometry.propagator.pre_fft

                ## compute log-likelihood
                if self.p.compute_log_likelihood:
                    t1 = time.time()
                    AWK.build_aux_no_ex(aux, addr, ob, pr)
                    aux *= kern.aux_pre_fft

                    aux[:] = FW(aux)
                    FUK.log_likelihood(aux, addr, mag, ma, err_phot)
                    self.benchmark.F_LLerror += time.time() - t1

                ## build auxilliary wave
                t1 = time.time()
                AWK.make_aux(aux, addr, ob, pr, ex, c_po=self._c, c_e=1-self._c)
                self.benchmark.A_Build_aux += time.time() - t1
                aux *= kern.aux_pre_fft

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
                aux *= np.conjugate(kern.aux_pre_fft)

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
            for dID, block in self.di.S.items():

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

                # setup pre_fft
                for i, view in enumerate(block.views):
                    kern.aux_pre_fft[i] = view.pod.geometry.propagator.pre_fft

                # We need to re-calculate the current error
                PCK.build_aux(aux, addr, ob, pr)
                aux *= kern.aux_pre_fft

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
                    aux *= kern.aux_pre_fft

                    aux[:] = FW(aux)
                    if self.p.position_refinement.metric == "fourier":
                        PCK.fourier_error(aux, mangled_addr, mag, ma, ma_sum)
                        PCK.error_reduce(mangled_addr, err_fourier)
                    if self.p.position_refinement.metric == "photon":
                        PCK.log_likelihood(aux, mangled_addr, mag, ma, err_fourier)
                    PCK.update_addr_and_error_state(addr, error_state, mangled_addr, err_fourier)

                prep.err_fourier = error_state
                prep.addr = addr


@register()
class DM_scanning_mirror_serial(_ProjectionEngine_scanning_mirror_serial, DMMixin):
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
        _ProjectionEngine_scanning_mirror_serial.__init__(self, ptycho_parent, pars)
        DMMixin.__init__(self, self.p.alpha)
        ptycho_parent.citations.add_article(**self.article)