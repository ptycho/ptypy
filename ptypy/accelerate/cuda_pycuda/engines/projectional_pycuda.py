# -*- coding: utf-8 -*-
"""
Difference Map reconstruction engine.

This file is part of the PTYPY package.

    :copyright: Copyright 2014 by the PTYPY team, see AUTHORS.
    :license: see LICENSE for details.
"""

import numpy as np
import time
from pycuda import gpuarray
import pycuda.driver as cuda

from ptypy import utils as u
from ptypy.utils.verbose import logger, log
from ptypy.utils import parallel
from ptypy.engines import register
from ptypy.engines.projectional import DMMixin, RAARMixin
from ptypy.accelerate.base.engines import projectional_serial
from .. import get_context
from ..kernels import FourierUpdateKernel, AuxiliaryWaveKernel, PoUpdateKernel, PositionCorrectionKernel
from ..kernels import PropagationKernel, RealSupportKernel, FourierSupportKernel
from ..array_utils import ArrayUtilsKernel, GaussianSmoothingKernel,\
TransposeKernel, ClipMagnitudesKernel, MassCenterKernel, Abs2SumKernel,\
InterpolatedShiftKernel
from ..mem_utils import make_pagelocked_paired_arrays as mppa
from ..multi_gpu import get_multi_gpu_communicator

__all__ = ['DM_pycuda', 'RAAR_pycuda']

class _ProjectionEngine_pycuda(projectional_serial._ProjectionEngine_serial):

    """
    Defaults:

    [probe_update_cuda_atomics]
    default = False
    type = bool
    help = For GPU, use the atomics version for probe update kernel

    [object_update_cuda_atomics]
    default = True
    type = bool
    help = For GPU, use the atomics version for object update kernel

    [fft_lib]
    default = reikna
    type = str
    help = Choose the pycuda-compatible FFT module.
    doc = One of:
      - ``'reikna'`` : the reikna packaga (fast load, competitive compute for streaming)
      - ``'cuda'`` : ptypy's cuda wrapper (delayed load, but fastest compute if all data is on GPU)
      - ``'skcuda'`` : scikit-cuda (fast load, slowest compute due to additional store/load stages)
    choices = 'reikna','cuda','skcuda'
    userlevel = 2

    """
    def __init__(self, ptycho_parent, pars=None):
        """
        Difference map reconstruction engine.
        """
        super().__init__(ptycho_parent, pars)
        self.multigpu = None

    def engine_initialize(self):
        """
        Prepare for reconstruction.
        """
        # Context, Multi GPU communicator and Stream (needs to be in this order)
        self.context, self.queue = get_context(new_context=True, new_queue=False)
        self.multigpu = get_multi_gpu_communicator()
        self.context, self.queue = get_context(new_context=False, new_queue=True)

        # Gaussian Smoothing Kernel
        self.GSK = GaussianSmoothingKernel(queue=self.queue)

        # Real/Fourier Support Kernel
        self.RSK = {}
        self.FSK = {}

        # Clip Magnitudes Kernel
        self.CMK = ClipMagnitudesKernel(queue=self.queue)

        # initialise kernels for centring probe if required
        if self.p.probe_center_tol is not None:
            # mass center kernel
            self.MCK = MassCenterKernel(queue=self.queue)
            # absolute sum kernel
            self.A2SK = Abs2SumKernel(dtype=self.pr.dtype, queue=self.queue)
            # interpolated shift kernel
            self.ISK = InterpolatedShiftKernel(queue=self.queue)

        super().engine_initialize()

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
            kern.aux = gpuarray.to_gpu(aux)

            # setup kernels, one for each SCAN.
            log(4, "Setting up FourierUpdateKernel")
            kern.FUK = FourierUpdateKernel(aux, nmodes, queue_thread=self.queue)
            kern.FUK.allocate()

            log(4, "Setting up PoUpdateKernel")
            kern.POK = PoUpdateKernel(queue_thread=self.queue)
            kern.POK.allocate()

            log(4, "Setting up AuxiliaryWaveKernel")
            kern.AWK = AuxiliaryWaveKernel(queue_thread=self.queue)
            kern.AWK.allocate()

            log(4, "Setting up ArrayUtilsKernel")
            kern.AUK = ArrayUtilsKernel(queue=self.queue)

            log(4, "Setting up TransposeKernel")
            kern.TK = TransposeKernel(queue=self.queue)

            log(4, "Setting up PropagationKernel")
            kern.PROP = PropagationKernel(aux, geo.propagator, self.queue, self.p.fft_lib)
            kern.PROP.allocate()
            kern.resolution = geo.resolution[0]

            if self.do_position_refinement:
                log(4, "Setting up PositionCorrectionKernel")
                kern.PCK = PositionCorrectionKernel(aux, nmodes, self.p.position_refinement, geo.resolution, queue_thread=self.queue)
                kern.PCK.allocate()
            log(4, "Kernel setup completed")

    def engine_prepare(self):

        super().engine_prepare()

        for name, s in self.ob.S.items():
            s.gpu = gpuarray.to_gpu(s.data)
        for name, s in self.ob_buf.S.items():
            s.gpu, s.data = mppa(s.data)
        for name, s in self.ob_nrm.S.items():
            s.gpu, s.data = mppa(s.data)
        for name, s in self.pr.S.items():
            s.gpu, s.data = mppa(s.data)
        for name, s in self.pr_buf.S.items():
            s.gpu, s.data = mppa(s.data)
        for name, s in self.pr_nrm.S.items():
            s.gpu, s.data = mppa(s.data)

        use_tiles = (not self.p.probe_update_cuda_atomics) or (not self.p.object_update_cuda_atomics)

        # TODO : like the serialization this one is needed due to object reformatting
        for label, d in self.di.storages.items():
            prep = self.diff_info[d.ID]
            prep.addr_gpu = gpuarray.to_gpu(prep.addr)
            if use_tiles:
                prep.addr2 = np.ascontiguousarray(np.transpose(prep.addr, (2, 3, 0, 1)))
                prep.addr2_gpu = gpuarray.to_gpu(prep.addr2)
            if self.do_position_refinement:
                prep.mangled_addr_gpu = prep.addr_gpu.copy()

        for label, d in self.ptycho.new_data:
            prep = self.diff_info[d.ID]
            pID, oID, eID = prep.poe_IDs
            s = self.ex.S[eID]
            s.gpu = gpuarray.to_gpu(s.data)
            s = self.ma.S[d.ID]
            s.gpu = gpuarray.to_gpu(s.data.astype(np.float32))

            prep.mag = gpuarray.to_gpu(prep.mag)
            prep.ma_sum = gpuarray.to_gpu(prep.ma_sum)
            prep.err_fourier_gpu = gpuarray.to_gpu(prep.err_fourier)
            prep.err_phot_gpu = gpuarray.to_gpu(prep.err_phot)
            prep.err_exit_gpu = gpuarray.to_gpu(prep.err_exit)
            if self.do_position_refinement:
                prep.error_state_gpu = gpuarray.empty_like(prep.err_fourier_gpu)

    def engine_iterate(self, num=1):
        """
        Compute one iteration.
        """
        queue = self.queue

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
                PROP = kern.PROP

                # get addresses and buffers
                addr = prep.addr_gpu
                mag = prep.mag
                ma_sum = prep.ma_sum
                err_fourier = prep.err_fourier_gpu
                err_phot = prep.err_phot_gpu
                err_exit = prep.err_exit_gpu
                pbound = self.pbound_scan[prep.label]
                aux = kern.aux

                # local references
                ma = self.ma.S[dID].gpu
                ob = self.ob.S[oID].gpu
                pr = self.pr.S[pID].gpu
                ex = self.ex.S[eID].gpu

                ## compute log-likelihood
                if self.p.compute_log_likelihood:
                    AWK.build_aux_no_ex(aux, addr, ob, pr)
                    PROP.fw(aux, aux)
                    FUK.log_likelihood(aux, addr, mag, ma, err_phot)

                ## build auxilliary wave
                #AWK.build_aux(aux, addr, ob, pr, ex, alpha=self.p.alpha)
                AWK.make_aux(aux, addr, ob, pr, ex, c_po=self._c, c_e=1-self._c)

                ## forward FFT
                PROP.fw(aux, aux)

                ## Deviation from measured data
                FUK.fourier_error(aux, addr, mag, ma, ma_sum)
                FUK.error_reduce(addr, err_fourier)
                FUK.fmag_all_update(aux, addr, mag, ma, err_fourier, pbound)

                ## backward FFT
                PROP.bw(aux, aux)

                ## build exit wave
                #AWK.build_exit(aux, addr, ob, pr, ex, alpha=self.p.alpha)
                AWK.make_exit(aux, addr, ob, pr, ex, c_a=self._b, c_po=self._a, c_e=-(self._a + self._b))
                FUK.exit_error(aux, addr)
                FUK.error_reduce(addr, err_exit)

            parallel.barrier()

            sync = (self.curiter % 1 == 0)
            self.overlap_update()

            self.center_probe()

            parallel.barrier()
            self.position_update()

            self.curiter += 1
            queue.synchronize()

        for name, s in self.ob.S.items():
            s.data[:] = s.gpu.get()
        for name, s in self.pr.S.items():
            s.data[:] = s.gpu.get()

        # costly but needed to sync back with
        # for name, s in self.ex.S.items():
        #     s.data[:] = s.gpu.get()
        for dID, prep in self.diff_info.items():
            err_fourier = prep.err_fourier_gpu.get()
            err_phot = prep.err_phot_gpu.get()
            err_exit = prep.err_exit_gpu.get()
            errs = np.ascontiguousarray(np.vstack([err_fourier, err_phot, err_exit]).T)
            error.update(zip(prep.view_IDs, errs))

        self.error = error
        return error

    def position_update(self):
        """
        Position refinement
        """
        if not self.do_position_refinement or (not self.curiter):
            return
        do_update_pos = (self.p.position_refinement.stop > self.curiter >= self.p.position_refinement.start)
        do_update_pos &= (self.curiter % self.p.position_refinement.interval) == 0
        use_tiles = (not self.p.probe_update_cuda_atomics) or (not self.p.object_update_cuda_atomics)

        # Update positions
        if do_update_pos:
            """
            Iterates through all positions and refines them by a given algorithm.
            """
            log(4, "----------- START POS REF -------------")
            for dID in self.di.S.keys():

                prep = self.diff_info[dID]
                pID, oID, eID = prep.poe_IDs
                ma = self.ma.S[dID].gpu
                ob = self.ob.S[oID].gpu
                pr = self.pr.S[pID].gpu
                kern = self.kernels[prep.label]
                aux = kern.aux
                addr = prep.addr_gpu
                original_addr = prep.original_addr
                mangled_addr = prep.mangled_addr_gpu
                mag = prep.mag
                ma_sum = prep.ma_sum
                err_fourier = prep.err_fourier_gpu
                error_state = prep.error_state_gpu

                PCK = kern.PCK
                TK  = kern.TK
                PROP = kern.PROP

                # Keep track of object boundaries
                max_oby = ob.shape[-2] - aux.shape[-2] - 1
                max_obx = ob.shape[-1] - aux.shape[-1] - 1

                # We need to re-calculate the current error
                PCK.build_aux(aux, addr, ob, pr)
                PROP.fw(aux, aux)
                if self.p.position_refinement.metric == "fourier":
                    PCK.fourier_error(aux, addr, mag, ma, ma_sum)
                    PCK.error_reduce(addr, err_fourier)
                if self.p.position_refinement.metric == "photon":
                    PCK.log_likelihood(aux, addr, mag, ma, err_fourier)
                cuda.memcpy_dtod(dest=error_state.ptr,
                                    src=err_fourier.ptr,
                                    size=err_fourier.nbytes)

                PCK.mangler.setup_shifts(self.curiter, nframes=addr.shape[0])

                log(4, 'Position refinement trial: iteration %s' % (self.curiter))
                for i in range(PCK.mangler.nshifts):
                    PCK.mangler.get_address(i, addr, mangled_addr, max_oby, max_obx)
                    PCK.build_aux(aux, mangled_addr, ob, pr)
                    PROP.fw(aux, aux)
                    if self.p.position_refinement.metric == "fourier":
                        PCK.fourier_error(aux, mangled_addr, mag, ma, ma_sum)
                        PCK.error_reduce(mangled_addr, err_fourier)
                    if self.p.position_refinement.metric == "photon":
                        PCK.log_likelihood(aux, mangled_addr, mag, ma, err_fourier)
                    PCK.update_addr_and_error_state(addr, error_state, mangled_addr, err_fourier)

                cuda.memcpy_dtod(dest=err_fourier.ptr,
                                    src=error_state.ptr,
                                    size=err_fourier.nbytes)
                if use_tiles:
                    s1 = addr.shape[0] * addr.shape[1]
                    s2 = addr.shape[2] * addr.shape[3]
                    TK.transpose(addr.reshape(s1, s2), prep.addr2_gpu.reshape(s2, s1))


    def center_probe(self):
        if self.p.probe_center_tol is not None:
            for name, pr_s in self.pr.storages.items():
                psum_d = self.A2SK.abs2sum(pr_s.gpu)
                c1 = self.MCK.mass_center(psum_d).get()
                c2 = (np.asarray(pr_s.shape[-2:]) // 2).astype(c1.dtype)

                shift = c2 - c1
                # exit if the current center of mass is within the tolerance
                if u.norm(shift) < self.p.probe_center_tol:
                    break

                # shift the probe
                pr_s.gpu = self.ISK.interpolate_shift(pr_s.gpu, shift)

                # shift the object
                ob_s = pr_s.views[0].pod.ob_view.storage
                ob_s.gpu = self.ISK.interpolate_shift(ob_s.gpu, shift)

                # shift the exit waves
                for dID in self.di.S.keys():
                    prep = self.diff_info[dID]
                    pID, oID, eID = prep.poe_IDs
                    if pID == name:
                        self.ex.S[eID].gpu = self.ISK.interpolate_shift(
                                self.ex.S[eID].gpu, shift)

                log(4,'Probe recentered from %s to %s'
                            % (str(tuple(c1)), str(tuple(c2))))


    ## object update
    def object_update(self, MPI=False):
        use_atomics = self.p.object_update_cuda_atomics
        queue = self.queue
        queue.synchronize()
        for oID, ob in self.ob.storages.items():
            obn = self.ob_nrm.S[oID]
            cfact = self.ob_cfact[oID]

            if self.p.obj_smooth_std is not None:
                log(4, 'Smoothing object, cfact is %.2f' % cfact)
                obb = self.ob_buf.S[oID]
                smooth_mfs = [self.p.obj_smooth_std, self.p.obj_smooth_std]
                self.GSK.convolution(ob.gpu, smooth_mfs, tmp=obb.gpu)

            ob.gpu *= cfact
            obn.gpu.fill(cfact)
            queue.synchronize()

        # storage for-loop
        for dID in self.di.S.keys():
            prep = self.diff_info[dID]

            POK = self.kernels[prep.label].POK
            # find probe, object in exit ID in dependence of dID
            pID, oID, eID = prep.poe_IDs

            # scan for loop
            addr = prep.addr_gpu if use_atomics else prep.addr2_gpu
            ev = POK.ob_update(addr,
                               self.ob.S[oID].gpu,
                               self.ob_nrm.S[oID].gpu,
                               self.pr.S[pID].gpu,
                               self.ex.S[eID].gpu,
                               atomics = use_atomics)
            queue.synchronize()

        for oID, ob in self.ob.storages.items():
            obn = self.ob_nrm.S[oID]
            self.multigpu.allReduceSum(ob.gpu)
            self.multigpu.allReduceSum(obn.gpu)
            ob.gpu /= obn.gpu

            self.clip_object(ob.gpu)
            queue.synchronize()

    ## probe update
    def probe_update(self, MPI=False):
        queue = self.queue

        # storage for-loop
        change_gpu = gpuarray.zeros((1,), dtype=np.float32)
        cfact = self.p.probe_inertia
        use_atomics = self.p.probe_update_cuda_atomics
        for pID, pr in self.pr.storages.items():
            prn = self.pr_nrm.S[pID]
            cfact = self.pr_cfact[pID]
            pr.gpu *= cfact
            prn.gpu.fill(cfact)

        for dID in self.di.S.keys():
            prep = self.diff_info[dID]

            POK = self.kernels[prep.label].POK
            # find probe, object in exit ID in dependence of dID
            pID, oID, eID = prep.poe_IDs

            # scan for-loop
            addr = prep.addr_gpu if use_atomics else prep.addr2_gpu
            ev = POK.pr_update(addr,
                               self.pr.S[pID].gpu,
                               self.pr_nrm.S[pID].gpu,
                               self.ob.S[oID].gpu,
                               self.ex.S[eID].gpu,
                               atomics=use_atomics)
            queue.synchronize()

        for pID, pr in self.pr.storages.items():

            buf = self.pr_buf.S[pID]
            prn = self.pr_nrm.S[pID]

            self.multigpu.allReduceSum(pr.gpu)
            self.multigpu.allReduceSum(prn.gpu)
            pr.gpu /= prn.gpu
            self.support_constraint(pr)

            ## calculate change on GPU
            queue.synchronize()
            AUK = self.kernels[list(self.kernels)[0]].AUK
            buf.gpu -= pr.gpu
            change_gpu += (AUK.norm2(buf.gpu) / AUK.norm2(pr.gpu))
            buf.gpu[:] = pr.gpu
            self.multigpu.allReduceSum(change_gpu)
            change = change_gpu.get().item() / parallel.size

        return np.sqrt(change)

    def support_constraint(self, storage=None):
        """
        Enforces 2D support constraint on probe.
        """
        if storage is None:
            for s in self.pr.storages.values():
                self.support_constraint(s)

        # Fourier space
        support = self._probe_fourier_support.get(storage.ID)
        if support is not None:
            if storage.ID not in self.FSK:
                supp = support.astype(np.complex64)
                self.FSK[storage.ID] = FourierSupportKernel(supp, self.queue, self.p.fft_lib)
                self.FSK[storage.ID].allocate()
            self.FSK[storage.ID].apply_fourier_support(storage.gpu)

        # Real space
        support = self._probe_support.get(storage.ID)
        if support is not None:
            if storage.ID not in self.RSK:
                self.RSK[storage.ID] = RealSupportKernel(support.astype(np.complex64))
                self.RSK[storage.ID].allocate()
            self.RSK[storage.ID].apply_real_support(storage.gpu)

    def clip_object(self, ob):
        """
        Clips magnitudes of object into given range.
        """
        if self.p.clip_object is not None:
            cmin, cmax = self.p.clip_object
            self.CMK.clip_magnitudes_to_range(ob, cmin, cmax)

    def engine_finalize(self):
        """
        clear GPU data and destroy context.
        """
        for name, s in self.ob.S.items():
            del s.gpu
        for name, s in self.ob_buf.S.items():
            del s.gpu
        for name, s in self.ob_nrm.S.items():
            del s.gpu
        for name, s in self.pr.S.items():
            del s.gpu
        for name, s in self.pr_buf.S.items():
            del s.gpu
        for name, s in self.pr_nrm.S.items():
            del s.gpu

        # copy addr to cpu
        for dID, prep in self.diff_info.items():
            prep.addr = prep.addr_gpu.get()

        # copy data to cpu
        # this kills the pagelock memory (otherwise we get segfaults in h5py)
        for name, s in self.pr.S.items():
            s.data = np.copy(s.data)

        self.context.pop()
        self.context.detach()

        # we don't need the  "benchmarking" in DM_serial
        super().engine_finalize(benchmark=False)


@register(name="DM_pycuda_nostream")
class DM_pycuda(_ProjectionEngine_pycuda, DMMixin):
    """
    A full-fledged Difference Map engine accelerated with pycuda.

    Defaults:

    [name]
    default = DM_pycuda
    type = str
    help =
    doc =

    """

    def __init__(self, ptycho_parent, pars=None):
        _ProjectionEngine_pycuda.__init__(self, ptycho_parent, pars)
        DMMixin.__init__(self, self.p.alpha)
        ptycho_parent.citations.add_article(**self.article)


@register(name="RAAR_pycuda_nostream")
class RAAR_pycuda(_ProjectionEngine_pycuda, RAARMixin):
    """
    A RAAR engine in accelerated with pycuda.

    Defaults:

    [name]
    default = RAAR_pycuda
    type = str
    help =
    doc =

    """

    def __init__(self, ptycho_parent, pars=None):
        _ProjectionEngine_pycuda.__init__(self, ptycho_parent, pars)
        RAARMixin.__init__(self, self.p.beta)
