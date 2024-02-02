import time

import numpy as np
from pycuda import gpuarray
import pycuda.driver as cuda

from ..engines import register
from .WASP_serial import WASP_serial
from ..utils.verbose import logger, log
from ..utils import parallel
from .. import utils as u

from ..accelerate.base.engines import projectional_serial
from ..accelerate.cuda_pycuda import get_context
from ..accelerate.cuda_pycuda.kernels import (FourierUpdateKernel,
    AuxiliaryWaveKernel, PoUpdateKernel, PositionCorrectionKernel,
    PropagationKernel, RealSupportKernel, FourierSupportKernel)
from ..accelerate.cuda_pycuda.array_utils import (ArrayUtilsKernel,
    GaussianSmoothingKernel, TransposeKernel, ClipMagnitudesKernel,
    MaxAbs2Kernel, MassCenterKernel, Abs2SumKernel,InterpolatedShiftKernel)
from ..accelerate.cuda_pycuda.mem_utils import make_pagelocked_paired_arrays as mppa
from ..accelerate.cuda_pycuda.mem_utils import GpuDataManager
from ..accelerate.cuda_pycuda.multi_gpu import get_multi_gpu_communicator


__all__ = ['WASP_pycuda']

EX_MA_BLOCKS_RATIO = 2
MAX_BLOCKS = 99999  # can be used to limit the number of blocks, simulating that they don't fit
#MAX_BLOCKS = 10  # can be used to limit the number of blocks, simulating that they don't fit

@register()
class WASP_pycuda(WASP_serial):
    """
    Weighted Average of Sequential Projections

    Defaults:

    [name]
    default = WASP_pycuda
    type = str
    help =
    doc =

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
        Weighted Average of Sequential Projections
        """
        super().__init__(ptycho_parent, pars)
        self.ma_data = None
        self.mag_data = None
        self.ex_data = None
        self.multigpu = None

    def engine_initialize(self):
        """
        Prepare for reconstruction.
        """
        # Context, Multi GPU communicator and Stream (needs to be in this order)
        self.context, self.queue = get_context(new_queue=False)
        self.multigpu = get_multi_gpu_communicator()
        self.context, self.queue = get_context(new_queue=True)

        # initialise kernels for centring probe if required
        if self.p.probe_center_tol is not None:
            # mass center kernel
            self.MCK = MassCenterKernel(queue=self.queue)
            # absolute sum kernel
            self.A2SK = Abs2SumKernel(dtype=self.pr.dtype, queue=self.queue)
            # interpolated shift kernel
            self.ISK = InterpolatedShiftKernel(queue=self.queue)

        # Clip Magnitudes Kernel
        self.CMK = ClipMagnitudesKernel(queue=self.queue)

        super().engine_initialize()
        self.qu_htod = cuda.Stream()
        self.qu_dtoh = cuda.Stream()

    def _setup_kernels(self):
        """
        Setup kernels, one for each scan. Derive scans from ptycho class
        """
        fpc = 0

        # get the scans
        for label, scan in self.ptycho.model.scans.items():

            kern = u.Param()
            kern.scanmodel = type(scan).__name__
            self.kernels[label] = kern
            # TODO: needs to be adapted for broad bandwidth
            geo = scan.geometries[0]

            # Get info to shape buffer arrays
            fpc = max(scan.max_frames_per_block, fpc)

            # TODO : make this more foolproof
            try:
                nmodes = scan.p.coherence.num_probe_modes * \
                         scan.p.coherence.num_object_modes
            except:
                nmodes = 1

            # create buffer arrays
            ash = (nmodes,) + tuple(geo.shape)
            aux = np.zeros(ash, dtype=np.complex64)
            kern.aux = gpuarray.to_gpu(aux)

            # setup kernels, one for each SCAN.
            log(4, "Setting up FourierUpdateKernel")
            kern.FUK = FourierUpdateKernel(aux, nmodes, queue_thread=self.queue)
            kern.FUK.fshape = (1,) + kern.FUK.fshape[1:]
            kern.FUK.allocate()

            log(4, "Setting up PoUpdateKernel")
            kern.POK = PoUpdateKernel(queue_thread=self.queue)
            kern.POK.allocate()

            log(4, "Setting up AuxiliaryWaveKernel")
            kern.AWK = AuxiliaryWaveKernel(queue_thread=self.queue)
            kern.AWK.allocate()

            log(4, "Setting up ArrayUtilsKernel")
            kern.AUK = ArrayUtilsKernel(queue=self.queue)

            #log(4, "Setting up TransposeKernel")
            #kern.TK = TransposeKernel(queue=self.queue)

            log(4, "setting up MaxAbs2Kernel")
            kern.MAK = MaxAbs2Kernel(queue=self.queue)

            log(4, "Setting up PropagationKernel")
            kern.PROP = PropagationKernel(aux, geo.propagator, self.queue, self.p.fft_lib)
            kern.PROP.allocate()
            kern.resolution = geo.resolution[0]

            if self.do_position_refinement:
                log(4, "Setting up position correction")
                kern.PCK = PositionCorrectionKernel(aux, nmodes, self.p.position_refinement, geo.resolution, queue_thread=self.queue)
                kern.PCK.allocate()

        ex_mem = 0
        mag_mem = 0
        for scan, kern in self.kernels.items():
            if kern.scanmodel in ("GradFull", "BlockGradFull"):
                ex_mem = max(kern.aux.nbytes * 1, ex_mem)
            else:
                ex_mem = max(kern.aux.nbytes * fpc, ex_mem)
            mag_mem = max(kern.FUK.gpu.fdev.nbytes * fpc, mag_mem)
        ma_mem = mag_mem
        mem = cuda.mem_get_info()[0]
        blk = ex_mem * EX_MA_BLOCKS_RATIO + ma_mem + mag_mem
        fit = int(mem - 200 * 1024 * 1024) // blk  # leave 200MB room for safety
        if not fit:
            log(1,"Cannot fit memory into device, if possible reduce frames per block. Exiting...")
            raise SystemExit("ptypy has been exited.")

        # TODO grow blocks dynamically
        nex = min(fit * EX_MA_BLOCKS_RATIO, MAX_BLOCKS)
        nma = min(fit, MAX_BLOCKS)

        log(3, 'PyCUDA max blocks fitting on GPU: exit arrays={}, ma_arrays={}'.format(nex, nma))
        # reset memory or create new
        self.ex_data = GpuDataManager(ex_mem, 0, nex, True)
        self.ma_data = GpuDataManager(ma_mem, 0, nma, False)
        self.mag_data = GpuDataManager(mag_mem, 0, nma, False)
        log(4, "Kernel setup completed")

    def engine_prepare(self):
        super().engine_prepare()

        for name, s in self.ob.S.items():
            s.gpu, s.data = mppa(s.data)
        for name, s in self.pr.S.items():
            s.gpu, s.data = mppa(s.data)

        for label, d in self.di.storages.items():
            prep = self.diff_info[d.ID]
            prep.addr_gpu = gpuarray.to_gpu(prep.addr)
            if self.do_position_refinement:
                prep.mangled_addr_gpu = prep.addr_gpu.copy()

        for label, d in self.ptycho.new_data:
            dID = d.ID
            prep = self.diff_info[dID]
            pID, oID, eID = prep.poe_IDs

            prep.ma_sum_gpu = gpuarray.to_gpu(prep.ma_sum)
            prep.err_fourier_gpu = gpuarray.to_gpu(prep.err_fourier)
            prep.err_phot_gpu = gpuarray.to_gpu(prep.err_phot)
            prep.err_exit_gpu = gpuarray.to_gpu(prep.err_exit)
            if self.do_position_refinement:
                prep.error_state_gpu = gpuarray.empty_like(prep.err_fourier_gpu)

            # these are the sum for averaging the global object/probe
            # they are added for each 'successive projection'
            # nmr and dnm stand for numerator and denominator respectively
            prep.ob_sum_nmr = gpuarray.to_gpu(prep.ob_sum_nmr)
            prep.ob_sum_dnm = gpuarray.to_gpu(prep.ob_sum_dnm )
            prep.pr_sum_nmr = gpuarray.to_gpu(prep.pr_sum_nmr)
            prep.pr_sum_dnm = gpuarray.to_gpu(prep.pr_sum_dnm )

            # prepare page-locked mems:
            ma = self.ma.S[dID].data.astype(np.float32)
            prep.ma = cuda.pagelocked_empty(ma.shape, ma.dtype, order="C", mem_flags=4)
            prep.ma[:] = ma
            ex = self.ex.S[eID].data
            prep.ex = cuda.pagelocked_empty(ex.shape, ex.dtype, order="C", mem_flags=4)
            prep.ex[:] = ex
            mag = prep.mag
            prep.mag = cuda.pagelocked_empty(mag.shape, mag.dtype, order="C", mem_flags=4)
            prep.mag[:] = mag

            self.ex_data.add_data_block()
            self.ma_data.add_data_block()
            self.mag_data.add_data_block()

    def engine_iterate(self, num=1):
        """
        Compute one iteration.
        """
        self.dID_list = list(self.di.S.keys())
        error = {}
        for it in range(num):

            for iblock, dID in enumerate(self.dID_list):

                # find probe, object and exit ID in dependence of dID
                prep = self.diff_info[dID]
                pID, oID, eID = prep.poe_IDs

                # references for kernels
                kern = self.kernels[prep.label]
                FUK = kern.FUK
                AWK = kern.AWK
                POK = kern.POK
                MAK = kern.MAK
                PROP = kern.PROP

                # get aux buffer
                aux = kern.aux

                # local references
                ob = self.ob.S[oID].gpu
                pr = self.pr.S[pID].gpu

                # the copy is important to prevent vieworder being modified,
                # which is always sorted
                vieworder_all = prep.view_IDs_all.copy()
                prep.rng.shuffle(vieworder_all)

                # reset the accumulated sum of object/probe before going
                # through all the diffraction view for this iteration
                ob_sum_nmr = prep.ob_sum_nmr
                ob_sum_dnm = prep.ob_sum_dnm
                pr_sum_nmr = prep.pr_sum_nmr
                pr_sum_dnm = prep.pr_sum_dnm
                ob_sum_nmr.fill(0)
                ob_sum_dnm.fill(0)
                pr_sum_nmr.fill(0)
                pr_sum_dnm.fill(0)

                # Schedule ex, ma, mag to device
                ev_ex, ex_full, data_ex = self.ex_data.to_gpu(prep.ex, dID, self.qu_htod)
                ev_mag, mag_full, data_mag = self.mag_data.to_gpu(prep.mag, dID, self.qu_htod)
                ev_ma, ma_full, data_ma = self.ma_data.to_gpu(prep.ma, dID, self.qu_htod)

                # Reference to ex, ma and mag
                prep.ex_full = ex_full
                prep.mag_full = mag_full
                prep.ma_full = ma_full

                ## synchronize h2d stream with compute stream
                self.queue.wait_for_event(ev_ex)

                # Iterate through views
                for vname in vieworder_all:
                    # only proceed for active view, which is in prep.view_IDs
                    # for this particular rank
                    if vname not in prep.view_IDs:
                        continue

                    # Get local adress and arrays
                    i = prep.view_IDs.index(vname)
                    addr = prep.addr_gpu[i,None]
                    ex_from, ex_to = prep.addr_ex[i]
                    ex = prep.ex_full[ex_from:ex_to]
                    mag = prep.mag_full[i,None]
                    ma = prep.ma_full[i,None]
                    ma_sum = prep.ma_sum_gpu[i,None]
                    err_phot = prep.err_phot_gpu[i,None]
                    err_fourier = prep.err_fourier_gpu[i,None]
                    err_exit = prep.err_exit_gpu[i,None]

                    ## build auxilliary wave
                    AWK.make_aux(aux, addr, ob, pr, ex, c_po=self._c, c_e=1-self._c)

                    ## forward FFT
                    PROP.fw(aux, aux)

                    ## Deviation from measured data
                    self.queue.wait_for_event(ev_mag)
                    if self.p.compute_fourier_error:
                        self.queue.wait_for_event(ev_ma)
                        FUK.fourier_error(aux, addr, mag, ma, ma_sum)
                        FUK.error_reduce(addr, err_fourier)
                    else:
                        FUK.fourier_deviation(aux, addr, mag)
                        self.queue.wait_for_event(ev_ma)
                    FUK.fmag_update_nopbound(aux, addr, mag, ma)

                    ## backward FFT
                    PROP.bw(aux, aux)

                    ## build exit wave
                    AWK.make_exit(aux, addr, ob, pr, ex, c_a=self._b, c_po=self._a, c_e=-(self._a + self._b))
                    if self.p.compute_exit_error:
                        FUK.exit_error(aux,addr)
                        FUK.error_reduce(addr, err_exit)

                    ## build auxilliary wave (ob * pr product)
                    AWK.build_aux2_no_ex(aux, addr, ob, pr)

                    # WASP ob and pr local update
                    ob_old = ob.copy()
                    POK.ob_update_wasp(addr, ob, pr, ex, aux, ob_sum_nmr,
                                       ob_sum_dnm, alpha=self.p.alpha)
                    POK.pr_update_wasp(addr, pr, ob_old, ex, aux, pr_sum_nmr,
                                       pr_sum_dnm, beta=self.p.beta)


                    ## compute log-likelihood
                    if self.p.compute_log_likelihood:
                        PROP.fw(aux, aux)
                        FUK.log_likelihood2(aux, addr, mag, ma, err_phot)

                # WASP averaging

                # collect the sums
                self.multigpu.allReduceSum(ob_sum_nmr)
                self.multigpu.allReduceSum(ob_sum_dnm)
                self.multigpu.allReduceSum(pr_sum_nmr)
                self.multigpu.allReduceSum(pr_sum_dnm)

                POK.avg_wasp(ob, ob_sum_nmr, ob_sum_dnm)
                POK.avg_wasp(pr, pr_sum_nmr, pr_sum_dnm)

                # Clip object
                if self.p.clip_object is not None:
                    self.clip_object(ob)
                    self.queue.synchronize()

                data_ex.record_done(self.queue, 'compute')
                if iblock + len(self.ex_data) < len(self.dID_list):
                    data_ex.from_gpu(self.qu_dtoh)

            # swap direction
            self.dID_list.reverse()

            # Re-center probe
            self.center_probe()

            # position update
            self.position_update()

            self.curiter += 1
            self.ex_data.syncback = False

        # finish all the compute
        self.queue.synchronize()

        for name, s in self.ob.S.items():
            s.gpu.get_async(stream=self.qu_dtoh, ary=s.data)
        for name, s in self.pr.S.items():
            s.gpu.get_async(stream=self.qu_dtoh, ary=s.data)

        for dID, prep in self.diff_info.items():
            err_fourier = prep.err_fourier_gpu.get()
            err_phot = prep.err_phot_gpu.get()
            err_exit = prep.err_exit_gpu.get()
            errs = np.ascontiguousarray(np.vstack([err_fourier, err_phot, err_exit]).T)
            error.update(zip(prep.view_IDs, errs))

        # wait for the async transfers
        self.qu_dtoh.synchronize()

        self.error = error
        return error

    def clip_object(self, ob):
        """
        Clips magnitudes of object into given range.
        """
        cmin, cmax = self.p.clip_object
        self.CMK.clip_magnitudes_to_range(ob, cmin, cmax)

    def position_update(self):
        """
        Position refinement
        (Copied from _ProjectionEngine_pycuda)
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
                        prep.ex_full = self.ISK.interpolate_shift(prep.ex_full,
                                shift)

                log(4,'Probe recentered from %s to %s'
                            % (str(tuple(c1)), str(tuple(c2))))

    def engine_finalize(self):
        """
        clear GPU data and destroy context.
        """
        self.ex_data = None
        self.ma_data = None
        self.mag_data = None

        for name, s in self.ob.S.items():
            del s.gpu
        for name, s in self.pr.S.items():
            del s.gpu
        for dID, prep in self.diff_info.items():
            prep.addr = prep.addr_gpu.get()

        # copy data to cpu
        # this kills the pagelock memory (otherwise we get segfaults in h5py)
        for name, s in self.pr.S.items():
            s.data = np.copy(s.data)
        for name, s in self.ob.S.items():
            s.data = np.copy(s.data)

        super().engine_finalize()
