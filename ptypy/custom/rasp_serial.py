import numpy as np
import time

from ..engines import register
from .rasp import RASP
from ..utils.verbose import logger, log
from ..utils import parallel
from .. import utils as u
from ..accelerate.base.kernels import FourierUpdateKernel, AuxiliaryWaveKernel, PoUpdateKernel, PositionCorrectionKernel
from ..accelerate.base import array_utils as au
from ..accelerate.base.engines import projectional_serial

__all__ = ['RASP_serial']


@register()
class RASP_serial(RASP):
    """
    Regularised Average Successive Projections

    Defaults:

    [name]
    default = RASP_serial
    type = str
    help =
    doc =

    [compute_exit_error]
    default = False
    type = bool
    help = A switch for computing the exitwave error (this can impact the performance of the engine)

    [compute_fourier_error]
    default = False
    type = bool
    help = A switch for computing the fourier error (this can impact the performance of the engine)
    """

    def __init__(self, ptycho_parent, pars=None):
        super().__init__(ptycho_parent, pars)

        self._a = 0.
        self._b = 1.
        self._c = 1.

        self.benchmark = u.Param()

        # Stores all information needed with respect to the diffraction storages.
        self.diff_info = {}
        self.kernels = {}

    def engine_initialize(self):
        """
        Prepare for reconstruction. (Copied from _ProjectionEngine_serial)
        """

        super().engine_initialize()
        self._reset_benchmarks()
        self._setup_kernels()

    def _reset_benchmarks(self):
        """(Copied from _ProjectionEngine_serial, almost)
        """
        self.benchmark.A_Build_aux = 0.
        self.benchmark.B_Prop = 0.
        self.benchmark.C_Fourier_update = 0.
        self.benchmark.D_iProp = 0.
        self.benchmark.E_Build_exit = 0.
        self.benchmark.F_LLerror = 0.
        self.benchmark.rasp_ob_pr_update = 0.
        self.benchmark.rasp_averaging = 0.
        self.benchmark.calls_fourier = 0
        self.benchmark.calls_rasp_ob_pr_update = 0
        self.benchmark.calls_rasp_averaging = 0

    def _setup_kernels(self):
        """
        Setup kernels, one for each scan. Derive scans from ptycho class
        (Copied from _ProjectionEngine_serial)
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
        """
        Last minute initialization.

        Everything that needs to be recalculated when new data arrives.
        """
        if self.ptycho.new_data:
            # recalculate everything
            mean_power = 0.
            max_power = 0.
            for s in self.di.storages.values():
                mean_power += s.mean_power
                if s.max_power > max_power:
                    max_power = s.max_power
            self.mean_power = mean_power / len(self.di.storages)
            self.max_power = max_power

        ## Serialize new data ##
        for label, d in self.ptycho.new_data:
            prep = u.Param()
            prep.label = label
            self.diff_info[d.ID] = prep
            prep.mag = np.sqrt(np.abs(d.data))
            prep.ma = self.ma.S[d.ID].data.astype(np.float32)
            prep.ma_sum = prep.ma.sum(-1).sum(-1)
            prep.err_phot = np.zeros_like(prep.ma_sum)
            prep.err_fourier = np.zeros_like(prep.ma_sum)
            prep.err_exit = np.zeros_like(prep.ma_sum)

        # Unfortunately this needs to be done for all pods, since
        # the shape of the probe / object was modified.
        # TODO: possible scaling issue, remove the need for padding
        for label, d in self.di.storages.items():
            prep = self.diff_info[d.ID]
            prep.view_IDs, prep.poe_IDs, prep.addr = projectional_serial.serialize_array_access(d)
            if self.do_position_refinement:
                prep.original_addr = np.zeros_like(prep.addr)
                prep.original_addr[:] = prep.addr
            pID, oID, eID = prep.poe_IDs

            # Keep a list of view indices
            prep.rng = np.random.default_rng()
            prep.vieworder = np.arange(prep.addr.shape[0])

            # Modify addresses, copy pa into ea and remove da/ma
            prep.addr_ex = np.vstack([prep.addr[:,0,2,0], prep.addr[:,-1,2,0]+1]).T
            prep.addr[:,:,2] = prep.addr[:,:,0]
            prep.addr[:,:,3:,0] = 0

            # Reference to ex
            prep.ex = self.ex.S[eID].data

            # these are the sum for averaging the global object/probe
            # they are added for each 'successive projection'
            # nmr and dnm stand for numerator and denominator respectively
            prep.ob_sum_nmr = np.zeros_like(self.ob.S[oID].data, dtype=np.complex64)
            prep.ob_sum_dnm = np.zeros_like(self.ob.S[oID].data, dtype=np.float32)
            prep.pr_sum_nmr = np.zeros_like(self.pr.S[pID].data, dtype=np.complex64)
            prep.pr_sum_dnm = np.zeros_like(self.pr.S[pID].data, dtype=np.float32)

            if self.p.probe_power_correction:
                pr_S = self.pr.S[pID]
                probe_sz = pr_S.data.size
                pr_S.data *= np.sqrt(self.max_power / (probe_sz * np.sum(au.abs2(pr_S.data))))

    def engine_iterate(self, num=1):
        """
        Compute one iteration.
        """
        for it in range(num):

            error_dct = {}

            for dID in self.di.S.keys():

                # find probe, object and exit ID in dependence of dID
                prep = self.diff_info[dID]
                pID, oID, eID = prep.poe_IDs

                # references for kernels
                kern = self.kernels[prep.label]
                FUK = kern.FUK
                AWK = kern.AWK
                POK = kern.POK
                FW = kern.FW
                BW = kern.BW

                # global aux buffer
                aux = kern.aux

                # references for ob, pr
                ob = self.ob.S[oID].data
                pr = self.pr.S[pID].data

                # shuffle view order
                vieworder = prep.vieworder
                prep.rng.shuffle(vieworder)

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

                # Iterate through views
                for i in vieworder:
                    # Get local adress and arrays
                    addr = prep.addr[i,None]
                    ex_from, ex_to = prep.addr_ex[i]
                    ex = prep.ex[ex_from:ex_to]
                    mag = prep.mag[i,None]
                    ma = prep.ma[i,None]
                    ma_sum = prep.ma_sum[i,None]
                    err_phot = prep.err_phot[i,None]
                    err_fourier = prep.err_fourier[i,None]
                    err_exit = prep.err_exit[i,None]

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
                    if self.p.compute_fourier_error:
                        FUK.fourier_error(aux, addr, mag, ma, ma_sum)
                        FUK.error_reduce(addr, err_fourier)
                    else:
                        FUK.fourier_deviation(aux, addr, mag)
                    FUK.fmag_update_nopbound(aux, addr, mag, ma)
                    self.benchmark.C_Fourier_update += time.time() - t1

                    ## backward FFT
                    t1 = time.time()
                    aux[:] = BW(aux)
                    self.benchmark.D_iProp += time.time() - t1

                    ## build exit wave
                    t1 = time.time()
                    AWK.make_exit(aux, addr, ob, pr, ex, c_a=self._b, c_po=self._a, c_e=-(self._a+self._b))
                    if self.p.compute_exit_error:
                        FUK.exit_error(aux,addr)
                        FUK.error_reduce(addr, err_exit)
                    self.benchmark.E_Build_exit += time.time() - t1
                    self.benchmark.calls_fourier += 1

                    ## build auxilliary wave (ob * pr product)
                    t1 = time.time()
                    AWK.build_aux_no_ex(aux, addr, ob, pr)
                    self.benchmark.A_Build_aux += time.time() - t1

                    # RASP ob and pr local update
                    t1 = time.time()

                    POK.rasp_ob_pr_update(addr, ob, pr, ex, aux, ob_sum_nmr,
                                          ob_sum_dnm, pr_sum_nmr, pr_sum_dnm,
                                          alpha=self.p.alpha, beta=self.p.beta)

                    self.benchmark.rasp_ob_pr_update += time.time() - t1
                    self.benchmark.calls_rasp_ob_pr_update += 1

                    ## compute log-likelihood
                    if self.p.compute_log_likelihood:
                        t1 = time.time()
                        aux[:] = FW(aux)
                        FUK.log_likelihood(aux, addr, mag, ma, err_phot)
                        self.benchmark.F_LLerror += time.time() - t1

                # update errors
                errs = np.ascontiguousarray(np.vstack([np.hstack(prep.err_fourier),
                                                       np.hstack(prep.err_phot),
                                                       np.hstack(prep.err_exit)]).T)
                error_dct.update(zip(prep.view_IDs, errs))

                # RASP averaging
                t1 = time.time()

                # collect the sums
                parallel.allreduce(ob_sum_nmr)
                parallel.allreduce(ob_sum_dnm)
                parallel.allreduce(pr_sum_nmr)
                parallel.allreduce(pr_sum_dnm)

                POK.rasp_averaging(ob, pr, ob_sum_nmr, ob_sum_dnm, pr_sum_nmr, pr_sum_dnm)

                self.benchmark.rasp_averaging += time.time() - t1
                self.benchmark.calls_rasp_averaging += 1

                # Clip object (This call takes like one ms. Not time critical)
                if self.p.clip_object is not None:
                    clip_min, clip_max = self.p.clip_object
                    ampl_obj = np.abs(ob)
                    phase_obj = np.exp(1j * np.angle(ob))
                    too_high = (ampl_obj > clip_max)
                    too_low = (ampl_obj < clip_min)
                    ob[too_high] = clip_max * phase_obj[too_high]
                    ob[too_low] = clip_min * phase_obj[too_low]

            # Re-center the probe
            self.center_probe()

            # position update
            self.position_update()

            self.curiter += 1

        #error = parallel.gather_dict(error_dct)
        return error_dct

    def position_update(self):
        """
        Position refinement
        (Copied from _ProjectionEngine_serial)
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
