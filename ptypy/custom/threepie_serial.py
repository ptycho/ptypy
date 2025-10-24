# -*- coding: utf-8 -*-
"""
Serialized multislice reconstruction engine.

This file is part of the PTYPY package.

    :copyright: Copyright 2014 by the PTYPY team, see AUTHORS.
    :license: see LICENSE for details.
"""
import h5py
import numpy as np
import time

from ptypy.engines import register
from ptypy.accelerate.base.engines.stochastic import _StochasticEngineSerial, EPIEMixin
from ptypy.utils.verbose import logger, log
from ptypy.utils import parallel
from ptypy import utils as u
from ptypy.accelerate.base.kernels import FourierUpdateKernel, AuxiliaryWaveKernel, PoUpdateKernel, PositionCorrectionKernel
from ptypy.accelerate.base import array_utils as au
from ptypy.accelerate.base.engines import projectional_serial
from ptypy import io
from ptypy.core import geometry
from ptypy.utils import Param

__all__ = ['ThreePIE_serial']


@register()
class ThreePIE_serial(_StochasticEngineSerial, EPIEMixin):
    """
    An extension of EPIE to include multislice

    Defaults:

    [name]
    default = ThreePIE_serial
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
    
    [number_of_slices]
    default = 2
    type = int
    help = The number of slices
    doc = Defines how many slices are used for the multi-slice object.

    [slice_thickness]
    default = 1e-6
    type = float, list, tuple
    help = Thickness of a single slice in meters
    doc = A single float value or a list of float values. If a single value is used, all the slice will be assumed to be of the same thickness.

    [slice_start_iteration]
    default = 0
    type = int, list, tuple
    help = iteration number to start using a specific slice
    doc =

    [fslices]
    default = slices.h5
    type = str
    help = File path for the slice data
    doc =
    
    """
    
    def __init__(self, ptycho_parent, pars=None):
        super(ThreePIE_serial, self).__init__(ptycho_parent, pars)
        
        # keep track of timings
        self.benchmark = u.Param()
        self.diff_info = {}
        self.kernels = {}
        
        self.article = dict(
            title='{Ptychographic transmission microscopy in three dimensions using a multi-slice approach',
            author='A. M. Maiden et al.',
            journal='J. Opt. Soc. Am. A',
            volume=29,
            year=2012,
            page=1606,
            doi='10.1364/JOSAA.29.001606',
            comment='The 3PIE reconstruction algorithm',
        )
        self.ptycho.citations.add_article(**self.article)

    def engine_initialize(self):
        """
        Prepare for reconstruction. 
        """
        super().engine_initialize()
        self._reset_benchmarks()
        self._setup_kernels()
        self._setup_multislice_kernels()

    def _reset_benchmarks(self):
        """(Copied from _ProjectionEngine_serial)
        """
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

    def _setup_multislice_kernels(self):
        """
        Setup kernels for multislice reconstruction.
        """
        for label, scan in self.ptycho.model.scans.items():
            kern_msk = u.Param()
            kern_msk.scanmodel = type(scan).__name__
            self.kernels[label+'_msk'] = kern_msk
            
            # get scan geometry
            geo = scan.geometries[0]
            kern_msk.energy = geo.energy
            kern_msk.distance = self.p.slice_thickness
            kern_msk.psize = geo.resolution
            kern_msk.shape = geo.shape
            kern_msk.propagation = "nearfield"
            
            kern_msk.FW = []
            kern_msk.BW = []
            
            if type(self.p.slice_thickness) in [list, tuple]:
                assert(len(self.p.slice_thickness) == self.p.number_of_slices-1)
                for thickness in self.p.slice_thickness:
                    kern_msk.distance = thickness
                    G = geometry.Geo(owner=None, pars=kern_msk)
                    kern_msk.FW.append(G.propagator.fw)
                    kern_msk.BW.append(G.propagator.bw)
            else:
                kern_msk.distance = self.p.slice_thickness
                G = geometry.Geo(owner=None, pars=kern_msk)
                kern_msk.FW = [G.propagator.fw for i in range(self.p.number_of_slices-1)]
                kern_msk.BW = [G.propagator.bw for i in range(self.p.number_of_slices-1)]

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
        ## Inherit from stochastic serial engine ##
        if self.ptycho.new_data:

            # recalculate everything
            mean_power = 0.
            for s in self.di.storages.values():
                mean_power += s.mean_power
            self.mean_power = mean_power / len(self.di.storages)
        
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
        
        # Handle slice start iterations
        if isinstance(self.p.slice_start_iteration, int):
            self.p.slice_start_iteration = np.ones(self.p.number_of_slices) * self.p.slice_start_iteration
        
        # Create slice arrays at class level (like original ThreePIE)
        self._object = [None] * self.p.number_of_slices
        self._probe  = [None] * self.p.number_of_slices
        self._exits  = [None] * self.p.number_of_slices

        # Unfortunately this needs to be done for all pods, since
        # the shape of the probe / object was modified.
        # TODO: possible scaling issue, remove the need for padding
        for label, d in self.di.storages.items():
            prep = self.diff_info[d.ID]
            # print("DEBUG: Processing d.ID =", d.ID, "d =", d)
            result = projectional_serial.serialize_array_access(d)           
            prep.view_IDs, prep.poe_IDs, prep.addr = result
            
            if self.do_position_refinement:
                prep.original_addr = np.zeros_like(prep.addr)
                prep.original_addr[:] = prep.addr
            pID, oID, eID = prep.poe_IDs
            
            # Initialize slice arrays with proper shapes
            for i in range(self.p.number_of_slices):
                if self._object[i] is None:
                    self._object[i] = np.zeros_like(self.ob.S[oID].data, dtype=np.complex64)
                    self._probe[i]  = np.zeros_like(self.pr.S[pID].data, dtype=np.complex64)
                    self._exits[i]  = np.zeros_like(self.ex.S[eID].data, dtype=np.complex64)
            
            # Keep a list of view indices
            prep.rng = np.random.default_rng()
            prep.vieworder = np.arange(prep.addr.shape[0])

            # Modify addresses, copy pa into ea and remove da/ma
            prep.addr_ex = np.vstack([prep.addr[:,0,2,0], prep.addr[:,-1,2,0]+1]).T
            prep.addr[:,:,2] = prep.addr[:,:,0]
            prep.addr[:,:,3:,0] = 0

            # Reference to ex
            prep.ex = self.ex.S[eID].data
            # Object / probe norm
            prep.obn = np.zeros_like(prep.mag[0,None], dtype=np.float32)
            prep.prn = np.zeros_like(prep.mag[0,None], dtype=np.float32)   

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

                # references for multislice kernels
                ker_msk = self.kernels[prep.label+'_msk']
                FW_msk = ker_msk.FW
                BW_msk = ker_msk.BW

                # references for ob, pr
                ob = self.ob.S[oID].data
                pr = self.pr.S[pID].data
                self.pr.S[pID].data = np.random.rand(*self.pr.S[pID].data.shape).astype(self.pr.S[pID].data.dtype)
                # print(f"DEBUG: {ob.shape}, {pr.shape}")
                
                # shuffle view order
                vieworder = prep.vieworder
                prep.rng.shuffle(vieworder)
                
                # Iterate through views
                for i in vieworder:
                    # Get local adress and arrays
                    addr = prep.addr[i,None]
                    ex_from, ex_to = prep.addr_ex[i]
                    ex = prep.ex[ex_from:ex_to]
                    mag = prep.mag[i,None]
                    ma = prep.ma[i,None]
                    ma_sum = prep.ma_sum[i,None]
                    
                    obn = prep.obn
                    prn = prep.prn
                    err_phot = prep.err_phot[i,None]
                    err_fourier = prep.err_fourier[i,None]
                    err_exit = prep.err_exit[i,None]
                
                    # global aux buffer
                    aux = kern.aux
                    self._object, self._probe, self._exits = POK.multislice_fw(aux, addr, self._object, self._probe, self._exits, FW_msk, it, self.p.slice_start_iteration)

                    
                    ob, pr, ex = POK.last_slice_copy_to_ptypy(aux, addr, ob_last = self._object[-1], pr_last = self._probe[-1], exit_last = self._exits[-1], pr = pr, ob = ob, ex = ex, curiter = it, slice_update_iter = self.p.slice_start_iteration)
                    
                    # print(f'After prshape: {pr.shape}, ob.shape: {ob.shape}, ex.shape: {ex.shape}')
                    # with h5py.File('/home/litang/multislice/debug.h5', 'w') as f:
                    #     f.create_dataset('pr', data=pr)
                    #     f.create_dataset('ob', data=ob)
                    #     f.create_dataset('ex', data=ex)
                    
                    # position update
                    self.position_update_local(prep,i)

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
                    

                    self._object[-1], self._probe[-1] = POK.ptypy_copy_to_last_slice(aux, addr, self._probe[-1], self._object[-1], pr, ob, ex, it, self.p.slice_start_iteration)


                    # # object update
                    # t1 = time.time()
                    # POK.pr_norm_local(addr, pr, prn)
                    # POK.ob_update_local(addr, ob, pr, ex, aux, prn, a=self._ob_a, b=self._ob_b)
                    # self.benchmark.object_update += time.time() - t1
                    # self.benchmark.calls_object += 1

                    # # probe update
                    # t1 = time.time()
                    # if self._object_norm_is_global and self._pr_a == 0:
                    #     obn_max = au.max_abs2(ob)
                    #     obn[:] = 0
                    # else:
                    #     POK.ob_norm_local(addr, ob, obn)
                    #     obn_max = obn.max()
                    # if self.p.probe_update_start <= self.curiter:
                    #     POK.pr_update_local(addr, pr, ob, ex, aux, obn, obn_max, a=self._pr_a, b=self._pr_b)
                    # self.benchmark.probe_update += time.time() - t1
                    # self.benchmark.calls_probe += 1
                    
                    self._object, self._probe, self._exits = POK.multislice_bw(aux, addr, self._object, self._probe, self._exits, ob, pr, ex, BW_msk, it, self.p.slice_start_iteration)
 
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
            # Re-center the probe
            self.center_probe()

            self.curiter += 1

        #error = parallel.gather_dict(error_dct)
        return error_dct


    def position_update_local(self, prep, i):
        """
        Position refinement update for current view.
        """
        if not self.do_position_refinement:
            return
        do_update_pos = (self.p.position_refinement.stop > self.curiter >= self.p.position_refinement.start)
        do_update_pos &= (self.curiter % self.p.position_refinement.interval) == 0

        # Update positions
        if do_update_pos:
            """Positions and refines them by a given algorithm.
            """
            #log(4, "----------- START POS REF -------------")
            pID, oID, eID = prep.poe_IDs
            mag = prep.mag[i,None]
            ma = prep.ma[i,None]
            ma_sum = prep.ma_sum[i,None]
            ob = self.ob.S[oID].data
            pr = self.pr.S[pID].data
            kern = self.kernels[prep.label]
            aux = kern.aux
            addr = prep.addr[i,None]
            original_addr = prep.original_addr[i,None]
            mangled_addr = addr.copy()
            err_fourier = prep.err_fourier[i,None]

            PCK = kern.PCK
            FW = kern.FW

            # Keep track of object boundaries
            max_oby = ob.shape[-2] - aux.shape[-2] - 1
            max_obx = ob.shape[-1] - aux.shape[-1] - 1

            # We first need to calculate the current error
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

            #log(4, 'Position refinement trial: iteration %s' % (self.curiter))
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

    def engine_finalize(self):
        """
        Finalize the multislice reconstruction and save results.
        """
        # Create final object as product of all slices
        self.ob.fill(self._object[0])
        for i in range(1, self.p.number_of_slices):
            self.ob *= self._object[i]

        # Save the slices
        slices_info = Param()
        slices_info.number_of_slices = self.p.number_of_slices
        slices_info.slice_thickness = self.p.slice_thickness

        slices_info.objects = {ob.ID: {ID: S._to_dict() for ID, S in ob.storages.items()} for ob in self._object}
        slices_info.slice_start_iteration = self.p.slice_start_iteration

        header = {'description': 'multi-slices result details.'}

        h5opt = io.h5options['UNSUPPORTED']
        io.h5options['UNSUPPORTED'] = 'ignore'
        logger.info(f'Saving to {self.p.fslices}')
        io.h5write(self.p.fslices, header=header, content=slices_info)
        io.h5options['UNSUPPORTED'] = h5opt

        return super().engine_finalize()