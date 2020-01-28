# -*- coding: utf-8 -*-
"""
Difference Map reconstruction engine.

This file is part of the PTYPY package.

    :copyright: Copyright 2014 by the PTYPY team, see AUTHORS.
    :license: GPLv2, see LICENSE for details.
"""

import numpy as np
import time
from pycuda import gpuarray
import pycuda.driver as cuda

from .. import utils as u
from ..utils.verbose import logger, log
from ..utils import parallel
from . import register, DM_pycuda
from ..accelerate import py_cuda as gpu
from ..accelerate.py_cuda.kernels import FourierUpdateKernel, AuxiliaryWaveKernel, PoUpdateKernel

from pycuda.tools import DeviceMemoryPool

MPI = parallel.size > 1
MPI = True

BLOCKS_ON_DEVICE = 2

__all__ = ['DM_pycuda_stream']

@register()
class DM_pycuda_stream(DM_pycuda.DM_pycuda):

    def __init__(self, ptycho_parent, pars = None):

        super(DM_pycuda_stream, self).__init__(ptycho_parent, pars)
        self.dmp = DeviceMemoryPool()
        self.qu2 = cuda.Stream()
        self.qu3 = cuda.Stream()

        self._ex_blocks_on_device = {}
        self._data_blocks_on_device = {}

    def engine_prepare(self):

        super(DM_pycuda.DM_pycuda, self).engine_prepare()

        for name, s in self.ob.S.items():
            s.gpu = gpuarray.to_gpu(s.data)
        for name, s in self.ob_buf.S.items():
            s.gpu = gpuarray.to_gpu(s.data)
        for name, s in self.ob_nrm.S.items():
            s.gpu = gpuarray.to_gpu(s.data)
        for name, s in self.pr.S.items():
            s.gpu = gpuarray.to_gpu(s.data)
        for name, s in self.pr_nrm.S.items():
            s.gpu = gpuarray.to_gpu(s.data)

        for label, d in self.ptycho.new_data:
            dID = d.ID
            prep = self.diff_info[dID]
            pID, oID, eID = prep.poe_IDs
            prep.addr_gpu = gpuarray.to_gpu(prep.addr)
            prep.ma_sum_gpu = gpuarray.to_gpu(prep.ma_sum)
            # prepare page-locked mems:
            prep.err_fourier_gpu = gpuarray.to_gpu(prep.err_fourier)
            ma = self.ma.S[dID].data.astype(np.float32)
            prep.ma = cuda.pagelocked_empty(ma.shape, ma.dtype, order="C", mem_flags=4)
            prep.ma[:] = ma            
            ex = self.ex.S[eID].data
            prep.ex = cuda.pagelocked_empty(ex.shape, ex.dtype, order="C", mem_flags=4)
            prep.ex[:] = ex
            mag = prep.mag
            prep.mag = cuda.pagelocked_empty(mag.shape, mag.dtype, order="C", mem_flags=4)
            prep.mag[:] = mag


    @property
    def ex_is_full(self):
        exl = self._ex_blocks_on_device
        return len([e for e in exl.values() if e > 1]) > BLOCKS_ON_DEVICE 

    @property
    def data_is_full(self):
        exl = self._data_blocks_on_device
        return len([e for e in exl.values() if e > 1]) > BLOCKS_ON_DEVICE

    def gpu_swap_ex(self, swaps=1, upload=True):
        """
        Find an exit wave block to transfer until. Delete block on device if full
        """
        s = 0
        for tID in self.dID_list:
            stat = self._ex_blocks_on_device[tID]
            prep = self.diff_info[tID]
            if stat == 3 and self.ex_is_full:
                # release data if already used and device full
                #print('Ex Free : ' + str(tID))
                if upload:
                    prep.ex_gpu.get_async(self.qu3, prep.ex)
                del prep.ex_gpu
                del prep.ex_ev
                self._ex_blocks_on_device[tID] = 0
            elif stat == 1 and not self.ex_is_full and s<=swaps:
                #print('Ex H2D : ' + str(tID))
                # not on device but there is space -> queue for stream
                prep.ex_gpu = gpuarray.to_gpu_async(prep.ex, allocator=self.dmp.allocate, stream=self.qu2)
                prep.ex_ev = cuda.Event()
                prep.ex_ev.record(self.qu2)
                # mark transfer
                self._ex_blocks_on_device[tID] = 2
                s+=1
            else:
                continue

    def gpu_swap_data(self, swaps=1):
        """
        Find an exit wave block to transfer until. Delete block on device if full
        """
        s = 0
        for tID in self.dID_list:
            stat = self._data_blocks_on_device[tID]
            if stat == 3 and self.data_is_full:
                # release data if already used and device full
                #rint('Data Free : ' + str(tID))
                del self.diff_info[tID].ma_gpu
                del self.diff_info[tID].mag_gpu
                del self.diff_info[tID].data_ev
                self._data_blocks_on_device[tID] = 0
            elif stat == 1 and not self.data_is_full and s<=swaps:
                #print('Data H2D : ' + str(tID))
                # not on device but there is space -> queue for stream
                prep = self.diff_info[tID]
                prep.mag_gpu = gpuarray.to_gpu_async(prep.mag, allocator=self.dmp.allocate, stream=self.qu2)
                prep.ma_gpu = gpuarray.to_gpu_async(prep.ma, allocator=self.dmp.allocate, stream=self.qu2)     
                prep.data_ev = cuda.Event()
                prep.data_ev.record(self.qu2)
                # mark transfer
                self._data_blocks_on_device[tID] = 2
                s+=1
            else:
                continue

    def engine_iterate(self, num=1):
        """
        Compute one iteration.
        """
        #ma_buf = ma_c = np.zeros(FUK.fshape, dtype=np.float32)
        self.dID_list = list(self.di.S.keys())
        self._ex_blocks_on_device = dict.fromkeys(self.dID_list,1)
        self._data_blocks_on_device = dict.fromkeys(self.dID_list,1)
        # 0: used, freed
        # 1: unused, not on device
        # 2: transfer to or on device
        # 3: used, on device
        for it in range(num):

            error = {}

            for inner in range(self.p.overlap_max_iterations):

                change = 0

                do_update_probe = (self.curiter >= self.p.probe_update_start)
                do_update_object = (self.p.update_object_first or (inner > 0) or not do_update_probe)
                do_update_fourier = (inner == 0)

                # initialize probe and object buffer to receive an update
                if do_update_object:
                    for oID, ob in self.ob.storages.items():
                        cfact = self.ob_cfact[oID]
                        obn = self.ob_nrm.S[oID]
                        obb = self.ob_buf.S[oID]
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
                        obb.gpu[:] = ob.gpu
                        obb.gpu *= cfact
                        obn.gpu.fill(cfact)

                # First cycle: Fourier + object update
                for dID in self.dID_list:
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
                    addr = prep.addr_gpu
                    err_fourier = prep.err_fourier_gpu
                    ma_sum = prep.ma_sum_gpu

                    # local references
                    ob = self.ob.S[oID].gpu
                    obn = self.ob_nrm.S[oID].gpu
                    obb = self.ob_buf.S[oID].gpu
                    pr = self.pr.S[pID].gpu


                    self.gpu_swap_ex()
                    prep.ex_ev.synchronize()
                    ex = prep.ex_gpu
                    

                    # Fourier update.
                    if do_update_fourier:
                        log(4, '----- Fourier update -----', True)
                        
                        self.gpu_swap_data()

                        t1 = time.time()
                        AWK.build_aux(aux, addr, ob, pr, ex, alpha=self.p.alpha)
                        self.benchmark.A_Build_aux += time.time() - t1


                        ## FFT
                        t1 = time.time()
                        FW(aux, aux)
                        self.benchmark.B_Prop += time.time() - t1


                        prep.data_ev.synchronize()
                        ma = prep.ma_gpu
                        mag = prep.mag_gpu

                        ## Deviation from measured data
                        t1 = time.time()
                        FUK.fourier_error(aux, addr, mag, ma, ma_sum)
                        FUK.error_reduce(addr, err_fourier)
                        FUK.fmag_all_update(aux, addr, mag, ma, err_fourier, pbound)
                        self.benchmark.C_Fourier_update += time.time() - t1
                        
                        # Mark computed
                        self._data_blocks_on_device[dID] = 3
                        
                        t1 = time.time()
                        BW(aux, aux)
                        self.benchmark.D_iProp += time.time() - t1

                        ## apply changes #2
                        t1 = time.time()
                        AWK.build_exit(aux, addr, ob, pr, ex)
                        self.benchmark.E_Build_exit += time.time() - t1

                        #queue.synchronize()
                        self.benchmark.calls_fourier += 1

                    prestr = '%d Iteration (Overlap) #%02d:  ' % (parallel.rank, inner)

                    # Update object
                    if do_update_object:
                        # Update object
                        log(4, prestr + '----- object update -----', True)
                        t1 = time.time()

                        # scan for loop
                        ev = POK.ob_update(addr, obb, obn, pr, ex)

                        self.benchmark.object_update += time.time() - t1
                        self.benchmark.calls_object += 1

                    # mark as computed
                    self._ex_blocks_on_device[dID] = 3

                for _dID, stat in self._ex_blocks_on_device.items():
                    if stat == 3: self._ex_blocks_on_device[_dID] = 2
                    elif stat == 0: self._ex_blocks_on_device[_dID] = 1

                for _dID, stat in self._data_blocks_on_device.items():
                    if stat == 3: self._data_blocks_on_device[_dID] = 2
                    elif stat == 0: self._data_blocks_on_device[_dID] = 1

                # swap direction
                if do_update_fourier:
                    self.dID_list.reverse()

                if do_update_object:
                    for oID, ob in self.ob.storages.items():
                        obn = self.ob_nrm.S[oID]
                        obb = self.ob_buf.S[oID]
                        # MPI test
                        if MPI:
                            obb.data[:] = obb.gpu.get()
                            obn.data[:] = obn.gpu.get()
                            #queue.synchronize()
                            parallel.allreduce(obb.data)
                            parallel.allreduce(obn.data)
                            obb.data /= obn.data
                            self.clip_object(obb)
                            ob.gpu.set(obb.data)
                        else:
                            obb.gpu /= obn.gpu
                            ob.gpu[:] = obb.gpu

                    #queue.synchronize()
                # Exit if probe should not yet be updated
                if not do_update_probe:
                    break

                # Update probe
                log(4, prestr + '----- probe update -----', True)
                change = self.probe_update(MPI=MPI)
                # change = self.probe_update(MPI=(parallel.size>1 and MPI))

                log(4, prestr + 'change in probe is %.3f' % change, True)

                # stop iteration if probe change is small
                if change < self.p.overlap_converge_factor: break

            #queue.synchronize()
            parallel.barrier()
            self.curiter += 1

        for name, s in self.ob.S.items():
            s.data[:] = s.gpu.get()
        for name, s in self.pr.S.items():
            s.data[:] = s.gpu.get()

        # costly but needed to sync back with
        # for name, s in self.ex.S.items():
        #     s.data[:] = s.gpu.get()
        for dID, prep in self.diff_info.items():
            err_fourier = prep.err_fourier_gpu.get()
            err_phot = np.zeros_like(err_fourier)
            err_exit = np.zeros_like(err_fourier)
            errs = np.ascontiguousarray(np.vstack([err_fourier, err_phot, err_exit]).T)
            error.update(zip(prep.view_IDs, errs))

        self.error = error
        return error

    ## probe update
    def probe_update(self, MPI=False):
        t1 = time.time()
        queue = self.queue

        # storage for-loop
        change = 0
        cfact = self.p.probe_inertia
        for pID, pr in self.pr.storages.items():
            prn = self.pr_nrm.S[pID]
            cfact = self.pr_cfact[pID]
            pr.gpu *= cfact
            prn.gpu.fill(cfact)

        for dID in self.dID_list:
            prep = self.diff_info[dID]

            POK = self.kernels[prep.label].POK
            # find probe, object in exit ID in dependence of dID
            pID, oID, eID = prep.poe_IDs

            self.gpu_swap_ex(upload=True)
            prep.ex_ev.synchronize()
            # scan for-loop
            ev = POK.pr_update(prep.addr_gpu,
                               self.pr.S[pID].gpu,
                               self.pr_nrm.S[pID].gpu,
                               self.ob.S[oID].gpu,
                               prep.ex_gpu)

            # mark as computed
            self._ex_blocks_on_device[dID] = 3

        for _dID, stat in self._ex_blocks_on_device.items():
            if stat == 3:
                self._ex_blocks_on_device[_dID] = 2
            elif stat == 0:
                self._ex_blocks_on_device[_dID] = 1

        #self.dID_list.reverse()

        for pID, pr in self.pr.storages.items():

            buf = self.pr_buf.S[pID]
            prn = self.pr_nrm.S[pID]

            # MPI test
            if MPI:
                # if False:
                pr.data[:] = pr.gpu.get()
                prn.data[:] = prn.gpu.get()
                #queue.synchronize()
                parallel.allreduce(pr.data)
                parallel.allreduce(prn.data)
                pr.data /= prn.data

                self.support_constraint(pr)

                pr.gpu.set(pr.data)
            else:
                pr.gpu /= prn.gpu
                # ca. 0.3 ms
                # self.pr.S[pID].gpu = probe_gpu
                pr.data[:] = pr.gpu.get()

            ## this should be done on GPU

            #queue.synchronize()
            change += u.norm2(pr.data - buf.data) / u.norm2(pr.data)
            buf.data[:] = pr.data
            if MPI:
                change = parallel.allreduce(change) / parallel.size

        # print 'probe update: ' + str(time.time()-t1)
        self.benchmark.probe_update += time.time() - t1
        self.benchmark.calls_probe += 1

        return np.sqrt(change)

