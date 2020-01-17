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

MPI = parallel.size > 1
MPI = True

__all__ = ['DM_pycuda_stream']

@register()
class DM_pycuda_stream(DM_pycuda.DM_pycuda):

    def __init__(self, ptycho_parent, pars = None):

        super(DM_pycuda_stream, self).__init__(ptycho_parent, pars)

        self.qu2 = cuda.Stream()

    def engine_prepare(self):

        super(DM_pycuda.DM_pycuda, self).engine_prepare()

        ## The following should be restricted to new data

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

        for prep in self.diff_info.values():
            prep.addr_gpu = gpuarray.to_gpu(prep.addr)
            prep.ma_sum_gpu = gpuarray.to_gpu(prep.ma_sum)
            prep.err_fourier_gpu = gpuarray.to_gpu(prep.err_fourier)
            self.dummy_error = np.zeros_like(prep.err_fourier)

    @property
    def gpu_is_full(self):
        return False

    def engine_iterate(self, num=1):
        """
        Compute one iteration.
        """
        #ma_buf = ma_c = np.zeros(FUK.fshape, dtype=np.float32)

        for it in range(num):

            queue = self.queue
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
                    addr = prep.addr_gpu
                    err_fourier = prep.err_fourier_gpu
                    ma_sum = prep.ma_sum_gpu

                    # stuff to be cycled
                    #mag = gpuarray.to_gpu(prep.mag)
                    #ma = gpuarray.to_gpu(self.ma.S[dID].data)


                    # local references
                    ob = self.ob.S[oID].gpu
                    obn = self.ob_nrm.S[oID].gpu
                    obb = self.ob_buf.S[oID].gpu
                    pr = self.pr.S[pID].gpu

                    # cycle exit in and out, cause it's used by both
                    if 'ex_gpu' in prep:
                        # print('got it')
                        ex = prep.ex_gpu
                    elif not self.gpu_is_full:
                        # print('new')
                        N, a, b  = self.ex.S[eID].data.shape
                        #ex_c = np.zeros(aux.shape, dtype=np.complex64)
                        #ex_c[:N] = self.ex.S[eID].data
                        ex = gpuarray.to_gpu_async(self.ex.S[eID].data, stream=self.qu2)
                        prep.ex_gpu = ex
                    else:
                        # print('steal')
                        # get a buffer
                        for tID, p in self.diff_info.items():
                            if not 'ex' in p:
                                continue
                            else:
                                ex = p.pop('ex')
                                eID = p.poe_IDs[2]
                                break
                        ex_t = self.ex.S[eID].data
                        ex_t[:] = ex.get()[:ex_t.shape[0]]
                        N, a, b  = self.ex.S[eID].data.shape
                        ex_c = np.zeros_like(aux)
                        ex_c[:N] = self.ex.S[eID].data
                        ex.set(ex_c)
                        prep.ex_gpu = ex

                    # Fourier update.
                    if do_update_fourier:
                        log(4, '----- Fourier update -----', True)
                        t1 = time.time()
                        AWK.build_aux(aux, addr, ob, pr, ex, alpha=self.p.alpha)
                        self.benchmark.A_Build_aux += time.time() - t1

                        ## FFT
                        t1 = time.time()
                        FW(aux, aux)
                        self.benchmark.B_Prop += time.time() - t1

                        # cycle exit in and out, cause it's used by both
                        if 'ma_gpu' in prep:
                            # print('got it ma')
                            ma = prep.ma_gpu
                            mag = prep.mag_gpu
                        elif not self.gpu_is_full:
                            # print('new ma', self.ma.S[dID].data.dtype)
                            N, a, b = prep.mag.shape
                            #ma_c = np.zeros(FUK.fshape, dtype=np.float32)
                            #mag_c = np.zeros(FUK.fshape, dtype=np.float32)
                            #ma_c[:N] = self.ma.S[dID].data
                            #mag_c[:N] = prep.mag
                            mag = gpuarray.to_gpu_async(prep.mag, stream=self.qu2)
                            ma = gpuarray.to_gpu_async(self.ma.S[dID].data.astype(np.float32), stream=self.qu2)
                            # print(ma.shape, mag.shape)
                            prep.ma_gpu = ma
                            prep.mag_gpu = mag
                        else:
                            # print('steal ma')
                            # get a buffer
                            for tID, p in self.diff_info.items():
                                if not 'ma_gpu' in p:
                                    continue
                                else:
                                    ma = p.pop('ma_gpu')
                                    mag = p.pop('mag_gpu')
                                    break
                            N, a, b = prep.mag.shape
                            ma_c = np.zeros(FUK.fshape, dtype=np.float32)
                            mag_c = np.zeros(FUK.fshape, dtype=np.float32)
                            ma_c[:N] = self.ma.S[dID].data
                            mag_c[:N] = prep.mag
                            ma.set(ma_c)
                            mag.set(mag_c)
                            prep.ma_gpu = ma
                            prep.mag_gpu = mag

                        ## Deviation from measured data
                        t1 = time.time()
                        FUK.fourier_error(aux, addr, mag, ma, ma_sum)
                        FUK.error_reduce(addr, err_fourier)
                        FUK.fmag_all_update(aux, addr, mag, ma, err_fourier, pbound)
                        self.benchmark.C_Fourier_update += time.time() - t1

                        t1 = time.time()
                        BW(aux, aux)
                        self.benchmark.D_iProp += time.time() - t1

                        ## apply changes #2
                        t1 = time.time()
                        AWK.build_exit(aux, addr, ob, pr, ex)
                        self.benchmark.E_Build_exit += time.time() - t1

                        #err_phot = np.zeros_like(err_fourier)
                        #err_exit = np.zeros_like(err_fourier)
                        errs = np.array(list(zip(self.dummy_error, self.dummy_error, self.dummy_error)))

                        #errs = np.ascontiguousarray(np.vstack([err_fourier.get(), err_phot, err_exit]).T)
                        error.update(zip(prep.view_IDs, errs))
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

                if do_update_object:
                    for oID, ob in self.ob.storages.items():
                        obn = self.ob_nrm.S[oID]
                        obb = self.ob_buf.S[oID]
                        # MPI test
                        if MPI:
                            obb.data[:] = obb.gpu.get()
                            obn.data[:] = obn.gpu.get()
                            # queue.synchronize()    // get synchronises automatically
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

            queue.synchronize()
            parallel.barrier()
            self.curiter += 1

        for name, s in self.ob.S.items():
            s.data[:] = s.gpu.get()
        for name, s in self.pr.S.items():
            s.data[:] = s.gpu.get()

        # costly but needed to sync back with
        # for name, s in self.ex.S.items():
        #     s.data[:] = s.gpu.get()


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

        for dID in self.di.S.keys():
            prep = self.diff_info[dID]

            POK = self.kernels[prep.label].POK
            # find probe, object in exit ID in dependence of dID
            pID, oID, eID = prep.poe_IDs

            # scan for-loop
            ev = POK.pr_update(prep.addr_gpu,
                               self.pr.S[pID].gpu,
                               self.pr_nrm.S[pID].gpu,
                               self.ob.S[oID].gpu,
                               prep.ex_gpu)
            #queue.synchronize()

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

            queue.synchronize()
            change += u.norm2(pr.data - buf.data) / u.norm2(pr.data)
            buf.data[:] = pr.data
            if MPI:
                change = parallel.allreduce(change) / parallel.size

        # print 'probe update: ' + str(time.time()-t1)
        self.benchmark.probe_update += time.time() - t1
        self.benchmark.calls_probe += 1

        return np.sqrt(change)

