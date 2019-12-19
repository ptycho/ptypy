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


    def engine_prepare(self):

        super(DM_pycuda.DM_pycuda, self).engine_prepare()

        ## The following should be restricted to new data

        # recursive copy to gpu
        for _cname, c in self.ptycho.containers.items():
            for _sname, s in c.S.items():
                # convert data here
                if s.data.dtype.name == 'bool':
                    data = s.data.astype(np.float32)
                else:
                    data = s.data
                s.gpu = gpuarray.to_gpu(data)

        for prep in self.diff_info.values():
            prep.addr = gpuarray.to_gpu(prep.addr)
            prep.mag = gpuarray.to_gpu(prep.mag)
            prep.ma_sum = gpuarray.to_gpu(prep.ma_sum)
            prep.err_fourier = gpuarray.to_gpu(prep.err_fourier)
            self.dummy_error = np.zeros_like(prep.err_fourier)

    def engine_iterate(self, num=1):
        """
        Compute one iteration.
        """

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
                    addr = prep.addr
                    mag = prep.mag
                    ma_sum = prep.ma_sum
                    err_fourier = prep.err_fourier

                    # local references
                    ma = self.ma.S[dID].gpu
                    ob = self.ob.S[oID].gpu
                    obn = self.ob_nrm.S[oID].gpu
                    obb = self.ob_buf.S[oID].gpu
                    pr = self.pr.S[pID].gpu
                    ex = self.ex.S[eID].gpu

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
                        queue.synchronize()
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
                            queue.synchronize()
                            parallel.allreduce(obb.data)
                            parallel.allreduce(obn.data)
                            obb.data /= obn.data
                            self.clip_object(obb)
                            ob.gpu.set(obb.data)
                        else:
                            obb.gpu /= obn.gpu
                            ob.gpu[:] = obb.gpu

                    queue.synchronize()
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

    ## object update
    def object_update(self, MPI=False):
        t1 = time.time()
        queue = self.queue
        queue.synchronize()
        for oID, ob in self.ob.storages.items():
            obn = self.ob_nrm.S[oID]
            """
            if self.p.obj_smooth_std is not None:
                logger.info('Smoothing object, cfact is %.2f' % cfact)
                t2 = time.time()
                self.prg.gaussian_filter(queue, (info[3],info[4]), None, obj_gpu.data, self.gauss_kernel_gpu.data)
                queue.synchronize()
                obj_gpu *= cfact
                print 'gauss: '  + str(time.time()-t2)
            else:
                obj_gpu *= cfact
            """
            cfact = self.ob_cfact[oID]
            ob.gpu *= cfact
            # obn.gpu[:] = cfact
            obn.gpu.fill(cfact)
            queue.synchronize()

        # storage for-loop
        for dID in self.di.S.keys():
            prep = self.diff_info[dID]

            POK = self.kernels[prep.label].POK
            # find probe, object in exit ID in dependence of dID
            pID, oID, eID = prep.poe_IDs

            # scan for loop
            ev = POK.ob_update(prep.addr,
                               self.ob.S[oID].gpu,
                               self.ob_nrm.S[oID].gpu,
                               self.pr.S[pID].gpu,
                               self.ex.S[eID].gpu)
            queue.synchronize()

        for oID, ob in self.ob.storages.items():
            obn = self.ob_nrm.S[oID]
            # MPI test
            if MPI:
                ob.data[:] = ob.gpu.get()
                obn.data[:] = obn.gpu.get()
                queue.synchronize()
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
                ob.gpu.set(ob.data)
            else:
                ob.gpu /= obn.gpu

            queue.synchronize()

        # print 'object update: ' + str(time.time()-t1)
        self.benchmark.object_update += time.time() - t1
        self.benchmark.calls_object += 1

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
            ev = POK.pr_update(prep.addr,
                               self.pr.S[pID].gpu,
                               self.pr_nrm.S[pID].gpu,
                               self.ob.S[oID].gpu,
                               self.ex.S[eID].gpu)
            queue.synchronize()

        for pID, pr in self.pr.storages.items():

            buf = self.pr_buf.S[pID]
            prn = self.pr_nrm.S[pID]

            # MPI test
            if MPI:
                # if False:
                pr.data[:] = pr.gpu.get()
                prn.data[:] = prn.gpu.get()
                queue.synchronize()
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

