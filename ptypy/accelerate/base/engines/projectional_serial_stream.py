# -*- coding: utf-8 -*-
"""
Difference Map reconstruction engine.

This file is part of the PTYPY package.

    :copyright: Copyright 2014 by the PTYPY team, see AUTHORS.
    :license: see LICENSE for details.
"""

# from .. import core
from __future__ import division

import numpy as np
import time

from ptypy import utils as u
from ptypy.utils.verbose import logger, log
from ptypy.utils import parallel
from ptypy.engines import register
from .projectional_serial import DM_serial

### TODOS 
# 
# - The Propagator needs to be made somewhere else
# - Get it running faster with MPI (partial sync)
# - implement "batching" when processing frames to lower the pressure on memory
# - Be smarter about the engine.prepare() part
# - Propagator needs to be reconfigurable for a certain batch size, gpyfft hates that.
# - Fourier_update_kernel needs to allow batched execution


__all__ = ['DM_serial_stream']

parallel = u.parallel
MPI = (parallel.size > 1)


@register()
class DM_serial_stream(DM_serial):
    """
    A full-fledged Difference Map engine that uses numpy arrays instead of iteration.
    """
    def engine_iterate(self, num=1):
        """
        Compute one iteration.
        """

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
                        obb.data[:] = ob.data
                        obb.data *= cfact
                        obn.data[:] = cfact

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
                    ma = self.ma.S[dID].data
                    ob = self.ob.S[oID].data
                    obn = self.ob_nrm.S[oID].data
                    obb = self.ob_buf.S[oID].data
                    pr = self.pr.S[pID].data
                    ex = self.ex.S[eID].data

                    # Fourier update.
                    if do_update_fourier:
                        log(4, '----- Fourier update -----', True)
                        t1 = time.time()
                        AWK.make_aux(aux, addr, ob, pr, ex, c_po=self._c, c_e=1-self._c)
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
                        AWK.make_exit(aux, addr, ob, pr, ex, c_a=self._b, c_po=self._a, c_e=-(self._a+self._b))
                        self.benchmark.E_Build_exit += time.time() - t1

                        err_phot = np.zeros_like(err_fourier)
                        err_exit = np.zeros_like(err_fourier)
                        errs = np.ascontiguousarray(np.vstack([err_fourier, err_phot, err_exit]).T)
                        error.update(zip(prep.view_IDs, errs))

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
                            parallel.allreduce(obb.data)
                            parallel.allreduce(obn.data)
                            obb.data /= obn.data
                        else:
                            obb.data /= obn.data

                        self.clip_object(obb)
                        ob.data[:] = obb.data

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
                """
                # Exit if probe should not yet be updated
                if do_update_probe:
                    # Update probe
                    log(4, prestr + '----- probe update -----', True)

                    # init probe
                    for pID, pr in self.pr.storages.items():
                        prn = self.pr_nrm.S[pID]
                        cfact = self.pr_cfact[pID]
                        pr.data[:] = pr.data
                        prn.data.fill(cfact)

                    # second cycle
                    for dID in self.di.S.keys():
                        prep = self.diff_info[dID]
                        # find probe, object in exit ID in dependence of dID
                        pID, oID, eID = prep.poe_IDs

                        # references for kernels
                        kern = self.kernels[prep.label]
                        POK = kern.POK

                        # get addresses and auxilliary array
                        addr = prep.addr

                        # local references
                        ob = self.ob.S[oID].data
                        pr = self.pr.S[pID].data
                        ex = self.ex.S[eID].data
                        prn = self.pr_nrm.S[pID].data

                        t1 = time.time()

                        # scan for-loop
                        ev = POK.pr_update(addr, pr, prn, ob, ex)

                        self.benchmark.probe_update += time.time() - t1
                        self.benchmark.calls_probe += 1

                    # synchronize
                    for pID, pr in self.pr.storages.items():

                        prn = self.pr_nrm.S[pID]
                        buf = self.pr_buf.S[pID]
                        # MPI test
                        if MPI:
                            # if False:
                            parallel.allreduce(pr.data)
                            parallel.allreduce(prn.data)
                            pr.data /= prn.data
                        else:
                            pr.data /= prn.data

                        self.support_constraint(pr)

                        change += u.norm2(pr.data - buf.data) / u.norm2(buf.data)
                        buf.data[:] = pr.data
                        if MPI:
                            change = parallel.allreduce(change) / parallel.size

                change = np.sqrt(change)

                log(4, prestr + 'change in probe is %.3f' % change, True)

                # stop iteration if probe change is small
                if change < self.p.overlap_converge_factor: break
                """

            parallel.barrier()
            self.curiter += 1

        self.error = error
        return error
