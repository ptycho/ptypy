# -*- coding: utf-8 -*-
"""
Difference Map reconstruction engine.

This file is part of the PTYPY package.

    :copyright: Copyright 2014 by the PTYPY team, see AUTHORS.
    :license: see LICENSE for details.
"""

import numpy as np
import time
import pyopencl as cl

from ptypy import utils as u
from ptypy.utils.verbose import logger, log
from ptypy.utils import parallel
from ptypy.engines import BaseEngine, register
from ptypy.accelerate.base.engines import projectional_serial

from pyopencl import array as cla
from ptypy.accelerate.ocl_pyopencl.ocl_kernels import FourierUpdateKernel, AuxiliaryWaveKernel, PoUpdateKernel
from ptypy.accelerate import ocl_pyopencl as gpu

### TODOS
# 
# - Get it running faster with MPI (partial sync)
# - implement "batching" when processing frames to lower the pressure on memory
# - Be smarter about the engine.prepare() part
# - Propagator needs to be reconfigurable for a certain batch size, gpyfft hates that.
# - Fourier_update_kernel needs to allow batched execution

## for debugging
#from matplotlib import pyplot as plt

__all__ = ['DM_ocl']

parallel = u.parallel

serialize_array_access = projectional_serial.serialize_array_access
gaussian_kernel = projectional_serial.gaussian_kernel


@register()
class DM_ocl(projectional_serial.DM_serial):

    def __init__(self, ptycho_parent, pars=None):
        """
        Difference map reconstruction engine.
        """

        super(DM_ocl, self).__init__(ptycho_parent, pars)

        self.queue = gpu.get_ocl_queue()

        # allocator for READ only buffers
        # self.const_allocator = cl.tools.ImmediateAllocator(queue, cl.mem_flags.READ_ONLY)
        ## gaussian filter
        # dummy kernel
        if not self.p.obj_smooth_std:
            gauss_kernel = gaussian_kernel(1, 1).astype(np.float32)
        else:
            gauss_kernel = gaussian_kernel(self.p.obj_smooth_std, self.p.obj_smooth_std).astype(np.float32)

        self.gauss_kernel_gpu = cla.to_device(self.queue, gauss_kernel)

    def engine_initialize(self):
        """
        Prepare for reconstruction.
        """
        super(DM_ocl, self).engine_initialize()

        self.error = []

        def constbuffer(nbytes):
            return cl.Buffer(self.queue.context, cl.mem_flags.READ_ONLY, size=nbytes)

        self.ob_cfact_gpu = {}
        self.pr_cfact_gpu = {}

    def _setup_kernels(self):
        """
        Setup kernels, one for each scan. Derive scans from ptycho class
        """
        # get the scans
        for label, scan in self.ptycho.model.scans.items():

            kern = u.Param()
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
            kern.aux = cla.to_device(self.queue, aux)

            # setup kernels, one for each SCAN.
            kern.FUK = FourierUpdateKernel(aux, nmodes)
            kern.FUK.allocate()

            kern.POK = PoUpdateKernel()
            kern.POK.allocate()

            kern.AWK = AuxiliaryWaveKernel()
            kern.AWK.allocate()

            from ptypy.accelerate.ocl_pyopencl.ocl_fft import FFT_2D_ocl_reikna as FFT
            kern.FW = FFT(self.queue, aux,
                          pre_fft=geo.propagator.pre_fft,
                          post_fft=geo.propagator.post_fft,
                          inplace=True,
                          symmetric=True).ft
            kern.BW = FFT(self.queue, aux,
                          pre_fft=geo.propagator.pre_ifft,
                          post_fft=geo.propagator.post_ifft,
                          inplace=True,
                          symmetric=True).ift
            self.queue.finish()

    def engine_prepare(self):

        super(DM_ocl, self).engine_prepare()

        ## The following should be restricted to new data

        # For Streaming / Queuing: Limit to data that stays on GPU like pr & ob
        # recursive copy to gpu
        for name, c in self.ptycho.containers.items():
            for name, s in c.S.items():
                ## convert data here
                if s.data.dtype.name == 'bool':
                    data = s.data.astype(np.float32)
                else:
                    data = s.data
                s.gpu = cla.to_device(self.queue, data)

        # For streaming, part of this needs to be moved to engine_iterate
        # this contains stuff that aligns with the data
        # also only new data should be considered here.
        for prep in self.diff_info.values():
            prep.addr = cla.to_device(self.queue, prep.addr)
            prep.mag = cla.to_device(self.queue, prep.mag)
            prep.ma_sum = cla.to_device(self.queue, prep.ma_sum)
            prep.err_fourier = cla.to_device(self.queue, prep.err_fourier)
            ## potentially
            #prep.ex = ...
            #prep.ma = ...

        # finish init queue
        self.queue.finish()

    def engine_iterate(self, num=1):
        """
        Compute one iteration.
        """

        for it in range(num):

            error_dct = {}

            for dID in self.di.S.keys():
                t1 = time.time()

                prep = self.diff_info[dID]
                # find probe, object in exit ID in dependence of dID
                pID, oID, eID = prep.poe_IDs

                # references for kernels
                kern = self.kernels[prep.label]
                FUK = kern.FUK
                AWK = kern.AWK

                pbound = self.pbound_scan[prep.label]
                aux = kern.aux
                FW = kern.FW
                BW = kern.BW

                # get addresses 
                addr = prep.addr
                mag = prep.mag
                ma_sum = prep.ma_sum
                err_fourier = prep.err_fourier

                # local references
                ma = self.ma.S[dID].gpu
                ob = self.ob.S[oID].gpu
                pr = self.pr.S[pID].gpu
                ex = self.ex.S[eID].gpu

                queue = self.queue

                t1 = time.time()
                AWK.build_aux(aux, addr, ob, pr, ex, alpha=self.p.alpha)
                queue.finish()

                self.benchmark.A_Build_aux += time.time() - t1

                ## FFT
                t1 = time.time()
                FW(aux, aux)
                queue.finish()
                self.benchmark.B_Prop += time.time() - t1

                ## Deviation from measured data
                t1 = time.time()
                FUK.fourier_error(aux, addr, mag, ma, ma_sum)
                FUK.error_reduce(addr, err_fourier)
                FUK.fmag_all_update(aux, addr, mag, ma, err_fourier, pbound)
                queue.finish()
                self.benchmark.C_Fourier_update += time.time() - t1

                ## iFFT
                t1 = time.time()
                BW(aux, aux)
                queue.finish()

                self.benchmark.D_iProp += time.time() - t1

                ## apply changes #2
                t1 = time.time()
                AWK.build_exit(aux, addr, ob, pr, ex)
                queue.finish()
                self.benchmark.E_Build_exit += time.time() - t1

                err_phot = np.zeros_like(err_fourier)
                err_exit = np.zeros_like(err_fourier)
                errs = np.array(list(zip(err_fourier.get(self.queue), err_phot, err_exit)))
                error = dict(zip(prep.view_IDs, errs))

                self.benchmark.calls_fourier += 1

            parallel.barrier()

            sync = (self.curiter % 1 == 0)
            self.overlap_update(MPI=True)

            parallel.barrier()
            self.curiter += 1
            queue.finish()

        for name, s in self.ob.S.items():
            s.data[:] = s.gpu.get(queue=self.queue)
        for name, s in self.pr.S.items():
            s.data[:] = s.gpu.get(queue=self.queue)

        # costly but needed to sync back with 
        for name, s in self.ex.S.items():
            s.data[:] = s.gpu.get(queue=self.queue)

        self.queue.finish()

        self.error = error
        return error

    ## object update
    def object_update(self, MPI=False):
        t1 = time.time()
        queue = self.queue
        queue.finish()
        for oID, ob in self.ob.storages.items():
            obn = self.ob_nrm.S[oID]
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
            cfact = self.ob_cfact[oID]
            ob.gpu *= cfact
            # obn.gpu[:] = cfact
            obn.gpu.fill(cfact)
            queue.finish()

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
            queue.finish()

        for oID, ob in self.ob.storages.items():
            obn = self.ob_nrm.S[oID]
            # MPI test
            if MPI:
                ob.data[:] = ob.gpu.get(queue=queue)
                obn.data[:] = obn.gpu.get(queue=queue)
                queue.finish()
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

            queue.finish()

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
            queue.finish()

        for pID, pr in self.pr.storages.items():

            buf = self.pr_buf.S[pID]
            prn = self.pr_nrm.S[pID]

            # MPI test
            if MPI:
                # if False:
                pr.data[:] = pr.gpu.get(queue=queue)
                prn.data[:] = prn.gpu.get(queue=queue)
                queue.finish()
                parallel.allreduce(pr.data)
                parallel.allreduce(prn.data)
                pr.data /= prn.data

                self.support_constraint(pr)

                pr.gpu.set(pr.data)
            else:
                pr.gpu /= prn.gpu
                # ca. 0.3 ms
                # self.pr.S[pID].gpu = probe_gpu
                pr.data[:] = pr.gpu.get(queue=queue)

            ## this should be done on GPU

            queue.finish()
            change += u.norm2(pr.data - buf.data) / u.norm2(pr.data)
            buf.data[:] = pr.data
            if MPI:
                change = parallel.allreduce(change) / parallel.size

        # print 'probe update: ' + str(time.time()-t1)
        self.benchmark.probe_update += time.time() - t1
        self.benchmark.calls_probe += 1

        return np.sqrt(change)

    def engine_finalize(self):
        """
        try deleting ever helper contianer
        """
        super(DM_ocl, self).engine_finalize()
        self.queue.finish()

        # delete local references to container buffer copies
