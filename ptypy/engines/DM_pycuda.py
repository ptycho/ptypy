# -*- coding: utf-8 -*-
"""
Difference Map reconstruction engine.

This file is part of the PTYPY package.

    :copyright: Copyright 2014 by the PTYPY team, see AUTHORS.
    :license: GPLv2, see LICENSE for details.
"""

import numpy as np
import time
import pycuda
import pycuda.driver as cuda
from .. import utils as u
from ..utils.verbose import logger, log
from ..utils import parallel
from . import BaseEngine, register, DM_serial, DM

from pycuda import gpuarray
from ..accelerate import py_cuda as gpu
from ..accelerate.py_cuda.fourier_update_kernel import FourierUpdateKernel
from ..accelerate.py_cuda.auxiliary_wave_kernel import AuxiliaryWaveKernel
from ..accelerate.py_cuda.po_update_kernel import PoUpdateKernel

### TODOS
#
# - Get it running faster with MPI (partial sync)
# - implement "batching" when processing frames to lower the pressure on memory
# - Be smarter about the engine.prepare() part
# - Propagator needs to be reconfigurable for a certain batch size, gpyfft hates that.
# - Fourier_update_kernel needs to allow batched execution

## for debugging
from matplotlib import pyplot as plt

__all__ = ['DM_pycuda']

parallel = u.parallel

serialize_array_access = DM_serial.serialize_array_access
gaussian_kernel = DM_serial.gaussian_kernel


@register()
class DM_pycuda(DM_serial.DM_serial):

    def __init__(self, ptycho_parent, pars=None):
        """
        Difference map reconstruction engine.
        """

        super(DM_pycuda, self).__init__(ptycho_parent, pars)

        self.context, self.queue = gpu.get_context()
        self.queue = cuda.Stream()
        # allocator for READ only buffers
        # self.const_allocator = cl.tools.ImmediateAllocator(queue, cl.mem_flags.READ_ONLY)
        ## gaussian filter
        # dummy kernel
        if not self.p.obj_smooth_std:
            gauss_kernel = gaussian_kernel(1, 1).astype(np.float32)
        else:
            gauss_kernel = gaussian_kernel(self.p.obj_smooth_std, self.p.obj_smooth_std).astype(np.float32)

        self.gauss_kernel_gpu = gpuarray.to_gpu(gauss_kernel)

    def engine_initialize(self):
        """
        Prepare for reconstruction.
        """
        super(DM_pycuda, self).engine_initialize()

        self.error = []

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
            # TODO: make this part of the engine rather than scan
            fpc = self.ptycho.frames_per_block

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
            kern.FUK = FourierUpdateKernel(aux, nmodes, queue_thread=self.queue)
            kern.FUK.allocate()

            kern.POK = PoUpdateKernel(queue_thread=self.queue)
            kern.POK.allocate()

            kern.AWK = AuxiliaryWaveKernel(queue_thread=self.queue)
            kern.AWK.allocate()

            from ptypy.accelerate.py_cuda.fft import FFT
            kern.FW = FFT(aux, self.queue,
                          pre_fft=geo.propagator.pre_fft,
                          post_fft=geo.propagator.post_fft,
                          inplace=True,
                          symmetric=True).ft
            kern.BW = FFT(aux, self.queue,
                          pre_fft=geo.propagator.pre_ifft,
                          post_fft=geo.propagator.post_ifft,
                          inplace=True,
                          symmetric=True).ift
            self.queue.synchronize()

    def engine_prepare(self):

        super(DM_pycuda, self).engine_prepare()

        ## The following should be restricted to new data

        # recursive copy to gpu
        for name, c in self.ptycho.containers.items():
            for name, s in c.S.items():
                ## convert data here
                if s.data.dtype.name == 'bool':
                    data = s.data.astype(np.float32)
                else:
                    data = s.data
                s.gpu = gpuarray.to_gpu(data)

        for prep in self.diff_info.values():
            prep.addr = gpuarray.to_gpu(prep.addr)
            prep.mag = gpuarray.to_gpu(prep.mag)
            prep.mask_sum = gpuarray.to_gpu(prep.mask_sum)
            prep.err_fourier = gpuarray.to_gpu(prep.err_fourier)

        """
        for dID, diffs in self.di.S.items():
            prep = u.Param()
            self.diff_info[dID] = prep

            prep.view_IDs, prep.poe_IDs, addr = serialize_array_access(diffs)

            all_modes = addr.shape[1]
            # master pod
            mpod = self.di.V[prep.view_IDs[0]].pod
            pr = mpod.pr_view.storage
            ob = mpod.ob_view.storage
            ex = mpod.ex_view.storage

            prep.addr_gpu = gpuarray.to_gpu(addr)
            prep.addr = addr

            ## auxiliary wave buffer
            aux = np.zeros_like(ex.data)
            prep.aux_gpu = gpuarray.to_gpu(aux)
            prep.aux = aux
            self.queue.synchronize()
        """
        # finish init queue
        self.queue.synchronize()

    def engine_iterate(self, num=1):
        """
        Compute one iteration.
        """

        for it in range(num):

            error_dct = {}

            for dID in self.di.S.keys():
                #print("DID is: %s" % dID)
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
                mask_sum = prep.mask_sum
                err_fourier = prep.err_fourier

                # local references
                ma = self.ma.S[dID].gpu
                ob = self.ob.S[oID].gpu
                pr = self.pr.S[pID].gpu
                ex = self.ex.S[eID].gpu

                queue = self.queue

                t1 = time.time()
                AWK.build_aux(aux, addr, ob, pr, ex, alpha=np.float32(self.p.alpha))
                queue.synchronize()

                self.benchmark.A_Build_aux += time.time() - t1

                ## FFT
                t1 = time.time()
                FW(aux, aux)

                queue.synchronize()
                self.benchmark.B_Prop += time.time() - t1

                ## Deviation from measured data
                t1 = time.time()
                FUK.fourier_error(aux, addr, mag, ma, mask_sum)
                FUK.error_reduce(addr, err_fourier)
                FUK.fmag_all_update(aux, addr, mag, ma, err_fourier, pbound)
                queue.synchronize()
                self.benchmark.C_Fourier_update += time.time() - t1
                ## iFFT
                t1 = time.time()
                BW(aux, aux)

                #print("The context is: %s" % self.context)
                queue.synchronize()
                #print("Here")
                self.benchmark.D_iProp += time.time() - t1

                ## apply changes #2
                t1 = time.time()
                AWK.build_exit(aux, addr, ob, pr, ex)
                queue.synchronize()
                self.benchmark.E_Build_exit += time.time() - t1

                err_phot = np.zeros_like(err_fourier)
                err_exit = np.zeros_like(err_fourier)
                errs = np.array(list(zip(err_fourier.get(), err_phot, err_exit)))
                error = dict(zip(prep.view_IDs, errs))

                self.benchmark.calls_fourier += 1

            parallel.barrier()

            sync = (self.curiter % 1 == 0)
            self.overlap_update(MPI=True)

            parallel.barrier()
            self.curiter += 1
            queue.synchronize()

        for name, s in self.ob.S.items():
            s.data[:] = s.gpu.get()
        for name, s in self.pr.S.items():
            s.data[:] = s.gpu.get()

        # costly but needed to sync back with
        for name, s in self.ex.S.items():
            s.data[:] = s.gpu.get()

        self.queue.synchronize()

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

    def engine_finalize(self):
        """
        try deleting ever helper contianer
        """
        super(DM_pycuda, self).engine_finalize()
        self.queue.synchronize()
        self.context.detach()

        # delete local references to container buffer copies
