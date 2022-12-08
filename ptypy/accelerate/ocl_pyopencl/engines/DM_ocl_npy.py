# -*- coding: utf-8 -*-
"""
Difference Map reconstruction engine.

This file is part of the PTYPY package.

    :copyright: Copyright 2014 by the PTYPY team, see AUTHORS.
    :license: see LICENSE for details.
"""

import os.path
import numpy as np
import time
import pyopencl as cl

from ptypy import utils as u
from ptypy.utils.verbose import logger, log
from ptypy.utils import parallel
from ptypy.engines import BaseEngine, register, projectional
from ptypy.accelerate import ocl_pyopencl as gpu

from pyopencl import array as cla

### TODOS
# 
# - Get it running faster with MPI (partial sync)
# - implement "batching" when processing frames to lower the pressure on memory
# - Be smarter about the engine.prepare() part
# - Propagator needs to be reconfigurable for a certain batch size, gpyfft hates that.
# - Fourier_update_kernel needs to allow batched execution

## for debugging
#from matplotlib import pyplot as plt

__all__ = ['DM_ocl_npy']

parallel = u.parallel


def gaussian_kernel(sigma, size=None, sigma_y=None, size_y=None):
    size = int(size)
    sigma = np.float(sigma)
    if not size_y:
        size_y = size
    if not sigma_y:
        sigma_y = sigma

    x, y = np.mgrid[-size:size + 1, -size_y:size_y + 1]

    g = np.exp(-(x ** 2 / (2 * sigma ** 2) + y ** 2 / (2 * sigma_y ** 2)))
    return g / g.sum()


def serialize_array_access(diff_storage):
    # Sort views according to layer in diffraction stack 
    views = diff_storage.views
    dlayers = [view.dlayer for view in views]
    views = [views[i] for i in np.argsort(dlayers)]
    view_IDs = [view.ID for view in views]

    # Master pod
    mpod = views[0].pod

    # Determine linked storages for probe, object and exit waves
    pr = mpod.pr_view.storage
    ob = mpod.ob_view.storage
    ex = mpod.ex_view.storage

    poe_ID = (pr.ID, ob.ID, ex.ID)

    addr = []
    for view in views:
        address = []

        for pname, pod in view.pods.items():
            ## store them for each pod
            # create addresses
            a = np.array(
                [(pod.pr_view.dlayer, pod.pr_view.dlow[0], pod.pr_view.dlow[1]),
                 (pod.ob_view.dlayer, pod.ob_view.dlow[0], pod.ob_view.dlow[1]),
                 (pod.ex_view.dlayer, pod.ex_view.dlow[0], pod.ex_view.dlow[1]),
                 (pod.di_view.dlayer, pod.di_view.dlow[0], pod.di_view.dlow[1]),
                 (pod.ma_view.dlayer, pod.ma_view.dlow[0], pod.ma_view.dlow[1])])

            address.append(a)

            if pod.pr_view.storage.ID != pr.ID:
                log(1, "Splitting probes for one diffraction stack is not supported in " + __name__)
            if pod.ob_view.storage.ID != ob.ID:
                log(1, "Splitting objects for one diffraction stack is not supported in " + __name__)
            if pod.ex_view.storage.ID != ex.ID:
                log(1, "Splitting exit stacks for one diffraction stack is not supported in " + __name__)

        ## store data for each view
        # adresses
        addr.append(address)

    # store them for each storage
    return view_IDs, poe_ID, np.array(addr).astype(np.int32)


@register()
class DM_ocl_npy(projectional.DM):

    def __init__(self, ptycho_parent, pars=None):
        """
        Difference map reconstruction engine.
        """

        super(DM_ocl_npy, self).__init__(ptycho_parent, pars)

        self.queue = gpu.get_ocl_queue()

        # allocator for READ only buffers
        # self.const_allocator = cl.tools.ImmediateAllocator(queue, cl.mem_flags.READ_ONLY)
        ## gaussian filter
        # dummy kernel
        if not self.p.obj_smooth_std:
            gauss_kernel = gaussian_kernel(1, 1).astype(np.float32)
        else:
            gauss_kernel = gaussian_kernel(self.p.obj_smooth_std, self.p.obj_smooth_std).astype(np.float32)
        kernel_pars = {'kernel_sh_x': gauss_kernel.shape[0], 'kernel_sh_y': gauss_kernel.shape[1]}

        self.gauss_kernel_gpu = cla.to_device(self.queue, gauss_kernel)

    def engine_initialize(self):
        """
        Prepare for reconstruction.
        """
        super(DM_ocl_npy, self).engine_initialize()

        self.benchmark = u.Param()
        self.benchmark.A_Build_aux = 0.
        self.benchmark.B_Prop = 0.
        self.benchmark.C_Fourier_update = 0.
        self.benchmark.D_iProp = 0.
        self.benchmark.E_Build_exit = 0.
        self.benchmark.probe_update = 0.
        self.benchmark.object_update = 0.
        self.benchmark.calls_fourier = 0
        self.benchmark.calls_object = 0
        self.benchmark.calls_probe = 0
        self.dattype = np.complex64

        self.error = []

        self.diff_info = {}
        self.ob_cfact = {}
        self.pr_cfact = {}

        def constbuffer(nbytes):
            return cl.Buffer(self.queue.context, cl.mem_flags.READ_ONLY, size=nbytes)

        self.ob_cfact_gpu = {}
        self.pr_cfact_gpu = {}

    def engine_prepare(self):

        super(DM_ocl_npy, self).engine_prepare()

        # object padding on high side (due to 16x16 wg size)    
        for oID, ob in self.ob.storages.items():
            obn = self.ob_nrm.S[oID]
            obv = self.ob_viewcover.S[oID]
            misfit = np.asarray(ob.shape[-2:]) % 32
            if (misfit != 0).any():
                pad = 32 - np.asarray(ob.shape[-2:]) % 32
                ob.data = u.crop_pad(ob.data, [[0, pad[0]], [0, pad[1]]], axes=[-2, -1], filltype='project')
                obv.data = u.crop_pad(obv.data, [[0, pad[0]], [0, pad[1]]], axes=[-2, -1], filltype='project')
                obn.data = u.crop_pad(obn.data, [[0, pad[0]], [0, pad[1]]], axes=[-2, -1], filltype='project')
                ob.shape = ob.data.shape
                obv.shape = obv.data.shape
                obn.shape = obn.data.shape
            ## calculating cfacts. This should actually belong to the parent class
            #cfact = self.p.object_inertia * self.mean_power * \
            #        (obv.data + 1.)
            #cfact /= u.parallel.size
            #self.ob_cfact[oID] = cfact
            #self.ob_cfact_gpu[oID] = cla.to_device(self.queue, cfact)
            self.ob_cfact[oID] = self.p.object_inertia * self.mean_power / u.parallel.size

        for pID, pr in self.pr.storages.items():
            cfact = self.p.probe_inertia * len(pr.views) / pr.data.shape[0]
            self.pr_cfact[pID] = cfact / u.parallel.size

        ## The following should be restricted to new data

        # recursive copy to gpu
        for name, c in self.ptycho.containers.items():
            for name, s in c.S.items():
                ## convert data here
                if s.data.dtype.name == 'bool':
                    data = s.data.astype(np.float32)
                else:
                    data = s.data
                s.gpu = cla.to_device(self.queue, data)

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

            prep.addr_gpu = cla.to_device(self.queue, addr)
            prep.addr = addr

            ## auxiliary wave buffer
            aux = np.zeros_like(ex.data)
            prep.aux_gpu = cla.to_device(self.queue, aux)
            prep.aux = aux
            self.queue.finish()

            ## setup kernels
            from ptypy.accelerate.ocl_pyopencl.ocl_kernels import Fourier_update_kernel as FUK
            prep.fourier_kernel = FUK(self.queue, nmodes=all_modes, pbound=self.pbound[dID])
            mask = self.ma.S[dID].data.astype(np.float32)
            prep.fourier_kernel.configure(diffs.data, mask, aux)

            from ptypy.accelerate.ocl_pyopencl.ocl_kernels import Auxiliary_wave_kernel as AWK
            prep.aux_ex_kernel = AWK(self.queue)
            prep.aux_ex_kernel.configure(ob.data, addr, self.p.alpha)

            from ptypy.accelerate.ocl_pyopencl.ocl_kernels import PO_update_kernel as PUK
            prep.po_kernel = PUK(self.queue)
            prep.po_kernel.configure(ob.data, pr.data, addr)

            geo = mpod.geometry
            # you cannot use gpyfft multiple times due to
            if not hasattr(geo, 'transform'):
                from ptypy.accelerate.ocl_pyopencl.ocl_fft import FFT_2D_ocl_reikna as FFT

                geo.transform = FFT(self.queue, aux,
                                    pre_fft=geo.propagator.pre_fft,
                                    post_fft=geo.propagator.post_fft,
                                    inplace=True,
                                    symmetric=True)
                geo.itransform = FFT(self.queue, aux,
                                     pre_fft=geo.propagator.pre_ifft,
                                     post_fft=geo.propagator.post_ifft,
                                     inplace=True,
                                     symmetric=True)
                self.queue.finish()
            prep.geo = geo

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

                # get addresses 
                addr = prep.addr

                # local references
                ma = self.ma.S[dID].data
                ob = self.ob.S[oID].data
                pr = self.pr.S[pID].data
                ex = self.ex.S[eID].data

                aux = prep.aux

                geo = prep.geo
                queue = self.queue

                t1 = time.time()
                ev = prep.aux_ex_kernel.npy_build_aux(aux, ob, pr, ex, addr)
                queue.finish()

                self.benchmark.A_Build_aux += time.time() - t1

                ## FFT
                t1 = time.time()
                #geo.transform.ft(aux, aux)
                aux[:] = geo.propagator.fw(aux)
                queue.finish()
                self.benchmark.B_Prop += time.time() - t1

                ## Deviation from measured data
                t1 = time.time()
                prep.fourier_kernel.npy.f = aux
                err_fourier = prep.fourier_kernel.execute_npy()
                queue.finish()
                self.benchmark.C_Fourier_update += time.time() - t1

                ## iFFT
                t1 = time.time()
                #geo.itransform.ift(aux, aux)
                aux[:] = geo.propagator.bw(aux)
                queue.finish()

                self.benchmark.D_iProp += time.time() - t1

                ## apply changes #2
                t1 = time.time()
                ev = prep.aux_ex_kernel.npy_build_exit(aux, ob, pr, ex, addr)
                queue.finish()

                # self.prg.reduce_one_step(queue, (shape_merged[0],64), (1,64), info_gpu.data, err_temp.data, err_exit.data)
                # queue.finish()

                self.benchmark.E_Build_exit += time.time() - t1

                err_phot = np.zeros_like(err_fourier)
                err_exit = np.zeros_like(err_fourier)
                errs = np.array(list(zip(err_fourier, err_phot, err_exit)))
                error = dict(zip(prep.view_IDs, errs))

                self.benchmark.calls_fourier += 1

            parallel.barrier()

            sync = (self.curiter % 1 == 0)
            self.overlap_update(MPI=True)

            parallel.barrier()
            self.curiter += 1
            queue.finish()
        """
        for name, s in self.ob.S.items():
            s.data[:] = s.gpu.get(queue=self.queue)
        for name, s in self.pr.S.items():
            s.data[:] = s.gpu.get(queue=self.queue)

        # costly but needed to sync back with 
        for name, s in self.ex.S.items():
            s.data[:] = s.gpu.get(queue=self.queue)
        """
        self.queue.finish()

        self.error = error
        return error

    def overlap_update(self, MPI=True):
        """
        DM overlap constraint update.
        """
        change = 1.
        # Condition to update probe
        do_update_probe = (self.p.probe_update_start <= self.curiter)

        for inner in range(self.p.overlap_max_iterations):
            prestr = '%d Iteration (Overlap) #%02d:  ' % (parallel.rank, inner)
            # Update object first
            if self.p.update_object_first or (inner > 0):
                # Update object
                log(4, prestr + '----- object update -----', True)
                self.object_update(MPI=(parallel.size > 1 and MPI))

            # Exit if probe should not yet be updated
            if not do_update_probe: break

            # Update probe
            log(4, prestr + '----- probe update -----', True)
            change = self.probe_update(MPI=(parallel.size > 1 and MPI))
            # change = self.probe_update(MPI=(parallel.size>1 and MPI))

            log(4, prestr + 'change in probe is %.3f' % change, True)

            # stop iteration if probe change is small
            if change < self.p.overlap_converge_factor: break

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
            ob.data *= cfact
            #obn.gpu[:] = cfact
            obn.data.fill(cfact)
            queue.finish()

        # storage for-loop
        for dID in self.di.S.keys():
            prep = self.diff_info[dID]

            # find probe, object in exit ID in dependence of dID
            pID, oID, eID = prep.poe_IDs

            # scan for loop
            ev = prep.po_kernel.npy_ob_update(self.ob.S[oID].data,
                                              self.ob_nrm.S[oID].data,
                                              self.pr.S[pID].data,
                                              self.ex.S[eID].data,
                                              prep.addr)

            queue.finish()

        for oID, ob in self.ob.storages.items():
            obn = self.ob_nrm.S[oID]
            # MPI test
            if MPI:
                #ob.data[:] = ob.gpu.get(queue=queue)
                #obn.data[:] = obn.gpu.get(queue=queue)
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
                #ob.gpu.set(ob.data)
            else:
                ob.data /= obn.data

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
            pr.data *= cfact
            prn.data.fill(cfact)

        for dID in self.di.S.keys():
            prep = self.diff_info[dID]

            # find probe, object in exit ID in dependence of dID
            pID, oID, eID = prep.poe_IDs

            # scan for-loop
            ev = prep.po_kernel.npy_pr_update(self.pr.S[pID].data,
                                              self.pr_nrm.S[pID].data,
                                              self.ob.S[oID].data,
                                              self.ex.S[eID].data,
                                              prep.addr)

            queue.finish()

        for pID, pr in self.pr.storages.items():

            buf = self.pr_buf.S[pID]
            prn = self.pr_nrm.S[pID]

            # MPI test
            if MPI:
                # if False:
                #pr.data[:] = pr.gpu.get(queue=queue)
                #prn.data[:] = prn.gpu.get(queue=queue)
                queue.finish()
                parallel.allreduce(pr.data)
                parallel.allreduce(prn.data)
                pr.data /= prn.data

                self.support_constraint(pr)
                # Apply probe support if requested
                #support = self.probe_support.get(pID)
                #if support is not None:
                #    pr.data *= support

                # Apply probe support in Fourier space (This could be better done on GPU)
                #support = self.probe_fourier_support.get(pID)
                #if support is not None:
                #    pr.data[:] = np.fft.ifft2(support * np.fft.fft2(pr.data))

                #pr.gpu.set(pr.data)
            else:
                pr.data /= prn.data
                # ca. 0.3 ms
                # self.pr.S[pID].gpu = probe_gpu
                #pr.data[:] = pr.gpu.get(queue=queue)

            ## this should be done on GPU

            queue.finish()
            # change += u.norm2(pr[i]-buf_pr[i]) / u.norm2(pr[i])
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
        self.queue.finish()
        if parallel.master:
            print("----- BENCHMARKS ----")
            acc = 0.
            for name in sorted(self.benchmark.keys()):
                t = self.benchmark[name]
                if name[0] in 'ABCDEFGHI':
                    print('%20s : %1.3f ms per iteration' % (name, t / self.benchmark.calls_fourier * 1000))
                    acc += t
                elif str(name) == 'probe_update':
                    # pass
                    print('%20s : %1.3f ms per call. %d calls' % (
                        name, t / self.benchmark.calls_probe * 1000, self.benchmark.calls_probe))
                elif str(name) == 'object_update':
                    print('%20s : %1.3f ms per call. %d calls' % (
                        name, t / self.benchmark.calls_object * 1000, self.benchmark.calls_object))

            print('%20s : %1.3f ms per iteration. %d calls' % (
                'Fourier_total', acc / self.benchmark.calls_fourier * 1000, self.benchmark.calls_fourier))

            """
            for name, s in self.ob.S.items():
                plt.figure('obj')
                d = s.gpu.get()
                #print np.abs(d[0][300:-300,300:-300]).mean()
                plt.imshow(u.imsave(d[0][400:-400,400:-400]))
            for name, s in self.pr.S.items():
                d = s.gpu.get()
                for l in d:
                    plt.figure()
                    plt.imshow(u.imsave(l))
                #print u.norm2(d)

            plt.show()
            """

        for original in [self.pr, self.ob, self.ex, self.di, self.ma]:
            original.delete_copy()

        # delete local references to container buffer copies
