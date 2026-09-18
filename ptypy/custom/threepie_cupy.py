# -*- coding: utf-8 -*-
"""
Accelerated multislice ePIE (3PIE) engine for the cupy backend.

The engine keeps the object of every slice in its own storage layer, sweeps
the exit wave forward through the slices, applies the far-field constraint at
the last one and sweeps back, updating object and probe slice by slice. The
update of one scan position is issued as a single CUDA graph by default, see
the ``cuda_graphs`` parameter.

This file is part of the PTYPY package.

    :copyright: Copyright 2014 by the PTYPY team, see AUTHORS.
    :license: see LICENSE for details.
"""
import os

import numpy as np
import cupy as cp

from ptypy import io
from ptypy.core import geometry
from ptypy.utils import Param
from ptypy.utils import parallel
from ptypy.utils.verbose import logger
from ptypy.engines import register
from ptypy.engines.stochastic import EPIEMixin
from ptypy.accelerate.cuda_cupy.engines.stochastic import _StochasticEngineCupy
from ptypy.accelerate.cuda_cupy.kernels import PropagationKernel, ThreePIEWaveKernel
from ptypy.accelerate.cuda_cupy.array_utils import MaxKernel
from ptypy.accelerate.cuda_cupy.mem_utils import \
    make_pagelocked_paired_arrays as mppa
from ptypy.custom.multislice_utils import (
    normalize_slice_pad, slice_bandlimit, PaddedSlicePropagationKernel)

__all__ = ['ThreePIE_cupy']

@register()
class ThreePIE_cupy(_StochasticEngineCupy, EPIEMixin):
    """
    An accelerated implementation of multislice ePIE / 3PIE.

    Defaults:

    [name]
    default = ThreePIE_cupy
    type = str
    help =
    doc =

    [number_of_slices]
    default = 2
    type = int
    help = The number of slices
    doc = Defines how many slices are used for the multi-slice object.

    [slice_thickness]
    default = 1e-6
    type = float, list, tuple
    help = Thickness of a single slice in meters
    doc = A single float value or a list of float values. If a single value
          is given, all slices get the same thickness.

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

    [slice_pad]
    default = 1
    type = int, str
    help = Zero-padding factor for inter-slice near-field propagation
    doc = Positive integer or ``"auto"``. Padding keeps the real-space pixel
          size and enlarges the propagation grid, which raises the
          angular-spectrum sampling limit for a fixed slice spacing.

    [slice_bandlimit]
    default = True
    type = bool
    help = Apply angular-spectrum anti-aliasing support for slice propagation
    doc =

    [cuda_graphs]
    default = True
    type = bool
    help = Record the per-view update into CUDA graphs and replay them
    doc = The kernel launches of one view are captured once per view (per
          data block and per set of active slices) and replayed with a
          single launch afterwards. The result is identical to launching
          the kernels one by one. Set to False to run the plain loop.
          Ignored when position refinement is on.

    """

    def __init__(self, ptycho_parent, pars=None):
        _StochasticEngineCupy.__init__(self, ptycho_parent, pars)
        EPIEMixin.__init__(self, self.p.alpha, self.p.beta)
        self.article = dict(
            title='Ptychographic transmission microscopy in three dimensions using a multi-slice approach',
            author='A. M. Maiden et al.',
            journal='J. Opt. Soc. Am. A',
            volume=29,
            year=2012,
            page=1606,
            doi='10.1364/JOSAA.29.001606',
            comment='The 3PIE reconstruction algorithm',
        )
        ptycho_parent.citations.add_article(**self.article)

    def engine_initialize(self):
        """Create the per-slice object, probe and exit-wave containers."""
        super().engine_initialize()

        if self.p.number_of_slices < 1:
            raise ValueError("number_of_slices must be at least 1")

        self._object = [None] * self.p.number_of_slices
        self._probe = [None] * self.p.number_of_slices
        self._exits = [None] * self.p.number_of_slices
        for i in range(self.p.number_of_slices):
            self._object[i] = self.ob.copy(self.ob.ID + "_o_" + str(i))
            self._probe[i] = self.pr.copy(self.pr.ID + "_p_" + str(i))
            self._exits[i] = self.pr.copy(self.pr.ID + "_e_" + str(i))

        if isinstance(self.p.slice_start_iteration, int):
            self.p.slice_start_iteration = (
                np.ones(self.p.number_of_slices, dtype=np.int32)
                * self.p.slice_start_iteration
            )
        elif len(self.p.slice_start_iteration) != self.p.number_of_slices:
            raise ValueError(
                "slice_start_iteration must have one value per slice"
            )

    def _setup_kernels(self):
        super()._setup_kernels()
        self._setup_slice_propagators()

    def _setup_slice_propagators(self):
        """Build the padded inter-slice propagators and per-slice buffers."""
        if isinstance(self.p.slice_thickness, (list, tuple)):
            if len(self.p.slice_thickness) != self.p.number_of_slices - 1:
                raise ValueError(
                    "slice_thickness must contain number_of_slices - 1 values"
                )
            thicknesses = self.p.slice_thickness
        else:
            thicknesses = [self.p.slice_thickness] * (self.p.number_of_slices - 1)

        for label, scan in self.ptycho.model.scans.items():
            geo = scan.geometries[0]
            pad = normalize_slice_pad(
                self.p.slice_pad, geo.shape, geo.resolution, geo.energy,
                self.p.slice_thickness)
            g = Param()
            g.energy = geo.energy
            g.psize = geo.resolution
            g.shape = tuple(int(v) * pad for v in geo.shape)
            g.propagation = "nearfield"

            kern = self.kernels[label]
            kern.slice_PROP = []
            aux_shape = tuple(kern.aux.shape[:-2]) + tuple(
                int(v) * pad for v in kern.aux.shape[-2:])
            aux = np.zeros(aux_shape, dtype=kern.aux.dtype)
            for thickness in thicknesses:
                g.distance = thickness
                G = geometry.Geo(owner=None, pars=g)
                if self.p.slice_bandlimit:
                    support = slice_bandlimit(
                        G.propagator.kernel.shape, geo.resolution, geo.energy,
                        thickness)
                    G.propagator.kernel *= support
                    G.propagator.ikernel *= support
                prop = PropagationKernel(
                    aux, G.propagator, self.queue, self.p.fft_lib
                )
                prop.allocate()
                kern.slice_PROP.append(PaddedSlicePropagationKernel(
                    prop, kern.aux.shape[-2:], pad=pad))
            kern.slice_exits = [
                cp.empty_like(kern.aux) for _ in range(self.p.number_of_slices)
            ]
            kern.slice_tmp = cp.empty_like(kern.aux)
            kern.slice_back = cp.empty_like(kern.aux)
            kern.TWK = ThreePIEWaveKernel(queue_thread=self.queue)
            # object-norm maxima per slice in preallocated buffers (no
            # allocation inside the view update, so it can be captured)
            kern.obn_max = [cp.zeros((1,), dtype=np.float32)
                            for _ in range(self.p.number_of_slices)]
            kern.prn_max = [cp.zeros((1,), dtype=np.float32)
                            for _ in range(self.p.number_of_slices)]
            kern.MAXK = MaxKernel(queue=self.queue)

    def engine_prepare(self):
        """Move the per-slice containers to the device; captured graphs expire."""
        super().engine_prepare()
        # captured graphs refer to the per-block device arrays that prepare
        # (re)allocates, so they are invalid from here on
        self._graph_cache = {}
        for container in self._object + self._probe + self._exits:
            for storage in container.S.values():
                if not hasattr(storage, "gpu"):
                    storage.gpu, storage.data = mppa(storage.data)

    def _slice_active(self, index):
        return self.curiter >= self.p.slice_start_iteration[index]

    def _last_view_record(self):
        """ID, frame index and position of the last view this engine processed."""
        name = getattr(self, "_last_view", None)
        if name is None:
            return None
        view = self.di.views[name]
        return {"ID": str(name), "layer": int(view.layer),
                "coord": np.asarray(view.pod.ob_view.coord, dtype=float)}

    def _sync_primary_gpu_arrays(self):
        for oID, storage in self.ob.S.items():
            cp.copyto(storage.gpu, self._object[0].S[oID].gpu)
            for i in range(1, self.p.number_of_slices):
                storage.gpu *= self._object[i].S[oID].gpu
        for pID, storage in self.pr.S.items():
            cp.copyto(storage.gpu, self._probe[0].S[pID].gpu)

    def _update_view(self, prep, kern, i, ob_layers, pr_layers):
        """
        One multislice ePIE update of view ``i`` of the block ``prep``: the
        forward sweep through the slices, the far-field constraint on the
        last slice and the backward sweep with the object and probe updates.
        Every operation is a stream operation whose buffers outlive the
        call (the reductions write into preallocated buffers), so the
        sequence can be recorded into a CUDA graph; the only device memory
        created on a first call is kernel scratch space that the kernel
        objects keep alive.
        """
        FUK = kern.FUK
        AWK = kern.AWK
        POK = kern.POK
        MAK = kern.MAK
        PROP = kern.PROP
        TWK = kern.TWK
        aux = kern.aux
        nslices = self.p.number_of_slices

        addr = prep.addr_gpu[i, None]
        ex_from, ex_to = prep.addr_ex[i]
        ex = prep.ex_full[ex_from:ex_to]
        mag = prep.mag_full[i, None]
        ma = prep.ma_full[i, None]
        ma_sum = prep.ma_sum_gpu[i, None]
        obn = prep.obn
        prn = prep.prn
        err_phot = prep.err_phot_gpu[i, None]
        err_fourier = prep.err_fourier_gpu[i, None]
        err_exit = prep.err_exit_gpu[i, None]

        for s in range(nslices):
            old_exit = kern.slice_exits[s]
            if self._slice_active(s):
                AWK.build_aux2_no_ex(old_exit, addr, ob_layers[s], pr_layers[s])
            else:
                TWK.pr_to_aux(old_exit, pr_layers[s], addr)
            if s < nslices - 1:
                kern.slice_PROP[s].fw(old_exit, kern.slice_tmp)
                TWK.aux_to_pr(pr_layers[s + 1], kern.slice_tmp, addr)

        cp.copyto(ex, kern.slice_exits[-1][:ex.shape[0]])
        AWK.make_aux(aux, addr, ob_layers[-1], pr_layers[-1], ex,
                     c_po=self._c, c_e=1 - self._c)
        PROP.fw(aux, aux)
        if self.p.compute_fourier_error:
            FUK.fourier_error(aux, addr, mag, ma, ma_sum)
            FUK.error_reduce(addr, err_fourier)
        else:
            FUK.fourier_deviation(aux, addr, mag)
        FUK.fmag_update_nopbound(aux, addr, mag, ma)
        PROP.bw(aux, aux)
        AWK.make_exit(aux, addr, ob_layers[-1], pr_layers[-1], ex,
                      c_a=self._b, c_po=self._a, c_e=-(self._a + self._b))
        if self.p.compute_exit_error:
            FUK.exit_error(aux, addr)
            FUK.error_reduce(addr, err_exit)
        if self.p.compute_log_likelihood:
            AWK.build_aux2_no_ex(aux, addr, ob_layers[-1], pr_layers[-1])
            PROP.fw(aux, aux)
            FUK.log_likelihood2(aux, addr, mag, ma, err_phot)

        back_wave = ex
        for s in range(nslices - 1, -1, -1):
            if s < nslices - 1:
                TWK.pr_to_aux(kern.slice_tmp, pr_layers[s + 1], addr)
                kern.slice_PROP[s].bw(kern.slice_tmp, kern.slice_back)
                back_wave = kern.slice_back
            if self._slice_active(s):
                POK.pr_norm_local(addr, pr_layers[s], prn)
                kern.MAXK.max(prn, kern.prn_max[s])
                POK.ob_update_local(
                    addr, ob_layers[s], pr_layers[s], back_wave,
                    kern.slice_exits[s], prn, a=self._ob_a, b=self._ob_b,
                    prn_max=kern.prn_max[s])
                obn_max = kern.obn_max[s]
                if self._object_norm_is_global and self._pr_a == 0:
                    MAK.max_abs2(ob_layers[s], obn_max)
                    obn.fill(np.float32(0.))
                else:
                    POK.ob_norm_local(addr, ob_layers[s], obn)
                    kern.MAXK.max(obn, obn_max)
                if self.p.probe_update_start <= self.curiter:
                    POK.pr_update_local(
                        addr, pr_layers[s], ob_layers[s], back_wave,
                        kern.slice_exits[s], obn, obn_max,
                        a=self._pr_a, b=self._pr_b)
            else:
                TWK.aux_to_pr(pr_layers[s], back_wave, addr)

    def _view_graphs(self, prep, kern, dID, ob_layers, pr_layers):
        """
        CUDA graphs of the per-view update, one per view of the block,
        captured on first use for the current set of active slices, probe
        update state and GPU data buffers, and cached on the engine.
        """
        state = (tuple(bool(self._slice_active(s))
                       for s in range(self.p.number_of_slices)),
                 bool(self.p.probe_update_start <= self.curiter),
                 float(self._c), float(self._a), float(self._b),
                 float(self._ob_a), float(self._ob_b),
                 float(self._pr_a), float(self._pr_b))
        buffers = tuple(int(a.data.ptr) for a in (
            prep.ex_full, prep.mag_full, prep.ma_full, prep.addr_gpu,
            prep.obn, prep.prn, prep.ma_sum_gpu, prep.err_fourier_gpu,
            prep.err_phot_gpu, prep.err_exit_gpu))
        key = (dID, state, buffers)
        cache = getattr(self, "_graph_cache", None)
        if cache is None:
            cache = self._graph_cache = {}
        if key in cache:
            return cache[key]
        # graphs of an earlier state (other active slices, probe not yet
        # updated, other coefficients) can never be replayed again
        for old in [k for k in cache if k[1] != state]:
            del cache[old]
        # relaxed capture mode: a kernel's first call may compile it and
        # allocate its scratch space while the stream is being captured
        mode = cp.cuda.runtime.streamCaptureModeRelaxed
        graphs = []
        for i in range(len(prep.view_IDs)):
            self.queue.begin_capture(mode=mode)
            try:
                self._update_view(prep, kern, i, ob_layers, pr_layers)
            except Exception:
                try:
                    self.queue.end_capture()
                except Exception:
                    pass
                raise
            graphs.append(self.queue.end_capture())
        cache[key] = graphs
        logger.info("ThreePIE_cupy: captured %d CUDA graphs for block %s"
                    % (len(graphs), dID))
        return graphs

    def engine_iterate(self, num=1):
        """
        Compute one multislice ePIE iteration on the GPU.
        """
        self.dID_list = list(self.di.S.keys())
        use_graphs = bool(self.p.cuda_graphs) and not self.do_position_refinement
        if self.p.cuda_graphs and self.do_position_refinement \
                and not getattr(self, "_graphs_warned", False):
            logger.warning("ThreePIE_cupy: cuda_graphs is ignored while "
                           "position refinement is active")
            self._graphs_warned = True

        for it in range(num):
            reduced_error = np.zeros((3,))
            reduced_error_count = 0
            local_error = {}

            for iblock, dID in enumerate(self.dID_list):
                prep = self.diff_info[dID]
                pID, oID, eID = prep.poe_IDs
                kern = self.kernels[prep.label]

                ob_layers = [self._object[i].S[oID].gpu
                             for i in range(self.p.number_of_slices)]
                pr_layers = [self._probe[i].S[pID].gpu
                             for i in range(self.p.number_of_slices)]

                vieworder = prep.vieworder
                prep.rng.shuffle(vieworder)

                ev_ex, ex_full, data_ex = self.ex_data.to_gpu(
                    prep.ex, dID, self.qu_htod)
                ev_mag, mag_full, data_mag = self.mag_data.to_gpu(
                    prep.mag, dID, self.qu_htod)
                ev_ma, ma_full, data_ma = self.ma_data.to_gpu(
                    prep.ma, dID, self.qu_htod)

                prep.ex_full = ex_full
                prep.mag_full = mag_full
                prep.ma_full = ma_full

                # the block's data must be on the device before the first
                # view; waiting here (instead of inside the view) keeps the
                # per-view sequence free of event dependencies
                self.queue.wait_event(ev_ex)
                self.queue.wait_event(ev_mag)
                self.queue.wait_event(ev_ma)

                graphs = (self._view_graphs(prep, kern, dID, ob_layers,
                                            pr_layers) if use_graphs else None)

                for i in vieworder:
                    if graphs is not None:
                        graphs[i].launch(self.queue)
                    else:
                        if self.do_position_refinement:
                            # position refinement reads the primary object
                            # and probe, so keep them current per view
                            self._sync_primary_gpu_arrays()
                        self.position_update_local(prep, i)
                        if self.do_position_refinement:
                            prep.addr[i, None] = prep.addr_gpu[i, None].get()
                        self._update_view(prep, kern, i, ob_layers, pr_layers)
                    self._last_view = prep.view_IDs[i]

                data_ex.record_done(self.queue, 'compute')
                if iblock + len(self.ex_data) < len(self.dID_list):
                    data_ex.from_gpu(self.qu_dtoh)

            self.dID_list.reverse()
            # product object and entrance probe for output/plotting, once per
            # iteration: nothing inside the view loop reads them
            self._sync_primary_gpu_arrays()
            self.curiter += 1
            self.ex_data.syncback = False

        self.queue.synchronize()

        for name, s in self.ob.S.items():
            cp.cuda.runtime.memcpyAsync(dst=s.data.ctypes.data,
                            src=s.gpu.data.ptr,
                            size=s.gpu.nbytes,
                            kind=2,
                            stream=self.queue.ptr)
        for name, s in self.pr.S.items():
            cp.cuda.runtime.memcpyAsync(dst=s.data.ctypes.data,
                            src=s.gpu.data.ptr,
                            size=s.gpu.nbytes,
                            kind=2,
                            stream=self.queue.ptr)

        for dID, prep in self.diff_info.items():
            err_fourier = prep.err_fourier_gpu.get()
            err_phot = prep.err_phot_gpu.get()
            err_exit = prep.err_exit_gpu.get()
            errs = np.ascontiguousarray(
                np.vstack([err_fourier, err_phot, err_exit]).T)
            if self.p.record_local_error:
                local_error.update(zip(prep.view_IDs, errs))
            else:
                reduced_error += errs.sum(axis=0)
                reduced_error_count += errs.shape[0]

        if self.p.record_local_error:
            error = local_error
        else:
            error = parallel.allreduce(reduced_error)
            count = parallel.allreduce(reduced_error_count)
            error /= count

        self.qu_dtoh.synchronize()

        return error

    def engine_finalize(self):
        """Write the slice file and release the device arrays and graphs."""
        self._sync_primary_gpu_arrays()
        self.queue.synchronize()

        for container in self._object + self._probe + self._exits:
            for storage in container.S.values():
                storage.data = storage.gpu.get()

        slices_info = Param()
        slices_info.number_of_slices = self.p.number_of_slices
        slices_info.slice_thickness = self.p.slice_thickness
        slices_info.objects = {
            ob.ID: {ID: S._to_dict() for ID, S in ob.storages.items()}
            for ob in self._object
        }
        # incident wave per slice: probes[0] is the illumination, probes[s > 0]
        # the wave that entered slice s for the last view that was processed
        slices_info.probes = {
            pr.ID: {ID: S._to_dict() for ID, S in pr.storages.items()}
            for pr in self._probe
        }
        # which scan position the probes[s > 0] waves belong to
        record = self._last_view_record()
        if record is not None:
            slices_info.last_view = record
        slices_info.slice_start_iteration = self.p.slice_start_iteration

        header = {'description': 'multi-slices result details.'}
        h5opt = io.h5options['UNSUPPORTED']
        io.h5options['UNSUPPORTED'] = 'ignore'
        logger.info(f'Saving to {self.p.fslices}')
        io.h5write(self.p.fslices, header=header, content=slices_info)
        io.h5options['UNSUPPORTED'] = h5opt

        for container in self._object + self._probe + self._exits:
            for storage in container.S.values():
                del storage.gpu
        self._graph_cache = {}

        return super().engine_finalize()


