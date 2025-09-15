# -*- coding: utf-8 -*-
"""
ThreePIE_cupy: Multislice ePIE on GPU (CuPy) for PTYPY.
Located at: ptypy/custom/threepie_cupy.py
"""

import numpy as np
import cupy as cp

from ptypy.engines import register
from ptypy.core import geometry
from ptypy.utils import Param
from ptypy.utils.verbose import logger

from ptypy.accelerate.cuda_cupy.engines.stochastic import EPIE_cupy
from ptypy.accelerate.cuda_cupy.mem_utils import make_pagelocked_paired_arrays as mppa
from ptypy.accelerate.cuda_cupy.kernels import PropagationKernel

from ptypy import defaults_tree
from ptypy.utils.descriptor import EvalDescriptor

def _roi(view_slice):
    """Robustly get the ROI (slice tuple) for a ptypy View across versions."""
    # Many PTYpy versions expose a hidden getter for the slice:
    if hasattr(view_slice, "_getitem"):
        return view_slice._getitem()
    # Fallback: try 'sl' or 'slices' attributes if present (rare)
    for key in ("sl", "slices"):
        if hasattr(view_slice, key):
            return getattr(view_slice, key)
    raise RuntimeError("Cannot obtain ROI slice for this PTYPY View; update _roi() if your API differs.")

@register()
class ThreePIE_cupy(EPIE_cupy):
    """
    GPU multislice extension of EPIE.

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
    type = float
    help = Thickness of a single slice in meters
    doc = A single float value. All slices will be assumed to be of the same thickness.

    [slice_start_iteration]
    default = 0
    type = int
    help = iteration number to start using a specific slice
    doc =

    [fslices]
    default = slices.h5
    type = str
    help = File path for the slice data
    doc =

    [object_regularization_rate]
    default = 0.0
    type = float
    help = Regularization strength for object updates
    doc =
    """

    def __init__(self, ptycho_parent, pars=None):
        super().__init__(ptycho_parent, pars)
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

        self._object = []
        self._probe  = []
        self._exits  = []
        self._PROP_fw = []
        self._PROP_bw = []
        self._aux_by_scan = {}
        self._slices_initialized = False
        self._propagators_initialized = False

    # ---------- setup ----------
    def engine_initialize(self):
        """
        Called early in the PTYpy initialization sequence.
        Parent class sets up basic structures but not containers yet.
        """
        # Call parent initialization - sets up kernels and basic structures
        super().engine_initialize()

        # Set up diffraction interface reference if available
        if hasattr(self.ptycho, 'diff'):
            self.di = self.ptycho.diff
        else:
            logger.debug("Diffraction interface not available during engine_initialize")

        logger.info(f"ThreePIE_cupy engine_initialize complete (containers will be set up in engine_prepare)")

    def engine_prepare(self):
        """
        Called after model is complete and containers exist.
        This is where the parent class sets up self.ob and self.pr.
        """
        # Call parent's prepare - this sets up self.ob and self.pr
        super().engine_prepare()
        
        # Now we can initialize our multislice components
        if not self._slices_initialized:
            self._initialize_slices()
            
        if not self._propagators_initialized:
            self._initialize_propagators()
            
        logger.info(f"ThreePIE_cupy engine_prepare complete: "
                   f"slices={self._slices_initialized}, "
                   f"propagators={self._propagators_initialized}")

    def _initialize_slices(self):
        """Initialize the multi-slice storage containers."""
        # Check if containers are available - they come from the parent class
        if not hasattr(self, 'ob') or self.ob is None or not hasattr(self, 'pr') or self.pr is None:
            logger.warning("Object/probe containers not yet available from parent class")
            return

        # normalize starts
        if isinstance(self.p.slice_start_iteration, int):
            self.p.slice_start_iteration = np.ones(
                int(self.p.number_of_slices), dtype=int
            ) * int(self.p.slice_start_iteration)
        else:
            self.p.slice_start_iteration = np.asarray(self.p.slice_start_iteration, dtype=int)
            assert len(self.p.slice_start_iteration) == self.p.number_of_slices, \
                "slice_start_iteration must match number_of_slices."

        # per-slice storages, attach pagelocked host + GPU arrays
        self._object = [self.ob.copy(self.ob.ID + f"_o_{i}") for i in range(self.p.number_of_slices)]
        self._probe  = [self.pr.copy(self.pr.ID + f"_p_{i}") for i in range(self.p.number_of_slices)]
        self._exits  = [self.pr.copy(self.pr.ID + f"_e_{i}") for i in range(self.p.number_of_slices)]

        for ob_i, pr_i, ex_i in zip(self._object, self._probe, self._exits):
            for _, S in ob_i.S.items():
                S.gpu, S.data = mppa(S.data.astype(np.complex64, copy=False))
            for _, S in pr_i.S.items():
                S.gpu, S.data = mppa(S.data.astype(np.complex64, copy=False))
            for _, S in ex_i.S.items():
                S.gpu, S.data = mppa(S.data.astype(np.complex64, copy=False))

        self._slices_initialized = True
        logger.info(f"Initialized {len(self._object)} multi-slice storage containers")

    def _initialize_propagators(self):
        """Initialize inter-slice propagators."""
        # Check if we have kernels from parent class
        if not hasattr(self, 'kernels') or not self.kernels:
            logger.warning("Parent kernels not available yet, deferring propagator setup")
            return

        # Need scan geometry info
        if not hasattr(self.ptycho, 'model') or not self.ptycho.model.scans:
            logger.warning("Scan geometry not available, deferring propagator setup")
            return

        # distances between slices
        scan = list(self.ptycho.model.scans.values())[0]
        geom0 = scan.geometries[0]
        g = Param()
        g.energy      = geom0.energy
        g.psize       = geom0.resolution
        g.shape       = geom0.shape
        g.propagation = "nearfield"

        if isinstance(self.p.slice_thickness, (list, tuple, np.ndarray)):
            zlist = list(self.p.slice_thickness)
            assert len(zlist) == self.p.number_of_slices - 1, \
                "slice_thickness list must be N_slices-1 long."
        else:
            zlist = [float(self.p.slice_thickness)] * (self.p.number_of_slices - 1)

        # Get aux buffers from parent kernels
        self._aux_by_scan.clear()
        for label, kern in self.kernels.items():
            if hasattr(kern, 'aux'):
                self._aux_by_scan[label] = kern.aux  # cupy complex64

        if not self._aux_by_scan:
            logger.error("No aux buffers available from parent kernels")
            return

        aux_ref = next(iter(self._aux_by_scan.values()))

        # allocate inter-slice propagators
        self._PROP_fw.clear()
        self._PROP_bw.clear()
        for dz in zlist:
            g.distance = dz
            G = geometry.Geo(owner=None, pars=g)
            PKf = PropagationKernel(aux_ref, G.propagator, self.queue, self.p.fft_lib)
            PKb = PropagationKernel(aux_ref, G.propagator, self.queue, self.p.fft_lib)
            PKf.allocate()
            PKb.allocate()
            self._PROP_fw.append(PKf)
            self._PROP_bw.append(PKb)

        self._propagators_initialized = True
        logger.info(f"Initialized {len(self._PROP_fw)} inter-slice propagators")

    # ---------- iterate ----------
    def engine_iterate(self, num=1):
        """
        Compute iterations. Overrides parent's method to add multislice.
        """
        # Make sure components are initialized (in case engine_prepare wasn't called)
        if not self._slices_initialized:
            self._initialize_slices()
            
        if not self._propagators_initialized:
            self._initialize_propagators()

        # Check if we can proceed with multislice
        if not self._slices_initialized:
            logger.warning("Slices not initialized, falling back to parent EPIE")
            return super().engine_iterate(num)

        # Ensure diffraction interface is available
        if not hasattr(self, 'di') or self.di is None:
            if hasattr(self.ptycho, 'diff'):
                self.di = self.ptycho.diff
            else:
                logger.error("No diffraction interface available")
                return {}

        # Use parent's curiter if we don't have our own
        if not hasattr(self, 'curiter') or self.curiter is None:
            if hasattr(super(), 'curiter'):
                self.curiter = super().curiter
            else:
                self.curiter = 0

        vieworder = list(self.di.views.keys())
        vieworder.sort()
        rng = np.random.default_rng()

        error_out = {}
        for _ in range(num):
            rng.shuffle(vieworder)
            for name in vieworder:
                view = self.di.views[name]
                if not view.active:
                    continue
                    
                # Use appropriate update method based on initialization state
                if self._propagators_initialized and self.p.number_of_slices > 1:
                    error_out[name] = self._multislice_update_gpu(view)
                else:
                    # Fallback to single-slice if propagators aren't ready
                    logger.debug(f"Using single-slice fallback for view {name}")
                    error_out[name] = self._single_slice_fallback(view)
                    
            self.curiter += 1
        return error_out

    def _single_slice_fallback(self, view):
        """Fallback to single-slice update when propagators aren't available."""
        if not self._slices_initialized or not self._object:
            logger.warning("Cannot perform single-slice fallback: slices not initialized")
            # Last resort: use parent's methods directly
            for _, pod in view.pods.items():
                pass  # Parent should handle this
            return self.fourier_update(view) if hasattr(self, 'fourier_update') else 0.0
        
        # Use only the first slice as a regular EPIE update
        for _, pod in view.pods.items():
            prS = self._probe[0].S[pod.pr_view.storage.ID].gpu
            obS = self._object[0].S[pod.ob_view.storage.ID].gpu
            exS = self._exits[0].S[pod.pr_view.storage.ID].gpu

            pr_sl = _roi(pod.pr_view)
            ob_sl = _roi(pod.ob_view)
            ex_sl = _roi(pod.pr_view)

            exS[ex_sl] = prS[pr_sl] * obS[ob_sl]
            pod.probe  = prS[pr_sl]
            pod.object = obS[ob_sl]
            pod.exit   = exS[ex_sl]

        # Use parent class Fourier update
        error_np = self.fourier_update(view)

        # Update using parent class methods
        exits_map = {pod.ID: self._exits[0].S[pod.pr_view.storage.ID].gpu[_roi(pod.pr_view)]
                     for _, pod in view.pods.items()}
        self.object_update(view, exits_map)
        self.probe_update(view, exits_map)

        for _, pod in view.pods.items():
            prS = self._probe[0].S[pod.pr_view.storage.ID].gpu
            obS = self._object[0].S[pod.ob_view.storage.ID].gpu
            prS[_roi(pod.pr_view)] = pod.probe
            obS[_roi(pod.ob_view)] = pod.object

        # Update main object container
        self.ob.fill(self._object[0])

        return error_np

    # ---------- finalize ----------
    def engine_finalize(self):
        """Clean up and finalize reconstruction."""
        if self._slices_initialized and len(self._object) > 0:
            self.ob.fill(self._object[0])
            for i in range(1, min(len(self._object), self.p.number_of_slices)):
                self.ob *= self._object[i]
        return super().engine_finalize()

    # ---------- core GPU multislice ----------
    def _multislice_update_gpu(self, view):
        """Full multislice update on GPU."""
        # Safety check
        if not self._aux_by_scan:
            logger.warning("No aux buffers available, falling back to single slice")
            return self._single_slice_fallback(view)

        aux = next(iter(self._aux_by_scan.values()))

        # forward: 0..N-2
        for s in range(self.p.number_of_slices - 1):
            for _, pod in view.pods.items():
                prS = self._probe[s].S[pod.pr_view.storage.ID].gpu
                obS = self._object[s].S[pod.ob_view.storage.ID].gpu
                exS = self._exits[s].S[pod.pr_view.storage.ID].gpu

                # Try to get slice info, with fallback to direct indexing
                try:
                    pr_sl = _roi(pod.pr_view)
                    ob_sl = _roi(pod.ob_view)
                    ex_sl = _roi(pod.pr_view)
                except:
                    # Fallback: use the pod's direct addressing
                    # This assumes the view references a single frame
                    logger.debug("Using fallback indexing for slices")
                    pr_sl = (slice(None), slice(None), slice(None))  # Use full array
                    ob_sl = (slice(None), slice(None), slice(None))  # Use full array
                    ex_sl = (slice(None), slice(None), slice(None))  # Use full array
                    
                    # Alternative: get from pod.pr_view.shape and pod.pr_view.dlow if available
                    if hasattr(pod.pr_view, 'shape') and hasattr(pod.pr_view, 'dlow'):
                        dlow = pod.pr_view.dlow
                        shape = pod.pr_view.shape
                        pr_sl = tuple(slice(d, d + s) for d, s in zip(dlow, shape))
                        ex_sl = pr_sl
                    if hasattr(pod.ob_view, 'shape') and hasattr(pod.ob_view, 'dlow'):
                        dlow = pod.ob_view.dlow
                        shape = pod.ob_view.shape
                        ob_sl = tuple(slice(d, d + s) for d, s in zip(dlow, shape))

                if self.curiter >= self.p.slice_start_iteration[s]:
                    exS[ex_sl] = prS[pr_sl] * obS[ob_sl]
                else:
                    exS[ex_sl] = prS[pr_sl]

                aux[...] = 0
                aux[0, ...] = exS[ex_sl]
                self._PROP_fw[s].fw(aux, aux)

                pr_next = self._probe[s+1].S[pod.pr_view.storage.ID].gpu
                pr_next[pr_sl] = aux[0, ...]

        # last slice: set pod state for Fourier update
        for _, pod in view.pods.items():
            prS = self._probe[-1].S[pod.pr_view.storage.ID].gpu
            obS = self._object[-1].S[pod.ob_view.storage.ID].gpu
            exS = self._exits[-1].S[pod.pr_view.storage.ID].gpu

            try:
                pr_sl = _roi(pod.pr_view)
                ob_sl = _roi(pod.ob_view)
                ex_sl = _roi(pod.pr_view)
            except:
                pr_sl = (slice(None), slice(None), slice(None))
                ob_sl = (slice(None), slice(None), slice(None))
                ex_sl = (slice(None), slice(None), slice(None))
                
                if hasattr(pod.pr_view, 'shape') and hasattr(pod.pr_view, 'dlow'):
                    dlow = pod.pr_view.dlow
                    shape = pod.pr_view.shape
                    pr_sl = tuple(slice(d, d + s) for d, s in zip(dlow, shape))
                    ex_sl = pr_sl
                if hasattr(pod.ob_view, 'shape') and hasattr(pod.ob_view, 'dlow'):
                    dlow = pod.ob_view.dlow
                    shape = pod.ob_view.shape
                    ob_sl = tuple(slice(d, d + s) for d, s in zip(dlow, shape))

            if self.curiter >= self.p.slice_start_iteration[-1]:
                exS[ex_sl] = prS[pr_sl] * obS[ob_sl]
            else:
                exS[ex_sl] = prS[pr_sl]

            pod.probe  = prS[pr_sl]
            pod.object = obS[ob_sl]
            pod.exit   = exS[ex_sl]

        # Fourier update on GPU
        error_np = self.fourier_update(view)

        # update last slice
        if self.curiter >= self.p.slice_start_iteration[-1]:
            exits_map = {}
            for _, pod in view.pods.items():
                try:
                    ex_sl = _roi(pod.pr_view)
                except:
                    if hasattr(pod.pr_view, 'shape') and hasattr(pod.pr_view, 'dlow'):
                        dlow = pod.pr_view.dlow
                        shape = pod.pr_view.shape
                        ex_sl = tuple(slice(d, d + s) for d, s in zip(dlow, shape))
                    else:
                        ex_sl = (slice(None), slice(None), slice(None))
                exits_map[pod.ID] = self._exits[-1].S[pod.pr_view.storage.ID].gpu[ex_sl]
                
            self.object_update(view, exits_map)
            self.probe_update(view, exits_map)
            for _, pod in view.pods.items():
                prS = self._probe[-1].S[pod.pr_view.storage.ID].gpu
                obS = self._object[-1].S[pod.ob_view.storage.ID].gpu
                try:
                    pr_sl = _roi(pod.pr_view)
                    ob_sl = _roi(pod.ob_view)
                except:
                    pr_sl = (slice(None), slice(None), slice(None))
                    ob_sl = (slice(None), slice(None), slice(None))
                    if hasattr(pod.pr_view, 'shape') and hasattr(pod.pr_view, 'dlow'):
                        dlow = pod.pr_view.dlow
                        shape = pod.pr_view.shape
                        pr_sl = tuple(slice(d, d + s) for d, s in zip(dlow, shape))
                    if hasattr(pod.ob_view, 'shape') and hasattr(pod.ob_view, 'dlow'):
                        dlow = pod.ob_view.dlow
                        shape = pod.ob_view.shape
                        ob_sl = tuple(slice(d, d + s) for d, s in zip(dlow, shape))
                prS[pr_sl] = pod.probe
                obS[ob_sl] = pod.object
        else:
            for _, pod in view.pods.items():
                prS = self._probe[-1].S[pod.pr_view.storage.ID].gpu
                exS = self._exits[-1].S[pod.pr_view.storage.ID].gpu
                try:
                    pr_sl = _roi(pod.pr_view)
                except:
                    pr_sl = (slice(None), slice(None), slice(None))
                    if hasattr(pod.pr_view, 'shape') and hasattr(pod.pr_view, 'dlow'):
                        dlow = pod.pr_view.dlow
                        shape = pod.pr_view.shape
                        pr_sl = tuple(slice(d, d + s) for d, s in zip(dlow, shape))
                prS[pr_sl] = exS[pr_sl]

        # backward: N-2..0
        for s in range(self.p.number_of_slices - 2, -1, -1):
            if self.curiter < self.p.slice_start_iteration[s]:
                for _, pod in view.pods.items():
                    prS  = self._probe[s].S[pod.pr_view.storage.ID].gpu
                    prSn = self._probe[s+1].S[pod.pr_view.storage.ID].gpu
                    try:
                        pr_sl = _roi(pod.pr_view)
                    except:
                        pr_sl = (slice(None), slice(None), slice(None))
                        if hasattr(pod.pr_view, 'shape') and hasattr(pod.pr_view, 'dlow'):
                            dlow = pod.pr_view.dlow
                            shape = pod.pr_view.shape
                            pr_sl = tuple(slice(d, d + s) for d, s in zip(dlow, shape))
                    aux[...] = 0
                    aux[0, ...] = prSn[pr_sl]
                    self._PROP_bw[s].bw(aux, aux)
                    prS[pr_sl] = aux[0, ...]
                continue

            for _, pod in view.pods.items():
                prS  = self._probe[s].S[pod.pr_view.storage.ID].gpu
                prSn = self._probe[s+1].S[pod.pr_view.storage.ID].gpu
                obS  = self._object[s].S[pod.ob_view.storage.ID].gpu

                try:
                    pr_sl = _roi(pod.pr_view)
                    ob_sl = _roi(pod.ob_view)
                except:
                    pr_sl = (slice(None), slice(None), slice(None))
                    ob_sl = (slice(None), slice(None), slice(None))
                    if hasattr(pod.pr_view, 'shape') and hasattr(pod.pr_view, 'dlow'):
                        dlow = pod.pr_view.dlow
                        shape = pod.pr_view.shape
                        pr_sl = tuple(slice(d, d + s) for d, s in zip(dlow, shape))
                    if hasattr(pod.ob_view, 'shape') and hasattr(pod.ob_view, 'dlow'):
                        dlow = pod.ob_view.dlow
                        shape = pod.ob_view.shape
                        ob_sl = tuple(slice(d, d + s) for d, s in zip(dlow, shape))

                aux[...] = 0
                aux[0, ...] = prSn[pr_sl]
                self._PROP_bw[s].bw(aux, aux)

                pod.exit   = aux[0, ...]
                pod.probe  = prS[pr_sl]
                pod.object = obS[ob_sl]

            exits_map_s = {}
            for _, pod in view.pods.items():
                try:
                    ex_sl = _roi(pod.pr_view)
                except:
                    if hasattr(pod.pr_view, 'shape') and hasattr(pod.pr_view, 'dlow'):
                        dlow = pod.pr_view.dlow
                        shape = pod.pr_view.shape
                        ex_sl = tuple(slice(d, d + s) for d, s in zip(dlow, shape))
                    else:
                        ex_sl = (slice(None), slice(None), slice(None))
                exits_map_s[pod.ID] = self._exits[s].S[pod.pr_view.storage.ID].gpu[ex_sl]
                
            self.object_update(view, exits_map_s)
            self.probe_update(view,  exits_map_s)

            for _, pod in view.pods.items():
                prS = self._probe[s].S[pod.pr_view.storage.ID].gpu
                obS = self._object[s].S[pod.ob_view.storage.ID].gpu
                try:
                    pr_sl = _roi(pod.pr_view)
                    ob_sl = _roi(pod.ob_view)
                except:
                    pr_sl = (slice(None), slice(None), slice(None))
                    ob_sl = (slice(None), slice(None), slice(None))
                    if hasattr(pod.pr_view, 'shape') and hasattr(pod.pr_view, 'dlow'):
                        dlow = pod.pr_view.dlow
                        shape = pod.pr_view.shape
                        pr_sl = tuple(slice(d, d + s) for d, s in zip(dlow, shape))
                    if hasattr(pod.ob_view, 'shape') and hasattr(pod.ob_view, 'dlow'):
                        dlow = pod.ob_view.dlow
                        shape = pod.ob_view.shape
                        ob_sl = tuple(slice(d, d + s) for d, s in zip(dlow, shape))
                prS[pr_sl] = pod.probe
                obS[ob_sl] = pod.object

        # live product for plotting
        self.ob.fill(self._object[0])
        for k in range(1, self.p.number_of_slices):
            self.ob *= self._object[k]

        if getattr(self.p, "object_regularization_rate", 0.0) > 0:
            self._apply_object_regularization_gpu()

        return error_np

    # ---------- optional regularization (GPU FFT) ----------
    def _apply_object_regularization_gpu(self):
        """Apply regularization to object slices."""
        if not self._slices_initialized or len(self._object) < 2:
            return
            
        assert self.p.number_of_slices > 1
        assert isinstance(self.p.slice_thickness, float)

        sid = next(iter(self._object[0].S))
        shp = self._object[0].S[sid].data.shape[1:]
        psize = self._object[0].S[sid].psize[0]

        stack = cp.stack([self._object[i].S[sid].gpu[0, ...]
                          for i in range(self.p.number_of_slices)], axis=0)

        kz = cp.fft.fftfreq(self.p.number_of_slices, self.p.slice_thickness)[:, None, None]
        ky = cp.fft.fftfreq(shp[0], psize)[:, None]
        kx = cp.fft.fftfreq(shp[1], psize)

        w = 1 - 2 * cp.arctan2(
            (self.p.object_regularization_rate ** 2) * kz**2,
            (kx**2 + ky**2 + cp.finfo(cp.float32).eps)
        ) / cp.pi

        cur = cp.fft.ifftn(cp.fft.fftn(stack) * w).astype(cp.complex64)

        for i in range(self.p.number_of_slices):
            self._object[i].S[sid].gpu[0, ...] = cur[i, ...]