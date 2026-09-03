# -*- coding: utf-8 -*-
"""
Serialized (NumPy) implementation of the multislice ePIE / 3PIE algorithm.

This engine sits between the pod/view CPU reference
(``ptypy.custom.threepie.ThreePIE``) and the GPU engine
(``ptypy.accelerate.cuda_cupy.engines.stochastic.ThreePIE_cupy``).

It runs on the CPU but uses the same serialized address layout and kernel
set as the GPU engine (``AuxiliaryWaveKernel``, ``PoUpdateKernel``,
``FourierUpdateKernel`` and ``ThreePIEWaveKernel`` from
``ptypy.accelerate.base.kernels``). The ``engine_iterate`` multislice sweep
is therefore a line-for-line NumPy mirror of the CuPy version. The algorithm
can be checked here without a GPU, and the GPU port is a small diff.

Reference: A. M. Maiden, M. J. Humphry, J. M. Rodenburg,
"Ptychographic transmission microscopy in three dimensions using a multi-slice
approach", J. Opt. Soc. Am. A 29, 1606 (2012). DOI: 10.1364/JOSAA.29.001606.
"""
import numpy as np

from ptypy.engines import register
from ptypy.core import geometry
from ptypy.utils import Param
from ptypy.utils.verbose import logger
from ptypy import io
from ptypy.engines.stochastic import EPIEMixin
from ptypy.accelerate.base.engines.stochastic import _StochasticEngineSerial
from ptypy.accelerate.base.kernels import ThreePIEWaveKernel
from ptypy.accelerate.base import array_utils as au

__all__ = ["ThreePIE_serial", "normalize_slice_pad"]


def normalize_slice_pad(value, shape, resolution, energy, slice_thickness):
    """
    Normalize the ThreePIE slice padding option to a positive integer.

    ``"auto"`` chooses the smallest pad factor that satisfies the angular
    spectrum sampling limit for the largest requested slice spacing, capped at
    four to keep memory growth bounded.
    """
    if value is None:
        return 1
    if isinstance(value, str):
        if value.lower() != "auto":
            raise ValueError('slice_pad must be a positive integer or "auto"')
        if isinstance(slice_thickness, (list, tuple)):
            distance = max(abs(float(d)) for d in slice_thickness)
        else:
            distance = abs(float(slice_thickness))
        n = int(min(shape[-2:]))
        dx = float(np.mean(resolution))
        wavelength = geometry.Geo._keV2m / float(energy)
        ratio = distance / (n * dx * dx / wavelength)
        return min(max(1, int(np.ceil(ratio))), 4)
    pad = int(value)
    if pad < 1:
        raise ValueError("slice_pad must be a positive integer")
    return pad


def slice_bandlimit(shape, resolution, energy, distance):
    """Angular-spectrum support mask for alias-free multislice propagation."""
    nrows, ncols = shape[-2:]
    dy, dx = resolution
    wavelength = geometry.Geo._keV2m / float(energy)
    distance = abs(float(distance))
    if distance == 0.0:
        return np.ones((nrows, ncols), dtype=np.complex64)
    vlim_y = 1.0 / np.sqrt((2.0 * distance / (nrows * dy)) ** 2 + 1.0)
    vlim_x = 1.0 / np.sqrt((2.0 * distance / (ncols * dx)) ** 2 + 1.0)
    y = ((np.arange(nrows) + nrows // 2) % nrows) - nrows // 2
    x = ((np.arange(ncols) + ncols // 2) % ncols) - ncols // 2
    vy = y * (wavelength / (nrows * dy))
    vx = x * (wavelength / (ncols * dx))
    VY, VX = np.meshgrid(vy, vx, indexing="ij")
    keep = (np.abs(VY) <= vlim_y) & (np.abs(VX) <= vlim_x)
    return keep.astype(np.complex64)


def crop_pad_last2(array, target_shape):
    """Centered crop/pad on the last two axes."""
    target_shape = tuple(int(v) for v in target_shape)
    out = np.zeros(array.shape[:-2] + target_shape, dtype=array.dtype)
    src_slices = []
    dst_slices = []
    for src_n, dst_n in zip(array.shape[-2:], target_shape):
        n = min(src_n, dst_n)
        src0 = (src_n - n) // 2
        dst0 = (dst_n - n) // 2
        src_slices.append(slice(src0, src0 + n))
        dst_slices.append(slice(dst0, dst0 + n))
    out[(...,) + tuple(dst_slices)] = array[(...,) + tuple(src_slices)]
    return out


class _PaddedSlicePROP:
    """NumPy slice propagator with optional centered zero-padding."""

    def __init__(self, propagator, shape, pad=1):
        self.propagator = propagator
        self._pad_factor = int(pad)
        self._shape = tuple(int(v) for v in shape)
        self._padded_shape = tuple(int(v) * self._pad_factor for v in self._shape)

    def _run(self, wave, direction):
        if self._pad_factor == 1:
            return direction(wave)
        padded = crop_pad_last2(wave, self._padded_shape)
        propagated = direction(padded)
        return crop_pad_last2(propagated, self._shape)

    def fw(self, wave):
        return self._run(wave, self.propagator.fw)

    def bw(self, wave):
        return self._run(wave, self.propagator.bw)


@register()
class ThreePIE_serial(_StochasticEngineSerial, EPIEMixin):
    """
    A serialized (NumPy) implementation of multislice ePIE / 3PIE.

    Defaults:

    [name]
    default = ThreePIE_serial
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
    doc = A single float value or a list of values (length number_of_slices-1).

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

    """

    def __init__(self, ptycho_parent, pars=None):
        _StochasticEngineSerial.__init__(self, ptycho_parent, pars)
        EPIEMixin.__init__(self, self.p.alpha, self.p.beta)
        self.article = dict(
            title='Ptychographic transmission microscopy in three dimensions '
                  'using a multi-slice approach',
            author='A. M. Maiden et al.',
            journal='J. Opt. Soc. Am. A',
            volume=29,
            year=2012,
            page=1606,
            doi='10.1364/JOSAA.29.001606',
            comment='The 3PIE reconstruction algorithm',
        )
        ptycho_parent.citations.add_article(**self.article)

    # ------------------------------------------------------------------ setup
    def engine_initialize(self):
        super().engine_initialize()

        nslices = self.p.number_of_slices
        if nslices < 1:
            raise ValueError("number_of_slices must be at least 1")

        # one object / probe container per slice (probe[0] is the
        # illumination, probe[s>0] hold the propagated incident waves)
        self._object = [None] * nslices
        self._probe = [None] * nslices
        for i in range(nslices):
            self._object[i] = self.ob.copy(self.ob.ID + "_o_" + str(i))
            self._probe[i] = self.pr.copy(self.pr.ID + "_p_" + str(i))

        if isinstance(self.p.slice_start_iteration, int):
            self.p.slice_start_iteration = (
                np.ones(nslices, dtype=np.int32) * self.p.slice_start_iteration)
        elif len(self.p.slice_start_iteration) != nslices:
            raise ValueError(
                "slice_start_iteration must have one value per slice")

    def _setup_kernels(self):
        super()._setup_kernels()
        self._setup_slice_propagators()

    def _setup_slice_propagators(self):
        nslices = self.p.number_of_slices
        if isinstance(self.p.slice_thickness, (list, tuple)):
            if len(self.p.slice_thickness) != nslices - 1:
                raise ValueError(
                    "slice_thickness must contain number_of_slices - 1 values")
            thicknesses = list(self.p.slice_thickness)
        else:
            thicknesses = [self.p.slice_thickness] * (nslices - 1)

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
            kern.slice_FW = []
            kern.slice_BW = []
            for thickness in thicknesses:
                g.distance = thickness
                G = geometry.Geo(owner=None, pars=g)
                if self.p.slice_bandlimit:
                    support = slice_bandlimit(
                        G.propagator.kernel.shape, geo.resolution, geo.energy,
                        thickness)
                    G.propagator.kernel *= support
                    G.propagator.ikernel *= support
                prop = _PaddedSlicePROP(G.propagator, geo.shape, pad=pad)
                kern.slice_FW.append(prop.fw)
                kern.slice_BW.append(prop.bw)

            kern.slice_exits = [np.zeros_like(kern.aux) for _ in range(nslices)]
            kern.slice_tmp = np.zeros_like(kern.aux)
            kern.slice_back = np.zeros_like(kern.aux)
            kern.TWK = ThreePIEWaveKernel()
            kern.TWK.allocate()

    # --------------------------------------------------------------- helpers
    def _slice_active(self, index):
        return self.curiter >= self.p.slice_start_iteration[index]

    def _sync_primary_arrays(self, oID, pID):
        """Product object + entrance probe, for live plotting / output."""
        ob = self.ob.S[oID].data
        ob[:] = self._object[0].S[oID].data
        for s in range(1, self.p.number_of_slices):
            ob[:] *= self._object[s].S[oID].data
        self.pr.S[pID].data[:] = self._probe[0].S[pID].data

    # ---------------------------------------------------------------- iterate
    def engine_iterate(self, num=1):
        """Compute one multislice ePIE iteration on the CPU (serialized)."""
        nslices = self.p.number_of_slices

        for it in range(num):
            error_dct = {}

            for dID in self.di.S.keys():
                prep = self.diff_info[dID]
                pID, oID, eID = prep.poe_IDs

                kern = self.kernels[prep.label]
                FUK = kern.FUK
                AWK = kern.AWK
                POK = kern.POK
                TWK = kern.TWK
                FW = kern.FW
                BW = kern.BW
                aux = kern.aux

                ob_layers = [self._object[s].S[oID].data for s in range(nslices)]
                pr_layers = [self._probe[s].S[pID].data for s in range(nslices)]

                vieworder = prep.vieworder
                prep.rng.shuffle(vieworder)

                for i in vieworder:
                    addr = prep.addr[i, None]
                    ex_from, ex_to = prep.addr_ex[i]
                    ex = prep.ex[ex_from:ex_to]
                    mag = prep.mag[i, None]
                    ma = prep.ma[i, None]
                    ma_sum = prep.ma_sum[i, None]
                    obn = prep.obn
                    prn = prep.prn
                    err_phot = prep.err_phot[i, None]
                    err_fourier = prep.err_fourier[i, None]
                    err_exit = prep.err_exit[i, None]

                    self.position_update_local(prep, i)

                    # ---- forward multislice sweep --------------------------
                    for s in range(nslices):
                        old_exit = kern.slice_exits[s]
                        if self._slice_active(s):
                            # exit wave Psi_s = O_s * P_s
                            AWK.build_aux_no_ex(old_exit, addr,
                                                ob_layers[s], pr_layers[s])
                        else:
                            # slice inactive: pass the probe straight through
                            TWK.pr_to_aux(old_exit, pr_layers[s], addr)

                        if s < nslices - 1:
                            # near-field propagate to the next slice:
                            # P_{s+1} = F_dz{ Psi_s }
                            kern.slice_tmp[:] = kern.slice_FW[s](old_exit)
                            TWK.aux_to_pr(pr_layers[s + 1], kern.slice_tmp, addr)

                    # ---- last slice: far-field Fourier constraint ----------
                    ex[:] = kern.slice_exits[-1][:ex.shape[0]]
                    AWK.make_aux(aux, addr, ob_layers[-1], pr_layers[-1], ex,
                                 c_po=self._c, c_e=1 - self._c)

                    aux[:] = FW(aux)
                    if self.p.compute_fourier_error:
                        FUK.fourier_error(aux, addr, mag, ma, ma_sum)
                        FUK.error_reduce(addr, err_fourier)
                    else:
                        FUK.fourier_deviation(aux, addr, mag)
                    FUK.fmag_update_nopbound(aux, addr, mag, ma)
                    aux[:] = BW(aux)

                    AWK.make_exit(aux, addr, ob_layers[-1], pr_layers[-1], ex,
                                  c_a=self._b, c_po=self._a,
                                  c_e=-(self._a + self._b))
                    if self.p.compute_exit_error:
                        FUK.exit_error(aux, addr)
                        FUK.error_reduce(addr, err_exit)

                    if self.p.compute_log_likelihood:
                        AWK.build_aux_no_ex(aux, addr, ob_layers[-1],
                                            pr_layers[-1])
                        aux[:] = FW(aux)
                        FUK.log_likelihood(aux, addr, mag, ma, err_phot)

                    # ---- backward sweep (update O_s, P_s) ------------------
                    back_wave = ex
                    for s in range(nslices - 1, -1, -1):
                        if s < nslices - 1:
                            # new exit estimate Psi'_s = F_dz^{-1}{ P_{s+1} }
                            TWK.pr_to_aux(kern.slice_tmp, pr_layers[s + 1], addr)
                            kern.slice_back[:] = kern.slice_BW[s](kern.slice_tmp)
                            back_wave = kern.slice_back

                        if self._slice_active(s):
                            POK.pr_norm_local(addr, pr_layers[s], prn)
                            POK.ob_update_local(
                                addr, ob_layers[s], pr_layers[s], back_wave,
                                kern.slice_exits[s], prn,
                                a=self._ob_a, b=self._ob_b)

                            if self._object_norm_is_global and self._pr_a == 0:
                                obn_max = au.max_abs2(ob_layers[s])
                                obn[:] = 0
                            else:
                                POK.ob_norm_local(addr, ob_layers[s], obn)
                                obn_max = obn.max()
                            if self.p.probe_update_start <= self.curiter:
                                POK.pr_update_local(
                                    addr, pr_layers[s], ob_layers[s], back_wave,
                                    kern.slice_exits[s], obn, obn_max,
                                    a=self._pr_a, b=self._pr_b)
                        else:
                            TWK.aux_to_pr(pr_layers[s], back_wave, addr)

                    self._sync_primary_arrays(oID, pID)

                # accumulate per-view errors (matches base serial engine)
                errs = np.ascontiguousarray(
                    np.vstack([np.hstack(prep.err_fourier),
                               np.hstack(prep.err_phot),
                               np.hstack(prep.err_exit)]).T)
                error_dct.update(zip(prep.view_IDs, errs))

            self.curiter += 1

        return error_dct

    # --------------------------------------------------------------- finalize
    def engine_finalize(self):
        # Build the product object into the primary container for output.
        for oID in self.ob.S.keys():
            self.ob.S[oID].data[:] = self._object[0].S[oID].data
            for s in range(1, self.p.number_of_slices):
                self.ob.S[oID].data[:] *= self._object[s].S[oID].data

        # Save the per-slice objects.
        slices_info = Param()
        slices_info.number_of_slices = self.p.number_of_slices
        slices_info.slice_thickness = self.p.slice_thickness
        slices_info.slice_start_iteration = self.p.slice_start_iteration
        slices_info.objects = {
            ob.ID: {ID: S._to_dict() for ID, S in ob.storages.items()}
            for ob in self._object}

        header = {'description': 'multi-slices result details.'}
        h5opt = io.h5options['UNSUPPORTED']
        io.h5options['UNSUPPORTED'] = 'ignore'
        logger.info(f'Saving to {self.p.fslices}')
        io.h5write(self.p.fslices, header=header, content=slices_info)
        io.h5options['UNSUPPORTED'] = h5opt

        return super().engine_finalize()
