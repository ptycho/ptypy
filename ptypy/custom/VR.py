"""
Variance-reduced stochastic maximum likelihood reconstruction engine.

Minimises the Poisson negative log-likelihood with NAdam on randomly drawn
mini-batches of diffraction patterns, using a hand-derived gradient and a
learning rate that is reduced on plateau.

This file is part of the PTYPY package.

    :copyright: Copyright 2014 by the PTYPY team, see AUTHORS.
    :license: see LICENSE for details.
"""
import numpy as np

from ptypy import utils as u
from ptypy.utils.verbose import logger
from ptypy.utils import parallel
from ptypy.engines import register
from ptypy.engines.base import PositionCorrectionEngine
from ptypy.core.manager import Full, Vanilla, BlockVanilla, BlockFull
from ptypy.accelerate.base.engines.projectional_serial import serialize_array_access

try:
    import torch
except ImportError:
    torch = None

import math

__all__ = ['VR']

EPS = 1e-8  # torch.nn.PoissonNLLLoss default eps


def _intensity(ft):
    """
    Detector intensity (B, Py, Px): |ft|^2 summed over the probe modes.
    """
    return torch.view_as_real(ft).square().sum(-1).sum(1)


def _dloss(I, T, w, const):
    """
    Per-pixel dL/dI and the per-frame loss, with L = w * (I - T log(I+eps) + const).
    """
    denom = I + EPS
    g = w * (1.0 - T / denom)
    per_frame = (w * (I - T * torch.log(denom) + const)).flatten(1).sum(1)
    return g, per_frame


def _shift_factors(frac, fy, fx, dtype):
    """
    The two separable factors (B, Py, 1) and (B, 1, Px) of the sub-pixel
    Fourier-shift ramp exp(i 2pi (dy fy + dx fx)); `frac` is (B, 2) in pixels.
    """
    ry = torch.exp((2j * math.pi) * frac[:, 0, None, None] * fy).to(dtype)
    rx = torch.exp((2j * math.pi) * frac[:, 1, None, None] * fx).to(dtype)
    return ry, rx


def _shift(patch, ramps, adjoint=False):
    """
    Shift object patches by the sub-pixel ramps, or apply the adjoint.
    """
    ry, rx = ramps
    if adjoint:
        ry, rx = ry.conj(), rx.conj()
    spectrum = torch.fft.fft2(patch)
    return torch.fft.ifft2(spectrum.mul_(ry).mul_(rx))


def _scatter_add_rows(dst, index, src):
    """
    dst[index] += src (rows), accumulating repeated indices. Atomics on CUDA,
    or the deterministic sort-based path if deterministic algorithms are on.
    """
    if torch.are_deterministic_algorithms_enabled():
        dst.index_put_((index,), src, accumulate=True)
    else:
        dst.index_add_(0, index, src)


def _poisson_loss_grad(fm, idx, grads):
    """
    Masked-mean Poisson NLL of the mini-batch `idx` and its gradient.

    Hand-derived Wirtinger gradient of

        I_b = sum_m | FFT( S_b[O] * P_m * pre * s_bm ) * post |^2

    where S_b[O] is the (sub-pixel shifted) object patch, P_m the probe modes
    and s_bm the per-frame probe scaling. The gradients of obj, probe and
    scale are written into `grads` (each fully overwritten, so no zeroing is
    needed between calls) for the optimiser to step on. torch's complex
    gradient is dL/dRe + i dL/dIm = 2 dL/d(conj z), so the adjoint of a linear
    map is applied directly: N * ifftn for the unnormalised fftn, and
    ifft2 . diag(conj ramp) . fft2 for the sub-pixel shift.

    Returns (loss, per_frame): the masked-mean Poisson NLL with the Stirling
    constant (as PoissonNLLLoss(full=True)), and the (B,) per-frame
    contributions, which sum to the loss.
    """
    B = idx.numel()
    M, py, px = fm.probe.shape
    ox = fm.obj.shape[1]
    obj, probe, scale = fm.obj.detach(), fm.probe.detach(), fm.scale.detach()
    T = fm.data[idx]

    # ---- forward ------------------------------------------------------- #
    pos = fm.pos[idx].long()
    iy = pos[:, 0, None] + torch.arange(py, device=pos.device)
    ix = pos[:, 1, None] + torch.arange(px, device=pos.device)
    flat = iy[:, :, None] * ox + ix[:, None, :]                  # (B, Py, Px) flat object index
    patch = obj.reshape(-1)[flat.reshape(-1)].view(B, py, px)   # gather

    ramps = None
    if fm.use_shift:
        ramps = _shift_factors(fm.frac[idx], fm.patch_fy, fm.patch_fx, patch.dtype)
        patch = _shift(patch, ramps)

    s = scale[idx]                                               # (B, M, 1, 1)
    Q = (patch.unsqueeze(1) * probe.unsqueeze(0)).mul_(fm.pre)   # patch * P * pre
    ft = torch.fft.fftn(Q * s, dim=(-2, -1))
    ft.mul_(fm.post)
    I = _intensity(ft)                                           # (B, Py, Px)

    # ---- loss and dL/dI ------------------------------------------------ #
    w = fm.mask / (fm.mask.sum() * B)
    Tc = T.clamp_min(1.0)
    const = torch.where(T > 1, Tc * torch.log(Tc) - Tc + 0.5 * torch.log(2 * math.pi * Tc),
                        torch.zeros_like(T))
    g, per_frame = _dloss(I, T, w, const)
    del I, T, Tc, const

    # ---- backward ------------------------------------------------------ #
    ft.mul_(g.unsqueeze(1) * 2.0)                                # dL/d ft
    del g
    ft.mul_(fm.post.conj())
    gpsi = torch.fft.ifftn(ft, dim=(-2, -1)).mul_(py * px)       # adjoint of the unnormalised fftn
    del ft

    # psi = Q * s
    grads['scale'].zero_()
    gs = torch.einsum('bmyx,bmyx->bm', Q.conj(), gpsi)
    _scatter_add_rows(grads['scale'].view(-1, M), idx, gs)
    del gs, Q
    gpsi.mul_(s.conj())                                          # -> grad wrt Q
    gpsi.mul_(fm.pre.conj())                                     # -> grad wrt (patch * P)

    # resolve_conj: the operands are small (one unit / the probe), and some
    # backends' complex einsum ignores a lazy conjugate flag
    grads['probe'].copy_(torch.einsum('byx,bmyx->myx', patch.conj().resolve_conj(), gpsi))
    gpatch = torch.einsum('myx,bmyx->byx', probe.conj().resolve_conj(), gpsi)
    del gpsi, patch

    if ramps is not None:                                        # adjoint of the shift
        gpatch = _shift(gpatch, ramps, adjoint=True)

    gobj = torch.view_as_real(grads['obj'])                      # (Oy, Ox, 2)
    gobj.zero_()
    _scatter_add_rows(gobj.view(-1, 2), flat.reshape(-1), torch.view_as_real(gpatch).reshape(-1, 2))
    return per_frame.sum(), per_frame

@register()
class VR(PositionCorrectionEngine):
    """
    Variance-reduced mini-batch maximum likelihood reconstruction engine.

    Defaults:

    [name]
    default = VR
    type = str
    help =
    doc =

    [minimum_batch_size]
    default = None
    type = int, None
    lowlim = 1
    help = Number of diffraction patterns per mini-batch
    doc = Initial mini-batch size. If None, ``minimum_batch_fraction`` of the number of frames, rounded to the nearest power of two. It doubles at learning-rate decreases if ``batch_doubling_lr_fraction`` is set. This is dataset dependent.

    [minimum_batch_fraction]
    default = 0.041
    type = float
    lowlim = 0.0
    uplim = 1.0
    help = Mini-batch size as a fraction of the number of frames
    doc = Used only if ``minimum_batch_size`` is None. The batch size is this fraction of the frames rounded to the nearest power of two (on a logarithmic scale), and at least 1.

    [uniform_sampling]
    default = False
    type = bool
    help = Sample patterns uniformly
    doc = If False, patterns are drawn with replacement with probability proportional to their squared Frobenius norm. If True, each epoch is a uniform random permutation of the full dataset.

    [sampler_seed]
    default = None
    type = int, None
    help = Seed of the mini-batch sampler
    doc = Keeps the sampling stream independent of the global random number generator. If None, the global stream is used.

    [batches_per_iteration]
    default = None
    type = int, None
    lowlim = 1
    help = Mini-batch updates per iteration
    doc = If None, an iteration is one epoch, i.e. the patterns are visited once in ceil(N / batch size) updates. If set, an iteration is exactly that many updates (1 gives a single parameter update per iteration). The learning-rate schedule and batch doubling advance once per epoch either way.

    [lr]
    default = 1.0
    type = float
    lowlim = 0.0
    help = Initial NAdam learning rate
    doc = Tuned for a probe normalised to unit mean modulus, which the engine applies internally and undoes on finalisation.

    [momentum_decay]
    default = 0.05
    type = float
    lowlim = 0.0
    help = NAdam momentum decay

    [scheduler_factor]
    default = 0.95
    type = float
    lowlim = 0.0
    uplim = 1.0
    help = Learning-rate reduction factor on plateau
    doc = Dataset dependent.

    [scheduler_patience]
    default = 1
    type = int
    lowlim = 0
    help = Epochs without improvement before the learning rate is reduced

    [scheduler_cooldown]
    default = 1
    type = int
    lowlim = 0
    help = Epochs to wait after a reduction before monitoring again

    [scheduler_threshold]
    default = 0.45
    type = float
    lowlim = 0.0
    help = Relative improvement counted as progress

    [min_lr]
    default = 1e-12
    type = float
    lowlim = 0.0
    help = Lower bound of the learning rate

    [use_subpixel_shift]
    default = True
    type = bool
    help = Apply the sub-pixel part of each scan position
    doc = Shifts each object patch in Fourier space by the fractional remainder of its scan position.

    [batch_doubling_lr_fraction]
    default = None
    type = float, None
    lowlim = 0.0
    uplim = 1.0
    help = Double the batch once the learning rate falls to this fraction of its initial value
    doc = If None, the batch size stays fixed.

    [max_batch_doublings]
    default = 1
    type = int
    lowlim = 0
    help = Maximum number of batch doublings

    [reconstruction_device]
    default = None
    type = str, None
    help = Torch device for the reconstruction
    doc = If None, ``cuda:0`` is used when available, otherwise ``cpu``.

    """

    SUPPORTED_MODELS = [Full, Vanilla, BlockVanilla, BlockFull]

    def __init__(self, ptycho_parent, pars=None):
        """
        Variance-reduced maximum likelihood reconstruction engine.
        """
        super().__init__(ptycho_parent, pars)

        if torch is None:
            raise ImportError('The VR engine requires PyTorch.')

        # Dense tensors extracted from the ptypy containers
        self.forward_model = None
        self.optimiser = None
        self.scheduler = None
        self.device = None
        self._grads = None

        # Working variables
        self.batch_size = None
        self.n_batch_doublings = 0
        self._storage_slices = {}   # diffraction storage ID -> its frames in the concatenated dataset

        # Normalisation applied to the probe, undone in engine_finalize
        self.probe_norm = None

        # Sampling state, persistent across engine_iterate calls
        self._n_frames = None
        self._probabilities = None
        self._generator = None
        self._order = None          # patterns of the current epoch
        self._cursor = 0            # position in self._order
        self._epoch_loss = None     # running sum of mini-batch losses
        self._epoch_batches = 0
        self.epoch = 0              # completed epochs
        self._lr0 = None            # initial and previous learning rate
        self._lr = None

        # Per-pattern Poisson NLL, reported in ptypy's error dictionary
        self._view_names = None
        self._frame_error = None
        self._error_cpu = None

        self.ptycho.citations.add_article(
            title='Stochastic optimisation for scalable ptychographic image reconstruction',
            author='Muriithi A. W., Tanner J., Fowkes J. M. and Daurer B. J.',
            journal='IUCr (under review)',
            volume='',
            year=2026,
            comment='The variance-reduced (VR) stochastic maximum likelihood reconstruction algorithm',
        )

    def engine_initialize(self):
        """
        Check the supported configuration and set up the sampler.
        """
        if parallel.size > 1:
            raise NotImplementedError('The VR engine does not support MPI.')
        if len(self.ptycho.model.scans) != 1:
            raise NotImplementedError('The VR engine supports a single scan.')
        if len(self.ob.storages) != 1 or len(self.pr.storages) != 1:
            raise NotImplementedError('The VR engine supports a single object and probe storage.')

        if self.p.reconstruction_device is None:
            self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(self.p.reconstruction_device)

        if self.p.sampler_seed is not None:
            self._generator = torch.Generator(device=self.device)
            self._generator.manual_seed(int(self.p.sampler_seed))

        self.batch_size = self.p.minimum_batch_size

    def engine_prepare(self):
        """
        Extract the dense tensors the reconstruction runs on.

        The diffraction storages (blocks) of the single scan are concatenated
        into one resident dataset. The probe is normalised to unit mean
        modulus, which `engine_finalize` undoes.
        """
        if self.forward_model is not None:
            raise NotImplementedError('The VR engine does not support data arriving during a reconstruction.')
        if not self.di.storages:
            raise NotImplementedError('The VR engine needs the diffraction data before it starts.')

        dev = self.device
        c64 = torch.complex64
        pr = next(iter(self.pr.storages.values()))
        ob = next(iter(self.ob.storages.values()))
        n_modes, py, px = pr.data.shape
        if ob.data.shape[0] != 1:
            raise NotImplementedError('The VR engine supports a single object mode.')

        # Far-field propagation: fw(w) = post_fft * sc * fft(pre_fft * w). The
        # shift ramps are inside pre_fft/post_fft, so ptypy's diffraction data
        # stay DC-centred and no fftshift is needed.
        prop = next(iter(self.ptycho.model.scans.values())).geometries[0].propagator
        if getattr(prop, 'pre_fft', None) is None or (prop.crop_pad != 0).any():
            raise NotImplementedError('The VR engine supports far-field propagation without crop/pad only.')

        frames, positions, subpixel, masks = [], [], [], []
        self._view_names = []
        for d in self.di.storages.values():
            view_IDs, poe_IDs, addr = serialize_array_access(d)
            # addr[view, pod, (probe, object, exit, diff, mask), (layer, y, x)]
            if poe_IDs[0] != pr.ID or poe_IDs[1] != ob.ID:
                raise NotImplementedError('The VR engine supports a single probe and object storage.')
            if (addr.shape[1] != n_modes
                    or (np.sort(addr[:, :, 0, 0], axis=1) != np.arange(n_modes)).any()
                    or (addr[:, :, 1, 0] != 0).any()
                    or (addr[:, :, 0, 1:] != 0).any()
                    or d.data.shape[1:] != (py, px)):
                raise NotImplementedError('The VR engine needs every view to see all probe modes, '
                                          'one object mode and a probe the size of the frame.')

            views = [self.di.views[v] for v in view_IDs]
            start = sum(len(f) for f in frames)
            frames.append(d.data[[v.dlayer for v in views]])
            self._storage_slices[d.ID] = slice(start, start + len(frames[-1]))
            positions.append(addr[:, 0, 1, 1:])                           # (n, 2) integer (y, x)
            subpixel.append(np.array([v.pod.ob_view.sp for v in views]))  # true - rounded, in pixels
            self._view_names += view_IDs

            ma = self.ma.S[d.ID].data
            if (ma != ma[0]).any():
                raise NotImplementedError('The VR engine supports a single detector mask.')
            masks.append(ma[0])

        if any((m != masks[0]).any() for m in masks):
            raise NotImplementedError('The VR engine supports a single detector mask.')

        fm = u.Param()
        fm.data = torch.from_numpy(np.concatenate(frames).astype(np.float32)).to(dev)
        fm.pos = torch.from_numpy(np.concatenate(positions)).to(dev)
        fm.frac = torch.from_numpy(np.concatenate(subpixel).astype(np.float32)).to(dev)
        fm.mask = torch.from_numpy(masks[0].astype(np.float32)).to(dev)
        fm.use_shift = bool(self.p.use_subpixel_shift)
        fm.pre = torch.from_numpy(prop.pre_fft).to(dev, c64)
        fm.post = torch.from_numpy(prop.post_fft * prop.sc).to(dev, c64)
        fm.patch_fy = torch.fft.fftfreq(py, device=dev).view(1, -1, 1)
        fm.patch_fx = torch.fft.fftfreq(px, device=dev).view(1, 1, -1)
        self._n_frames = fm.data.shape[0]

        # Mini-batch size: given, or a fraction of the frames rounded to a power of two
        if self.p.minimum_batch_size is not None:
            self.batch_size = self.p.minimum_batch_size
        else:
            fraction = self.p.minimum_batch_fraction * self._n_frames
            self.batch_size = int(2 ** round(np.log2(max(fraction, 1.))))
        self.batch_size = min(self.batch_size, self._n_frames)

        # Parameters, in the units the learning rate is tuned for
        probe = torch.from_numpy(pr.data.copy()).to(dev, c64)
        self.probe_norm = probe.abs().mean().item()
        fm.probe = torch.nn.Parameter(probe / self.probe_norm)
        fm.obj = torch.nn.Parameter(torch.from_numpy(ob.data[0].copy()).to(dev, c64))
        fm.scale = torch.nn.Parameter(torch.ones((self._n_frames, n_modes, 1, 1), dtype=c64, device=dev))
        self.forward_model = fm

        # Persistent gradient buffers, overwritten by the gradient every update
        self._grads = {}
        for name in ('obj', 'probe', 'scale'):
            prm = getattr(fm, name)
            self._grads[name] = torch.zeros_like(prm)
            prm.grad = self._grads[name]

        self.optimiser = torch.optim.NAdam([fm.obj, fm.probe, fm.scale], lr=self.p.lr,
                                           momentum_decay=self.p.momentum_decay, eps=1e-10)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimiser, mode='min', factor=self.p.scheduler_factor,
            patience=self.p.scheduler_patience, min_lr=self.p.min_lr,
            cooldown=self.p.scheduler_cooldown, threshold=self.p.scheduler_threshold)
        self._lr0 = self._lr = self.optimiser.param_groups[0]['lr']

        # Sampling and reporting state
        norms = fm.data.flatten(1).norm(dim=1).square()
        self._probabilities = norms / norms.sum()
        self._order = None
        self._epoch_loss = torch.zeros((), device=dev)
        self._epoch_batches = 0
        self._frame_error = torch.zeros(self._n_frames, device=dev)
        self._error_cpu = np.zeros(self._n_frames)

        logger.info('VR: %d frames of %dx%d, %d probe mode(s), batch %d, device %s'
            % (self._n_frames, py, px, n_modes, self.batch_size, dev))

    def engine_iterate(self, num=1):
        """
        Compute `num` iterations.

        An iteration is `batches_per_iteration` mini-batch updates, or one
        epoch if that is None.
        """
        for it in range(num):
            if self.p.batches_per_iteration is None:
                epoch_done = False
                while not epoch_done:
                    epoch_done = self._step()
            else:
                for _ in range(self.p.batches_per_iteration):
                    self._step()

            self.curiter += 1

        # ptypy reads the storages for autosaves and plots, so they must be current
        self._write_back()
        return self._error_dict()

    def _step(self):
        """
        One mini-batch update. Returns True if it completed an epoch.
        """
        if self._order is None:
            self._order = self._sample_order()
            self._cursor = 0

        idx = self._order[self._cursor:self._cursor + self.batch_size]
        self._cursor += idx.numel()

        loss = self._batch_update(idx)
        self._epoch_loss += loss
        self._epoch_batches += 1

        if self._cursor < self._order.numel():
            return False

        self._end_epoch()
        return True

    def _sample_order(self):
        """
        Patterns of one epoch: a uniform permutation, or N draws with
        replacement weighted by squared norm.
        """
        n = self._n_frames
        if self.p.uniform_sampling:
            return torch.randperm(n, device=self.device, generator=self._generator)
        return torch.multinomial(self._probabilities, n, replacement=True,
                                 generator=self._generator)

    def _batch_update(self, idx):
        """
        Gradient, loss and NAdam step of the mini-batch `idx`.

        The per-frame Poisson NLL is stored for the error dictionary, scored at
        the estimate before this update. `w` in the gradient carries a 1/B, so
        the factor B makes each entry that frame's own masked-mean loss; their
        batch mean is exactly the returned loss.
        """
        with torch.no_grad():
            loss, per_frame = _poisson_loss_grad(self.forward_model, idx, self._grads)
            self._frame_error[idx] = per_frame * idx.numel()
        self.optimiser.step()
        return loss

    def _end_epoch(self):
        """
        Epoch bookkeeping: scheduler, batch doubling, error snapshot.
        """
        mean_loss = (self._epoch_loss / self._epoch_batches).item()
        self.scheduler.step(mean_loss)
        self.epoch += 1

        lr = self.optimiser.param_groups[0]['lr']
        frac = self.p.batch_doubling_lr_fraction
        lr_cut = lr < self._lr * (1. - 1e-9)
        if (frac is not None and lr_cut
                and self.n_batch_doublings < self.p.max_batch_doublings
                and lr <= frac * self._lr0 * (1. + 1e-9)):
            doubled = min(2 * self.batch_size, self._n_frames)
            if doubled != self.batch_size:
                self.batch_size = doubled
                self.n_batch_doublings += 1
        self._lr = lr

        # One GPU synchronisation per epoch, not per update
        self._error_cpu = self._frame_error.cpu().numpy()

        logger.info('Epoch %d: Poisson NLL %.4f, lr %.3g, batch %d'
                    % (self.epoch, mean_loss, lr, self.batch_size))

        self._order = None
        self._epoch_loss.zero_()
        self._epoch_batches = 0

    def _error_dict(self):
        """
        ptypy error dictionary: Poisson NLL per view as of the last completed epoch.
        """
        return {name: np.array([0., self._error_cpu[i], 0.])
                for i, name in enumerate(self._view_names)}

    def engine_finalize(self):
        """
        Write the final estimate back, record the residual per-frame probe
        scaling in the runtime and release the GPU memory.
        """
        if self.forward_model is None:
            return

        self._write_back()

        with torch.no_grad():
            scale = self.forward_model.scale[:, :, 0, 0]   # (N, M)
            residual = (scale / scale.mean(dim=0)).cpu().numpy()
        self.ptycho.runtime['vr_probe_scaling'] = {
            ID: residual[sl] for ID, sl in self._storage_slices.items()}

        self.forward_model = None
        self.optimiser = None
        self.scheduler = None
        self._grads = None
        self._probabilities = None
        self._frame_error = None
        self._epoch_loss = None
        self._order = None
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
