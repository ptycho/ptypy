# -*- coding: utf-8 -*-
"""
Maximum Likelihood reconstruction engine (refractive index).

TODO.

  * Implement other regularizers

This file is part of the PTYPY package.

    :copyright: Copyright 2014 by the PTYPY team, see AUTHORS.
    :license: see LICENSE for details.
"""
import numpy as np

from ptypy import utils as u
from ptypy.utils.verbose import logger
from ptypy.utils import parallel
from ptypy.engines import register
from ptypy.engines.ML import ML, BaseModel
from ptypy.core.manager import Full, Vanilla, Bragg3dModel, BlockVanilla, BlockFull, GradFull, BlockGradFull


__all__ = ['ML_ref']


@register()
class ML_ref(ML):
    """
    Maximum likelihood reconstruction engine (refractive index).


    Defaults:

    [name]
    default = ML_ref
    type = str
    help =
    doc =

    [ML_type]
    default = 'gaussian'
    type = str
    help = Likelihood model
    choices = ['gaussian','poisson','euclid']
    doc = One of ‘gaussian’, poisson’ or ‘euclid’. Only 'gaussian' is implemented.

    [floating_intensities]
    default = False
    type = bool
    help = Adaptive diffraction pattern rescaling
    doc = If True, allow for adaptative rescaling of the diffraction pattern intensities (to correct for incident beam intensity fluctuations).

    [intensity_renormalization]
    default = 1.
    type = float
    lowlim = 0.0
    help = Rescales the intensities so they can be interpreted as Poisson counts.

    [reg_del2]
    default = False
    type = bool
    help = Whether to use a Gaussian prior (smoothing) regularizer

    [reg_del2_amplitude]
    default = .01
    type = float
    lowlim = 0.0
    help = Amplitude of the Gaussian prior if used

    [smooth_gradient]
    default = 0.0
    type = float
    help = Smoothing preconditioner
    doc = Sigma for gaussian filter (turned off if 0.)

    [smooth_gradient_decay]
    default = 0.
    type = float
    help = Decay rate for smoothing preconditioner
    doc = Sigma for gaussian filter will reduce exponentially at this rate

    [scale_precond]
    default = False
    type = bool
    help = Whether to use the object/probe scaling preconditioner
    doc = This parameter can give faster convergence for weakly scattering samples.

    [scale_probe_object]
    default = 1.
    type = float
    lowlim = 0.0
    help = Relative scale of probe to object

    [probe_update_start]
    default = 2
    type = int
    lowlim = 0
    help = Number of iterations before probe update starts

    [poly_line_coeffs]
    default = quadratic
    type = str
    help = How many coefficients to be used in the the linesearch
    doc = choose between the 'quadratic' approximation (default) or 'all'

    [wavefield_precond]
    default = False
    type = bool
    help = Whether to use the wavefield preconditioner
    doc = This parameter can give faster convergence.

    [wavefield_delta_object]
    default = 0.1
    type = float
    help = Wavefield preconditioner damping constant for the object.

    [wavefield_delta_probe]
    default = 0.1
    type = float
    help = Wavefield preconditioner damping constant for the probe.

    """

    SUPPORTED_MODELS = [Full, Vanilla, Bragg3dModel, BlockVanilla, BlockFull, GradFull, BlockGradFull]

    def __init__(self, ptycho_parent, pars=None):
        """
        Maximum likelihood reconstruction engine (refractive index).
        """
        super().__init__(ptycho_parent, pars)

    def _initialize_model(self):

        # Create noise model
        if self.p.ML_type.lower() == "gaussian":
            self.ML_model = GaussianModel(self)
        # elif self.p.ML_type.lower() == "poisson":
        #     self.ML_model = PoissonModel(self)
        # elif self.p.ML_type.lower() == "euclid":
        #     self.ML_model = EuclidModel(self)
        else:
            raise RuntimeError("Unsupported ML_type: '%s'" % self.p.ML_type)

class GaussianModel(BaseModel):
    """
    Gaussian noise model.
    TODO: feed actual statistical weights instead of using the Poisson statistic heuristic.
    """

    def __init__(self, MLengine):
        """
        Core functions for ML computation using a Gaussian model.
        """
        BaseModel.__init__(self, MLengine)

        # Gaussian model requires weights
        # TODO: update this part of the code once actual weights are passed in the PODs
        self.weights = self.engine.di.copy(self.engine.di.ID + '_weights')
        # FIXME: This part needs to be updated once statistical weights are properly
        # supported in the data preparation.
        for name, di_view in self.di.views.items():
            if not di_view.active:
                continue
            self.weights[di_view] = (self.Irenorm * di_view.pod.ma_view.data
                                     / (1./self.Irenorm + di_view.data))

    def __del__(self):
        """
        Clean up routine
        """
        BaseModel.__del__(self)
        del self.engine.ptycho.containers[self.weights.ID]
        del self.weights

    def new_grad(self):
        """
        Compute a new gradient direction according to a Gaussian noise model.

        Note: The negative log-likelihood and local errors are also computed
        here.
        """
        self.ob_grad.fill(0.)
        self.pr_grad.fill(0.)
        if self.p.wavefield_precond:
            self.ob_fln.fill(0.)
            self.pr_fln.fill(0.)

        # We need an array for MPI
        LL = np.array([0.])
        error_dct = {}

        # Outer loop: through diffraction patterns
        for dname, diff_view in self.di.views.items():
            if not diff_view.active:
                continue

            # Weights and intensities for this view
            w = self.weights[diff_view]
            I = diff_view.data

            Imodel = np.zeros_like(I)
            f = {}

            # First pod loop: compute total intensity
            for name, pod in diff_view.pods.items():
                if not pod.active:
                    continue
                f[name] = pod.fw(pod.probe *  np.exp(1j * pod.object))
                Imodel += pod.downsample(u.abs2(f[name]))

            # Floating intensity option
            if self.p.floating_intensities:
                self.float_intens_coeff[dname] = ((w * Imodel * I).sum()
                                                / (w * Imodel**2).sum())
                Imodel *= self.float_intens_coeff[dname]

            DI = np.double(Imodel) - I

            # Second pod loop: gradients computation
            LLL = np.sum((w * DI**2).astype(np.float64))
            for name, pod in diff_view.pods.items():
                if not pod.active:
                    continue
                xi = pod.bw(pod.upsample(w*DI) * f[name])
                self.ob_grad[pod.ob_view] += 2. * xi * (1j * np.exp(1j * pod.object) * pod.probe).conj()
                self.pr_grad[pod.pr_view] += 2. * xi * np.exp(1j * pod.object).conj()

                # Compute fluence maps for object and probe
                if self.p.wavefield_precond:
                    self.ob_fln[pod.ob_view] += u.abs2(pod.probe)
                    self.pr_fln[pod.pr_view] += u.abs2(pod.object)

            diff_view.error = LLL
            error_dct[dname] = np.array([0, LLL / np.prod(DI.shape), 0])
            LL += LLL

        # MPI reduction of gradients
        self.ob_grad.allreduce()
        self.pr_grad.allreduce()
        if self.p.wavefield_precond:
            self.ob_fln.allreduce()
            self.pr_fln.allreduce()
        parallel.allreduce(LL)

        # Object regularizer
        if self.regularizer:
            for name, s in self.ob.storages.items():
                self.ob_grad.storages[name].data += self.regularizer.grad(
                    s.data)
                LL += self.regularizer.LL
        self.LL = LL / self.tot_measpts

        return error_dct

    def poly_line_coeffs(self, ob_h, pr_h):
        """
        Compute the coefficients of the polynomial for line minimization
        in direction h
        """

        B = np.zeros((3,), dtype=np.longdouble)
        Brenorm = 1. / self.LL[0]**2

        # Outer loop: through diffraction patterns
        for dname, diff_view in self.di.views.items():
            if not diff_view.active:
                continue

            # Weights and intensities for this view
            w = self.weights[diff_view]
            I = diff_view.data

            A0 = None
            A1 = None
            A2 = None

            for name, pod in diff_view.pods.items():
                if not pod.active:
                    continue
                f = pod.fw(pod.probe * np.exp(1j * pod.object))
                a = pod.fw(pod.probe * ob_h[pod.ob_view]
                           + pr_h[pod.pr_view] * np.exp(1j * pod.object))
                b = pod.fw(pr_h[pod.pr_view] * ob_h[pod.ob_view])

                if A0 is None:
                    A0 = u.abs2(f).astype(np.longdouble)
                    A1 = 2 * np.real(f * a.conj()).astype(np.longdouble)
                    A2 = (2 * np.real(f * b.conj()).astype(np.longdouble)
                          + u.abs2(a).astype(np.longdouble))
                else:
                    A0 += u.abs2(f)
                    A1 += 2 * np.real(f * a.conj())
                    A2 += 2 * np.real(f * b.conj()) + u.abs2(a)

            if self.p.floating_intensities:
                A0 *= self.float_intens_coeff[dname]
                A1 *= self.float_intens_coeff[dname]
                A2 *= self.float_intens_coeff[dname]

            A0 = np.double(A0) - pod.upsample(I)
            #A0 -= pod.upsample(I)
            w = pod.upsample(w)

            B[0] += np.dot(w.flat, (A0**2).flat) * Brenorm
            B[1] += np.dot(w.flat, (2*A0*A1).flat) * Brenorm
            B[2] += np.dot(w.flat, (A1**2 + 2*A0*A2).flat) * Brenorm

        parallel.allreduce(B)

        # Object regularizer
        if self.regularizer:
            for name, s in self.ob.storages.items():
                B += Brenorm * self.regularizer.poly_line_coeffs(
                    ob_h.storages[name].data, s.data)

        if np.isinf(B).any() or np.isnan(B).any():
            logger.warning(
                'Warning! inf or nan found! Trying to continue...')
            B[np.isinf(B)] = 0.
            B[np.isnan(B)] = 0.

        self.B = B

        return B

    def poly_line_all_coeffs(self, ob_h, pr_h):
            """
            Compute all the coefficients of the polynomial for line minimization
            in direction h
            """

            B = np.zeros((9,), dtype=np.longdouble)
            Brenorm = 1. / self.LL[0]**2

            # Outer loop: through diffraction patterns
            for dname, diff_view in self.di.views.items():
                if not diff_view.active:
                    continue

                # Weights and intensities for this view
                w = self.weights[diff_view]
                I = diff_view.data

                A0 = None
                A1 = None
                A2 = None
                A3 = None
                A4 = None

                for name, pod in diff_view.pods.items():
                    if not pod.active:
                        continue
                    f = pod.fw(pod.probe * np.exp(1j * pod.object))
                    a = pod.fw(pod.probe * ob_h[pod.ob_view]
                            + pr_h[pod.pr_view] * np.exp(1j * pod.object))
                    b = pod.fw(pr_h[pod.pr_view] * ob_h[pod.ob_view])

                    if A0 is None:
                        A0 = u.abs2(f).astype(np.longdouble)
                        A1 = 2 * np.real(f * a.conj()).astype(np.longdouble)
                        A2 = (2 * np.real(f * b.conj()).astype(np.longdouble)
                            + u.abs2(a).astype(np.longdouble))
                        A3 = 2 * np.real(a * b.conj()).astype(np.longdouble)
                        A4 = u.abs2(b).astype(np.longdouble)
                    else:
                        A0 += u.abs2(f)
                        A1 += 2 * np.real(f * a.conj())
                        A2 += 2 * np.real(f * b.conj()) + u.abs2(a)
                        A3 += 2 * np.real(a * b.conj())
                        A4 += u.abs2(b)

                if self.p.floating_intensities:
                    A0 *= self.float_intens_coeff[dname]
                    A1 *= self.float_intens_coeff[dname]
                    A2 *= self.float_intens_coeff[dname]
                    A3 *= self.float_intens_coeff[dname]
                    A4 *= self.float_intens_coeff[dname]

                A0 = np.double(A0) - pod.upsample(I)
                #A0 -= pod.upsample(I)
                w = pod.upsample(w)

                B[0] += np.dot(w.flat, (A0**2).flat) * Brenorm
                B[1] += np.dot(w.flat, (2*A0*A1).flat) * Brenorm
                B[2] += np.dot(w.flat, (A1**2 + 2*A0*A2).flat) * Brenorm
                B[3] += np.dot(w.flat, (2*A0*A3 + 2*A1*A2).flat) * Brenorm
                B[4] += np.dot(w.flat, (A2**2 + 2*A1*A3 + 2*A0*A4).flat) * Brenorm
                B[5] += np.dot(w.flat, (2*A1*A4 + 2*A2*A3).flat) * Brenorm
                B[6] += np.dot(w.flat, (A3**2 + 2*A2*A4).flat) * Brenorm
                B[7] += np.dot(w.flat, (2*A3*A4).flat) * Brenorm
                B[8] += np.dot(w.flat, (A4**2).flat) * Brenorm

            parallel.allreduce(B)

            # Object regularizer
            if self.regularizer:
                for name, s in self.ob.storages.items():
                    B[:3] += Brenorm * self.regularizer.poly_line_coeffs(
                        ob_h.storages[name].data, s.data)

            if np.isinf(B).any() or np.isnan(B).any():
                logger.warning(
                    'Warning! inf or nan found! Trying to continue...')
                B[np.isinf(B)] = 0.
                B[np.isnan(B)] = 0.

            self.B = B

            return B
