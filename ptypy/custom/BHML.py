# -*- coding: utf-8 -*-
"""
Bilinear Hessian Maximum Likelihood reconstruction engine.

TODO.

  * Implement other regularizers

This file is part of the PTYPY package.

    :copyright: Copyright 2025 by the PTYPY team, see AUTHORS.
    :license: see LICENSE for details.
"""
import numpy as np
import time

from ptypy import utils as u
from ptypy.utils.verbose import logger
from ptypy.utils import parallel
from ptypy.engines.utils import Cnorm2, Cdot
from ptypy.engines import register
from ptypy.engines.ML import ML, BaseModel, GaussianModel
from ptypy.core.manager import Full, Vanilla, Bragg3dModel, BlockVanilla, BlockFull, GradFull, BlockGradFull


__all__ = ['BHML']


@register()
class BHML(ML):
    """
    Bilinear Hessian Maximum likelihood reconstruction engine.


    Defaults:

    [name]
    default = BHML
    type = str
    help =
    doc =

    [ML_type]
    default = 'gaussian'
    type = str
    help = Likelihood model
    choices = ['gaussian','poisson','euclid']
    doc = One of ‘gaussian’, poisson’ or ‘euclid’.

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

    #FIXME: support the other noise models
    def _initialize_model(self):

        # Create noise model
        if self.p.ML_type.lower() == "gaussian":
            self.ML_model = GaussianModel(self)
        else:
            raise RuntimeError("Unsupported ML_type: '%s'" % self.p.ML_type)


    #FIXME: refactor to remove new_obj_grad and new_probe_grad as BHML does not need previous gradients
    def engine_iterate(self, num=1):
        """
        Compute `num` iterations.
        """
        ########################
        # Compute new gradient
        ########################
        tg = 0.
        tc = 0.
        ta = time.time()
        for it in range(num):
            t1 = time.time()
            error_dct = self.ML_model.new_grad()
            new_ob_grad, new_pr_grad = self.ob_grad_new, self.pr_grad_new
            tg += time.time() - t1

            if self.p.probe_update_start <= self.curiter:
                # Apply probe support if needed
                for name, s in new_pr_grad.storages.items():
                    self.support_constraint(s)
                    #support = self.probe_support.get(name)
                    #if support is not None:
                    #    s.data *= support
            else:
                new_pr_grad.fill(0.)

            # Wavefield preconditioner
            if self.p.wavefield_precond:
                for name, s in new_ob_grad.storages.items():
                    new_ob_grad.storages[name].data /= np.sqrt(self.ob_fln.storages[name].data + self.p.wavefield_delta_object)
                    new_pr_grad.storages[name].data /= np.sqrt(self.pr_fln.storages[name].data + self.p.wavefield_delta_probe)

            # Smoothing preconditioner
            if self.smooth_gradient:
                self.smooth_gradient.sigma *= (1. - self.p.smooth_gradient_decay)
                for name, s in new_ob_grad.storages.items():
                    s.data[:] = self.smooth_gradient(s.data)

            # probe/object rescaling
            if self.p.scale_precond:
                cn2_new_pr_grad = Cnorm2(new_pr_grad)
                cn2_new_ob_grad = Cnorm2(new_ob_grad)
                if cn2_new_pr_grad > 1e-5:
                    scale_p_o = (self.p.scale_probe_object * cn2_new_ob_grad
                                 / cn2_new_pr_grad)
                else:
                    scale_p_o = self.p.scale_probe_object
                if self.scale_p_o is None:
                    self.scale_p_o = scale_p_o
                else:
                    self.scale_p_o = self.scale_p_o ** self.scale_p_o_memory
                    self.scale_p_o *= scale_p_o ** (1-self.scale_p_o_memory)
                logger.debug('Scale P/O: %6.3g' % scale_p_o)
            else:
                self.scale_p_o = self.p.scale_probe_object

            ############################
            # Compute next conjugate
            ############################
            if self.curiter == 0:
                bt = 0.
            else: # NB: in the below need to do h/tmin as did h*tmin when taking steps (don't you just love containers?)
                bt = self.ML_model.compute_beta(self.ob_h / self.tmin, self.pr_h / self.tmin, self.scale_p_o)
                bt = max(0, bt)

                bt_num = (self.scale_p_o
                          * (Cnorm2(new_pr_grad)
                             - np.real(Cdot(new_pr_grad, self.pr_grad)))
                          + (Cnorm2(new_ob_grad)
                             - np.real(Cdot(new_ob_grad, self.ob_grad))))
                bt_denom = self.scale_p_o*Cnorm2(self.pr_grad) + Cnorm2(self.ob_grad)
                print("beta_old: %f" % (bt_num/bt_denom))

            # logger.info('Polak-Ribiere coefficient: %f ' % bt)

            self.ob_grad << new_ob_grad
            self.pr_grad << new_pr_grad

            dt = self.ptycho.FType

            # 3. Next conjugate
            self.ob_h *= bt / self.tmin
            # Wavefield preconditioner for the object (with and without smoothing preconditioner)
            if self.p.wavefield_precond:
                for name, s in self.ob_h.storages.items():
                    if self.smooth_gradient:
                        s.data[:] -= self.smooth_gradient(self.ob_grad.storages[name].data
                                      / np.sqrt(self.ob_fln.storages[name].data + self.p.wavefield_delta_object))
                    else:
                        s.data[:] -= (self.ob_grad.storages[name].data
                                      / np.sqrt(self.ob_fln.storages[name].data + self.p.wavefield_delta_object))
            else:
                # Smoothing preconditioner for the object
                if self.smooth_gradient:
                    for name, s in self.ob_h.storages.items():
                        s.data[:] -= self.smooth_gradient(self.ob_grad.storages[name].data)
                else:
                    self.ob_h -= self.ob_grad

            self.pr_h *= bt / self.tmin
            self.pr_grad *= self.scale_p_o
            # Wavefield preconditioner for the probe
            if self.p.wavefield_precond:
                for name, s in self.pr_h.storages.items():
                    s.data[:] -= (self.pr_grad.storages[name].data
                                  / np.sqrt(self.pr_fln.storages[name].data + self.p.wavefield_delta_probe))
            else:
                self.pr_h -= self.pr_grad

            # In principle, the way things are now programmed this part
            # could be iterated over in a real Newton-Raphson style.
            t2 = time.time()
            if self.p.poly_line_coeffs == "all":
                B = self.ML_model.poly_line_all_coeffs(self.ob_h, self.pr_h)
                diffB = np.arange(1,len(B))*B[1:] # coefficients of poly derivative
                roots = np.roots(np.flip(diffB.astype(np.double))) # roots only supports double
                real_roots = np.real(roots[np.isreal(roots)]) # not interested in complex roots
                if real_roots.size == 1: # single real root
                    self.tmin = dt(real_roots[0])
                else: # find real root with smallest poly objective
                    evalp = lambda root: np.polyval(np.flip(B),root)
                    self.tmin = dt(min(real_roots, key=evalp)) # root with smallest poly objective
            elif self.p.poly_line_coeffs == "quadratic":
                B = self.ML_model.poly_line_coeffs(self.ob_h, self.pr_h)
                # same as above but quicker when poly quadratic
                self.tmin = dt(-0.5 * B[1] / B[2])
            else:
                raise NotImplementedError("poly_line_coeffs should be 'quadratic' or 'all'")

            tc += time.time() - t2

            self.ob_h *= self.tmin
            self.pr_h *= self.tmin
            self.ob += self.ob_h
            self.pr += self.pr_h
            # Newton-Raphson loop would end here

            # Position correction
            self.position_update()

            # Allow for customized modifications at the end of each iteration
            self._post_iterate_update()

            # increase iteration counter
            self.curiter +=1

        logger.info('Time spent in gradient calculation: %.2f' % tg)
        logger.info('  ....  in coefficient calculation: %.2f' % tc)
        return error_dct  # np.array([[self.ML_model.LL[0]] * 3])

class BaseModel(BaseModel):

    def compute_beta(self, ob_h, pr_h, scale_p_o):
        """
        Compute CG beta parameter using the bilinear Hessian
        """
        raise NotImplementedError

class GaussianModel(GaussianModel):

    def compute_beta(self, ob_h, pr_h, scale_p_o):
        """
        Compute CG beta parameter using the bilinear Hessian
        """

        # We need arrays for MPI
        beta_num = np.array([0.])
        beta_denom = np.array([0.])

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
                f[name] = pod.fw(pod.probe * pod.object)
                Imodel += pod.downsample(u.abs2(f[name]))

            # Floating intensity option
            if self.p.floating_intensities:
                self.float_intens_coeff[dname] = ((w * Imodel * I).sum()
                                                / (w * Imodel**2).sum())
                Imodel *= self.float_intens_coeff[dname]

            DI = np.double(Imodel) - I

            A_num = None
            A_denom = None
            v1_num = None
            v2_num = None
            v1_denom = None
            v2_denom = None

            # Second pod loop: beta computation
            for name, pod in diff_view.pods.items():
                if not pod.active:
                    continue
                xi = pod.bw(pod.upsample(w*DI) * f[name])
                I0 = I / (np.abs(f[name]) + 1e-6)
                f0 = f[name] / (np.abs(f[name]) + 1e-6)

                d2mt = (scale_p_o * self.pr_grad[pod.pr_view] * ob_h[pod.ob_view]
                        + self.ob_grad[pod.ob_view] * pr_h[pod.pr_view] )
                d2mb = 2 * pr_h[pod.pr_view] * ob_h[pod.ob_view]

                Ddm1 = pod.fw(scale_p_o * self.pr_grad[pod.pr_view] * pod.object
                            + self.ob_grad[pod.ob_view] * pod.probe )
                Ddm2 = pod.fw( pr_h[pod.pr_view] * pod.object
                            + ob_h[pod.ob_view] * pod.probe )

                if A_num is None:
                    A_num = np.sum(np.real(xi * d2mt.conj())).astype(np.longdouble)
                    A_denom = np.sum(np.real(xi * d2mb.conj())).astype(np.longdouble)
                    v1_num = np.sum(w * (1 - I0) * np.real(Ddm1 * Ddm2.conj())).astype(np.longdouble)
                    v2_num = np.sum(w * I0 * np.real(f0 * Ddm1.conj()) * np.real(f0 * Ddm2.conj())).astype(np.longdouble)
                    v1_denom = np.sum(w * (1 - I0) * u.abs2(Ddm2)).astype(np.longdouble)
                    v2_denom = np.sum(w * I0 * np.real(f0 * Ddm2.conj()) ** 2).astype(np.longdouble)
                else:
                    A_num += np.sum(np.real(xi * d2mt.conj()))
                    A_denom += np.sum(np.real(xi * d2mb.conj()))
                    v1_num += np.sum(w * (1 - I0) * np.real(Ddm1 * Ddm2.conj()))
                    v2_num += np.sum(w * I0 * np.real(f0 * Ddm1.conj()) * np.real(f0 * Ddm2.conj()))
                    v1_denom += np.sum(w * (1 - I0) * u.abs2(Ddm2))
                    v2_denom += np.sum(w * I0 * np.real(f0 * Ddm2.conj()) ** 2)

            if self.p.floating_intensities:
                A_num *= self.float_intens_coeff[dname]
                A_denom *= self.float_intens_coeff[dname]
                v1_num *= self.float_intens_coeff[dname]
                v2_num *= self.float_intens_coeff[dname]
                v1_denom *= self.float_intens_coeff[dname]
                v2_denom *= self.float_intens_coeff[dname]

            beta_num += A_num + 2 * (v1_num + v2_num)
            beta_denom += A_denom + 2 * (v1_denom + v2_denom)

        parallel.allreduce(beta_num)
        parallel.allreduce(beta_denom)

        beta = beta_num[0] / beta_denom[0]
        print("beta: %f" % beta)

        return beta
