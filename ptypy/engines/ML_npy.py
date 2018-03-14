# -*- coding: utf-8 -*-
"""
Maximum Likelihood reconstruction engine.

TODO.

  * Implement other regularizers

This file is part of the PTYPY package.

    :copyright: Copyright 2014 by the PTYPY team, see AUTHORS.
    :license: GPLv2, see LICENSE for details.
"""
import numpy as np
import time

from .. import utils as u
from ..utils.verbose import logger
from ..utils import parallel
from .utils import Cnorm2, Cdot
from ML import ML, ML_Gaussian
from ptypy.array_based import data_utils as du
from ptypy.array_based import FLOAT_TYPE, COMPLEX_TYPE
from ..utils.descriptor import defaults_tree
from ..core.manager import Full, Vanilla

__all__ = ['MLNpy']


@defaults_tree.parse_doc('engine.MLNpy')
class MLNpy(ML):
    """
    Maximum likelihood reconstruction engine.


    Defaults:

    [name]
    default = MLNpy
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

    """

    SUPPORTED_MODELS = [Full, Vanilla]

    def __init__(self, ptycho_parent, pars=None):
        """
        Difference map reconstruction engine.
        """
        super(MLNpy, self).__init__(ptycho_parent, pars)

    def engine_prepare(self):
        """
        Last minute initialization.

        Everything that needs to be recalculated when new data arrives.
        """
        super(MLNpy, self).engine_prepare()
        # and then something to convert the arrays to numpy
        self.vectorised_scan = {}
        self.propagator = {}
        for dID, _diffs in self.di.S.iteritems():
            self.vectorised_scan[dID] = du.pod_to_arrays(self, dID)
            first_view_id = self.vectorised_scan[dID]['meta']['view_IDs'][0]
            self.propagator[dID] = self.di.V[first_view_id].pod.geometry.propagator

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
            new_ob_grad, new_pr_grad, error_dct = self.ML_model.new_grad()
            tg += time.time() - t1

            if self.p.probe_update_start <= self.curiter:
                # Apply probe support if needed
                for name, s in new_pr_grad.storages.iteritems():
                    support = self.probe_support.get(name)
                    if support is not None:
                        s.data *= support
            else:
                new_pr_grad.fill(0.)

            # Smoothing preconditioner
            if self.smooth_gradient:
                self.smooth_gradient.sigma *= (1. - self.p.smooth_gradient_decay)
                for name, s in new_ob_grad.storages.iteritems():
                    s.data[:] = self.smooth_gradient(s.data)

            # probe/object rescaling
            if self.p.scale_precond:
                cn2_new_pr_grad = Cnorm2(new_pr_grad)
                if cn2_new_pr_grad > 1e-5:
                    scale_p_o = (self.p.scale_probe_object * Cnorm2(new_ob_grad)
                                 / Cnorm2(new_pr_grad))
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
            else:
                bt_num = (self.scale_p_o
                          * (Cnorm2(new_pr_grad)
                             - np.real(Cdot(new_pr_grad, self.pr_grad)))
                          + (Cnorm2(new_ob_grad)
                             - np.real(Cdot(new_ob_grad, self.ob_grad))))

                bt_denom = self.scale_p_o*Cnorm2(self.pr_grad) + Cnorm2(self.ob_grad)

                bt = max(0, bt_num/bt_denom)

            # verbose(3,'Polak-Ribiere coefficient: %f ' % bt)

            self.ob_grad << new_ob_grad
            self.pr_grad << new_pr_grad

            # 3. Next conjugate
            self.ob_h *= bt / self.tmin

            # Smoothing preconditioner
            if self.smooth_gradient:
                for name, s in self.ob_h.storages.iteritems():
                    s.data[:] -= self.smooth_gradient(self.ob_grad.storages[name].data)
            else:
                self.ob_h -= self.ob_grad
            self.pr_h *= bt / self.tmin
            self.pr_grad *= self.scale_p_o
            self.pr_h -= self.pr_grad

            t2 = time.time()
            B = self.ML_model.poly_line_coeffs(self.ob_h, self.pr_h)
            tc += time.time() - t2

            if np.isinf(B).any() or np.isnan(B).any():
                logger.warning(
                    'Warning! inf or nan found! Trying to continue...')
                B[np.isinf(B)] = 0.
                B[np.isnan(B)] = 0.

            self.tmin = -.5 * B[1] / B[2]
            self.ob_h *= self.tmin
            self.pr_h *= self.tmin
            self.ob += self.ob_h
            self.pr += self.pr_h

            # increase iteration conter
            self.curiter +=1

        logger.info('Time spent in gradient calculation: %.2f' % tg)
        logger.info('  ....  in coefficient calculation: %.2f' % tc)
        return error_dct



class ML_Gaussian_gpu(ML_Gaussian):

    def __init__(self, MLengine):
        super(ML_Gaussian_gpu, self).__init__(MLengine)

    def new_grad(self):
        """
        Compute a new gradient direction according to a Gaussian noise model.

        Note: The negative log-likelihood and local errors are also computed
        here.
        """
        self.ob_grad.fill(0.)
        self.pr_grad.fill(0.)

        # We need an array for MPI
        LL = np.array([0.])
        error_dct = {}

        # Outer loop: through diffraction patterns
        for dname, diff_view in self.di.views.iteritems():
            if not diff_view.active:
                continue

            # Weights and intensities for this view
            w = self.weights[diff_view]
            I = diff_view.data

            Imodel = np.zeros_like(I)
            f = {}

            # First pod loop: compute total intensity
            for name, pod in diff_view.pods.iteritems():
                if not pod.active:
                    continue
                f[name] = pod.fw(pod.probe * pod.object)
                Imodel += u.abs2(f[name])

            # Floating intensity option
            if self.p.floating_intensities:
                diff_view.float_intens_coeff = ((w * Imodel * I).sum()
                                                / (w * Imodel**2).sum())
                Imodel *= diff_view.float_intens_coeff

            DI = Imodel - I

            # Second pod loop: gradients computation
            LLL = np.sum((w * DI**2))
            for name, pod in diff_view.pods.iteritems():
                if not pod.active:
                    continue
                xi = pod.bw(w * DI * f[name])
                self.ob_grad[pod.ob_view] += 2. * xi * pod.probe.conj()
                self.pr_grad[pod.pr_view] += 2. * xi * pod.object.conj()

                # Negative log-likelihood term
                # LLL += (w * DI**2).sum()

            # LLL
            diff_view.error = LLL
            error_dct[dname] = np.array([0, LLL / np.prod(DI.shape), 0])
            LL += LLL


        # Object regularizer
        if self.regularizer:
            for name, s in self.ob.storages.iteritems():
                self.ob_grad.storages[name].data += self.regularizer.grad(
                    s.data)
                LL += self.regularizer.LL

        self.LL = LL / self.tot_measpts

        return self.ob_grad, self.pr_grad, error_dct

    def poly_line_coeffs(self, ob_h, pr_h):
        """
        Compute the coefficients of the polynomial for line minimization
        in direction h
        """

        B = np.zeros((3,), dtype=np.longdouble)
        Brenorm = 1. / self.LL[0]**2

        # Outer loop: through diffraction patterns
        for dname, diff_view in self.di.views.iteritems():
            if not diff_view.active:
                continue

            # Weights and intensities for this view
            w = self.weights[diff_view]
            I = diff_view.data

            A0 = None
            A1 = None
            A2 = None

            for name, pod in diff_view.pods.iteritems():
                if not pod.active:
                    continue
                f = pod.fw(pod.probe * pod.object)
                a = pod.fw(pod.probe * ob_h[pod.ob_view]
                           + pr_h[pod.pr_view] * pod.object)
                b = pod.fw(pr_h[pod.pr_view] * ob_h[pod.ob_view])

                if A0 is None:
                    A0 = u.abs2(f)
                    A1 = 2 * np.real(f * a.conj())
                    A2 = (2 * np.real(f * b.conj())
                          + u.abs2(a))
                else:
                    A0 += u.abs2(f)
                    A1 += 2 * np.real(f * a.conj())
                    A2 += 2 * np.real(f * b.conj()) + u.abs2(a)

            if self.p.floating_intensities:
                A0 *= diff_view.float_intens_coeff
                A1 *= diff_view.float_intens_coeff
                A2 *= diff_view.float_intens_coeff
            A0 -= I

            B[0] += np.dot(w.flat, (A0**2).flat) * Brenorm
            B[1] += np.dot(w.flat, (2 * A0 * A1).flat) * Brenorm
            B[2] += np.dot(w.flat, (A1**2 + 2*A0*A2).flat) * Brenorm

        parallel.allreduce(B)

        # Object regularizer
        if self.regularizer:
            for name, s in self.ob.storages.iteritems():
                B += Brenorm * self.regularizer.poly_line_coeffs(
                    ob_h.storages[name].data, s.data)

        self.B = B

        return B



class Regul_del2(object):
    """\
    Squared gradient regularizer (Gaussian prior).

    This class applies to any numpy array.
    """
    def __init__(self, amplitude, axes=[-2, -1]):
        # Regul.__init__(self, axes)
        self.axes = axes
        self.amplitude = amplitude
        self.delxy = None
        self.g = None
        self.LL = None

    def grad(self, x):
        """
        Compute and return the regularizer gradient given the array x.
        """
        ax0, ax1 = self.axes
        del_xf = u.delxf(x, axis=ax0)
        del_yf = u.delxf(x, axis=ax1)
        del_xb = u.delxb(x, axis=ax0)
        del_yb = u.delxb(x, axis=ax1)

        self.delxy = [del_xf, del_yf, del_xb, del_yb]
        self.g = 2. * self.amplitude*(del_xb + del_yb - del_xf - del_yf)

        self.LL = self.amplitude * (u.norm2(del_xf)
                               + u.norm2(del_yf)
                               + u.norm2(del_xb)
                               + u.norm2(del_yb))

        return self.g

    def poly_line_coeffs(self, h, x=None):
        ax0, ax1 = self.axes
        if x is None:
            del_xf, del_yf, del_xb, del_yb = self.delxy
        else:
            del_xf = u.delxf(x, axis=ax0)
            del_yf = u.delxf(x, axis=ax1)
            del_xb = u.delxb(x, axis=ax0)
            del_yb = u.delxb(x, axis=ax1)

        hdel_xf = u.delxf(h, axis=ax0)
        hdel_yf = u.delxf(h, axis=ax1)
        hdel_xb = u.delxb(h, axis=ax0)
        hdel_yb = u.delxb(h, axis=ax1)

        c0 = self.amplitude * (u.norm2(del_xf)
                               + u.norm2(del_yf)
                               + u.norm2(del_xb)
                               + u.norm2(del_yb))

        c1 = 2 * self.amplitude * np.real(np.vdot(del_xf, hdel_xf)
                                          + np.vdot(del_yf, hdel_yf)
                                          + np.vdot(del_xb, hdel_xb)
                                          + np.vdot(del_yb, hdel_yb))

        c2 = self.amplitude * (u.norm2(hdel_xf)
                               + u.norm2(hdel_yf)
                               + u.norm2(hdel_xb)
                               + u.norm2(hdel_yb))

        self.coeff = np.array([c0, c1, c2])
        return self.coeff

