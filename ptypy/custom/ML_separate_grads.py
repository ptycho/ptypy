# -*- coding: utf-8 -*-
"""
Maximum Likelihood separate gradients reconstruction engine.

TODO.

  * Implement other ML models (Poisson/Euclid)

This file is part of the PTYPY package.

    :copyright: Copyright 2024 by the PTYPY team, see AUTHORS.
    :license: see LICENSE for details.
"""
import numpy as np
import time

from .. import utils as u
from ..utils.verbose import logger
from ..utils import parallel
from ..engines.utils import Cnorm2, Cdot
from ..engines import register
from ..engines.base import BaseEngine, PositionCorrectionEngine
from ..core.manager import Full, Vanilla, Bragg3dModel, BlockVanilla, BlockFull, GradFull, BlockGradFull


__all__ = ['MLSeparateGrads']


@register()
class MLSeparateGrads(PositionCorrectionEngine):
    """
    Maximum likelihood reconstruction engine.


    Defaults:

    [name]
    default = ML
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

    [probe_update_start]
    default = 0
    type = int
    lowlim = 0
    help = Number of iterations before probe update starts
    # NOTE: probe_update_start doesn't work with this code, need to add some code to fix this

    [poly_line_coeffs]
    default = quadratic
    type = str
    help = How many coefficients to be used in the the linesearch
    choices = ['quadratic','all']
    doc = choose between the 'quadratic' approximation (default) or 'all'

    """

    SUPPORTED_MODELS = [Full, Vanilla, Bragg3dModel, BlockVanilla, BlockFull, GradFull, BlockGradFull]

    def __init__(self, ptycho_parent, pars=None):
        """
        Maximum likelihood reconstruction engine.
        """
        super(MLSeparateGrads, self).__init__(ptycho_parent, pars)

        # Instance attributes

        # Object gradient
        self.ob_grad = None

        # Object minimization direction
        self.ob_h = None

        # Probe gradient
        self.pr_grad = None

        # Probe minimization direction
        self.pr_h = None

        # Working variables
        # Object gradient
        self.ob_grad_new = None

        # Probe gradient
        self.pr_grad_new = None


        # Other
        self.tmin_ob = None
        self.tmin_pr = None
        self.ML_model = None
        self.smooth_gradient = None

        self.ptycho.citations.add_article(
            title='Maximum-likelihood refinement for coherent diffractive imaging',
            author='Thibault P. and Guizar-Sicairos M.',
            journal='New Journal of Physics',
            volume=14,
            year=2012,
            page=63004,
            doi='10.1088/1367-2630/14/6/063004',
            comment='The maximum likelihood reconstruction algorithm',
        )

    def engine_initialize(self):
        """
        Prepare for ML reconstruction.
        """
        super(MLSeparateGrads, self).engine_initialize()

        # Object gradient and minimization direction
        self.ob_grad = self.ob.copy(self.ob.ID + '_grad', fill=0.)
        self.ob_grad_new = self.ob.copy(self.ob.ID + '_grad_new', fill=0.)
        self.ob_h = self.ob.copy(self.ob.ID + '_h', fill=0.)

        # Probe gradient and minimization direction
        self.pr_grad = self.pr.copy(self.pr.ID + '_grad', fill=0.)
        self.pr_grad_new = self.pr.copy(self.pr.ID + '_grad_new', fill=0.)
        self.pr_h = self.pr.copy(self.pr.ID + '_h', fill=0.)

        self.tmin_ob = 1.
        self.tmin_pr = 1.

        # Other options
        self.smooth_gradient = prepare_smoothing_preconditioner(
            self.p.smooth_gradient)

        self._initialize_model()

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



    def engine_prepare(self):
        """
        Last minute initialization, everything, that needs to be recalculated,
        when new data arrives.
        """
        # - # fill object with coverage of views
        # - for name,s in self.ob_viewcover.S.items():
        # -    s.fill(s.get_view_coverage())
        self.ML_model.prepare()

    def engine_iterate(self, num=1):
        """
        Compute `num` iterations.
        """
        tg = 0.
        tc = 0.
        ta = time.time()
        for it in range(num):

            ########################
            # Compute new gradients
            # ob: new_ob_grad
            # pr: new_pr_grad
            ########################
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
            # FIXME: this hack doesn't work here as we step in probe and object separately
            # FIXME: really it's the probe step that should be zeroed out not the gradient
            else:
                new_pr_grad.fill(0.)

            # Smoothing preconditioner
            if self.smooth_gradient:
                self.smooth_gradient.sigma *= (1. - self.p.smooth_gradient_decay)
                for name, s in new_ob_grad.storages.items():
                    s.data[:] = self.smooth_gradient(s.data)

            ############################
            # Compute Polak-Ribiere betas
            # bt_ob = bt_num_ob/bt_denom_ob
            # bt_pr = bt_num_pr/bt_denom_pr
            ############################
            if self.curiter == 0:
                bt_ob = 0.
                bt_pr = 0.
            else:
                # For the object
                bt_num_ob = Cnorm2(new_ob_grad) - np.real(Cdot(new_ob_grad, self.ob_grad))
                bt_denom_ob = Cnorm2(self.ob_grad)
                bt_ob = max(0, bt_num_ob/bt_denom_ob)

                # For the probe
                bt_num_pr = Cnorm2(new_pr_grad) - np.real(Cdot(new_pr_grad, self.pr_grad))
                bt_denom_pr = Cnorm2(self.pr_grad)
                bt_pr = max(0, bt_num_pr/bt_denom_pr)

            self.ob_grad << new_ob_grad
            self.pr_grad << new_pr_grad

            ############################
            # Compute next conjugates
            # ob_h -= bt_ob * ob_grad
            # pr_h -= bt_pr * pr_grad
            # NB: in the below need to do h/tmin
            # as did h*tmin when taking steps
            # (don't you just love containers?)
            ############################
            self.ob_h *= bt_ob / self.tmin_ob

            # Smoothing preconditioner
            if self.smooth_gradient:
                for name, s in self.ob_h.storages.items():
                    s.data[:] -= self.smooth_gradient(self.ob_grad.storages[name].data)
            else:
                self.ob_h -= self.ob_grad

            self.pr_h *= bt_pr / self.tmin_pr
            self.pr_h -= self.pr_grad

            ########################
            # Compute step-sizes
            # ob: tmin_ob
            # pr: tmin_pr
            ########################
            dt = self.ptycho.FType
            t2 = time.time()

            if self.p.poly_line_coeffs == "quadratic":
                B_ob = self.ML_model.poly_line_coeffs_ob(self.ob_h)
                B_pr = self.ML_model.poly_line_coeffs_pr(self.pr_h)

                # same as above but quicker when poly quadratic
                self.tmin_ob = dt(-0.5 * B_ob[1] / B_ob[2])
                self.tmin_pr = dt(-0.5 * B_pr[1] / B_pr[2])

            else:
                raise NotImplementedError("poly_line_coeffs should be 'quadratic' or 'all'")

            tc += time.time() - t2

            ########################
            # Take conjugate steps
            # ob += tmin_ob * ob_h
            # pr += tmin_pr * pr_h
            ########################
            self.ob_h *= self.tmin_ob
            self.pr_h *= self.tmin_pr
            self.ob += self.ob_h
            self.pr += self.pr_h

            # Position correction
            self.position_update()

            # Allow for customized modifications at the end of each iteration
            self._post_iterate_update()

            # increase iteration counter
            self.curiter +=1

        logger.info('Time spent in gradient calculation: %.2f' % tg)
        logger.info('  ....  in coefficient calculation: %.2f' % tc)
        return error_dct  # np.array([[self.ML_model.LL[0]] * 3])

    def _post_iterate_update(self):
        """
        Enables modification at the end of each ML iteration.
        """
        pass

    def engine_finalize(self):
        """
        Delete temporary containers.
        """
        del self.ptycho.containers[self.ob_grad.ID]
        del self.ob_grad
        del self.ptycho.containers[self.ob_grad_new.ID]
        del self.ob_grad_new
        del self.ptycho.containers[self.ob_h.ID]
        del self.ob_h
        del self.ptycho.containers[self.pr_grad.ID]
        del self.pr_grad
        del self.ptycho.containers[self.pr_grad_new.ID]
        del self.pr_grad_new
        del self.ptycho.containers[self.pr_h.ID]
        del self.pr_h

        # Save floating intensities into runtime
        self.ptycho.runtime["float_intens"] = parallel.gather_dict(self.ML_model.float_intens_coeff)

        # Delete model
        del self.ML_model

class BaseModel(object):
    """
    Base class for log-likelihood models.
    """

    def __init__(self, MLengine):
        """
        Core functions for ML computation using a Gaussian model.
        """
        self.engine = MLengine

        # Transfer commonly used attributes from ML engine
        self.di = self.engine.di
        self.p = self.engine.p
        self.ob = self.engine.ob
        self.ob_grad = self.engine.ob_grad_new
        self.pr_grad = self.engine.pr_grad_new
        self.pr = self.engine.pr
        self.float_intens_coeff = {}

        if self.p.intensity_renormalization is None:
            self.Irenorm = 1.
        else:
            self.Irenorm = self.p.intensity_renormalization

        if self.p.reg_del2:
            self.regularizer = Regul_del2(self.p.reg_del2_amplitude)
        else:
            self.regularizer = None

        # Create working variables
        self.LL = 0.


    def prepare(self):
        # Useful quantities
        self.tot_measpts = sum(s.data.size
                               for s in self.di.storages.values())
        self.tot_power = self.Irenorm * sum(s.tot_power
                                            for s in self.di.storages.values())
        # Prepare regularizer
        if self.regularizer is not None:
            obj_Npix = self.ob.size
            expected_obj_var = obj_Npix / self.tot_power  # Poisson
            reg_rescale = self.tot_measpts / (8. * obj_Npix * expected_obj_var)
            logger.debug(
                'Rescaling regularization amplitude using '
                'the Poisson distribution assumption.')
            logger.debug('Factor: %8.5g' % reg_rescale)

            # TODO remove usage of .p. access
            self.regularizer.amplitude = self.p.reg_del2_amplitude * reg_rescale

    def __del__(self):
        """
        Clean up routine
        """
        # Remove working attributes
        for name, diff_view in self.di.views.items():
            if not diff_view.active:
                continue
            try:
                del diff_view.error
            except:
                pass

    def new_grad(self):
        """
        Compute new object and probe gradient directions according to the noise model.

        Note: The negative log-likelihood and local errors should also be computed here.
        """
        raise NotImplementedError

    def poly_line_coeffs_ob(self, ob_h):
        """
        Compute the coefficients of the polynomial for line minimization
        in direction h for the object
        """
        raise NotImplementedError

    def poly_line_coeffs_pr(self, pr_h):
        """
        Compute the coefficients of the polynomial for line minimization
        in direction h for the probe
        """
        raise NotImplementedError


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
                f[name] = pod.fw(pod.probe * pod.object)
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
                self.ob_grad[pod.ob_view] += 2. * xi * pod.probe.conj()
                self.pr_grad[pod.pr_view] += 2. * xi * pod.object.conj()

            diff_view.error = LLL
            error_dct[dname] = np.array([0, LLL / np.prod(DI.shape), 0])
            LL += LLL

        # MPI reduction of gradients
        self.ob_grad.allreduce()
        self.pr_grad.allreduce()
        parallel.allreduce(LL)

        # Object regularizer
        if self.regularizer:
            for name, s in self.ob.storages.items():
                self.ob_grad.storages[name].data += self.regularizer.grad(
                    s.data)
                LL += self.regularizer.LL
        self.LL = LL / self.tot_measpts

        return error_dct

    def poly_line_coeffs_ob(self, ob_h):
        """
        Compute the coefficients of the polynomial for line minimization
        in direction h for the object
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
                f = pod.fw(pod.probe * pod.object)
                a = pod.fw(pod.probe * ob_h[pod.ob_view])

                if A0 is None:
                    A0 = u.abs2(f).astype(np.longdouble)
                    A1 = 2 * np.real(f * a.conj()).astype(np.longdouble)
                    A2 = u.abs2(a).astype(np.longdouble)
                else:
                    A0 += u.abs2(f)
                    A1 += 2 * np.real(f * a.conj())
                    A2 += u.abs2(a)

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

    def poly_line_coeffs_pr(self, pr_h):
        """
        Compute the coefficients of the polynomial for line minimization
        in direction h for the probe
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

                f = pod.fw(pod.probe * pod.object)
                a = pod.fw(pr_h[pod.pr_view] * pod.object)

                if A0 is None:

                    A0 = u.abs2(f).astype(np.longdouble)
                    A1 = 2 * np.real(f * a.conj()).astype(np.longdouble)
                    A2 = u.abs2(a).astype(np.longdouble)
                else:
                    A0 += u.abs2(f)
                    A1 += 2 * np.real(f * a.conj())
                    A2 += u.abs2(a)

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

        if np.isinf(B).any() or np.isnan(B).any():
            logger.warning(
                'Warning! inf or nan found! Trying to continue...')
            B[np.isinf(B)] = 0.
            B[np.isnan(B)] = 0.

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


def prepare_smoothing_preconditioner(amplitude):
    """
    Factory for smoothing preconditioner.
    """
    if amplitude == 0.:
        return None

    class GaussFilt(object):
        def __init__(self, sigma):
            self.sigma = sigma

        def __call__(self, x):
            return u.c_gf(x, [0, self.sigma, self.sigma])

    # from scipy.signal import correlate2d
    # class HannFilt:
    #    def __call__(self, x):
    #        y = np.empty_like(x)
    #        sh = x.shape
    #        xf = x.reshape((-1,) + sh[-2:])
    #        yf = y.reshape((-1,) + sh[-2:])
    #        for i in range(len(xf)):
    #            yf[i] = correlate2d(xf[i],
    #                                np.array([[.0625, .125, .0625],
    #                                          [.125, .25, .125],
    #                                          [.0625, .125, .0625]]),
    #                                mode='same')
    #        return y

    if amplitude > 0.:
        logger.debug(
            'Using a smooth gradient filter (Gaussian blur - only for ML)')
        return GaussFilt(amplitude)

    elif amplitude < 0.:
        raise RuntimeError('Hann filter not implemented (negative smoothing amplitude not supported)')
        # logger.debug(
        #    'Using a smooth gradient filter (Hann window - only for ML)')
        # return HannFilt()
