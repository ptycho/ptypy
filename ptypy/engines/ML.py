# -*- coding: utf-8 -*-
"""
Maximum Likelihood reconstruction engine.

TODO.

  * Implement other regularizers

This file is part of the PTYPY package.

    :copyright: Copyright 2014 by the PTYPY team, see AUTHORS.
    :license: see LICENSE for details.
"""
import numpy as np
import time

from .. import utils as u
from ..utils.verbose import logger
from ..utils import parallel
from .utils import Cnorm2, Cdot
from . import register
from .base import BaseEngine, PositionCorrectionEngine
from ..core.manager import Full, Vanilla, Bragg3dModel, BlockVanilla, BlockFull, GradFull, BlockGradFull


__all__ = ['ML']


@register()
class ML(PositionCorrectionEngine):
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

    SUPPORTED_MODELS = [Full, Vanilla, Bragg3dModel, BlockVanilla, BlockFull, GradFull, BlockGradFull]

    def __init__(self, ptycho_parent, pars=None):
        """
        Maximum likelihood reconstruction engine.
        """
        super(ML, self).__init__(ptycho_parent, pars)

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
        self.tmin = None
        self.ML_model = None
        self.smooth_gradient = None
        self.scale_p_o = None
        self.scale_p_o_memory = .9

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
        super(ML, self).engine_initialize()

        # Object gradient and minimization direction
        self.ob_grad = self.ob.copy(self.ob.ID + '_grad', fill=0.)
        self.ob_grad_new = self.ob.copy(self.ob.ID + '_grad_new', fill=0.)
        self.ob_h = self.ob.copy(self.ob.ID + '_h', fill=0.)

        # Probe gradient and minimization direction
        self.pr_grad = self.pr.copy(self.pr.ID + '_grad', fill=0.)
        self.pr_grad_new = self.pr.copy(self.pr.ID + '_grad_new', fill=0.)
        self.pr_h = self.pr.copy(self.pr.ID + '_h', fill=0.)

        self.tmin = 1.

        # Other options
        self.smooth_gradient = prepare_smoothing_preconditioner(
            self.p.smooth_gradient)

        self._initialize_model()

    def _initialize_model(self):

        # Create noise model
        if self.p.ML_type.lower() == "gaussian":
            self.ML_model = GaussianModel(self)
        elif self.p.ML_type.lower() == "poisson":
            self.ML_model = PoissonModel(self)
        elif self.p.ML_type.lower() == "euclid":
            raise NotImplementedError('Euclid norm model not yet implemented')
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
            else:
                bt_num = (self.scale_p_o
                          * (Cnorm2(new_pr_grad)
                             - np.real(Cdot(new_pr_grad, self.pr_grad)))
                          + (Cnorm2(new_ob_grad)
                             - np.real(Cdot(new_ob_grad, self.ob_grad))))

                bt_denom = self.scale_p_o*Cnorm2(self.pr_grad) + Cnorm2(self.ob_grad)

                bt = max(0, bt_num/bt_denom)

            # logger.info('Polak-Ribiere coefficient: %f ' % bt)

            self.ob_grad << new_ob_grad
            self.pr_grad << new_pr_grad

            dt = self.ptycho.FType
            # 3. Next conjugate
            self.ob_h *= bt / self.tmin

            # Smoothing preconditioner
            if self.smooth_gradient:
                for name, s in self.ob_h.storages.items():
                    s.data[:] -= self.smooth_gradient(self.ob_grad.storages[name].data)
            else:
                self.ob_h -= self.ob_grad

            self.pr_h *= bt / self.tmin
            self.pr_grad *= self.scale_p_o
            self.pr_h -= self.pr_grad

            # In principle, the way things are now programmed this part
            # could be iterated over in a real Newton-Raphson style.
            t2 = time.time()
            B = self.ML_model.poly_line_coeffs(self.ob_h, self.pr_h)
            tc += time.time() - t2

            if np.isinf(B).any() or np.isnan(B).any():
                logger.warning(
                    'Warning! inf or nan found! Trying to continue...')
                B[np.isinf(B)] = 0.
                B[np.isnan(B)] = 0.

            self.tmin = dt(-.5 * B[1] / B[2])
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
        Compute a new gradient direction according to the noise model.

        Note: The negative log-likelihood and local errors should also be computed
        here.
        """
        raise NotImplementedError

    def poly_line_coeffs(self, ob_h, pr_h):
        """
        Compute the coefficients of the polynomial for line minimization
        in direction h
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
                f = pod.fw(pod.probe * pod.object)
                a = pod.fw(pod.probe * ob_h[pod.ob_view]
                           + pr_h[pod.pr_view] * pod.object)
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
            B[1] += np.dot(w.flat, (2 * A0 * A1).flat) * Brenorm
            B[2] += np.dot(w.flat, (A1**2 + 2*A0*A2).flat) * Brenorm

        parallel.allreduce(B)

        # Object regularizer
        if self.regularizer:
            for name, s in self.ob.storages.items():
                B += Brenorm * self.regularizer.poly_line_coeffs(
                    ob_h.storages[name].data, s.data)

        self.B = B

        return B


class PoissonModel(BaseModel):
    """
    Poisson noise model.
    """

    def __init__(self, MLengine):
        """
        Core functions for ML computation using a Gaussian model.
        """
        BaseModel.__init__(self, MLengine)
        from scipy import special
        self.LLbase = {}
        for name, di_view in self.di.views.items():
            if not di_view.active:
                continue
            self.LLbase[name] = special.gammaln(di_view.data+1).sum()

    def new_grad(self):
        """
        Compute a new gradient direction according to a Poisson noise model.

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

            # Mask and intensities for this view
            I = diff_view.data
            m = diff_view.pod.ma_view.data

            Imodel = np.zeros_like(I)
            f = {}

            # First pod loop: compute total intensity
            for name, pod in diff_view.pods.items():
                if not pod.active:
                    continue
                f[name] = pod.fw(pod.probe * pod.object)
                Imodel += u.abs2(f[name])

            # Floating intensity option
            if self.p.floating_intensities:
                self.float_intens_coeff[dname] = I.sum() / Imodel.sum()
                Imodel *= self.float_intens_coeff[dname]

            Imodel += 1e-6
            DI = m * (1. - I / Imodel)

            # Second pod loop: gradients computation
            LLL = self.LLbase[dname] + (m * (Imodel - I * np.log(Imodel))).sum().astype(np.float64)
            for name, pod in diff_view.pods.items():
                if not pod.active:
                    continue
                xi = pod.bw(DI * f[name])
                self.ob_grad[pod.ob_view] += 2 * xi * pod.probe.conj()
                self.pr_grad[pod.pr_view] += 2 * xi * pod.object.conj()

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

    def poly_line_coeffs(self, ob_h, pr_h):
        """
        Compute the coefficients of the polynomial for line minimization
        in direction h
        """
        B = np.zeros((3,), dtype=np.longdouble)
        Brenorm = 1/(self.tot_measpts * self.LL[0])**2

        # Outer loop: through diffraction patterns
        for dname, diff_view in self.di.views.items():
            if not diff_view.active:
                continue

            # Weights and intensities for this view
            I = diff_view.data
            m = diff_view.pod.ma_view.data

            A0 = None
            A1 = None
            A2 = None

            for name, pod in diff_view.pods.items():
                if not pod.active:
                    continue
                f = pod.fw(pod.probe * pod.object)
                a = pod.fw(pod.probe * ob_h[pod.ob_view]
                           + pr_h[pod.pr_view] * pod.object)
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

            A0 += 1e-6
            DI = 1. - I/A0

            B[0] += (self.LLbase[dname] + (m * (A0 - I * np.log(A0))).sum().astype(np.float64)) * Brenorm
            B[1] += np.dot(m.flat, (A1*DI).flat) * Brenorm
            B[2] += (np.dot(m.flat, (A2*DI).flat) + .5*np.dot(m.flat, (I*(A1/A0)**2.).flat)) * Brenorm

        parallel.allreduce(B)

        # Object regularizer
        if self.regularizer:
            for name, s in self.ob.storages.items():
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
