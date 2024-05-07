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
from ..engines.utils import Cnorm2, Cdot
from ..engines import register
from ..engines.base import BaseEngine, PositionCorrectionEngine
from ..core.manager import Full, Vanilla, Bragg3dModel, BlockVanilla, BlockFull, GradFull, BlockGradFull
from ..utils.tomo import AstraTomoWrapperViewBased
from scipy.ndimage.filters import gaussian_filter


__all__ = ['MLPtychoTomo']


@register()
class MLPtychoTomo(PositionCorrectionEngine):
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
    doc = choose between the 'quadratic' approximation (default) or 'all'

    """

    SUPPORTED_MODELS = [Full, Vanilla, Bragg3dModel, BlockVanilla, BlockFull, GradFull, BlockGradFull]

    def __init__(self, ptycho_parent, pars=None):
        """
        Maximum likelihood reconstruction engine.
        """
        super(MLPtychoTomo, self).__init__(ptycho_parent, pars)

        # Instance attributes

        # Volume
        self.rho = None

        # Projected volume
        self.projected_rho = None

        # Object gradient
        self.rho_grad = None

        # Object minimization direction
        self.rho_h = None

        # Probe gradient
        self.pr_grad = None

        # Probe minimization direction
        self.pr_h = None

        # Working variables
        # Object gradient
        self.rho_grad_new = None

        # Probe gradient
        self.pr_grad_new = None

        self.projector = None

        # Other
        self.tmin_rho = None
        self.tmin_pr = None
        self.ML_model = None
        # self.smooth_gradient = None

        self.omega = None

        self.pshape = list(self.ptycho.obj.S.values())[0].data.shape[-1]

        n_angles = len(self.ptycho.obj.S)
        angles = np.linspace(0, np.pi, n_angles, endpoint=True)

        self.angles_dict = {}
        for i,k in enumerate(self.ptycho.obj.S):
            self.angles_dict[k] = angles[i]

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
        super(MLPtychoTomo, self).engine_initialize()

        # Object gradient and minimization direction
        self.rho_grad = np.zeros(3*(self.pshape,), dtype=np.complex64)       # self.ob.copy(self.ob.ID + '_grad', fill=0.)
        self.rho_grad_new = np.zeros(3*(self.pshape,), dtype=np.complex64)   # self.ob.copy(self.ob.ID + '_grad_new', fill=0.)
        self.rho_h = np.zeros(3*(self.pshape,), dtype=np.complex64)          # self.ob.copy(self.ob.ID + '_h', fill=0.)

        # This is needed in poly_line_coeffs_rho
        self.omega = self.ob.copy(self.ob.ID + '_omega', fill=0.)

        # Volume  
        rho_real = np.load('real_vol_35it.npy')
        rho_imag = np.load('imag_vol_35it.npy')
        rho_real_br = gaussian_filter(rho_real, sigma=3.5)
        rho_imag_br = gaussian_filter(rho_imag, sigma=3.5)
        self.rho = rho_real_br + 1j * rho_imag_br 

        # This 
        stacked_views = np.array([v.data for v in self.ptycho.obj.views.values()])
        self.projected_rho = np.zeros_like((stacked_views), dtype=np.complex64)

        # Probe gradient and minimization direction
        self.pr_grad = self.pr.copy(self.pr.ID + '_grad', fill=0.)
        self.pr_grad_new = self.pr.copy(self.pr.ID + '_grad_new', fill=0.)
        self.pr_h = self.pr.copy(self.pr.ID + '_h', fill=0.)

        self.tmin_rho = 1.
        self.tmin_pr = 1.

        # Other options
        # self.smooth_gradient = prepare_smoothing_preconditioner(
        #     self.p.smooth_gradient)

        self.projector = AstraTomoWrapperViewBased (    
            obj=self.ptycho.obj, 
            vol=self.rho, 
            angles=self.angles_dict, 
            obj_is_refractive_index=False, 
            mask_threshold=35
            )

        self.projected_rho = self.projector.forward(self.rho)
        self.update_views()  

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
        self.ML_model.prepare()


    def update_views(self):

        """
        Updates the views so that the projected rho (non-exponentiated)  
        can be retrieved from pod.object .
        """
        for i, (k,v) in enumerate(self.ptycho.obj.views.items()):
            v.data[:] = self.projected_rho[i]


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
            self.ML_model.new_grad_rho()
            error_dct = self.ML_model.new_grad_pr()
            
            new_rho_grad, new_pr_grad = self.rho_grad_new, self.pr_grad_new

            # SCALING: needed for the regularizer to work on probe coeffs
            n_pixels = np.prod(np.shape(self.rho))
            scaling_factor = 0.57 * n_pixels         
            new_rho_grad /= scaling_factor

            print('rho_grad: %.2e' % np.linalg.norm(self.rho_grad))
            print('new_rho_grad: %.2e ' % np.linalg.norm(new_rho_grad))
            print('pr_grad: %.2e ' % np.sqrt(Cnorm2(self.pr_grad)))
            print('new_pr_grad: %.2e ' % np.sqrt(Cnorm2(new_pr_grad)))

            # PLOTTING
            # try:
            #     iter = self.ptycho.runtime.iter_info[-1]['iteration']
            # except:
            #     iter=0

            # if iter%10 == 0:
            #     self.projector.plot_complex_array(new_rho_grad[26, :, :], title='new_rho_grad computed by new_grad')

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
            # if self.smooth_gradient:
            #     self.smooth_gradient.sigma *= (1. - self.p.smooth_gradient_decay)
            #     for name, s in new_ob_grad.storages.items():
            #         s.data[:] = self.smooth_gradient(s.data)

            ############################
            # Compute next conjugate
            ############################
            if self.curiter == 0:
                bt_rho = 0.
                bt_pr = 0.

            else:
                # For the volume
                bt_num_rho = u.norm2(new_rho_grad) - np.real(np.vdot(new_rho_grad.flat, self.rho_grad.flat)) 
                bt_denom_rho = u.norm2(self.rho_grad)
                print('bt_num_rho, bt_denom_rho: (%.2e, %.2e) ' % (bt_num_rho, bt_denom_rho))
                
                if bt_denom_rho == 0:
                    bt_rho = 0
                else:
                    bt_rho = max(0, bt_num_rho/bt_denom_rho)

                # For the probe
                bt_num_pr = Cnorm2(new_pr_grad) - np.real(Cdot(new_pr_grad, self.pr_grad))
                bt_denom_pr = Cnorm2(self.pr_grad)
                bt_pr = max(0, bt_num_pr/bt_denom_pr)               

            self.rho_grad = new_rho_grad.copy()
            self.pr_grad << new_pr_grad

            dt = self.ptycho.FType
            # 3. Next conjugate
            print('bt_rho, self.tmin_rho: (%.2e,%.2e)' % (bt_rho, self.tmin_rho))
            self.rho_h *= bt_rho / self.tmin_rho

            # Smoothing preconditioner
            # if self.smooth_gradient:
            #     for name, s in self.ob_h.storages.items():
            #         s.data[:] -= self.smooth_gradient(self.ob_grad.storages[name].data)
            # else:
            self.rho_h -= self.rho_grad

            self.pr_h *= bt_pr / self.tmin_pr
            self.pr_h -= self.pr_grad

            t2 = time.time()

            if self.p.poly_line_coeffs == "quadratic":
                B_rho = self.ML_model.poly_line_coeffs_rho(self.rho_h, self.pr_h)
                B_pr = self.ML_model.poly_line_coeffs_pr(self.rho_h, self.pr_h)
                
                print('B_rho, B_pr', B_rho, B_pr)
 
                # same as above but quicker when poly quadratic
                self.tmin_rho = dt(-0.5 * B_rho[1] / B_rho[2])
                self.tmin_pr = dt(-0.5 * B_pr[1] / B_pr[2])

            else:
                B_rho = self.ML_model.poly_line_all_coeffs_rho(self.rho_h, self.pr_h)
                diffB_rho = np.arange(1,len(B_rho))*B_rho[1:] # coefficients of poly derivative
                roots = np.roots(np.flip(diffB_rho.astype(np.double))) # roots only supports double
                real_roots = np.real(roots[np.isreal(roots)]) # not interested in complex roots
                if real_roots.size == 1: # single real root
                    self.tmin_rho = dt(real_roots[0])
                else: # find real root with smallest poly objective
                    evalp = lambda root: np.polyval(np.flip(B_rho),root)
                    self.tmin_rho = dt(min(real_roots, key=evalp)) # root with smallest poly objective

                B_pr = self.ML_model.poly_line_all_coeffs_pr(self.rho_h, self.pr_h)
                diffB_pr = np.arange(1,len(B_pr))*B_pr[1:] # coefficients of poly derivative
                roots = np.roots(np.flip(diffB_pr.astype(np.double))) # roots only supports double
                real_roots = np.real(roots[np.isreal(roots)]) # not interested in complex roots
                if real_roots.size == 1: # single real root
                    self.tmin_pr = dt(real_roots[0])
                else: # find real root with smallest poly objective
                    evalp = lambda root: np.polyval(np.flip(B_pr),root)
                    self.tmin_pr = dt(min(real_roots, key=evalp)) # root with smallest poly objective

            tc += time.time() - t2
                
            print('self.tmin_pr, self.tmin_rho: (%.2e,%.2e)' % (self.tmin_pr, self.tmin_rho))

            self.rho_h *= self.tmin_rho
            self.pr_h *= self.tmin_pr

            self.rho += self.rho_h

            try:
                iter = self.ptycho.runtime.iter_info[-1]['iteration']
            except:
                iter = 0

            if iter%50 == 0:
                self.projector.plot_vol(self.rho)

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
        # del self.ptycho.containers[self.ob_grad.ID]
        del self.rho_grad
        # del self.ptycho.containers[self.ob_grad_new.ID]
        del self.rho_grad_new
        # del self.ptycho.containers[self.ob_h.ID]
        del self.rho_h
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
        self.rho = self.engine.rho
        self.rho_grad = self.engine.rho_grad_new
        self.pr_grad = self.engine.pr_grad_new
        self.omega = self.engine.omega

        self.pr = self.engine.pr
        self.float_intens_coeff = {}

        if self.p.intensity_renormalization is None:
            self.Irenorm = 1.
        else:
            self.Irenorm = self.p.intensity_renormalization

        self.projector = self.engine.projector

        if self.p.reg_del2:
            self.regularizer = Regul_del2(self.p.reg_del2_amplitude)
        else:
            self.regularizer = None

        self.LL = 0.


    def prepare(self):
        # Useful quantities
        self.tot_measpts = sum(s.data.size
                               for s in self.di.storages.values())
        self.tot_power = self.Irenorm * sum(s.tot_power
                                            for s in self.di.storages.values())
        # Prepare regularizer
        if self.regularizer is not None:
            obj_Npix = np.prod(np.shape(self.rho))    #self.ob.size 
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

    def poly_line_coeffs_rho(self, rho_h, pr_h):
        """
        Compute the coefficients of the polynomial for line minimization
        in direction h for the object
        """
        raise NotImplementedError

    def poly_line_coeffs_pr(self, rho_h, pr_h):
        """
        Compute the coefficients of the polynomial for line minimization
        in direction h
        in direction h for the probe
        """
        raise NotImplementedError

    def poly_line_all_coeffs_rho(self, rho_h, pr_h):
        """
        Compute all the coefficients of the polynomial for line minimization
        in direction h
        """
        raise NotImplementedError

    def poly_line_all_coeffs_pr(self, rho_h, pr_h):
        """
        Compute all the coefficients of the polynomial for line minimization
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


    def new_grad_rho(self):
        """
        Compute a new gradient direction according to a Gaussian noise model.

        Note: The negative log-likelihood and local errors are also computed
        here.
        """
        self.rho_grad.fill(0.)

        # Moved this here from the end of engine_iterate
        self.projected_rho = self.projector.forward(self.rho)
        self.engine.update_views()

        products_xi_psi_conj = []

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
                f[name] = pod.fw(pod.probe * np.exp(1j * pod.object))
                Imodel += pod.downsample(u.abs2(f[name]))

            # Floating intensity option
            if self.p.floating_intensities:
                self.float_intens_coeff[dname] = ((w * Imodel * I).sum()
                                                / (w * Imodel**2).sum())
                Imodel *= self.float_intens_coeff[dname]

            DI = np.double(Imodel) - I

            # Second pod loop: gradients computation
            for name, pod in diff_view.pods.items():
                if not pod.active:
                    continue
                xi = pod.bw(pod.upsample(w*DI) * f[name])
                psi = pod.probe * np.exp(1j * pod.object)   # CHANGING from pod.object 
                product_xi_psi_conj = xi * psi.conj() 
                products_xi_psi_conj.append(product_xi_psi_conj)

        self.rho_grad = 2 * self.projector.backward(np.moveaxis(np.array(products_xi_psi_conj), 1, 0))
        # print(self.rho_grad)
        # self.engine.rho_grad = self.rho_grad
        self.engine.rho_grad_new = self.rho_grad
        # print(np.linalg.norm(self.engine.rho_grad_new))
        # MPI reduction of gradients
        # self.ob_grad.allreduce()
        return 

    def new_grad_pr(self):
        """
        Compute a new gradient direction according to a Gaussian noise model.

        Note: The negative log-likelihood and local errors are also computed
        here.
        """
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
                f[name] = pod.fw(pod.probe * np.exp(1j * pod.object))
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
                self.pr_grad[pod.pr_view] += 2. * xi * np.exp(1j * pod.object).conj()

            diff_view.error = LLL
            error_dct[dname] = np.array([0, LLL / np.prod(DI.shape), 0])
            LL += LLL

        # MPI reduction of gradients
        self.pr_grad.allreduce()
        parallel.allreduce(LL)

        # Object regularizer
        if self.regularizer:
            # OLD
            # for name, s in self.ob.storages.items():
            #     self.rho_grad.storages[name].data += self.regularizer.grad(
            #         s.data)
            self.rho_grad += self.regularizer.grad(self.rho)
            LL += self.regularizer.LL

        self.LL = LL / self.tot_measpts

        return error_dct

    # NEEDS Updating, while poly_line_pr stays same
    def poly_line_coeffs_rho(self, rho_h, pr_h):
        """
        Compute the coefficients of the polynomial for line minimization
        in direction h for the object
        """

        B = np.zeros((3,), dtype=np.longdouble)
        Brenorm = 1. / self.LL[0]**2

        # try:
        #     iter = self.engine.ptycho.runtime.iter_info[-1]['iteration']
        # except:
        #     iter=0
        # if iter%10 == 0:
        #     self.projector.plot_complex_array(rho_h[26, :, :], title='rho_h passed to poly_line_rho')
        
        omega = self.projector.forward(rho_h)
        self.omega << omega

        i = 0
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

                psi = pod.probe * np.exp(1j * pod.object)   # exit_wave
                f = pod.fw(psi)  

                # Need to change this so omega can be accessed properly
                omega_i = omega[i]
                a = pod.fw(psi*omega_i)
                b = pod.fw(psi*(omega_i**2))

                if A0 is None:
                    A0 = u.abs2(f).astype(np.longdouble)
                    A1 = 2 * np.real(f * a.conj()).astype(np.longdouble)
                    A2 = np.real(f * b.conj()).astype(np.longdouble) + u.abs2(a).astype(np.longdouble)       
                else:
                    A0 += u.abs2(f)
                    A1 += 2 * np.real(f * a.conj())
                    A2 += (np.real(f * b.conj()) + u.abs2(a))

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

            i+=1

        parallel.allreduce(B)

        # Object regularizer
        if self.regularizer:
            # OLD
            # for name, s in self.ob.storages.items():
            #     B += Brenorm * self.regularizer.poly_line_coeffs(
            #         ob_h.storages[name].data, s.data)
            B += Brenorm * self.regularizer.poly_line_coeffs(rho_h, self.rho)

        if np.isinf(B).any() or np.isnan(B).any():
            logger.warning(
                'Warning! inf or nan found! Trying to continue...')
            B[np.isinf(B)] = 0.
            B[np.isnan(B)] = 0.

        self.B = B

        return B

    def poly_line_all_coeffs_rho(self, rho_h, pr_h):
        """
        Compute the coefficients of the polynomial for line minimization
        in direction h for the object
        """

        B = np.zeros((5,), dtype=np.longdouble)
        Brenorm = 1. / self.LL[0]**2
        
        omega = self.projector.forward(rho_h)

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

                psi = pod.probe * np.exp(1j * pod.object)   # exit_wave
                f = pod.fw(psi)  

                omega_i = omega[pod.ob_view] 
                a = pod.fw(psi*omega_i)
                b = pod.fw(psi*(omega_i**2))

                if A0 is None:
                    A0 = u.abs2(f).astype(np.longdouble)
                    A1 = 2 * np.real(f * a.conj()).astype(np.longdouble)
                    A2 = np.real(f * b.conj()).astype(np.longdouble) + u.abs2(a).astype(np.longdouble)       
                else:
                    A0 += u.abs2(f)
                    A1 += 2 * np.real(f * a.conj())
                    A2 += (np.real(f * b.conj()) + u.abs2(a))

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
            B[3] += np.dot(w.flat, (2*A1*A2).flat) * Brenorm
            B[4] += np.dot(w.flat, (A2**2).flat) * Brenorm
            
        parallel.allreduce(B)

        # Object regularizer
        if self.regularizer:
            # OLD
            # for name, s in self.ob.storages.items():
            #     B += Brenorm * self.regularizer.poly_line_coeffs(
            #         ob_h.storages[name].data, s.data)
            B += Brenorm * self.regularizer.poly_line_coeffs(rho_h, self.rho)

        if np.isinf(B).any() or np.isnan(B).any():
            logger.warning(
                'Warning! inf or nan found! Trying to continue...')
            B[np.isinf(B)] = 0.
            B[np.isnan(B)] = 0.

        self.B = B

        return B

    def poly_line_coeffs_pr(self, rho_h, pr_h):
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

                f = pod.fw(pod.probe * np.exp(1j * pod.object))
                a = pod.fw(pr_h[pod.pr_view] * np.exp(1j * pod.object))

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
            # OLD
            # for name, s in self.ob.storages.items():
            #     B += Brenorm * self.regularizer.poly_line_coeffs(
            #         ob_h.storages[name].data, s.data)
            B += Brenorm * self.regularizer.poly_line_coeffs(rho_h, self.rho)

        if np.isinf(B).any() or np.isnan(B).any():
            logger.warning(
                'Warning! inf or nan found! Trying to continue...')
            B[np.isinf(B)] = 0.
            B[np.isnan(B)] = 0.

        self.B = B
        return B


    def poly_line_all_coeffs_pr(self, rho_h, pr_h):
        """
        Compute the coefficients of the polynomial for line minimization
        in direction h for the probe
        """

        B = np.zeros((5,), dtype=np.longdouble)
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
                a = pod.fw(pr_h[pod.pr_view] * np.exp(1j * pod.object))

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
            B[3] += np.dot(w.flat, (2*A1*A2).flat) * Brenorm
            B[4] += np.dot(w.flat, (A2**2).flat) * Brenorm

        parallel.allreduce(B)

        # Object regularizer
        if self.regularizer:
            # OLD
            # for name, s in self.ob.storages.items():
            #     B += Brenorm * self.regularizer.poly_line_coeffs(
            #         ob_h.storages[name].data, s.data)
            B += Brenorm * self.regularizer.poly_line_coeffs(rho_h, self.rho)

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
    def __init__(self, amplitude, axes=[-3, -2, -1]):
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
        ax0, ax1, ax2 = self.axes
        del_xf = u.delxf(x, axis=ax0)
        del_yf = u.delxf(x, axis=ax1)
        del_zf = u.delxf(x, axis=ax2)

        del_xb = u.delxb(x, axis=ax0)
        del_yb = u.delxb(x, axis=ax1)
        del_zb = u.delxb(x, axis=ax2)

        self.delxy = [del_xf, del_yf, del_zf, del_xb, del_yb, del_zb]
        self.g = 2. * self.amplitude*(del_xb + del_yb + del_zb - del_xf - del_yf - del_zf)

        self.LL = self.amplitude * (u.norm2(del_xf)
                               + u.norm2(del_yf)
                               + u.norm2(del_zf)
                               + u.norm2(del_xb)
                               + u.norm2(del_yb)
                               + u.norm2(del_zb))

        return self.g

    def poly_line_coeffs(self, h, x=None):
        ax0, ax1, ax2 = self.axes
        if x is None:
            del_xf, del_yf, del_zf, del_xb, del_yb, del_zb = self.delxy
        else:
            del_xf = u.delxf(x, axis=ax0)
            del_yf = u.delxf(x, axis=ax1)
            del_zf = u.delxf(x, axis=ax2)
            del_xb = u.delxb(x, axis=ax0)
            del_yb = u.delxb(x, axis=ax1)
            del_zb = u.delxb(x, axis=ax2)

        hdel_xf = u.delxf(h, axis=ax0)
        hdel_yf = u.delxf(h, axis=ax1)
        hdel_zf = u.delxf(h, axis=ax2)
        hdel_xb = u.delxb(h, axis=ax0)
        hdel_yb = u.delxb(h, axis=ax1)
        hdel_zb = u.delxb(h, axis=ax2)

        c0 = self.amplitude * (u.norm2(del_xf)
                               + u.norm2(del_yf)
                               + u.norm2(del_zf)
                               + u.norm2(del_xb)
                               + u.norm2(del_yb)
                               + u.norm2(del_zb))

        c1 = 2 * self.amplitude * np.real(np.vdot(del_xf, hdel_xf)
                                          + np.vdot(del_yf, hdel_yf)
                                          + np.vdot(del_zf, hdel_zf)
                                          + np.vdot(del_xb, hdel_xb)
                                          + np.vdot(del_yb, hdel_yb)
                                          + np.vdot(del_zb, hdel_zb))

        c2 = self.amplitude * (u.norm2(hdel_xf)
                               + u.norm2(hdel_yf)
                               + u.norm2(hdel_zf)
                               + u.norm2(hdel_xb)
                               + u.norm2(hdel_yb)
                               + u.norm2(hdel_zb))

        self.coeff = np.array([c0, c1, c2])
        return self.coeff


# def prepare_smoothing_preconditioner(amplitude):
#     """
#     Factory forforfor smoothing preconditioner.
#     """
#     if amplitude == 0.:
#         return None

#     class GaussFilt(object):
#         def __init__(self, sigma):
#             self.sigma = sigma

#         def __call__(self, x):
#             return u.c_gf(x, [0, self.sigma, self.sigma])

#     # from scipy.signal import correlate2d
#     # class HannFilt:
#     #    def __call__(self, x):
#     #        y = np.empty_like(x)
#     #        sh = x.shape
#     #        xf = x.reshape((-1,) + sh[-2:])
#     #        yf = y.reshape((-1,) + sh[-2:])
#     #        for i in range(len(xf)):
#     #            yf[i] = correlate2d(xf[i],
#     #                                np.array([[.0625, .125, .0625],
#     #                                          [.125, .25, .125],
#     #                                          [.0625, .125, .0625]]),
#     #                                mode='same')
#     #        return y

#     if amplitude > 0.:
#         logger.debug(
#             'Using a smooth gradient filter (Gaussian blur - only for ML)')
#         return GaussFilt(amplitude)

#     elif amplitude < 0.:
#         raise RuntimeError('Hann filter not implemented (negative smoothing amplitude not supported)')
#         # logger.debug(
#         #    'Using a smooth gradient filter (Hann window - only for ML)')
#         # return HannFilt()
