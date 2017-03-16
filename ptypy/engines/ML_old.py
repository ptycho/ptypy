# -*- coding: utf-8 -*-
"""
Maximum Likelihood reconstruction engine.

TODO:
 * Implement other regularizers

This file is part of the PTYPY package.

    :copyright: Copyright 2014 by the PTYPY team, see AUTHORS.
    :license: GPLv2, see LICENSE for details.
"""
import numpy as np
from .. import utils as u
from ..utils.verbose import logger
from ..utils import parallel
from engine_utils import Cnorm2, Cdot
from . import BaseEngine
#from .. import core

__all__=['ML']
import warnings
warnings.simplefilter('always', DeprecationWarning)
warnings.warn('This module is deprecated and will be removed from the package on 30/11/16',DeprecationWarning)
DEFAULT = u.Param(
    ML_type = 'gaussian',
    floating_intensities = False,
    intensity_renormalization = 1.,
    reg_del2 = False,
    reg_del2_amplitude = .01,
    smooth_gradient = 0,
    scale_precond = False,
    scale_probe_object = 1.


    #overlap_converge_factor = .1,
    #overlap_max_iterations = 10,
    #probe_inertia = 1e-9,               # Portion of probe that is kept from iteraiton to iteration, formally cfact
    #object_inertia = 1e-4,              # Portion of object that is kept from iteraiton to iteration, formally DM_smooth_amplitude
    #obj_smooth_std = None,              # Standard deviation for smoothing of object between iterations
    #clip_object = None,                 # None or tuple(min,max) of desired limits of the object modulus
)


class ML(BaseEngine):# pragma: no cover
    
    DEFAULT = DEFAULT

    def __init__(self, ptycho_parent, pars=None):
        """
        Maximum likelihood reconstruction engine.
        """
        if pars is None:
            pars = DEFAULT.copy()
        super(ML, self).__init__(ptycho_parent, pars)
        
    def engine_initialize(self):
        """
        Prepare for ML reconstruction.
        """
        # Container for the "errors"
        #self.error = np.

        # Object gradient and minimization direction
        self.ob_grad = self.ob.copy(self.ob.ID+'_grad',fill=0.) # Object gradient
        self.ob_h = self.ob.copy(self.ob.ID+'_h',fill=0.) # Object minimization direction

        # Probe gradient
        self.pr_grad = self.pr.copy(self.pr.ID+'_grad',fill=0.) # Probe gradient
        self.pr_h = self.pr.copy(self.pr.ID+'_h',fill=0.) # Probe minimization direction
   
        # Create noise model
        if self.p.ML_type.lower() == "gaussian":
            self.ML_model = ML_Gaussian(self)
        elif self.p.ML_type.lower() == "poisson":
            self.ML_model = ML_Gaussian(self)
        elif self.p.ML_type.lower() == "euclid":
            self.ML_model = ML_Gaussian(self)
        else:
            raise RuntimeError("Unsupported ML_type: '%s'" % self.p.ML_type)
            
        # Other options
        self.smooth_gradient = prepare_smoothing_preconditioner(self.p.smooth_gradient)

    def engine_prepare(self):
        """
        last minute initialization, everything, that needs to be recalculated, when new data arrives
        """     
        #- # fill object with coverage of views
        #- for name,s in self.ob_viewcover.S.iteritems():
        #-    s.fill(s.get_view_coverage())
        pass
        
        
    def engine_iterate(self, num=1):
        """
        Compute `num` iterations.
        """
        ########################
        # Compute new gradient
        ########################
        for it in range(num):
            new_ob_grad, new_pr_grad, error_dct = self.ML_model.new_grad() 
    
            if (self.p.probe_update_start <= self.curiter):
                # Apply probe support if needed
                for name,s in new_pr_grad.S.iteritems():
                    support = self.probe_support.get(name)
                    if support is not None: 
                        s.data *= support
            else:
                new_pr_grad.fill(0.)
    
            # Smoothing preconditioner
            # !!! Lets make this consistent with the smoothing already done in DM
            #if self.smooth_gradient:
            #    for name,s in new_ob_grad.S.iteritems()
            #    s.data[:] = self.smooth_gradient(s.data)
    
            # probe/object rescaling
            if self.p.scale_precond:
                scale_p_o = self.p.scale_probe_object * Cnorm2(new_ob_grad) / Cnorm2(new_pr_grad)
                logger.debug('Scale P/O: %6.3g' % scale_p_o)
            else:
                scale_p_o = self.p.scale_probe_object
    
            ############################
            # Compute next conjugate
            ############################
            if self.curiter == 0:
                bt = 0.
            else:
                bt_num = scale_p_o * ( Cnorm2(new_pr_grad) - np.real(Cdot(new_pr_grad, self.pr_grad))) +\
                                     ( Cnorm2(new_ob_grad) - np.real(Cdot(new_ob_grad, self.ob_grad))) 
                bt_denom = scale_p_o * Cnorm2(self.pr_grad) + Cnorm2(self.ob_grad) 
    
                bt = max(0,bt_num/bt_denom)
    
            #verbose(3,'Polak-Ribiere coefficient: %f ' % bt)
    
            # It would be nice to have something more elegant than the following
            for name,s in self.ob_grad.S.iteritems():
                s.data[:] = new_ob_grad.S[name].data
            for name,s in self.pr_grad.S.iteritems():
                s.data[:] = new_pr_grad.S[name].data
            
            # 3. Next conjugate
            for name,s in self.ob_h.S.iteritems():
                s.data *= bt
                s.data -= self.ob_grad.S[name].data
                
            for name,s in self.pr_h.S.iteritems():
                s.data *= bt
                s.data -= scale_p_o * self.pr_grad.S[name].data
                
            # 3. Next conjugate
            #ob_h = self.ob_h
            #ob_h *= bt
            
            # Smoothing preconditioner not implemented.
            #if self.smooth_gradient:
            #    ob_h -= object_smooth_filter(grad_obj)
            #else:
            #    ob_h -= ob_grad
            
            #ob_h -= ob_grad
            #pr_h *= bt
            #pr_h -= scale_p_o * pr_grad
            
            # Minimize - for now always use quadratic approximation (i.e. single Newton-Raphson step)
            # In principle, the way things are now programmed this part could be iterated over in
            # a real NR style.
            B = self.ML_model.poly_line_coeffs(self.ob_h, self.pr_h)
    
            if np.isinf(B).any() or np.isnan(B).any():
                print 'Warning! inf or nan found! Trying to continue...'
                B[np.isinf(B)] = 0.
                B[np.isnan(B)] = 0.
                
            tmin = -.5*B[1]/B[2]
    
            for name,s in self.ob.S.iteritems():
                s.data += tmin*self.ob_h.S[name].data
            for name,s in self.pr.S.iteritems():
                s.data += tmin*self.pr_h.S[name].data
            # Newton-Raphson loop would end here

        return error_dct  #np.array([[self.ML_model.LL[0]] * 3]) 

    def engine_finalize(self):
        """
        Delete temporary containers.
        """
        del self.ptycho.containers[self.ob_grad.ID]
        del self.ob_grad
        del self.ptycho.containers[self.ob_h.ID]
        del self.ob_h
        del self.ptycho.containers[self.pr_grad.ID]
        del self.pr_grad
        del self.ptycho.containers[self.pr_h.ID]
        del self.pr_h
               
class ML_Gaussian(object):
    """
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
        self.pr = self.engine.pr

        # Create working variables
        self.ob_grad = self.engine.ob.copy(self.ob.ID+'_ngrad',fill=0.) # New object gradient
        self.pr_grad = self.engine.pr.copy(self.pr.ID+'_ngrad',fill=0.) # New probe gradient
        self.LL = 0.

        # Gaussian model requires weights
        self.weights = self.engine.di.copy(self.engine.di.ID+'_weights')
        for name,di_view in self.di.V.iteritems():
            if not di_view.active: continue
            self.weights[di_view] = di_view.pod.ma_view.data / (1. + di_view.data)

        # Useful quantities
        self.tot_measpts = len(self.di.V)
        self.tot_power = sum(s.tot_power for s in self.di.S.values())

        # Prepare regularizer
        if self.p.reg_del2:
            obj_Npix = self.ob.size
            expected_obj_var = obj_Npix / self.tot_power  # Poisson
            reg_rescale  = self.tot_measpts / (8. * obj_Npix * expected_obj_var) 
            logger.debug('Rescaling regularization amplitude using the Poisson distribution assumption.')
            logger.debug('Factor: %8.5g' % reg_rescale)
            reg_del2_amplitude = self.p.reg_del2_amplitude * reg_rescale
            self.regularizer = Regul_del2(amplitude=reg_del2_amplitude)
        else:
            self.regularizer = None

    def __del__(self):
        """
        Clean up routine
        """
        # Delete containers
        del self.engine.ptycho.containers[self.weights.ID]
        del self.weights
        del self.engine.ptycho.containers[self.ob_grad.ID]
        del self.ob_grad
        del self.engine.ptycho.containers[self.pr_grad.ID]
        del self.pr_grad
        
        # Remove working attributes
        for name,diff_view in self.di.V.iteritems():
            if not diff_view.active: continue
            try:
                del diff_view.float_intens_coeff
                del diff_view.error
            except:
                pass

    def new_grad(self):
        """
        Compute a new gradient direction according to a Gaussian noise model.

        Note: The negative log-likelihood and local errors are also computed
        here.
        """
        ob_grad = self.ob_grad
        pr_grad = self.pr_grad
        ob_grad.fill(0.)
        pr_grad.fill(0.)
        
        LL = np.array([0.]) # We need an array for MPI
        error_dct={}

        # Outer loop: through diffraction patterns
        for dname,diff_view in self.di.V.iteritems():
            if not diff_view.active: continue
            
            # Weights and intensities for this view
            w = self.weights[diff_view]
            I = diff_view.data
            
            Imodel = np.zeros_like(I)
            f = {}
            
            # First pod loop: compute total intensity
            for name,pod in diff_view.pods.iteritems():
                if not pod.active: continue
                f[name] = pod.fw(pod.probe*pod.object)
                Imodel += u.abs2(f[name])
        
            # Floating intensity option
            if self.p.floating_intensities:
                diff_view.float_intens_coeff = (w * Imodel * I).sum() / (w * Imodel**2).sum()
                Imodel *= diff_view.float_intens_coeff 
            
            DI = Imodel - I

            # Second pod loop: gradients computation
            LLL = np.sum((w * DI**2).astype(np.float64))
            #print LLL
            for name,pod in diff_view.pods.iteritems():
                if not pod.active: continue
                xi = pod.bw(w*DI*f[name])
                ob_grad[pod.ob_view] += 2. * xi * pod.probe.conj()
                pr_grad[pod.pr_view] += 2. * xi * pod.object.conj()

                # Negative log-likelihood term
                #LLL += (w * DI**2).sum()

            #LLL 
            diff_view.error = LLL
            error_dct[dname]= np.array([0,LLL / np.prod(DI.shape),0])
            LL += LLL

        # MPI reduction of gradients
        for name,s in ob_grad.S.iteritems():
            parallel.allreduce(s.data)
        for name,s in pr_grad.S.iteritems():
            parallel.allreduce(s.data)
        parallel.allreduce(LL)
        
        # Object regularizer
        if self.regularizer:
            for name,s in self.ob.S.iteritems():
                ob_grad.S[name].data += self.regularizer.grad(s.data)

        self.LL = LL / self.tot_measpts
        
        return ob_grad, pr_grad, error_dct

    def poly_line_coeffs(self, ob_h, pr_h):
        """
        Compute the coefficients of the polynomial for line minimization
        in direction h
        """
        
        B = np.zeros((3,))
        Brenorm = 1./ self.LL[0]**2
        
        # Outer loop: through diffraction patterns
        for dname,diff_view in self.di.V.iteritems():
            if not diff_view.active: continue
            
            # Weights and intensities for this view
            w = self.weights[diff_view]
            I = diff_view.data

            A0 = None
            A1 = None
            A2 = None
            
            for name,pod in diff_view.pods.iteritems():
                if not pod.active: continue
                f = pod.fw(pod.probe*pod.object)
                a = pod.fw(pod.probe * ob_h[pod.ob_view] + pr_h[pod.pr_view] * pod.object)
                b = pod.fw(pr_h[pod.pr_view] * ob_h[pod.ob_view])
    
                if A0 is None: 
                    A0 = u.abs2(f)
                    A1 = 2*np.real(f*a.conj())
                    A2 = 2*np.real(f*b.conj()) + u.abs2(a)
                else:
                    A0 += u.abs2(f)
                    A1 += 2*np.real(f*a.conj())
                    A2 += 2*np.real(f*b.conj()) + u.abs2(a)
                    
            if self.p.floating_intensities:
                A0 *= diff_view.float_intens_coeff
                A1 *= diff_view.float_intens_coeff
                A2 *= diff_view.float_intens_coeff
            A0 -= I
    
            B[0] += np.dot(w.flat,(A0**2).flat) * Brenorm
            B[1] += np.dot(w.flat,(2*A0*A1).flat) * Brenorm
            B[2] += np.dot(w.flat,(A1**2 + 2*A0*A2).flat) * Brenorm
 
        parallel.allreduce(B)

        # Object regularizer
        if self.regularizer:
            for name,s in self.ob.S.iteritems():
                B += Brenorm * self.regularizer.poly_line_coeffs(ob_h.S[name].data, s.data)

        self.B = B
        
        return B

# Regul class does not exist, replace by objectclass
#class Regul_del2(Regul):
class Regul_del2(object):
    """\
    Squared gradient regularizer (Gaussian prior).
    This class applies to any numpy array.
    """
    def __init__(self, amplitude, axes=[-2,-1]):
        #Regul.__init__(self, axes)
        self.axes = axes
        self.amplitude = amplitude
        self.delxy = None        
        
    def grad(self, x):
        """
        Compute and return the regularizer gradient given the array x.
        """
        ax0,ax1 = self.axes
        del_xf = u.delxf(x,axis=ax0)
        del_yf = u.delxf(x,axis=ax1)
        del_xb = u.delxb(x,axis=ax0)
        del_yb = u.delxb(x,axis=ax1)

        self.delxy = [del_xf, del_yf, del_xb, del_yb]
        self.g = 2. * self.amplitude*(del_xb + del_yb - del_xf - del_yf)

        return self.g
        
    def poly_line_coeffs(self, h, x=None):
        ax0,ax1 = self.axes
        if x is None:
            del_xf,del_yf,del_xb,del_yb = self.delxy
        else:
            del_xf = u.delxf(x,axis=ax0)
            del_yf = u.delxf(x,axis=ax1)
            del_xb = u.delxb(x,axis=ax0)
            del_yb = u.delxb(x,axis=ax1)
            
        hdel_xf = u.delxf(h,axis=ax0)
        hdel_yf = u.delxf(h,axis=ax1)
        hdel_xb = u.delxb(h,axis=ax0)
        hdel_yb = u.delxb(h,axis=ax1)
        
        c0 = self.amplitude * (u.norm2(del_xf) + u.norm2(del_yf) + u.norm2(del_xb) + u.norm2(del_yb))
        c1 = 2 * self.amplitude * np.real(np.vdot(del_xf, hdel_xf) + np.vdot(del_yf, hdel_yf) +\
                                          np.vdot(del_xb, hdel_xb) + np.vdot(del_yb, hdel_yb))
        c2 = self.amplitude * (u.norm2(hdel_xf) + u.norm2(hdel_yf) + u.norm2(hdel_xb) + u.norm2(hdel_yb))
        
        self.coeff = np.array([c0,c1,c2])
        return self.coeff
    
def prepare_smoothing_preconditioner(amplitude):
    """\
    Factory for smoothing preconditioner.
    """
    if amplitude == 0.: return None

    class GaussFilt:
        def __init__(self,sigma):
            self.sigma = sigma
        def __call__(self,x):
            y = np.empty_like(x)
            sh = x.shape
            xf = x.reshape((-1,)+sh[-2:])
            yf = y.reshape((-1,)+sh[-2:])
            for i in range(len(xf)):
                yf[i] = gaussian_filter(xf[i], self.sigma)
            return y

    from scipy.signal import correlate2d
    class HannFilt:
        def __call__(self,x):
            y = np.empty_like(x)
            sh = x.shape
            xf = x.reshape((-1,)+sh[-2:])
            yf = y.reshape((-1,)+sh[-2:])
            for i in range(len(xf)):
                yf[i] = correlate2d(xf[i], np.array([[.0625, .125, .0625], [.125, .25, .125], [.0625, .125, .0625]]), mode='same')
            return y

    if object_smooth_gradient > 0.:
        logger.debug('Using a smooth gradient filter (Gaussian blur - only for ML)')
        return GaussFilt(object_smooth_gradient)

    elif object_smooth_gradient < 0.:
        logger.debug('Using a smooth gradient filter (Hann window - only for ML)')
        return HannFilt()



