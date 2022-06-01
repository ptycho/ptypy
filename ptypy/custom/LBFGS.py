# -*- coding: utf-8 -*-
"""
Limited-memory BFGS reconstruction engine.

This file is part of the PTYPY package.

    :copyright: Copyright 2014 by the PTYPY team, see AUTHORS.
    :license: GPLv2, see LICENSE for details.
"""
import numpy as np
import time
import sys
sys.path.insert(0, "/home/uef75971/ptypy/")

#from ptypy import utils as u
from ptypy.utils.verbose import logger
#from ptypy.utils import parallel
from ptypy.engines.utils import Cnorm2, Cdot
from ptypy.engines import register
from ptypy.engines.ML import ML
from ptypy.core.manager import Full, Vanilla, Bragg3dModel, BlockVanilla, BlockFull, GradFull, BlockGradFull


__all__ = ['LBFGS']


@register()
class LBFGS(ML):
    """
    Limited-memory BFGS reconstruction engine.


    Defaults:

    [name]
    default = LBFGS
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

    [bfgs_memory_size]
    default = 5
    type = int
    lowlim = 2
    help = Number of BFGS updates to store
    """

    SUPPORTED_MODELS = [Full, Vanilla, Bragg3dModel, BlockVanilla, BlockFull, GradFull, BlockGradFull]

    def __init__(self, ptycho_parent, pars=None):
        """
        Limited-memory BFGS reconstruction engine.
        """
        super().__init__(ptycho_parent, pars)

        # Memory of object updates and gradient differences
        self.ob_s = [None]*self.p.bfgs_memory_size
        self.ob_y = [None]*self.p.bfgs_memory_size

        # Memory of probe updates and gradient differences
        self.pr_s = [None]*self.p.bfgs_memory_size
        self.pr_y = [None]*self.p.bfgs_memory_size

        # Other BFGS memories
        self.rho = np.zeros(self.p.bfgs_memory_size)
        self.alpha = np.zeros(self.p.bfgs_memory_size)

        self.ptycho.citations.add_article(
            title='L-BFGS for Ptychography paper',
            author='Fowkes J. and Daurer B.',
            journal='TBA',
            volume=None,
            year=None,
            page=None,
            doi='TBA',
            comment='The L-BFGS reconstruction algorithm',
        )

    def engine_initialize(self):
        """
        Prepare for reconstruction.
        """
        super().engine_initialize()

        # Create containers for memories of updates and gradient differences
        for i in range(self.p.bfgs_memory_size):
            self.ob_s[i] = self.ob.copy(self.ob.ID + '_s' + str(i), fill=0.)
            self.ob_y[i] = self.ob.copy(self.ob.ID + '_y' + str(i), fill=0.)
            self.pr_s[i] = self.pr.copy(self.pr.ID + '_s' + str(i), fill=0.)
            self.pr_y[i] = self.pr.copy(self.pr.ID + '_y' + str(i), fill=0.)



    def engine_prepare(self):
        """
        Last minute initialization, everything, that needs to be recalculated,
        when new data arrives.
        """
        super().engine_prepare()

    def engine_iterate(self, num=1):
        """
        Compute `num` iterations.
        """
        tg = 0.
        tc = 0.
        ta = time.time()
        for it in range(num):
            #######################################
            # Compute new gradient (same as ML)
            #######################################
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

            # probe/object rescaling (not used for now)
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

            # Smoothing preconditioner decay (once per iteration)
            if self.smooth_gradient:
                self.smooth_gradient.sigma *= (1. - self.p.smooth_gradient_decay)
                for name, s in new_ob_grad.storages.items():
                    s.data[:] = self.smooth_gradient(s.data)


            ############################
            # LBFGS Two Loop Recursion
            ############################
            if self.curiter == 0: # Initial steepest-descent step

                # Object steepest-descent step
                self.ob_h -= new_ob_grad

                # Probe steepest-descent step
                new_pr_grad *= self.scale_p_o # probe preconditioning
                self.pr_h -= new_pr_grad

            else: # Two-loop LBFGS recursion

                # Memory index
                mi = min(self.curiter,self.p.bfgs_memory_size)

                # Remember last object update and gradient difference
                self.ob_s[mi-1] << self.ob_h
                self.ob_y[mi-1] << new_ob_grad
                self.ob_y[mi-1] -= self.ob_grad

                # Remember last probe update and gradient difference
                self.pr_h /= np.sqrt(self.scale_p_o) # probe preconditioning
                self.pr_s[mi-1] << self.pr_h
                new_pr_grad *= np.sqrt(self.scale_p_o) # probe preconditioning
                self.pr_y[mi-1] << new_pr_grad
                self.pr_y[mi-1] -= self.pr_grad

                # Compute and store rho
                self.rho[mi-1] = 1. / ( np.real(Cdot(self.ob_y[mi-1], self.ob_s[mi-1]))
                                        + np.real(Cdot(self.pr_y[mi-1], self.pr_s[mi-1])) )
                # BFGS update
                self.ob_h << new_ob_grad
                self.pr_h << new_pr_grad
                # Compute right-hand side of BGFS product
                for i in reversed(range(mi)):
                    self.alpha[i] = self.rho[i]*( np.real(Cdot(self.ob_s[i], self.ob_h))
                                                  + np.real(Cdot(self.pr_s[i], self.pr_h)) )

                    #TODO: support operand * for 'float' and 'Container'
                    # (reusing self.ob_grad here is not efficient)
                    # self.ob_h -= self.alpha[i]*self.ob_y[i]
                    self.ob_grad << self.ob_y[i]
                    self.ob_grad *= self.alpha[i]
                    self.ob_h -= self.ob_grad
                    #TODO: support operand * for 'float' and 'Container'
                    # (reusing self.pr_grad here is not efficient)
                    # self.pr_h -= self.alpha[i]*self.pr_y[i]
                    self.pr_grad << self.pr_y[i]
                    self.pr_grad *= self.alpha[i]
                    self.pr_h -= self.pr_grad

                # Compute centre of BFGS product (scaled identity)
                c_num = ( np.real(Cdot(self.ob_s[mi-1], self.ob_y[mi-1]))
                         + np.real(Cdot(self.pr_s[mi-1], self.pr_y[mi-1])) )
                c_denom = Cnorm2(self.ob_y[mi-1]) + Cnorm2(self.pr_y[mi-1])
                self.ob_h *= (c_num/c_denom)
                self.pr_h *= (c_num/c_denom)


                # Compute left-hand side of BFGS product
                for i in range(mi):
                    beta = self.rho[i]*( np.real(Cdot(self.ob_y[i], self.ob_h))
                                         + np.real(Cdot(self.pr_y[i], self.pr_h)) )
                    #TODO: support operand * for 'float' and 'Container'
                    # (reusing self.ob_grad here is not efficient)
                    # self.ob_h += (self.alpha[i]-beta)*self.ob_s[i]
                    self.ob_grad << self.ob_s[i]
                    self.ob_grad *= (self.alpha[i]-beta)
                    self.ob_h += self.ob_grad

                    #TODO: support operand * for 'float' and 'Container'
                    # (reusing self.pr_grad here is not efficient)
                    # self.pr_h += (self.alpha[i]-beta)*self.pr_s[i]
                    self.pr_grad << self.pr_s[i]
                    self.pr_grad *= (self.alpha[i]-beta)
                    self.pr_h += self.pr_grad

                # Flip step direction for minimisation
                self.ob_h *= -1
                self.pr_h *= np.sqrt(self.scale_p_o) # probe preconditioning
                self.pr_h *= -1

            # update current gradients with new gradients
            self.ob_grad << new_ob_grad
            self.pr_grad << new_pr_grad

            # linesearch (same as ML)
            t2 = time.time()
            B = self.ML_model.poly_line_coeffs(self.ob_h, self.pr_h)
            tc += time.time() - t2


            if np.isinf(B).any() or np.isnan(B).any():
                logger.warning(
                    'Warning! inf or nan found! Trying to continue...')
                B[np.isinf(B)] = 0.
                B[np.isnan(B)] = 0.

            dt = self.ptycho.FType
            self.tmin = dt(-.5 * B[1] / B[2])

            # step update
            self.ob_h *= self.tmin
            self.pr_h *= self.tmin
            self.ob += self.ob_h
            self.pr += self.pr_h

            # Roll memory for overwriting
            if self.curiter >= self.p.bfgs_memory_size:
                self.ob_s.append(self.ob_s.pop(0))
                self.pr_s.append(self.pr_s.pop(0))
                self.ob_y.append(self.ob_y.pop(0))
                self.pr_y.append(self.pr_y.pop(0))
                self.rho = np.roll(self.rho,-1)

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
        Enables modification at the end of each LBFGS iteration.
        """
        pass

    def engine_finalize(self):
        """
        Delete temporary containers.
        """
        super().engine_finalize()

        # Delete containers for memories of updates and gradient differences
        for i in reversed(range(self.p.bfgs_memory_size)):
            del self.ptycho.containers[self.ob_s[i].ID]
            del self.ob_s[i]
            del self.ptycho.containers[self.ob_y[i].ID]
            del self.ob_y[i]
            del self.ptycho.containers[self.pr_s[i].ID]
            del self.pr_s[i]
            del self.ptycho.containers[self.pr_y[i].ID]
            del self.pr_y[i]
