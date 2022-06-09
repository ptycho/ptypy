# -*- coding: utf-8 -*-
"""
Limited-memory BFGS reconstruction engine.

TODO.

  * Implement other regularizers

This file is part of the PTYPY package.

    :copyright: Copyright 2014 by the PTYPY team, see AUTHORS.
    :license: GPLv2, see LICENSE for details.
"""
import numpy as np
import sys
sys.path.insert(0, "/home/uef75971/ptypy/")
import time

from ptypy.custom.LBFGS import LBFGS
from ptypy.engines.ML import ML, BaseModel
# from .projectional_serial import serialize_array_access
from ptypy.accelerate.base.engines.ML_serial import ML_serial, GaussianModel
from ptypy.accelerate.base.engines.projectional_serial import serialize_array_access
from ptypy import utils as u
from ptypy.utils.verbose import logger, log
from ptypy.utils import parallel
from ptypy.engines.utils import Cnorm2, Cdot
from ptypy.engines import register
from ptypy.accelerate.base.kernels import GradientDescentKernel, AuxiliaryWaveKernel, PoUpdateKernel, PositionCorrectionKernel
from ptypy.accelerate.base import address_manglers


__all__ = ['LBFGS_serial']

@register()
class LBFGS_serial(LBFGS, ML_serial):

    def __init__(self, ptycho_parent, pars=None):
        """
        Limited-memory BFGS reconstruction engine.
        """
        super(LBFGS_serial, self).__init__(ptycho_parent, pars)

        self.cdotr_ob_ys = [0] * self.p.bfgs_memory_size
        self.cdotr_pr_ys = [0] * self.p.bfgs_memory_size
        self.cdotr_ob_sh = [0] * self.p.bfgs_memory_size
        self.cdotr_pr_sh = [0] * self.p.bfgs_memory_size
        self.cdotr_ob_yh = [0] * self.p.bfgs_memory_size
        self.cdotr_pr_yh = [0] * self.p.bfgs_memory_size
        self.cn2_ob_y = [0] * self.p.bfgs_memory_size
        self.cn2_pr_y = [0] * self.p.bfgs_memory_size

    def engine_initialize(self):
        """
        Prepare for reconstruction.
        """
        super(LBFGS_serial, self).engine_initialize()

    def _initialize_model(self):
        # Create noise model
        if self.p.ML_type.lower() == "gaussian":
            self.ML_model = GaussianModel(self)
        elif self.p.ML_type.lower() == "poisson":
            raise NotImplementedError('Poisson norm model not yet implemented')
        elif self.p.ML_type.lower() == "euclid":
            raise NotImplementedError('Euclid norm model not yet implemented')
        else:
            raise RuntimeError("Unsupported ML_type: '%s'" % self.p.ML_type)

    def engine_prepare(self):
        super(LBFGS_serial, self).engine_prepare()

    def _get_smooth_gradient(self, data, sigma):
        return self.smooth_gradient(data)

    def _replace_ob_grad(self):
        new_ob_grad = self.ob_grad_new
        # Smoothing preconditioner
        if self.smooth_gradient:
            self.smooth_gradient.sigma *= (1. - self.p.smooth_gradient_decay)
            for name, s in new_ob_grad.storages.items():
                s.data[:] = self._get_smooth_gradient(s.data, self.smooth_gradient.sigma)

        norm = Cnorm2(new_ob_grad)
        dot = np.real(Cdot(new_ob_grad, self.ob_grad))

        return norm, dot

    def _replace_pr_grad(self):
        new_pr_grad = self.pr_grad_new
        # probe support
        if self.p.probe_update_start <= self.curiter:
            # Apply probe support if needed
            for name, s in new_pr_grad.storages.items():
                self.support_constraint(s)
        else:
            new_pr_grad.fill(0.)

        norm = Cnorm2(new_pr_grad)
        dot = np.real(Cdot(new_pr_grad, self.pr_grad))

        return norm, dot

    def _replace_ob_pr_ysh(self, mi):
        self.cdotr_ob_ys[mi-1] = np.real(Cdot(self.ob_y[mi-1],
            self.ob_s[mi-1]))
        self.cdotr_pr_ys[mi-1] = np.real(Cdot(self.pr_y[mi-1],
            self.pr_s[mi-1]))
        self.cn2_ob_y[mi-1] = Cnorm2(self.ob_y[mi-1])
        self.cn2_pr_y[mi-1] = Cnorm2(self.pr_y[mi-1])

        for i in reversed(range(mi)):
            self.cdotr_ob_sh[i] = np.real(Cdot(self.ob_s[i], self.ob_h))
            self.cdotr_pr_sh[i] = np.real(Cdot(self.pr_s[i], self.pr_h))

    def _replace_ob_pr_yh(self, mi):
        for i in range(mi):
            self.cdotr_ob_yh[i] = np.real(Cdot(self.ob_y[i], self.ob_h))
            self.cdotr_pr_yh[i] = np.real(Cdot(self.pr_y[i], self.pr_h))

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
            tg += time.time() - t1

            cn2_new_pr_grad, cdotr_pr_grad = self._replace_pr_grad()
            cn2_new_ob_grad, cdotr_ob_grad = self._replace_ob_grad()

            # probe/object rescaling
            if self.p.scale_precond:
                if cn2_new_pr_grad > 1e-5:
                    scale_p_o = (self.p.scale_probe_object * cn2_new_ob_grad
                                 / cn2_new_pr_grad)
                else:
                    scale_p_o = self.p.scale_probe_object
                if self.scale_p_o is None:
                    self.scale_p_o = scale_p_o
                else:
                    self.scale_p_o = self.scale_p_o ** self.scale_p_o_memory
                    self.scale_p_o *= scale_p_o ** (1 - self.scale_p_o_memory)
                logger.debug('Scale P/O: %6.3g' % scale_p_o)
            else:
                self.scale_p_o = self.p.scale_probe_object

            ############################
            # LBFGS Two Loop Recursion
            ############################
            if self.curiter == 0: # Initial steepest-descent step

                # Object steepest-descent step
                self.ob_h -= self.ob_grad_new

                # Probe steepest-descent step
                self.pr_grad_new *= self.scale_p_o # probe preconditioning
                self.pr_h -= self.pr_grad_new

            else: # Two-loop LBFGS recursion

                # Memory index
                mi = min(self.curiter, self.p.bfgs_memory_size)

                # Remember last object update and gradient difference
                self.ob_s[mi-1] << self.ob_h
                self.ob_y[mi-1] << self.ob_grad_new
                self.ob_y[mi-1] -= self.ob_grad

                # Remember last probe update and gradient difference
                self.pr_h /= np.sqrt(self.scale_p_o) # probe preconditioning
                self.pr_s[mi-1] << self.pr_h
                self.pr_grad_new *= np.sqrt(self.scale_p_o) # probe preconditioning
                self.pr_y[mi-1] << self.pr_grad_new
                self.pr_y[mi-1] -= self.pr_grad

                # BFGS update
                self.ob_h << self.ob_grad_new
                self.pr_h << self.pr_grad_new

                # Compute and store rho
                self._replace_ob_pr_ysh(mi)
                self.rho[mi-1] = 1. / (self.cdotr_ob_ys[mi-1] +
                        self.cdotr_pr_ys[mi-1])

                # Compute right-hand side of BGFS product
                for i in reversed(range(mi)):
                    self.alpha[i] = self.rho[i] * (self.cdotr_ob_sh[i] +
                            self.cdotr_pr_sh[i])

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
                c_num = self.cdotr_ob_ys[mi-1] + self.cdotr_pr_ys[mi-1]
                c_denom = self.cn2_ob_y[mi-1] + self.cn2_pr_y[mi-1]
                gamma = c_num/c_denom
                self.ob_h *= gamma
                self.pr_h *= gamma

                # Compute left-hand side of BFGS product
                self._replace_ob_pr_yh(mi)
                for i in range(mi):
                    beta = self.rho[i] * (self.cdotr_ob_yh[i] +
                            self.cdotr_pr_yh[i])


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
            self.ob_grad << self.ob_grad_new
            self.pr_grad << self.pr_grad_new

            self.cn2_ob_grad = cn2_new_ob_grad
            self.cn2_pr_grad = cn2_new_pr_grad

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
            self.curiter += 1

        logger.info('Time spent in gradient calculation: %.2f' % tg)
        logger.info('  ....  in coefficient calculation: %.2f' % tc)
        return error_dct  # np.array([[self.ML_model.LL[0]] * 3])

    def engine_finalize(self):
        super(LBFGS_serial, self).engine_finalize()
