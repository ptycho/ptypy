# -*- coding: utf-8 -*-
"""
Least Squares reconstruction engine.

TODO.

  * Implement other regularizers

This file is part of the PTYPY package.

    :copyright: Copyright 2022 by the PTYPY team, see AUTHORS.
    :license: see LICENSE for details.
"""
import numpy as np
import scipy as sp
import time

from ptypy import utils as u
from ptypy.utils.verbose import logger
from ptypy.utils import parallel
from ptypy.engines.utils import Cnorm2, Cdot
from ptypy.engines import register
from ptypy.engines.base import BaseEngine, PositionCorrectionEngine
from ptypy.core.manager import Full, Vanilla, Bragg3dModel, BlockVanilla, BlockFull, GradFull, BlockGradFull


__all__ = ['LSQML']


@register()
class LSQML(PositionCorrectionEngine):
    """
    Least Squares reconstruction engine.


    Defaults:

    [name]
    default = LSQML
    type = str
    help =
    doc =

    [ML_type]
    default = 'euclid'
    type = str
    help = Likelihood model
    choices = ['gaussian','poisson','euclid']
    doc = One of ‘gaussian’, poisson’ or ‘euclid’.

    """

    SUPPORTED_MODELS = [Full, Vanilla, Bragg3dModel, BlockVanilla, BlockFull, GradFull, BlockGradFull]

    def __init__(self, ptycho_parent, pars=None):
        """
        Least Squares reconstruction engine.
        """
        super(LSQML, self).__init__(ptycho_parent, pars)

        # Instance attributes

        # Object gradient and minimization direction
        self.ob_grad = None
        self.ob_h = None

        # Probe gradient and minimization direction
        self.pr_grad = None
        self.pr_h = None

        # Working variables

        # Object normalisation and buffer
        self.ob_nrm = None
        self.ob_buf = None

        # Probe normalisation and buffer
        self.pr_nrm = None
        self.pr_buf = None

        # Object and Probe step length dicts
        self.ob_tmin = None
        self.pr_tmin = None

        # Other
        self.ML_model = None

        self.ptycho.citations.add_article(
            title='Iterative least-squares solver for generalized maximum-likelihood ptychography',
            author='Odstrčil M., Menzel A. and Guizar-Sicairos M.',
            journal='Optics Express',
            volume=26,
            year=2018,
            page=3108,
            doi='10.1364/OE.26.003108',
            comment='The least squares reconstruction algorithm',
        )

    def engine_initialize(self):
        """
        Prepare for LSQML reconstruction.
        """
        super(LSQML, self).engine_initialize()

        # Object gradient and minimization direction
        self.ob_grad = self.ob.copy(self.ob.ID + '_grad', fill=0.)
        self.ob_h = self.ob.copy(self.ob.ID + '_h', fill=0.)
        # Object normalisation and buffer
        self.ob_nrm = self.ob.copy(self.ob.ID + '_nrm', fill=0., dtype='real')
        self.ob_buf = self.ob.copy(self.ob.ID + '_buf', fill=0.)
        self.ob_tmin = {} # need scalar per named pod

        # Probe gradient and minimization direction
        self.pr_grad = self.pr.copy(self.pr.ID + '_grad', fill=0.)
        self.pr_h = self.pr.copy(self.pr.ID + '_h', fill=0.)
        # Probe normalisation and buffer
        self.pr_nrm = self.pr.copy(self.pr.ID + '_nrm', fill=0., dtype='real')
        self.pr_buf = self.pr.copy(self.pr.ID + '_buf', fill=0.)
        self.pr_tmin = {} # need scalar per named pod

        self._initialize_model()

    def _initialize_model(self):

        # Create noise model
        if self.p.ML_type.lower() == "gaussian":
            raise NotImplementedError('Gaussian noise model not yet implemented')
        elif self.p.ML_type.lower() == "poisson":
            raise NotImplementedError('Poisson noise model not yet implemented')
        elif self.p.ML_type.lower() == "euclid":
            self.ML_model = EuclidModel(self)
        else:
            raise RuntimeError("Unsupported ML_type: '%s'" % self.p.ML_type)

    def engine_prepare(self):
        """
        Last minute initialization, everything, that needs to be recalculated,
        when new data arrives.
        """
        self.ML_model.prepare()

    def engine_iterate(self, num=1):
        """
        Compute `num` iterations.
        """
        tg = 0.
        ts = 0.
        tu = 0.
        for it in range(num):

            ########################
            # Compute new gradient #
            ########################
            t1 = time.time()
            error_dct = self.ML_model.new_grad()
            new_ob_grad, new_pr_grad = self.ob_buf, self.pr_buf
            tg += time.time() - t1

            ##########################
            # Compute next conjugate #
            ##########################
            if self.curiter == 0:
                bt = 0.
            else:
                bt_num = (  (Cnorm2(new_pr_grad)
                             - np.real(Cdot(new_pr_grad, self.pr_grad)))
                          + (Cnorm2(new_ob_grad)
                             - np.real(Cdot(new_ob_grad, self.ob_grad))))

                bt_denom = Cnorm2(self.pr_grad) + Cnorm2(self.ob_grad)

                bt = max(0, bt_num/bt_denom)

            # logger.info('Polak-Ribiere coefficient: %f ' % bt)

            self.ob_grad << new_ob_grad
            self.pr_grad << new_pr_grad

            # Next conjugate
            self.ob_h *= bt
            self.ob_h -= self.ob_grad
            self.pr_h *= bt
            self.ob_h -= self.ob_grad

            # ############################
            # # Compute steepest descent #
            # ############################
            # self.ob_h << new_ob_grad
            # self.ob_h *= -1
            # self.pr_h << new_pr_grad
            # self.pr_h *= -1

            ##########################
            # Average direction (25) #
            ##########################
            self.ob_nrm += 1e-6
            self.ob_h /= self.ob_nrm
            self.pr_nrm += 1e-6
            self.pr_h /= self.pr_nrm

            ########################
            # Compute step lengths #
            ########################
            t2 = time.time()
            self.ML_model.compute_step_lengths()
            ts += time.time() - t2

            ################################
            # Take weighted mean step (27) #
            ################################
            t3 = time.time()
            # compute step
            self.ML_model.new_step()
            # scale step
            self.ob_buf /= self.ob_nrm
            self.pr_buf /= self.pr_nrm
            # take step
            self.ob += self.ob_buf
            self.pr += self.pr_buf
            tu += time.time() - t3

            # ########################
            # # Take unweighted step #
            # ########################
            # t3 = time.time()
            # # compute step
            # self.ML_model.new_step_unweighted()
            # # take step
            # self.ob += self.ob_buf
            # self.pr += self.pr_buf
            # tu += time.time() - t3

            # ################
            # # Take ML step #
            # ################
            # t3 = time.time()
            # B = self.ML_model.poly_line_coeffs()
            # tmin = self.ptycho.FType(-.5 * B[1] / B[2])
            # self.ob_h *= tmin
            # self.pr_h *= tmin
            # self.ob += self.ob_h
            # self.pr += self.pr_h
            # tu += time.time() - t3

            # Position correction
            self.position_update()

            # Allow for customized modifications at the end of each iteration
            self._post_iterate_update()

            # increase iteration counter
            self.curiter +=1

        logger.info('Time spent in gradient calculation: %.2f' % tg)
        logger.info('  ....  in step length calculation: %.2f' % ts)
        logger.info('  ....  in actual step calculation: %.2f' % tu)
        return error_dct

    def _post_iterate_update(self):
        """
        Enables modification at the end of each LSQML iteration.
        """
        pass

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

        # Delete normalisation and buffer containers
        del self.ptycho.containers[self.ob_nrm.ID]
        del self.ob_nrm
        del self.ptycho.containers[self.ob_buf.ID]
        del self.ob_buf
        del self.ptycho.containers[self.pr_nrm.ID]
        del self.pr_nrm
        del self.ptycho.containers[self.pr_buf.ID]
        del self.pr_buf

        # Delete step length dicts
        del self.ob_tmin
        del self.pr_tmin

        # Delete noise model
        del self.ML_model


class BaseModel(object):
    """
    Base class for log-likelihood models.
    """

    def __init__(self, LSQMLengine):
        """
        Core functions for LSQML computation for all noise models.
        """
        self.engine = LSQMLengine

        # Transfer commonly used attributes from LSQML engine
        self.di = self.engine.di
        self.p = self.engine.p
        self.ob_h = self.engine.ob_h
        self.pr_h = self.engine.pr_h
        self.ob_nrm = self.engine.ob_nrm
        self.pr_nrm = self.engine.pr_nrm
        self.ob_buf = self.engine.ob_buf
        self.pr_buf = self.engine.pr_buf
        self.ob_tmin = self.engine.ob_tmin
        self.pr_tmin = self.engine.pr_tmin

        # Create working variables
        self.LL = 0.

    def prepare(self):
        # Useful quantities
        self.tot_measpts = sum(s.data.size
                               for s in self.di.storages.values())

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

        Note: The negative log-likelihood and local errors should also be computed here.
        """
        raise NotImplementedError

    def compute_step_lengths(self):
        """
        Compute optimization step lengths according to the noise model.
        """
        raise NotImplementedError

    def new_step(self):
        """
        Compute new step for probe and object using weighted minimisation directions and step lengths.
        """
        self.ob_buf.fill(0.)
        self.pr_buf.fill(0.)

        # Outer loop: through diffraction patterns
        for dname, diff_view in self.di.views.items():
            if not diff_view.active:
                continue

            # Fourth pod loop: compute new weighted step
            for name, pod in diff_view.pods.items():
                if not pod.active:
                    continue

                # Compute numerator of (27)
                self.ob_buf[pod.ob_view] += self.ob_tmin[name] * self.ob_h[pod.ob_view] * self.ob_nrm[pod.ob_view]
                self.pr_buf[pod.pr_view] += self.pr_tmin[name] * self.pr_h[pod.pr_view] * self.pr_nrm[pod.pr_view]

        # MPI reduction of weighted step
        self.ob_buf.allreduce()
        self.pr_buf.allreduce()

    def new_step_unweighted(self):
        """
        Compute new step for probe and object using minimisation directions and step lengths.
        """
        self.ob_buf.fill(0.)
        self.pr_buf.fill(0.)

        # Outer loop: through diffraction patterns
        for dname, diff_view in self.di.views.items():
            if not diff_view.active:
                continue

            # Fourth pod loop: compute final updates
            for name, pod in diff_view.pods.items():
                if not pod.active:
                    continue

                # Compute new step
                self.ob_buf[pod.ob_view] += self.ob_tmin[name] * self.ob_h[pod.ob_view]
                self.pr_buf[pod.pr_view] += self.pr_tmin[name] * self.pr_h[pod.pr_view]

        # MPI reduction of step
        self.ob_buf.allreduce()
        self.pr_buf.allreduce()

class EuclidModel(BaseModel):
    """
    Euclidean (Amplitude) noise model.
    TODO: feed actual statistical weights instead of using a fixed variance.
    """

    def __init__(self, LSQMLengine):
        """
        Core functions for LSQML computation using a Euclidean model.
        """
        BaseModel.__init__(self, LSQMLengine)

        # Euclidean model requires weights
        # TODO: update this part of the code once actual weights are passed in the PODs
        self.weights = self.engine.di.copy(self.engine.di.ID + '_weights')
        # FIXME: This part needs to be updated once statistical weights are properly
        # supported in the data preparation.
        for name, di_view in self.di.views.items():
            if not di_view.active:
                continue
            self.weights[di_view] = di_view.pod.ma_view.data # just the mask for now
            #self.weights[di_view] = (di_view.pod.ma_view.data
            #                         / (1. + stat_weights/di_view.data))

    def __del__(self):
        """
        Clean up routine
        """
        BaseModel.__del__(self)
        del self.engine.ptycho.containers[self.weights.ID]
        del self.weights

    def new_grad(self):
        """
        Compute a new gradient direction according to a Euclidean noise model.

        Note: The negative log-likelihood and local errors are also computed here.
        """
        self.ob_buf.fill(0.)
        self.pr_buf.fill(0.)
        self.ob_nrm.fill(0.)
        self.pr_nrm.fill(0.)

        # We need an array for MPI
        LL = np.array([0.])
        error_dct = {}

        # Outer loop: through diffraction patterns
        for dname, diff_view in self.di.views.items():
            if not diff_view.active:
                continue

            # Weights and amplitudes for this view
            w = self.weights[diff_view]
            A = np.sqrt(diff_view.data)

            Amodel = np.zeros_like(A)
            f = {}

            # First pod loop: compute total amplitude
            for name, pod in diff_view.pods.items():
                if not pod.active:
                    continue
                f[name] = pod.fw(pod.probe * pod.object)
                Amodel += np.sqrt(u.abs2(f[name]))

            Amodel += 1e-6 # cf Poisson model
            DA = (1. - A / Amodel)

            # Second pod loop: update direction computation
            LLL = np.sum((w * (Amodel - A)**2).astype(np.float64))
            for name, pod in diff_view.pods.items():
                if not pod.active:
                    continue

                # 1. Optimization in reciprocal (fourier) space
                # This is equivalent to applying the modulus constraint

                # Compute noise model gradient (12)
                #rec_grad = 2 * w*DA * f[name]

                # Calculate reciprocal step length (16)
                #rec_step = 0.5

                # Updated exit wave (14) and (18)
                #exit = pod.bw(f[name] - rec_step*rec_grad)

                # Updated exit wave (18), (14), (16), (12)
                #exit = pod.bw(f[name]*(1 - w*DA)) # all-in-one

                # 2. Optimization in real space
                # This is essentially a generalised overlap update

                # Calculate xi (19)
                #xi = exit - pod.probe * pod.object

                # Compute real-space gradients (24)
                #self.ob_buf[pod.ob_view] += -xi * pod.probe.conj()
                #self.pr_buf[pod.pr_view] += -xi * pod.object.conj()

                # ML gradients (equivalent to the above)
                xi = -pod.bw(w*DA * f[name])
                self.ob_buf[pod.ob_view] += -xi * pod.probe.conj()
                self.pr_buf[pod.pr_view] += -xi * pod.object.conj()

                # Store xi for later
                pod.exit = xi

                # Compute normalisations for object and probe
                self.ob_nrm[pod.ob_view] += u.abs2(pod.probe)
                self.pr_nrm[pod.pr_view] += u.abs2(pod.object)

            diff_view.error = LLL
            error_dct[dname] = np.array([0, LLL / np.prod(DA.shape), 0])
            LL += LLL

        # MPI reduction of minimization directions, normalisations, and gradients
        self.ob_buf.allreduce()
        self.pr_buf.allreduce()
        self.ob_nrm.allreduce()
        self.pr_nrm.allreduce()
        parallel.allreduce(LL)

        self.LL = LL / self.tot_measpts

        return error_dct

    def compute_step_lengths(self):
        """
        Compute optimization step lengths according to a Euclidean noise model.
        """

        # Outer loop: through diffraction patterns
        for dname, diff_view in self.di.views.items():
            if not diff_view.active:
                continue

            # Third pod loop: calculate real-space step lengths
            for name, pod in diff_view.pods.items():
                if not pod.active:
                    continue

                # Get xi
                xi = pod.exit

                # Get update directions
                ob_h = self.ob_h[pod.ob_view]
                pr_h = self.pr_h[pod.pr_view]

                # Compute cross-terms
                ob_h_pr = ob_h * pod.probe
                pr_h_ob = pr_h * pod.object

                # Calculate real-space step lengths (22)
                M = np.zeros((2,2), dtype=np.cdouble)
                rhs = np.zeros(2, dtype=np.double)
                M[0,0] = np.sum(u.abs2(ob_h_pr)) + 1e-6
                M[1,1] = np.sum(u.abs2(pr_h_ob)) + 1e-6
                M[0,1] = np.sum(ob_h_pr * pr_h_ob.conj())
                M[1,0] = np.sum(ob_h_pr.conj() * pr_h_ob)
                rhs[0] = np.sum(np.real(xi * ob_h_pr.conj()))
                rhs[1] = np.sum(np.real(xi * pr_h_ob.conj()))
                #self.ob_tmin[name], self.pr_tmin[name] = np.linalg.solve(M, rhs)
                self.ob_tmin[name], self.pr_tmin[name] = sp.linalg.solve(M, rhs, assume_a='her')

                # # Calculate approx real-space step lengths (23)
                # self.ob_tmin[name] = np.sum(np.real(xi * ob_h_pr.conj())) / (np.sum(u.abs2(ob_h_pr)) + 1e-6)
                # self.pr_tmin[name] = np.sum(np.real(xi * pr_h_ob.conj())) / (np.sum(u.abs2(pr_h_ob)) + 1e-6)

    def poly_line_coeffs(self):
        """
        Compute the coefficients of the polynomial for line minimization
        in direction h
        """
        ob_h = self.ob_h
        pr_h = self.pr_h

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

            A0 = np.double(A0) - I

            B[0] += np.dot(w.flat, (A0**2).flat) * Brenorm
            B[1] += np.dot(w.flat, (2 * A0 * A1).flat) * Brenorm
            B[2] += np.dot(w.flat, (A1**2 + 2*A0*A2).flat) * Brenorm

        parallel.allreduce(B)

        return B
