# -*- coding: utf-8 -*-
"""
Maximum Likelihood Tomographic Ptychography reconstruction engine.

TODO.

  * Implement other ML models (Poisson/Euclid)

This file is part of the PTYPY package.

    :copyright: Copyright 2024 by the PTYPY team, see AUTHORS.
    :license: see LICENSE for details.
"""
import numpy as np
import time
import h5py
import subprocess

from .. import utils as u
from ..utils.verbose import logger
from ..utils import parallel
from ..engines.utils import Cnorm2, Cdot
from ..engines import register
from ..engines.base import BaseEngine, PositionCorrectionEngine
from ..core.manager import Full, Vanilla, Bragg3dModel, BlockVanilla, BlockFull, GradFull, BlockGradFull
from ..utils.tomo import AstraTomoWrapperViewBased
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage import shift
from ..engines.utils import Cnorm2, Cdot, reduce_dimension


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

    [angles]
    default = None
    type = str
    help = Tomography angles
    doc = Path to the tomography angles.

    [shifts]
    default = None
    type = str
    help = Tomography angle shifts
    doc = Path to the tomography angle shifts (if provided).

    [init_vol_zero]
    default = True
    type = bool
    help = Zero initial volume
    doc = Start with a zero initial volume for the reconstruction.

    [init_vol_real]
    default = None
    type = str
    help = Initial volume (real part)
    doc = Path to real part of the starting volume for the reconstruction.

    [init_vol_imag]
    default = None
    type = str
    help = Initial volume (imaginary part)
    doc = Path to imaginary part of the starting volume for the reconstruction.

    [init_vol_blur]
    default = False
    type = bool
    help = Blur initial volume
    doc = Apply Gaussian blur to initial volume for the reconstruction.

    [init_vol_blur_sigma]
    default = 2.5
    type = float
    lowlim = 0.0
    help = StdDev for initial volume blur
    doc = Standard deviation for the initial volume Gaussian blur.

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

    [weight_gradient]
    default = False
    type = bool
    help = Coverage gradient weights
    doc = Weight the gradient based on the view coverage.

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

    [OPR]
    default = False
    type = bool
    help = Whether to enable orthogonal probe modes
    doc = This parameter enables orthogonal probe modes

    [OPR_modes]
    default = 1
    type = int
    lowlim = 1
    help = Number of orthogonal probe modes to use
    doc = This parameter sets the number of orthogonal probe modes to use

    [OPR_method]
    default = `second`
    type = str
    help = Type of OPR method to use
    choices = ['first','second']
    doc = One of ‘first’ or ‘second’.

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
        super(MLPtychoTomo, self).__init__(ptycho_parent, pars)

        # Instance attributes

        # FIXME: remove debug printing
        self.DEBUG = False

        # Volume
        self.rho = None

        # Projected volume
        self.projected_rho = None

        # Volume gradient
        self.rho_grad = None

        # Volume minimization direction
        self.rho_h = None

        # Probe gradient
        self.pr_grad = None

        # Probe minimization direction
        self.pr_h = None

        # Working variables
        # Volume gradient
        self.rho_grad_new = None

        # Probe gradient
        self.pr_grad_new = None

        # Tomography projectors
        self.projector = None
        self.projector2 = None

        # View coverage
        self.coverage = None

        # Other
        self.nangles = None
        self.tmin_rho = None
        self.tmin_pr = None
        self.ML_model = None
        self.smooth_gradient = None
        self.downsample = None

        self.omega = None

        # Get volume size
        self.pshape = list(self.ptycho.obj.S.values())[0].data.shape[-1]
        if self.DEBUG:
            print('self.pshape:   ', self.pshape)

        # Load tomography angles and create angles dictionary
        # FIXME: there should be a more efficient way to do this
        angles = np.load(self.p.angles)
        self.nangles = len(angles)
        self.angles_dict = {}
        for i,k in enumerate(self.ptycho.obj.S):
            self.angles_dict[k] = angles[i]

        # Load angle shifts if provided and create dictionary
        self.shifts_per_angle = None
        if self.p.shifts is not None:
            self.shifts_per_angle = {}
            self.shifts = np.load(self.p.shifts)
            for i,k in enumerate(self.ptycho.obj.S):
                self.shifts_per_angle[k]  =  self.shifts[:,i]  # dx, dy

        # FIXME: update with paper
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

        # Start with 2 x downsampled volume
        self.downsample = 2
        self.pshape = int(self.pshape / self.downsample)

        # Initialise volume gradient and minimization direction              # OLD using containers
        self.rho_grad = np.zeros(3*(self.pshape,), dtype=np.complex64)       # self.ob.copy(self.ob.ID + '_grad', fill=0.)
        self.rho_grad_new = np.zeros(3*(self.pshape,), dtype=np.complex64)   # self.ob.copy(self.ob.ID + '_grad_new', fill=0.)
        self.rho_h = np.zeros(3*(self.pshape,), dtype=np.complex64)          # self.ob.copy(self.ob.ID + '_h', fill=0.)

        # This is needed in poly_line_coeffs_rho
        self.omega = self.ex

        # Initialise volume
        if self.p.init_vol_zero: # starting from zero volume
            rho_real = np.zeros(3*(self.pshape,), dtype=np.complex64)
            rho_imag = np.zeros(3*(self.pshape,), dtype=np.complex64)
        # FIXME: allow passing in initial volume
        #else: # starting from given volume
        #    rho_real = np.load(self.p.init_vol_real)
        #    rho_imag = np.load(self.p.init_vol_imag)
        # FIXME: end allow passing in initial volume
        if self.p.init_vol_blur: # gaussian blur initial volume
            rho_real = gaussian_filter(rho_real, sigma=self.p.init_vol_blur_sigma)
            rho_imag = gaussian_filter(rho_imag, sigma=self.p.init_vol_blur_sigma)
        self.rho = rho_real + 1j * rho_imag

        # Initialise coverage mask
        if self.p.weight_gradient:
             self.coverage = list(self.ptycho.obj.S.values())[0].get_view_coverage()
             self.coverage = np.squeeze(np.real(self.coverage)) # extract
             # FIXME: parameterise coverage mask
             # Simulated data
             views_threshold = 25
             filter_sigma = 1
             # Real data
             #views_threshold = 50
             #filter_sigma = 2.5
             # FIXME: end parametrise coverage
             self.coverage[self.coverage <= views_threshold] = 0 # threshold zero
             self.coverage[self.coverage > views_threshold] = 1  # threshold one
             self.coverage = gaussian_filter(self.coverage, sigma=filter_sigma) # smooth
             #np.save('coverage_mask', self.coverage)

        # Initialise stacked views
        stacked_views = np.array([v.data for v in self.ptycho.obj.views.values() if v.pod.active])
        self.projected_rho = np.zeros_like((stacked_views), dtype=np.complex64)

        # Initialise probe gradient and minimization direction
        self.pr_grad = self.pr.copy(self.pr.ID + '_grad', fill=0.)
        self.pr_grad_new = self.pr.copy(self.pr.ID + '_grad_new', fill=0.)
        self.pr_h = self.pr.copy(self.pr.ID + '_h', fill=0.)

        # Initialise step sizes
        self.tmin_rho = 1.
        self.tmin_pr = 1.

        # Initialise smoothing preconditioner for volume
        self.smooth_gradient = prepare_smoothing_preconditioner(
            self.p.smooth_gradient)

        # Initialise ASTRA projector for volume
        self.projector = AstraTomoWrapperViewBased (
            obj=self.ptycho.obj,
            vol=np.zeros(3*(self.pshape*self.downsample,), dtype=np.complex64),
            angles=self.angles_dict,
            shifts=self.shifts_per_angle,
            obj_is_refractive_index=False,
            mask_threshold=35
            )

        # Initialise 2 x downsampled ASTRA projector
        self.projector2 = AstraTomoWrapperViewBased (
            obj=self.ptycho.obj,
            vol=self.rho,
            angles=self.angles_dict,
            shifts=self.shifts_per_angle,
            obj_is_refractive_index=False,
            mask_threshold=35,
            downsample=2
            )

        # Initialise 2 x downsampled projected volume and update views
        self.projected_rho = self.projector2.forward(self.rho)
        self.update_views()

        # Initialise ML noise model
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
        can be retrieved from pod.object.
        """
        for i, (k,v) in enumerate([(i,v) for i,v in self.ptycho.obj.views.items() if v.pod.active]):
            v.data[:] = self.projected_rho[i]


    def engine_iterate(self, num=1):
        """
        Compute `num` iterations.
        """
        tg = 0.
        tc = 0.
        ta = time.time()
        reset_cg = False
        for it in range(num):

            # Switch to upsampled volume
            if self.curiter == 24: # curiter starts at 0
                self.downsample = 1
                self.projector2.plot_vol(self.rho, title='before')
                self.projected_rho = self.projector2.forward(self.rho) #FIXME: should this be updated in new_grad?
                self.rho = self.projector.backward(np.moveaxis(self.projected_rho,1,0))
                self.rho_h = np.zeros_like(self.rho, dtype=np.complex64)
                self.projector.plot_vol(self.rho, title='after')
                reset_cg = True # as rho gradients differ in size

            ########################
            # Compute new gradients
            # vol: new_rho_grad
            # pr: new_pr_grad
            ########################
            t1 = time.time()

            # volume and probe gradient, volume regularizer, LL
            error_dct = self.ML_model.new_grad()
            new_rho_grad, new_pr_grad = self.rho_grad_new, self.pr_grad_new

            if self.DEBUG:
                print('rho_grad: %.2e' % np.linalg.norm(self.rho_grad))
                print('new_rho_grad: %.2e ' % np.linalg.norm(new_rho_grad))
                print('pr_grad: %.2e ' % np.sqrt(Cnorm2(self.pr_grad)))
                print('new_pr_grad: %.2e ' % np.sqrt(Cnorm2(new_pr_grad)))

            tg += time.time() - t1

            if self.p.probe_update_start <= self.curiter:
                # Apply probe support if needed
                for name, s in new_pr_grad.storages.items():
                    self.support_constraint(s)
                    #support = self.probe_support.get(name)
                    #if support is not None:
                    #    s.data *= support
            # FIXME: this hack doesn't work here as we step in probe and volume separately
            # FIXME: really it's the probe step that should be zeroed out not the gradient
            else:
                new_pr_grad.fill(0.)

            # Smoothing preconditioner for the volume
            if self.smooth_gradient:
                self.smooth_gradient.sigma *= (1. - self.p.smooth_gradient_decay)
                new_rho_grad = self.smooth_gradient(new_rho_grad)

            ############################
            # Compute Polak-Ribiere betas
            # bt_rho = bt_num_rho/bt_denom_rho
            # bt_pr = bt_num_pr/bt_denom_pr
            ############################
            if self.curiter == 0 or reset_cg:
                bt_rho = 0.
                bt_pr = 0.
                reset_cg = False
            else:
                # For the volume
                bt_num_rho = u.norm2(new_rho_grad) - np.real(np.vdot(new_rho_grad.flat, self.rho_grad.flat))
                bt_denom_rho = u.norm2(self.rho_grad)
                if self.DEBUG:
                    print('bt_num_rho, bt_denom_rho: (%.2e, %.2e) ' % (bt_num_rho, bt_denom_rho))

                # FIXME: this shouldn't be needed if we scale correctly
                if bt_denom_rho == 0:
                    bt_rho = 0
                else:
                    bt_rho = max(0, bt_num_rho/bt_denom_rho)

                # For the probe
                bt_num_pr = Cnorm2(new_pr_grad) - np.real(Cdot(new_pr_grad, self.pr_grad))
                bt_denom_pr = Cnorm2(self.pr_grad)
                bt_pr = max(0, bt_num_pr/bt_denom_pr)

            self.rho_grad = new_rho_grad.copy() # not a container
            self.pr_grad << new_pr_grad

            ############################
            # Compute next conjugates
            # rho_h -= bt_rho * rho_grad
            # pr_h -= bt_pr * pr_grad
            # NB: in the below need to do h/tmin
            # as did h*tmin when taking steps
            # (don't you just love containers?)
            ############################
            if self.DEBUG:
                print('bt_rho, self.tmin_rho: (%.2e,%.2e)' % (bt_rho, self.tmin_rho))
            self.rho_h *= bt_rho / self.tmin_rho

            # Smoothing preconditioner for the volume
            if self.smooth_gradient:
                self.rho_h -= self.smooth_gradient(self.rho_grad)
            else:
                self.rho_h -= self.rho_grad

            self.pr_h *= bt_pr / self.tmin_pr
            self.pr_h -= self.pr_grad

            ########################
            # Compute step-sizes
            # vol: tmin_rho
            # pr: tmin_pr
            ########################
            dt = self.ptycho.FType
            t2 = time.time()

            if self.p.poly_line_coeffs == "quadratic":
                B_rho = self.ML_model.poly_line_coeffs_rho(self.rho_h)
                B_pr = self.ML_model.poly_line_coeffs_pr(self.pr_h)

                if self.DEBUG:
                    print('B_rho, B_pr', B_rho, B_pr)

                # same as below but quicker when poly quadratic
                self.tmin_rho = dt(-0.5 * B_rho[1] / B_rho[2])
                self.tmin_pr = dt(-0.5 * B_pr[1] / B_pr[2])

            elif self.p.poly_line_coeffs == "all":
                B_rho = self.ML_model.poly_line_all_coeffs_rho(self.rho_h)
                diffB_rho = np.arange(1,len(B_rho))*B_rho[1:] # coefficients of poly derivative
                roots = np.roots(np.flip(diffB_rho.astype(np.double))) # roots only supports double
                real_roots = np.real(roots[np.isreal(roots)]) # not interested in complex roots
                if real_roots.size == 1: # single real root
                    self.tmin_rho = dt(real_roots[0])
                else: # find real root with smallest poly objective
                    evalp = lambda root: np.polyval(np.flip(B_rho),root)
                    self.tmin_rho = dt(min(real_roots, key=evalp)) # root with smallest poly objective

                B_pr = self.ML_model.poly_line_all_coeffs_pr(self.pr_h)
                diffB_pr = np.arange(1,len(B_pr))*B_pr[1:] # coefficients of poly derivative
                roots = np.roots(np.flip(diffB_pr.astype(np.double))) # roots only supports double
                real_roots = np.real(roots[np.isreal(roots)]) # not interested in complex roots
                if real_roots.size == 1: # single real root
                    self.tmin_pr = dt(real_roots[0])
                else: # find real root with smallest poly objective
                    evalp = lambda root: np.polyval(np.flip(B_pr),root)
                    self.tmin_pr = dt(min(real_roots, key=evalp)) # root with smallest poly objective

            else:
                raise NotImplementedError("poly_line_coeffs should be 'quadratic' or 'all'")

            tc += time.time() - t2

            if self.DEBUG:
                print('self.tmin_pr, self.tmin_rho: (%.2e,%.2e)' % (self.tmin_pr, self.tmin_rho))

            ########################
            # Take conjugate steps
            # rho += tmin_rho * rho_h
            # pr += tmin_pr * pr_h
            ########################

            self.rho_h *= self.tmin_rho
            self.pr_h *= self.tmin_pr

            self.rho += self.rho_h
            self.pr += self.pr_h

            # FIXME: move saving volumes to run script
            if parallel.master and self.curiter == 199: # curiter starts at zero
            # Get SLURM Job ID
                sid = subprocess.check_output("squeue -u $USER | tail -1| awk '{print $1}'", encoding="ascii", shell=True).strip()
            # Saving volumes when running Toy Problem (saves to png)
                self.projector.plot_vol(self.rho, title= '200iters_'+sid)
            # Saving volumes when running Real Data (saves to cmap)
            #    with h5py.File("/dls/science/users/iat69393/ptycho-tomo-project/SMALLER_recon_vol_ampl_HARDC_it200_"+sid+".cmap", "w") as f:
            #        f["data"] = np.imag(self.rho)[100:-100,100:-100,100:-100]
            #    with h5py.File("/dls/science/users/iat69393/ptycho-tomo-project/SMALLER_NEG_recon_vol_phase_HARDC_it200_"+sid+".cmap", "w") as f:
            #        f["data"] = -np.real(self.rho)[100:-100,100:-100,100:-100]
            # FIXME: end move saving volumes to run script

            # Position correction
            self.position_update()

            # Allow for customized modifications at the end of each iteration
            self._post_iterate_update()

            # increase iteration counter
            self.curiter +=1

        logger.info('Time spent in gradient calculation: %.2f' % tg)
        logger.info('  ....  in coefficient calculation: %.2f' % tc)
        return error_dct  # np.array([[self.ML_model.LL[0]] * 3])

    ########################
    # OPR Section
    ########################
    def _post_iterate_update(self):
        """
        Orthogonal Probe Relaxation (OPR) update.
        """

        # FIXME: this should be updated to use MPI
        if self.p.OPR:

            # FIRST OPR METHOD (probes not shifted) ###################
            if self.p.OPR_method.lower() == "first":

                # Assemble probes
                probe_size = list(self.pr.S.values())[0].shape[1:]
                pr_input = np.zeros((self.nangles,*probe_size))
                for i, (name, s) in enumerate(self.pr.storages.items()):
                    pr_input[i,:] = s.data

                # Apply low dimensional approximation to probes
                new_pr, modes, coeffs = reduce_dimension(pr_input, self.p.OPR_modes)

                # Update probes
                for i, (name, s) in enumerate(self.pr.storages.items()):

                    # Rescale new probes to match old probes (needed so that probes do not explode)
                    new_pr[i] = np.linalg.norm(pr_input[i]) * new_pr[i] / np.linalg.norm(new_pr[i])

                    # Update probes
                    s.data[:] = new_pr[i,:]

                # FIXME: move saving probes to run script
                if parallel.master and self.curiter == 199: # curiter starts at zero
                    sid = subprocess.check_output("squeue -u $USER | tail -1| awk '{print $1}'", encoding="ascii", shell=True).strip()
                    np.save('opr_probes_'+sid, new_pr)

                #if parallel.master:
                #    np.save('pr_input', pr_input)
                #    np.save('new_pr', new_pr)

            # SECOND OPR METHOD (probes are shifted) ###################
            elif self.p.OPR_method.lower() == "second":

                # Compute average probe mass center and assemble probes
                centers = np.zeros((self.nangles,2))
                probe_size = list(self.pr.S.values())[0].shape[1:]
                pr_input = np.zeros((self.nangles,*probe_size))
                for i, (name, s) in enumerate(self.pr.storages.items()):
                    centers[i,:] = u.mass_center(np.abs(s.data))[1:]
                    pr_input[i,:] = s.data
                mean_center = np.mean(centers, axis=0)

                # Shift probes
                for i, (name, s) in enumerate(self.pr.storages.items()):
                    pr_input[i,:] = shift(pr_input[i], mean_center-centers[i], mode='nearest')

                # Apply low dimensional approximation to probes
                new_pr, modes, coeffs = reduce_dimension(pr_input, self.p.OPR_modes)

                # Unshift probes and update
                for i, (name, s) in enumerate(self.pr.storages.items()):

                    # Rescale new probes to match old probes (needed so that probes do not explode)
                    new_pr[i] = np.linalg.norm(pr_input[i]) * new_pr[i] / np.linalg.norm(new_pr[i])

                    # Unshift new probes
                    new_pr[i,:] = shift(new_pr[i], centers[i]-mean_center, mode='nearest')

                    # Update probes
                    s.data[:] = new_pr[i,:]

                # FIXME: move saving probes to run script
                if parallel.master and self.curiter == 199: # curiter starts at zero
                    sid = subprocess.check_output("squeue -u $USER | tail -1| awk '{print $1}'", encoding="ascii", shell=True).strip()
                    np.save('opr_probes_'+sid, new_pr)

                #if parallel.master:
                #    print('mean_center:',end=None)
                #    print(mean_center)
                #    np.save('pr_input', pr_input)
                #    np.save('new_pr', new_pr)

            else:
                raise RuntimeError("Unsupported OPR_method: '%s'" % self.p.OPR_method)

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
        self.ex = self.engine.ex
        self.coverage = self.engine.coverage

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

        self.LL = 0.


    def prepare(self):
        # Useful quantities
        self.tot_measpts = sum(s.data.size
                               for s in self.di.storages.values())
        self.tot_power = self.Irenorm * sum(s.tot_power
                                            for s in self.di.storages.values())
        # Prepare regularizer
        if self.regularizer is not None:
            # obj_Npix = np.prod(np.shape(self.rho))    #self.ob.size
            # expected_obj_var = obj_Npix / self.tot_power  # Poisson
            # reg_rescale = self.tot_measpts / (8. * obj_Npix * expected_obj_var)
            # logger.debug(
            #     'Rescaling regularization amplitude using '
            #     'the Poisson distribution assumption.')
            # logger.debug('Factor: %8.5g' % reg_rescale)

            # Use amplitude directly for now
            reg_rescale = 1

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
        Compute new volume and probe gradient directions according to the noise model.

        Note: The negative log-likelihood and local errors should also be computed here.
        """
        raise NotImplementedError

    def poly_line_coeffs_rho(self, rho_h):
        """
        Compute the coefficients of the polynomial for line minimization
        in direction h for the volume
        """
        raise NotImplementedError

    def poly_line_coeffs_pr(self, pr_h):
        """
        Compute the coefficients of the polynomial for line minimization
        in direction h for the probe
        """
        raise NotImplementedError

    def poly_line_all_coeffs_rho(self, rho_h):
        """
        Compute all the coefficients of the polynomial for line minimization
        in direction h for the volume
        """
        raise NotImplementedError

    def poly_line_all_coeffs_pr(self, pr_h):
        """
        Compute all the coefficients of the polynomial for line minimization
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
        Compute new volume and probe gradient directions according to a Gaussian noise model.

        Note: The negative log-likelihood and local errors are also computed here.
        """
        self.rho_grad.fill(0.)
        self.pr_grad.fill(0.)

        # Use correct downsampled projector and rho
        if self.engine.downsample == 2:
            self.projector = self.engine.projector2
        else:
            self.projector = self.engine.projector
            self.rho = self.engine.rho # need downsampled rho (not a container)

        # We need an array for MPI
        LL = np.array([0.])
        error_dct = {}

        # Forward project volume
        self.projected_rho = self.projector.forward(self.rho)

        # Get refractive index per view
        i = 0
        for dname, diff_view in self.di.views.items():
            for name, pod in diff_view.pods.items():
                if pod.active:
                    pod.object = self.projected_rho[i]
                    i+=1

        # FIXME: this should be a container not a list
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
            LLL = np.sum((w * DI**2).astype(np.float64))
            for name, pod in diff_view.pods.items():
                if not pod.active:
                    continue
                xi = pod.bw(pod.upsample(w*DI) * f[name])
                expobj = np.exp(1j * pod.object)
                self.pr_grad[pod.pr_view] += 2. * xi * expobj.conj()
                product_xi_psi_conj = -1j * xi * (pod.probe * expobj).conj() / self.tot_power
                if self.p.weight_gradient: # apply coverage mask
                    product_xi_psi_conj *= self.coverage[pod.ob_view.slice[1:]]
                products_xi_psi_conj.append(product_xi_psi_conj)

            diff_view.error = LLL
            error_dct[dname] = np.array([0, LLL / np.prod(DI.shape), 0])
            LL += LLL

        # Back project volume
        self.rho_grad = 2 * self.projector.backward(np.moveaxis(np.array(products_xi_psi_conj), 1, 0))

        # MPI reduction of gradients
        parallel.allreduce(self.rho_grad)
        self.pr_grad.allreduce()
        parallel.allreduce(LL)

        # Volume regularizer
        if self.regularizer:
            # OLD using containers
            # for name, s in self.ob.storages.items():
            #     self.rho_grad.storages[name].data += self.regularizer.grad(
            #         s.data)

            # add volume regularizer gradient and regularizer log-likelihood
            self.rho_grad += self.regularizer.grad(self.rho)
            LL += self.regularizer.LL

        self.LL = LL / self.tot_measpts

        # FIXME: it is bizarre that this is actually needed
        # FIXME: is it because we are not using containers?
        self.engine.rho_grad_new = self.rho_grad
        #print('DEBUG rho_grad: %.2e ' % np.linalg.norm(self.rho_grad))

        return error_dct

    def poly_line_coeffs_rho(self, rho_h):
        """
        Compute the coefficients of the polynomial for line minimization
        in direction h for the volume
        """

        B = np.zeros((3,), dtype=np.longdouble)
        Brenorm = 1. / self.LL[0]**2

        # Forward project volume minimization direction
        omega = np.array(self.projector.forward(rho_h))

        # Store omega (so that we can use it later) and multiply it by the required 1j
        for i, (k,ov) in enumerate([(i,v) for i,v in self.omega.views.items() if v.pod.active]):
            ov.data[:] = 1j * omega[i]

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

                omega_i = self.omega[pod.ex_view]

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

        parallel.allreduce(B)

        # Volume regularizer
        if self.regularizer:
            # OLD using containers
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

    def poly_line_all_coeffs_rho(self, rho_h):
        """
        Compute the coefficients of the polynomial for line minimization
        in direction h for the volume
        """

        B = np.zeros((5,), dtype=np.longdouble)
        Brenorm = 1. / self.LL[0]**2

        # Forward project volume minimization direction
        omega = np.array(self.projector.forward(rho_h))

        # Store omega (so that we can use it later) and multiply it by the required 1j
        for i, (k,ov) in enumerate([(i,v) for i,v in self.omega.views.items() if v.pod.active]):
            ov.data[:] = 1j * omega[i]

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

                omega_i = self.omega[pod.ex_view]
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

        # Volume regularizer
        if self.regularizer:
            # OLD using containers
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

        if np.isinf(B).any() or np.isnan(B).any():
            logger.warning(
                'Warning! inf or nan found! Trying to continue...')
            B[np.isinf(B)] = 0.
            B[np.isnan(B)] = 0.

        self.B = B

        return B

    def poly_line_all_coeffs_pr(self, pr_h):
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

        if np.isinf(B).any() or np.isnan(B).any():
            logger.warning(
                'Warning! inf or nan found! Trying to continue...')
            B[np.isinf(B)] = 0.
            B[np.isnan(B)] = 0.

        self.B = B

        return B

class Regul_del2(object):
    """\
    Squared volume gradient regularizer (Gaussian prior).

    This class applies to any 3D numpy array.
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


def prepare_smoothing_preconditioner(amplitude):
    """
    Factory for 3D volume smoothing preconditioner.
    """
    if amplitude == 0.:
        return None

    class GaussFilt(object):
        def __init__(self, sigma):
            self.sigma = sigma

        def __call__(self, x):
            return u.c_gf(x, self.sigma) # blur all 3 dimensions of volume

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
