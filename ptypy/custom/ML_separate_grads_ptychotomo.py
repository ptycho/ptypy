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
import os
import time
from .. import io
from .. import utils as u
from ..utils.verbose import logger
from ..utils import parallel
from ..engines.utils import Cnorm2, Cdot
from ..engines import register
from ..engines.base import PositionCorrectionEngine
from ..core.manager import Full, Vanilla, Bragg3dModel, BlockVanilla, BlockFull, BlockFull3D, GradFull, BlockGradFull
from ..core import Container
from ..utils.tomo import AstraViewBased
from scipy.ndimage import gaussian_filter

__all__ = ['MLPtychoTomo']

# ID given to the storage of every volume container, i.e. the volume itself,
# its gradient and its minimization direction.
VOL_STORAGE_ID = "_rho"


def volume_data(container):
    """
    The data array of a volume container, e.g. the volume itself or its
    gradient.

    Each of those containers holds a single storage, which is read off here
    rather than looked up by the ID it was created with.
    """
    return next(iter(container.storages.values())).data


class PtypyTomoWrapper:
    def __init__(self, obj, vol, shifts=None):
        self._setup_projector(obj, vol, shifts)

    def _setup_projector(self, obj, vol, shifts=None):
        list_view_to_proj_vectors = []
        all_angles = []

        # For all blocks
        for i, (k,v) in enumerate([(i,v) for i,v in obj.views.items()]):
            y = v.dcoord[0] - v.storage.data.shape[-2]/2 
            x = v.dcoord[1] - v.storage.data.shape[-1]/2 
            if shifts is not None:
                y += shifts[i][1]
                x += shifts[i][0]
            list_view_to_proj_vectors.append((y, x))
            all_angles.append(v.extra['val'])

        view_to_proj_vectors = np.array(list_view_to_proj_vectors)
        #print(view_to_proj_vectors)

        self.projector = AstraViewBased(
            vol=vol,
            n_views = len(all_angles),
            view_shape = np.shape(list(obj.views.values())[0]),
            block_size = sum(1 for v in obj.views.values() if v.pod.active),
            angles = all_angles,
            shifts = None,
            view_to_proj_vectors = view_to_proj_vectors
        )

    def forward(self, vol, ind, output):
        """
        Computes the forward projection, so a 3d array of shape
        (n_active_views, view_shape_1, view_shape_2), and places
        it in the container passed as argument.

        Receives:
            vol     3d numpy array - the volume
            ind     list[int] - the indices of the active views
            output  container - to store the result
        """
        self.projector.vol = vol
        self.projector.ind_of_views = ind

        # Does not currently work with multiple storages
        output_proj_array = self.projector.forward()
        for ID, s in output.storages.items():
            s.data[:] = output_proj_array

    def backward(self, proj_array, ind, output):
        """
        Computes the backward projection, so a 3d array having
        same shape as volume, and places it in the container
        passed as argument.

        Receives:
            proj_array  3d numpy array - what we want to backward project
                        Has shape : (view_shape_1, n_active_views, view_shape_2)
            ind         list[int] - the indices of the active views
            output      container - to store the result
        """
        self.projector.proj_array = proj_array
        self.projector.ind_of_views = ind

        # Does not currently work with multiple storages
        output_vol = self.projector.backward()
        for ID, s in output.storages.items():
            s.data[:] = output_vol


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

    [n_angles]
    default = None
    type = int
    help = Number of tomography angles

    [shifts]
    default = None
    type = ndarray
    help = Tomography angle shifts
    doc = Path to the tomography angle shifts (if provided).

    [init_vol_zero]
    default = True
    type = bool
    help = Zero initial volume
    doc = Start with a zero initial volume for the reconstruction.

    [vol_size]
    default = None
    type = tuple
    help = Shape of volume
    doc = Size for the initial volume, if starting from a zero volume. If
          None, a cube with the side of the object views is used. Ignored
          unless ``init_vol_zero`` is True, the shape being taken from the
          loaded volume otherwise.

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

    [save_vol]
    default = None
    type = str
    help = Where to save the reconstructed volume
    doc = If None, the volume is never saved. Otherwise a path, taken
          relative to ``io.home`` unless it is absolute, which may contain
          the ``%(run)s``, ``%(engine)s`` and ``%(iteration)04d``
          placeholders, e.g. ``vols/%(run)s_%(iteration)04d.npy``. The file
          format follows the extension: ``.npy`` is written with numpy,
          anything else as hdf5 with the volume in the ``data`` entry.

    [save_vol_interval]
    default = 0
    type = int
    lowlim = 0
    help = Number of iterations between two saved volumes
    doc = If ``<=0``, the volume is only saved once the engine has finished.
          Has no effect unless ``save_vol`` is set.

    [floating_intensities]
    default = False
    type = bool
    help = Adaptive diffraction pattern rescaling
    doc = If True, allow for adaptative rescaling of the diffraction pattern intensities
          (to correct for incident beam intensity fluctuations).

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
    doc = probe_update_start doesn't work with MLPtychoTomo yet.

    [poly_line_coeffs]
    default = quadratic
    type = str
    help = How many coefficients to be used in the the linesearch
    choices = ['quadratic','all']
    doc = choose between the 'quadratic' approximation (default) or 'all'

    """

    SUPPORTED_MODELS = [Full, Vanilla, Bragg3dModel, BlockVanilla, BlockFull, BlockFull3D, GradFull, BlockGradFull]

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

        # Tomography projector
        self.projector = None

        # Other
        self.nangles = self.p.n_angles
        self.tmin_rho = None
        self.tmin_pr = None
        self.ML_model = None
        self.smooth_gradient = None

        self.omega = None

        # Side of the object views, used as the volume size when none is given
        self.view_shape = list(self.ptycho.obj.S.values())[0].data.shape[-1]

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

        # Shape of the volume, shared by the volume itself, its gradient and
        # its minimization direction
        if self.p.init_vol_zero: # starting from zero volume
            if self.p.vol_size:
                vol_shape = tuple(self.p.vol_size)
            else:
                vol_shape = 3*(self.view_shape,)
            rho_real = np.zeros(vol_shape, dtype=np.complex64)
            rho_imag = np.zeros(vol_shape, dtype=np.complex64)
        else: # starting from given volume, which sets the shape
            rho_real = np.load(self.p.init_vol_real)
            rho_imag = np.load(self.p.init_vol_imag)
            if rho_real.shape != rho_imag.shape:
                raise ValueError(
                    'The real and imaginary parts of the initial volume have '
                    'different shapes, %s and %s.'
                    % (rho_real.shape, rho_imag.shape))
            vol_shape = rho_real.shape

        if self.p.init_vol_blur: # gaussian blur initial volume
            rho_real = gaussian_filter(rho_real, sigma=self.p.init_vol_blur_sigma)
            rho_imag = gaussian_filter(rho_imag, sigma=self.p.init_vol_blur_sigma)

        # Initialise volume rho as container
        self.rho = Container()
        self.rho.new_storage(ID=VOL_STORAGE_ID, shape=vol_shape)
        self.rho.fill(rho_real + 1j * rho_imag)

        # Initialise volume gradient and minimization direction
        self.rho_grad = Container()
        self.rho_grad_new = Container()
        self.rho_h = Container()

        self.rho_grad.new_storage(ID=VOL_STORAGE_ID, shape=vol_shape)
        self.rho_grad_new.new_storage(ID=VOL_STORAGE_ID, shape=vol_shape)
        self.rho_h.new_storage(ID=VOL_STORAGE_ID, shape=vol_shape)

        # Needed in poly_line_coeffs_rho
        self.omega = self.ex
        self.projected_rho = self.ex.copy(self.ex.ID + '_proj_rho', fill=0.)

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

        self.tomo_wrapper = PtypyTomoWrapper(
            obj=self.ptycho.obj,
            vol=volume_data(self.rho),
            shifts=self.p.shifts
        )

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

    def _volume_path(self):
        """
        The path the volume is saved to at the current iteration, i.e. the
        `save_vol` template with its placeholders filled in and made
        absolute, relative to `io.home` if it was not already.
        """
        path = os.path.expanduser(self.p.save_vol)
        if not os.path.isabs(path):
            path = os.path.join(self.ptycho.paths.home, path)

        return os.path.abspath(path % {
            'run': self.ptycho.runtime.run,
            'engine': self.p.name,
            'iteration': self.curiter,
        })

    def _save_volume(self):
        """
        Write the current volume to `save_vol`, creating the directory it
        lives in if needed. Only the master node writes.
        """
        if not parallel.master:
            return

        path = self._volume_path()
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)

        logger.info('Saving volume to %s' % path)
        data = volume_data(self.rho)
        if path.endswith('.npy'):
            np.save(path, data)
        else:
            io.h5write(path, data=data)

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
            # vol: new_rho_grad
            # pr: new_pr_grad
            ########################
            t1 = time.time()

            # volume and probe gradient, volume regularizer, LL
            error_dct = self.ML_model.new_grad()
            new_rho_grad, new_pr_grad = self.rho_grad_new, self.pr_grad_new

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
                new_rho_grad_data = self.smooth_gradient(volume_data(new_rho_grad))
                new_rho_grad.fill(new_rho_grad_data)

            ############################
            # Compute Polak-Ribiere betas
            # bt_rho = bt_num_rho/bt_denom_rho
            # bt_pr = bt_num_pr/bt_denom_pr
            ############################
            if self.curiter == 0:
                bt_rho = 0.
                bt_pr = 0.
            else:
                # For the volume
                bt_num_rho = Cnorm2(new_rho_grad) - np.real(Cdot(new_rho_grad, self.rho_grad))
                bt_denom_rho = Cnorm2(self.rho_grad)
                bt_rho = max(0, bt_num_rho/bt_denom_rho)

                # For the probe
                bt_num_pr = Cnorm2(new_pr_grad) - np.real(Cdot(new_pr_grad, self.pr_grad))
                bt_denom_pr = Cnorm2(self.pr_grad)
                bt_pr = max(0, bt_num_pr/bt_denom_pr)

            self.rho_grad << new_rho_grad
            self.pr_grad << new_pr_grad

            ############################
            # Compute next conjugates
            # rho_h = bt_rho * rho_h - rho_grad
            # pr_h = bt_pr * pr_h - pr_grad
            # NB: in the below need to do h/tmin
            # as did h*tmin when taking steps
            # (don't you just love containers?)
            ############################
            self.rho_h *= bt_rho / self.tmin_rho

            # Smoothing preconditioner for the volume
            if self.smooth_gradient:
                self.rho_h -= self.smooth_gradient(volume_data(self.rho_grad))
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

            ########################
            # Take conjugate steps
            # rho += tmin_rho * rho_h
            # pr += tmin_pr * pr_h
            ########################

            self.rho_h *= self.tmin_rho
            self.pr_h *= self.tmin_pr

            self.rho += self.rho_h
            self.pr += self.pr_h

            # Position correction
            self.position_update()

            # increase iteration counter
            self.curiter +=1

            # Save an intermediate volume if one is due
            if (self.p.save_vol and self.p.save_vol_interval > 0
                    and self.curiter % self.p.save_vol_interval == 0):
                self._save_volume()

        logger.info('Time spent in gradient calculation: %.2f' % tg)
        logger.info('  ....  in coefficient calculation: %.2f' % tc)
        return error_dct  # np.array([[self.ML_model.LL[0]] * 3])

    def engine_finalize(self):
        """
        Save the final volume and delete temporary containers.
        """
        if self.p.save_vol:
            self._save_volume()

        del self.rho_grad
        del self.rho_grad_new
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
        self.projected_rho = self.engine.projected_rho

        self.pr = self.engine.pr
        self.float_intens_coeff = {}

        if self.p.intensity_renormalization is None:
            self.Irenorm = 1.
        else:
            self.Irenorm = self.p.intensity_renormalization

        self.projector = self.engine.projector
        self.tomo_wrapper = self.engine.tomo_wrapper

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

    def get_indexes_of_active_views(self):
        """
        The indices of the active pods, counted over every pod of every
        diffraction view in turn, active or not.

        These index into the view list the projector was built with in
        `PtypyTomoWrapper._setup_projector`, which is in object view order,
        so the two orderings have to stay in step.
        """
        ind_active_views = []
        i = 0
        for diff_view in self.di.views.values():
            for pod in diff_view.pods.values():
                if pod.active:
                    ind_active_views.append(i)
                i += 1

        return ind_active_views

    def new_grad(self):
        """
        Compute new volume and probe gradient directions according to a Gaussian noise model.

        Note: The negative log-likelihood and local errors are also computed here.
        """
        self.rho_grad.fill(0.)
        self.pr_grad.fill(0.)

        # We need an array for MPI
        LL = np.array([0.])
        error_dct = {}

        # Forward project volume
        self.tomo_wrapper.forward(
            vol=volume_data(self.rho),
            ind=self.get_indexes_of_active_views(),
            output=self.projected_rho
        )

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
                f[name] = pod.fw(pod.probe * np.exp(1j * self.projected_rho[pod.ex_view]))
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
                expobj = np.exp(1j * self.projected_rho[pod.ex_view])
                self.pr_grad[pod.pr_view] += 2. * xi * expobj.conj()
                prod_xi_psi_conj = -1j * xi * (pod.probe * expobj).conj() / self.tot_power
                self.projected_rho[pod.ex_view] = prod_xi_psi_conj

            diff_view.error = LLL
            error_dct[dname] = np.array([0, LLL / np.prod(DI.shape), 0])
            LL += LLL

        # Back project volume
        storage_key = next(iter(self.projected_rho.S)) #list(self.prods_container.S)[0]
        self.tomo_wrapper.backward(
            proj_array=np.moveaxis(self.projected_rho.S[storage_key].data, 1, 0),
            ind=self.get_indexes_of_active_views(),
            output=self.rho_grad
        )
        self.rho_grad *= 2

        # MPI reduction of gradients
        self.rho_grad.allreduce()
        self.pr_grad.allreduce()
        parallel.allreduce(LL)

        # Volume regularizer
        if self.regularizer:
            # add volume regularizer gradient and regularizer log-likelihood
            for name, s in self.rho.storages.items():
                self.rho_grad.storages[name].data += self.regularizer.grad(
                    s.data)
            LL += self.regularizer.LL

        self.LL = LL / self.tot_measpts


        return error_dct

    def poly_line_coeffs_rho(self, rho_h):
        """
        Compute the coefficients of the polynomial for line minimization
        in direction h for the volume
        """
        return self._poly_line_coeffs_rho(rho_h, order=2)

    def poly_line_all_coeffs_rho(self, rho_h):
        """
        Compute all the coefficients of the polynomial for line minimization
        in direction h for the volume
        """
        return self._poly_line_coeffs_rho(rho_h, order=4)

    def poly_line_coeffs_pr(self, pr_h):
        """
        Compute the coefficients of the polynomial for line minimization
        in direction h for the probe
        """
        return self._poly_line_coeffs_pr(pr_h, order=2)

    def poly_line_all_coeffs_pr(self, pr_h):
        """
        Compute all the coefficients of the polynomial for line minimization
        in direction h for the probe
        """
        return self._poly_line_coeffs_pr(pr_h, order=4)

    def _poly_line_coeffs_rho(self, rho_h, order):
        """
        Compute the coefficients of the polynomial of the given order for
        line minimization in direction rho_h for the volume.

        Receives:
            rho_h   container - the volume minimization direction
            order   int - 2 for the quadratic approximation, 4 for all
                    coefficients
        """

        B = np.zeros((order + 1,), dtype=np.longdouble)
        Brenorm = 1. / self.LL[0]**2
        ind = self.get_indexes_of_active_views()

        # Forward project volume minimization direction
        self.tomo_wrapper.forward(
            vol=volume_data(rho_h),
            ind=ind,
            output=self.omega
        )
        # Forward project the volume itself, as new_grad has since
        # overwritten projected_rho with the gradient product
        self.tomo_wrapper.forward(
            vol=volume_data(self.rho),
            ind=ind,
            output=self.projected_rho
        )
        # Multiply omega by the required 1j
        self.omega *= 1j

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

                psi = pod.probe * np.exp(1j * self.projected_rho[pod.ex_view]) # exit_wave
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

            self._accumulate_B(B, A0, A1, A2, w, I, pod, dname, Brenorm)

        parallel.allreduce(B)

        # Volume regularizer, which only contributes up to the quadratic term
        if self.regularizer:
            for name, s in self.rho.storages.items():
                B[:3] += Brenorm * self.regularizer.poly_line_coeffs(
                    rho_h.storages[name].data, s.data)

        self._zero_non_finite_B(B)

        self.B = B

        return B

    def _poly_line_coeffs_pr(self, pr_h, order):
        """
        Compute the coefficients of the polynomial of the given order for
        line minimization in direction pr_h for the probe.

        Receives:
            pr_h    container - the probe minimization direction
            order   int - 2 for the quadratic approximation, 4 for all
                    coefficients
        """

        B = np.zeros((order + 1,), dtype=np.longdouble)
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

                expobj = np.exp(1j * self.projected_rho[pod.ex_view])
                f = pod.fw(pod.probe * expobj)
                a = pod.fw(pr_h[pod.pr_view] * expobj)

                if A0 is None:
                    A0 = u.abs2(f).astype(np.longdouble)
                    A1 = 2 * np.real(f * a.conj()).astype(np.longdouble)
                    A2 = u.abs2(a).astype(np.longdouble)
                else:
                    A0 += u.abs2(f)
                    A1 += 2 * np.real(f * a.conj())
                    A2 += u.abs2(a)

            self._accumulate_B(B, A0, A1, A2, w, I, pod, dname, Brenorm)

        parallel.allreduce(B)

        self._zero_non_finite_B(B)

        self.B = B

        return B

    def _accumulate_B(self, B, A0, A1, A2, w, I, pod, dname, Brenorm):
        """
        Apply the floating intensity correction to the A coefficients of a
        single diffraction view, subtract the measured intensities and
        accumulate the resulting polynomial coefficients into B.

        The number of terms accumulated follows the length of B, i.e. 3 for
        the quadratic approximation and 5 for all coefficients.
        """
        if self.p.floating_intensities:
            A0 *= self.float_intens_coeff[dname]
            A1 *= self.float_intens_coeff[dname]
            A2 *= self.float_intens_coeff[dname]

        A0 = np.double(A0) - pod.upsample(I)
        w = pod.upsample(w)

        B[0] += np.dot(w.flat, (A0**2).flat) * Brenorm
        B[1] += np.dot(w.flat, (2*A0*A1).flat) * Brenorm
        B[2] += np.dot(w.flat, (A1**2 + 2*A0*A2).flat) * Brenorm

        if len(B) > 3:
            B[3] += np.dot(w.flat, (2*A1*A2).flat) * Brenorm
            B[4] += np.dot(w.flat, (A2**2).flat) * Brenorm

    @staticmethod
    def _zero_non_finite_B(B):
        """
        Zero out any inf or nan polynomial coefficient, warning if found.
        """
        if np.isinf(B).any() or np.isnan(B).any():
            logger.warning(
                'Warning! inf or nan found! Trying to continue...')
            B[np.isinf(B)] = 0.
            B[np.isnan(B)] = 0.

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

