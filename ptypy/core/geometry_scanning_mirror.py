"""
Geometry for scanning mirror data.
"""

from .. import utils as u
from ..utils.verbose import logger
from .geometry import Geo as _Geo
from .geometry import FFTchooser as _FFTchooser
from ..utils.descriptor import EvalDescriptor
from .classes import Container, Storage, View
import numpy as np
from scipy.ndimage.interpolation import map_coordinates

__all__ = ['Geo_ScanningMirror']


local_tree = EvalDescriptor('')
@local_tree.parse_doc()
class Geo_ScanningMirror(_Geo):
    """
    Class which presents a Geo analog valid for scanning mirror datasets.
    """

    def _initialize(self, p):
        """
        Parse input parameters, fill missing parameters and set up a
        propagator.
        """
        self.interact = False

        # Set distance
        if self.p.distance is None or self.p.distance == 0:
            raise ValueError(
                'Distance (geometry.distance) must not be None or 0')

        # Set frame shape
        if self.p.shape is None or (np.array(self.p.shape) == 0).any():
            raise ValueError(
                'Frame size (geometry.shape) must not be None or 0')
        else:
            self.p.shape = u.expect2(p.shape)

        # Set energy and wavelength
        if p.energy is None:
            if p.lam is None:
                raise ValueError(
                    'Wavelength (geometry.lam) and energy (geometry.energy)\n'
                    'must not both be None')
            else:
                self.lam = p.lam  # also sets energy
        else:
            if p.lam is not None:
                logger.debug('Energy and wavelength both specified. '
                             'Energy takes precedence over wavelength')

            self.energy = p.energy

        # Set initial geometrical misfit to 0
        self.p.misfit = u.expect2(0.)

        # Pixel size
        self.p.psize_is_fix = p.psize is not None
        self.p.resolution_is_fix = p.resolution is not None

        if not self.p.psize_is_fix and not self.p.resolution_is_fix:
            raise ValueError(
                'Pixel size in sample plane (geometry.resolution) and '
                'detector plane \n(geometry.psize) must not both be None')

        # Fill pixel sizes
        if self.p.resolution_is_fix:
            self.p.resolution = u.expect2(p.resolution)
        else:
            self.p.resolution = u.expect2(1.0)

        if self.p.psize_is_fix:
            self.p.psize = u.expect2(p.psize)
        else:
            self.p.psize = u.expect2(1.0)

        # Update other values
        self.update(False)

        # Attach propagator
        self._propagator = self._get_propagator()
        self.interact = True

        # Resampling
        self.resample = 1

    def _get_propagator(self):
        # attach desired datatype for propagator
        try:
            dt = self.owner.CType
        except:
            dt = np.complex64

        return ScanningMirrorFarfieldPropagator(self.p, ffttype=self.p["ffttype"], dtype=dt)


class ScanningMirrorFarfieldPropagator(object):
    """
    Modified version of BasicFarfieldPropagator. The modification applies a
    phase gradient before propagation to account for changes in the beam angle
    during the scan.
    """

    def __init__(self, geo_pars=None, ffttype='scipy', **kwargs):
        """
        Parameters
        ----------


        geo_pars : Param or dict
            Parameter dictionary as in :py:attr:`DEFAULT`.

        ffttype : str or tuple
            Type of FFT implementation. One of:

            - 'fftw' for pyFFTW
            - 'numpy' for numpy.fft.fft2
            - 'scipy' for scipy.fft.fft2
            - 2 or 4-tuple of (forward_fft2(), inverse_fft2(),
              [scaling, inverse_scaling])
        """
        # Instance attributes
        self.crop_pad = None
        self.sh = None
        self.grids_sam = None
        self.grids_det = None
        self.pre_curve = None
        self.pre_fft = None
        self.post_curve = None
        self.post_fft = None
        self.pre_ifft = None
        self.post_ifft = None

        # Get default parameters and update
        self.p = u.Param(_Geo.DEFAULT)
        if 'dtype' in kwargs:
            self.dtype = kwargs['dtype']
        else:
            self.dtype = np.complex128
        self.FFTch = _FFTchooser(ffttype)
        self.fft, self.ifft = self.FFTch.assign_fft()
        self.beam_shift = geo_pars.beam_shift

        self.update(geo_pars, **kwargs)



    def update(self, geo_pars=None, **kwargs):
        """
        Update internal p dictionary. Recompute all internal array buffers.
        """
        # Local reference to avoid excessive self. use
        p = self.p
        if geo_pars is not None:
            p.update(geo_pars)
        for k, v in kwargs.items():
            if k in p:
                p[k] = v

        # Wavelength * distance factor
        lz = p.lam * p.distance

        # Calculate real space pixel size.
        if p.resolution is not None:
            resolution = p.resolution
        else:
            resolution = lz / p.shape / p.psize

        # Calculate array shape from misfit
        mis = u.expect2(p.misfit)
        self.crop_pad = np.round(mis / 2.0).astype(int) * 2
        self.sh = p.shape + self.crop_pad

        # Undo rounding error
        lz /= (self.sh[0] + mis[0] - self.crop_pad[0]) / self.sh[0]

        # Calculate the grids
        if u.isstr(p.origin):
            c_sam = p.origin
        else:
            c_sam = p.origin + self.crop_pad / 2.

        if u.isstr(p.center):
            c_det = p.center
        else:
            c_det = p.center + self.crop_pad / 2.

        [X, Y] = u.grids(self.sh, resolution, c_sam)
        [V, W] = u.grids(self.sh, p.psize, c_det)

        # Maybe useful later. delete this references if space is short
        self.grids_sam = [X, Y]
        self.grids_det = [V, W]

        # Quadratic phase + shift factor before fft
        self.pre_curve = np.exp(
            1j * np.pi * (X**2 + Y**2) / lz).astype(self.dtype)

        # self.pre_check = np.exp(
        #     -2.0 * np.pi * 1j * ((X-X[0, 0]) * V[0, 0] +
        #                          (Y-Y[0, 0]) * W[0, 0]) / lz
        # ).astype(self.dtype)

        self.pre_fft = self.pre_curve * np.exp(
            -2.0 * np.pi * 1j * ((X-X[0, 0]) * V[0, 0] +
                                 (Y-Y[0, 0]) * W[0, 0]) / lz
        ).astype(self.dtype)

        # Quadratic phase + shift factor before fft
        self.post_curve = np.exp(
            1j * np.pi * (V**2 + W**2) / lz).astype(self.dtype)

        self.post_fft = self.post_curve * np.exp(
            -2.0 * np.pi * 1j * (X[0, 0]*V + Y[0, 0]*W) / lz
        ).astype(self.dtype)

        # modify to take phase gradient into account
        self.pre_fft *= self.generate_phase_grad(self.beam_shift, W.shape)

        # Factors for inverse operation
        self.pre_ifft = self.post_fft.conj()
        self.post_ifft = self.pre_fft.conj()

        self.sc, self.isc = self.FFTch.assign_scaling(self.sh)

    def generate_phase_grad(self, shift, shape):
        """
        Creates a phase ramp that will shift the fourier transform by the given
        amount of pixels.

        Parameters
        ----------
        shift : list
            List containing two floats, shift_i and shift_j, which are the
            number of pixels that the far field diffraction pattern is shifted
            on the detector. The index order is the same as in the far field
            detector image.

        shape : tuple
            The shape of the array.

        Returns
        -------
        numpy.ndarray
            The 2d phase gradient map.
        """
        i = np.linspace(0, 1, shape[0], endpoint=False)
        j = np.linspace(0, 1, shape[1], endpoint=False)
        phase_grad_i = np.exp(
            1j * 2 * np.pi * shift[0] * i
        )
        phase_grad_j = np.exp(
            1j * 2 * np.pi * shift[1] * j
        )

        phase_grad_2d = np.multiply.outer(phase_grad_i, phase_grad_j)
        return phase_grad_2d

    def fw(self, W):
        """
        Computes forward propagated wavefront of input wavefront W.

        Parameters
        ----------
        W : numpy.ndarray
            The wave front to be propagated.

        shift : list
            List containing two floats, shift_i and shift_j, which are the
            number of pixels that the far field diffraction pattern is shifted
            on the detector.

        Returns
        -------
        numpy.ndarray
            The propagated wave front.
        """
        # Check for cropping
        if (self.crop_pad != 0).any():
            w = u.crop_pad(W, self.crop_pad)
        else:
            w = W

        # Apply phase gradient
        # Now rolled into self.pre_fft
        #w *= self.generate_phase_grad(self.beam_shift, w.shape)

        w = self.post_fft * self.sc * self.fft(self.pre_fft * w)

        # Cropping again
        if (self.crop_pad != 0).any():
            return u.crop_pad(w, -self.crop_pad)
        else:
            return w

    def bw(self, W):
        """
        Computes backward propagated wavefront of input wavefront W.

        Parameters
        ----------
        W : numpy.ndarray
            The wave front to be backwards propagated.

        Returns
        -------
        numpy.ndarray
            The backwards propagated wave front.
        """
        # Check for cropping
        if (self.crop_pad != 0).any():
            w = u.crop_pad(W, self.crop_pad)
        else:
            w = W

        # Compute transform
        w = self.ifft(self.pre_ifft * w) * self.isc * self.post_ifft

        # Un-apply phase gradient
        # Now rolled into self.post_ifft
        #w /= self.generate_phase_grad(self.beam_shift, w.shape)

        # Cropping again
        if (self.crop_pad != 0).any():
            return u.crop_pad(w, -self.crop_pad)
        else:
            return w
