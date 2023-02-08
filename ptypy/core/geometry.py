"""
Geometry manager.

This file is part of the PTYPY package.

    :copyright: Copyright 2014 by the PTYPY team, see AUTHORS.
    :license: see LICENSE for details.
"""
import numpy as np
from scipy import fftpack

from .. import utils as u
from ..utils.verbose import logger
from .classes import Base, GEO_PREFIX
from ..utils.descriptor import EvalDescriptor

try:
    import pyfftw
    import pyfftw.interfaces.numpy_fft as fftw_np
except ImportError:
    pass
    #logger.warning("Unable to import pyFFTW! Will use a slower FFT method.")

__all__ = ['Geo', 'BasicNearfieldPropagator', 'BasicFarfieldPropagator']


_old2new = u.Param(
    # Distance from object to screen
    z='distance',
    # Pixel size (in meters) at detector plane
    psize_det='psize',
    # Pixel size (in meters) at sample plane
    psize_sam='resolution',
    # Number of detector pixels
    N='shape',
    prop_type='propagation',
    origin_det='center',
    origin_sam='origin',
)


local_tree = EvalDescriptor('')
@local_tree.parse_doc()
class Geo(Base):
    """
    Hold and keep consistent the information about experimental parameters.

    Keeps also reference to the Propagator and updates this reference
    when resolution, pixel size, etc. is changed in `Geo`. Reference to
    :py:data:`.io.paths.recons`.

    Attributes
    ----------
    interact : bool (True)
        If set to True, changes to properties like :py:meth:`energy`,
        :py:meth:`lam`, :py:meth:`shape` or :py:meth:`psize` will cause
        a call to :py:meth:`update`.
    

    Default geometry parameters. See also :py:data:`.scan.geometry`

    Defaults:

    [energy]
    type = float
    default = 6.2
    help = Energy (in keV)
    doc = If ``None``, uses `lam` instead.
    userlevel = 0
    lowlim = 0

    [lam]
    type = float
    default = None
    help = Wavelength (in meters)
    doc = Used only if `energy` is ``None``
    userlevel = 0
    lowlim = 0

    [distance]
    type = float
    default = 7.19
    help = Distance from object to detector (in meters)
    doc =
    userlevel = 0

    [psize]
    type = float
    default = 0.000172
    help = Pixel size in the detector plane (in meters)
    doc =
    userlevel = 1
    lowlim = 0

    [resolution]
    type = float
    default = None
    help = Pixel size in the sample plane
    doc = This parameter is used only for simulations
    userlevel = 2
    lowlim = 0

    [propagation]
    type = str
    default = farfield
    help = Propagation type
    doc = Choose between "farfield" or "nearfield"
    choices = 'farfield', 'nearfield'
    userlevel = 1

    [ffttype]
    type = str
    default = scipy
    help = FFT library
    doc = Choose which library to use for FFTs
    choices = 'numpy', 'scipy', 'fftw'
    userlevel = 1

    [shape]
    type = int, tuple
    default = 256
    help = Number of image pixels
    doc = Shape represents the number of pixels in the wavefield/image frames, and not necessarily
          the number of pixel in the raw diffraction patterns, as undersample or padding may be used.
          Valid entries are 2-tuple of int (Nx, Ny) or an int N, in which case it is interpreted as (N, N).
    userlevel = 1

    [misfit]
    type = float, tuple
    default = 0.
    help = TODO: WHAT IS MISFIT?
    doc =
    userlevel = 2

    [center]
    type = int, tuple, str
    default = 'fftshift'
    help = TODO: document this parameter.
    doc =
    userlevel = 2

    [origin]
    type = int, tuple, str
    default = 'fftshift'
    help = TODO: document this parameter.
    doc =
    userlevel = 2
    """

    _keV2m = 1.23984193e-09
    _PREFIX = GEO_PREFIX

    def __init__(self, owner=None, ID=None, pars=None, **kwargs):
        """
        Parameters
        ----------
        owner : Base or subclass
            Instance of a subclass of :any:`Base` or None. That is usually
            :any:`Ptycho` for a Geo instance.

        ID : str or int
            Identifier. Use ``ID=None`` for automatic ID assignment.

        pars : dict or Param
            The configuration parameters. See `Geo.DEFAULT`.
            Any other kwarg will update internal p dictionary
            if the key exists in `Geo.DEFAULT`.
        """
        super(Geo, self).__init__(owner, ID)

        # Starting parameters
        p = self.DEFAULT.copy(99)
        if pars is not None:
            p.update(pars)
            for k, v in p.items():
                if k in _old2new.keys():
                    p[_old2new[k]] = v
        for k, v in kwargs.items():
            if k in p:
                p[k] = v

        self.p = p
        self._initialize(p)

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

    def update(self, update_propagator=True):
        """
        Update the internal pixel sizes, giving precedence to the sample
        pixel size (resolution) if self.psize_is_fixed is True.
        """
        # 4 cases
        if not self.p.resolution_is_fix and not self.p.psize_is_fix:
            # This is a rare case
            logger.debug('No pixel size is marked as constant. '
                         'Setting detector pixel size as fix.')
            self.psize_is_fix = True
            self.update()
            return
        elif not self.p.resolution_is_fix and self.p.psize_is_fix:
            if self.p.propagation == 'farfield':
                self.p.resolution[:] = self.lz / self.p.psize / self.p.shape
            else:
                self.p.resolution[:] = self.p.psize
        elif self.p.resolution_is_fix and not self.p.psize_is_fix:
            if self.p.propagation == 'farfield':
                self.p.psize[:] = self.lz / self.p.resolution / self.p.shape
            else:
                self.p.psize[:] = self.p.resolution
        else:
            # Both psizes are fix
            if self.p.propagation == 'farfield':
                # Frame misfit that would make it work
                self.p.misfit[:] = (self.lz / self.p.resolution / self.p.psize
                                    - self.p.shape)
            else:
                self.p.misfit[:] = self.p.resolution - self.p.psize

        # Update the propagator too (optionally pass the dictionary,
        # but Geometry & Propagator should share a dict

        if update_propagator:
            self.propagator.update(self.p)

    @property
    def energy(self):
        """
        Get or set the energy (in keV). If self.interact is True, recompute
        dependent quantities accordingly.
        """
        return self.p.energy

    @energy.setter
    def energy(self, v):
        self.p.energy = v
        # actively change inner variables
        self.p.lam = self._keV2m / v
        if self.interact:
            self.update()

    @property
    def lam(self):
        """
        Get or set the wavelength (in meters).

        If self.interact is True, recompute dependent quantities accordingly. Changing wavelengths
        never changes the image extent (self.sh).
        """
        return self.p.lam

    @lam.setter
    def lam(self, v):
        self.p.lam = v
        self.p.energy = self._keV2m / v
        if self.interact:
            self.update()

    @property
    def resolution(self):
        """
        Get or set the pixel size in the image plane (in meters).

        If self.interact is True, recompute dependent quantities accordingly.
        """
        return self.p.resolution

    @resolution.setter
    def resolution(self, v):
        self.p.resolution[:] = u.expect2(v)
        if self.interact:
            self.update()

    @property
    def psize(self):
        """
        Get or set the pixel size in the propagated plane (in meters).

        If self.interact is True, recompute dependent quantities accordingly.
        """
        return self.p.psize

    @psize.setter
    def psize(self, v):
        self.p.psize[:] = u.expect2(v)
        if self.interact:
            self.update()

    @property
    def lz(self):
        """
        Product of wavelength and propagation distance (read-only)
        """
        return self.p.lam * self.p.distance

    @property
    def shape(self):
        """
        Get or set the frame dimensions.

        If self.interact is True, recompute dependent quantities accordingly.
        """
        return self.p.shape

    @shape.setter
    def shape(self, v):
        self.p.shape[:] = u.expect2(v).astype(int)
        if self.interact:
            self.update()

    @property
    def distance(self):
        """
        Propagation distance in meters (read-only)
        """
        return self.p.distance

    @property
    def propagator(self):
        """
        Propagator, created on-the-fly if needed (read-only)
        """
        if not hasattr(self, '_propagator'):
            self._propagator = self._get_propagator()

        return self._propagator

    def upsample(self, c):
        """
        Upsample the array c (float or complex).
        """
        if self.resample == 1:
            return c
        return u.zoom(c, self.resample, order=0) / (self.resample**2)

    def downsample(self, a):
        """
        Downsample the array a (float)
        """
        if self.resample == 1:
            return a
        return u.rebin_2d(a, self.resample)[0]

    def __str__(self):
        keys = list(self.p.keys())
        keys.sort()
        start = ""
        for key in keys:
            start += "%25s : %s\n" % (str(key), str(self.p[key]))
        return start

    def _to_dict(self):
        # Delete propagator reference
        try:
            del self._propagator
        except AttributeError:
            pass

        # Return internal dicts
        return self.__dict__.copy()

    def _get_propagator(self):
        # attach desired datatype for propagator
        try:
            dt = self.owner.CType
        except:
            dt = np.complex64

        return get_propagator(self.p, dtype=dt)


def get_propagator(geo_dct, **kwargs):
    """
    Helper function to determine propagator to be attached to Geometry class.
    """
    if geo_dct['propagation'] == 'farfield':
        return BasicFarfieldPropagator(geo_dct, ffttype=geo_dct["ffttype"], **kwargs)
    else:
        return BasicNearfieldPropagator(geo_dct, ffttype=geo_dct["ffttype"], **kwargs)


class FFTchooser(object):
    """
    Chooses the desired FFT algo, and assigns scaling.
    If pyFFTW is not available, falls back to scipy.
    """
    def __init__(self, ffttype='scipy'):
        """
        Parameters
        ----------
        ffttype : str or tuple
            Type of FFT implementation. One of:

            - 'fftw' for pyFFTW
            - 'numpy' for numpy.fft.fft2
            - 'scipy' for scipy.fft.fft2
            - 2 or 4-tuple of (forward_fft2(), inverse_fft2(),
              [scaling, inverse_scaling])
        """
        self.ffttype = ffttype

    def _FFTW_fft(self):
        pyfftw.interfaces.cache.enable()
        pyfftw.interfaces.cache.set_keepalive_time(15.0)
        pe = 'FFTW_MEASURE'
        self.fft = lambda x: fftw_np.fft2(x, planner_effort=pe)
        self.ifft = lambda x: fftw_np.ifft2(x, planner_effort=pe)

    def _scipy_fft(self):
        self.fft = lambda x: fftpack.fft2(x).astype(x.dtype)
        self.ifft = lambda x: fftpack.ifft2(x).astype(x.dtype)

    def _numpy_fft(self):
        self.fft = lambda x: np.ascontiguousarray(np.fft.fft2(x).astype(x.dtype))
        self.ifft = lambda x: np.ascontiguousarray(np.fft.ifft2(x).astype(x.dtype))

    def assign_scaling(self, shape):
        if isinstance(self.ffttype, tuple) and len(self.ffttype) > 2:
            self.sc = self.ffttype[2]
            self.isc = self.ffttype[3]
        else:
            self.sc = 1.0 / np.sqrt(np.prod(shape))
            self.isc = 1.0 / self.sc

        return (self.sc, self.isc)

    def assign_fft(self):
        if str(self.ffttype) == 'fftw':
            try:
                self._FFTW_fft()
            except NameError:
                self._scipy_fft()
        elif str(self.ffttype) == 'scipy':
            self._scipy_fft()
        elif str(self.ffttype) == 'numpy':
            self._numpy_fft()
        elif isinstance(self.ffttype, tuple):
            self.fft = self.ffttype[0]
            self.ifft = self.ffttype[1]

        return (self.fft, self.ifft)


class BasicFarfieldPropagator(object):
    """
    Basic single step Farfield Propagator.

    Includes quadratic phase factors and arbitrary origin in array.

    Be aware though, that if the origin is not in the center of the frame,
    coordinates are rolled periodically, just like in the conventional fft case.
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
        self.p = u.Param(Geo.DEFAULT)
        if 'dtype' in kwargs:
            self.dtype = kwargs['dtype']
        else:
            self.dtype = np.complex128
        self.FFTch = FFTchooser(ffttype)
        self.fft, self.ifft = self.FFTch.assign_fft()
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

        # self.post_check = np.exp(
        #     -2.0 * np.pi * 1j * (X[0, 0]*V + Y[0, 0]*W) / lz
        # ).astype(self.dtype)

        self.post_fft = self.post_curve * np.exp(
            -2.0 * np.pi * 1j * (X[0, 0]*V + Y[0, 0]*W) / lz
        ).astype(self.dtype)

        # Factors for inverse operation
        self.pre_ifft = self.post_fft.conj()
        self.post_ifft = self.pre_fft.conj()
        self.sc, self.isc = self.FFTch.assign_scaling(self.sh)


    def fw(self, W):
        """
        Computes forward propagated wavefront of input wavefront W.
        """
        # Check for cropping

        if (self.crop_pad != 0).any():
            w = u.crop_pad(W, self.crop_pad)
        else:
            w = W

        w = self.post_fft * self.sc * self.fft(self.pre_fft * w)

        # Cropping again
        if (self.crop_pad != 0).any():
            return u.crop_pad(w, -self.crop_pad)
        else:
            return w

    def bw(self, W):
        """
        Computes backward propagated wavefront of input wavefront W.
        """
        # Check for cropping
        if (self.crop_pad != 0).any():
            w = u.crop_pad(W, self.crop_pad)
        else:
            w = W

        # Compute transform
        w = self.ifft(self.pre_ifft * w) * self.isc * self.post_ifft

        # Cropping again
        if (self.crop_pad != 0).any():
            return u.crop_pad(w, -self.crop_pad)
        else:
            return w


def translate_to_pix(sh, center):
    """
    Translate arbitrary input to a pixel position with respect to sh.
    """
    sh = np.array(sh)
    if center == 'fftshift':
        cen = sh // 2.0
    elif center == 'geometric':
        cen = sh / 2.0 - 0.5
    elif center == 'fft':
        cen = sh * 0.0
    elif center is not None:
        # cen = sh * np.asarray(center) % sh - 0.5
        cen = np.asarray(center) % sh

    return cen


class BasicNearfieldPropagator(object):
    """
    Basic two step (i.e. two ffts) Nearfield Propagator.
    """

    def __init__(self, geo_pars=None, ffttype='scipy', **kwargs):
        """
        Parameters
        ----------
        geo_pars : Param or dict
            Parameter dictionary as in :py:data:`DEFAULT`.

        ffttype : str or tuple
            Type of FFT implementation. One of:

            - 'fftw' for pyFFTW
            - 'numpy' for numpy.fft.fft2
            - 'scipy' for scipy.fft.fft2
            - 2 or 4-tuple of (forward_fft2(),inverse_fft2(),
              [scaling,inverse_scaling])
        """
        # Instance attributes
        self.sh = None
        self.grids_sam = None
        self.grids_det = None
        self.kernel = None
        self.ikernel = None

        # Get default parameters and update
        self.p = u.Param(Geo.DEFAULT)
        self.dtype = kwargs['dtype'] if 'dtype' in kwargs else np.complex128
        self.update(geo_pars, **kwargs)
        self.FFTch = FFTchooser(ffttype)
        self.fft, self.ifft = self.FFTch.assign_fft()

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

        self.sh = p.shape

        # Calculate the grids
        [X, Y] = u.grids(self.sh, p.resolution, p.origin)

        # Maybe useful later. delete this references if space is short
        self.grids_sam = [X, Y]
        self.grids_det = [X, Y]

        # Calculating kernel
        # psize_fspace = p.lam * p.distance / p.shape / p.resolution
        # [V, W] = u.grids(self.sh, psize_fspace, 'fft')
        # a2 = (V**2 + W**2) / p.distance**2

        psize_fspace = p.lam / p.shape / p.resolution
        [V, W] = u.grids(self.sh, psize_fspace, 'fft')
        a2 = (V**2 + W**2)

        self.kernel = np.exp(
            2j * np.pi * (p.distance / p.lam) * (np.sqrt(1-a2) - 1)).astype(self.dtype)
        # self.kernel = np.fft.fftshift(self.kernel)
        self.ikernel = self.kernel.conj()

    def fw(self, W):
        """
        Computes forward propagated wavefront of input wavefront W.
        """
        return self.ifft(self.fft(W) * self.kernel)

    def bw(self, W):
        """
        Computes backward propagated wavefront of input wavefront W.
        """
        return self.ifft(self.fft(W) * self.ikernel)


############
# TESTING ##
############

if __name__ == "__main__":
    G = Geo()
    # G._initialize()
    GD = G._to_dict()
    G2 = Geo._from_dict(GD)
