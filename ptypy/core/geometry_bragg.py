"""
Geometry management and propagation for Bragg geometry.
"""

from .. import utils as u
from ..utils.verbose import logger
from .geometry import Geo as _Geo
from ..utils.descriptor import EvalDescriptor
from .classes import Container, Storage, View
import numpy as np
from scipy.ndimage.interpolation import map_coordinates

__all__ = ['Geo_Bragg']


local_tree = EvalDescriptor('')
@local_tree.parse_doc()
class Geo_Bragg(_Geo):
    """
    Class which presents a Geo analog valid for the 3d Bragg case.

    This class follows the naming convention of:
    Berenguer et al., Phys. Rev. B 88 (2013) 144101.

    Indexing into all q-space arrays and storages follows (q3, q1, q2),
    which corresponds to (r3, r1, r2) in the so-called natural real
    space coordinate system. These coordinates are transformed to 
    (x, z, y) as described below.

    Defaults:

    [psize]
    type = tuple
    default = (.065, 172e-6, 172e-6)
    help = Rocking curve step (in degrees) and pixel sizes (in meters)
    doc = First element is the rocking curve step.

    [propagation]
    doc = Only "farfield" is valid for Bragg

    [shape]
    type = tuple
    default = (31, 128, 128)
    help = Number of rocking curve positions and detector pixels
    doc = First element is the number of rocking curve positions.

    [theta_bragg]
    type = float
    default = 6.89
    help = Diffraction angle (theta, not two theta) in degrees

    [resolution]
    type = tuple
    default = None
    help = 3D sample pixel size (in meters)
    doc = Refers to the conjugate (natural) coordinate system as (r3, r1, r2).
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
            self.p.shape = u.expect3(p.shape)

        # Set energy and wavelength
        if p.energy is None:
            if p.lam is None:
                raise ValueError(
                    'Wavelength (geometry.lam) and energy (geometry.energy)\n'
                    'must not both be None')
            else:
                self.lam = p.lam  # also sets energy through a property
        else:
            if p.lam is not None:
                logger.debug('Energy and wavelength both specified. '
                             'Energy takes precedence over wavelength')

            self.energy = p.energy  # also sets lam through a property

        # Pixel size
        self.p.psize_is_fix = p.psize is not None
        self.p.resolution_is_fix = p.resolution is not None

        if not self.p.psize_is_fix and not self.p.resolution_is_fix:
            raise ValueError(
                'Pixel size in sample plane (geometry.resolution) and '
                'detector plane \n(geometry.psize) must not both be None')

        # Fill pixel sizes
        if self.p.resolution_is_fix:
            self.p.resolution = u.expect3(p.resolution)
        else:
            self.p.resolution = u.expect3(1.0)

        if self.p.psize_is_fix:
            self.p.psize = u.expect3(p.psize)
        else:
            self.p.psize = u.expect3(1.0)

        # Update other values
        self.update(False)

        # Attach propagator
        self._propagator = self._get_propagator()
        self.interact = True

    def update(self, update_propagator=True):
        """
        Update things which need a little computation
        """

        # Update the internal pixel sizes: 4 cases
        if not self.p.resolution_is_fix and not self.p.psize_is_fix:
            raise ValueError(
                'Neither pixel size nor sample resolution specified.')
        elif not self.p.resolution_is_fix and self.p.psize_is_fix:
            dq1 = self.psize[1] * 2 * np.pi / self.distance / self.lam
            dq2 = self.psize[2] * 2 * np.pi / self.distance / self.lam
            dq3 = np.deg2rad(self.psize[0]) * 4 * np.pi / self.lam * self.sintheta
            self.p.resolution[1] = 2 * np.pi / \
                (self.shape[1] * dq1 * self.costheta)
            self.p.resolution[2] = 2 * np.pi / (self.shape[2] * dq2)
            self.p.resolution[0] = 2 * np.pi / \
                (self.shape[0] * dq3 * self.costheta)
        elif self.p.resolution_is_fix and not self.p.psize_is_fix:
            dq1 = 2 * np.pi / \
                (self.shape[1] * self.resolution[1] * self.costheta)
            dq2 = 2 * np.pi / (self.shape[2] * self.resolution[2])
            dq3 = 2 * np.pi / \
                (self.shape[0] * self.resolution[0] * self.costheta)
            self.p.psize[1] = dq1 * self.distance * self.lam / (2 * np.pi)
            self.p.psize[2] = dq2 * self.distance * self.lam / (2 * np.pi)
            self.p.psize[0] = np.rad2deg(
                dq3 * self.lam / (4 * np.pi * self.sintheta))
        else:
            raise ValueError(
                'Both pixel size and sample resolution specified.')

        # These are useful to have on hand
        self.dq1 = dq1
        self.dq2 = dq2
        self.dq3 = dq3

        # Establish transforms between coordinate systems
        # ...from {x z y} to {r3 r1 r2}
        self.A_r3r1r2 = [[1 / self.costheta, 0, 0],
                         [-self.sintheta / self.costheta, 1, 0],
                         [0, 0, 1]]
        # ...from {r3 r1 r2} to {x z y}
        self.A_xzy = [[self.costheta, 0, 0],
                      [self.sintheta, 1, 0],
                      [0, 0, 1]]
        # ...from {qx qz qy} to {q3 q1 q2}
        self.A_q3q1q2 = [[1, self.sintheta / self.costheta, 0],
                         [0, 1 / self.costheta,             0],
                         [0, 0,                             1]]
        # ...from {q3 q1 q2} to {qx qz qy}
        self.A_qxqzqy = [[1, -self.sintheta, 0],
                         [0, self.costheta,  0],
                         [0, 0,              1]]

        # Update the propagator too
        if update_propagator:
            self.propagator.update()

    @property
    def theta_bragg(self):
        return self.p.theta_bragg

    @theta_bragg.setter
    def theta_bragg(self, v):
        self.p.theta_bragg = v
        if self.interact:
            self.update()

    @property
    def sintheta(self):
        return np.sin(np.deg2rad(self.theta_bragg))

    @property
    def costheta(self):
        return np.cos(np.deg2rad(self.theta_bragg))

    @property
    def tantheta(self):
        return np.tan(np.deg2rad(self.theta_bragg))

    @property
    def resolution(self):
        return self.p.resolution

    @resolution.setter
    def resolution(self, v):
        self.p.resolution[:] = u.expect3(v)
        if self.interact:
            self.update()

    @property
    def psize(self):
        return self.p.psize

    @psize.setter
    def psize(self, v):
        self.p.psize[:] = u.expect3(v)
        if self.interact:
            self.update()

    @property
    def shape(self):
        return self.p.shape

    @shape.setter
    def shape(self, v):
        self.p.shape[:] = u.expect3(v).astype(int)
        if self.interact:
            self.update()

    def _get_propagator(self):
        """
        The real space pixel size in the cartesian system.
        """
        prop = BasicBragg3dPropagator(self)
        return prop

    def _r3r1r2(self, p):
        """
        Transforms a single point from [x z y] to [r3 r1 r2]
        """
        return np.dot(self.A_r3r1r2, p)

    def _xzy(self, p):
        """
        Transforms a single point from [r3 r1 r2] to [x z y]
        """
        return np.dot(self.A_xzy, p)

    def _q3q1q2(self, p):
        """
        Transforms a single point from [qx qz qy] to [q3 q1 q2]
        """
        return np.dot(self.A_q3q1q2, p)

    def _qzqyqx(self, p):
        """
        Transforms a single point from [q3 q1 q2] to [qx qz qy]
        """
        return np.dot(self.A_qxqzqy, p)

    def transformed_grid(self, grids, input_space='real', input_system='natural'):
        """

        Transforms a coordinate grid between the cartesian and natural
        coordinate systems in real or reciprocal space.

        Parameters
        ----------
        grids : 3-tuple of 3-dimensional arrays: (x, z, y), 
                (r3, r1, r2), (qx, qz, qy), or (q3, q1, q2),
                or a 3-dimensional Storage instance.

        input_space: `real` or `reciprocal`

        input_system: `cartesian` or `natural`

        """

        if isinstance(grids, Storage):
            grids = grids.grids()

        # choose transformation operator: 4 cases
        if input_space == 'real' and input_system == 'natural':
            r3, r1, r2 = grids
            z = r1 + self.sintheta * r3
            y = r2
            x = self.costheta * r3
            return x, z, y
        elif input_space == 'real' and input_system == 'cartesian':
            x, z, y = grids
            r1 = z - self.sintheta / self.costheta * x
            r2 = y
            r3 = 1 / self.costheta * x
            return r3, r1, r2
        elif input_space == 'reciprocal' and input_system == 'natural':
            q3, q1, q2 = grids
            qz = self.costheta * q1
            qy = q2
            qx = q3 - self.sintheta * q1
            return qx, qz, qy
        elif input_space == 'reciprocal' and input_system == 'cartesian':
            qx, qz, qy = grids
            q1 = 1 / self.costheta * qz
            q2 = qy
            q3 = qx + self.sintheta / self.costheta * qz
            return q3, q1, q2
        else:
            raise ValueError('invalid options')


    def coordinate_shift(self, input_storage, input_space='real',
                         input_system='natural', keep_dims=True,
                         layer=0):
        """ 
        Transforms a 3D storage between the cartesian and natural
        coordinate systems in real or reciprocal space by simply rolling
        the axes. It tries to do this symmetrically so that the center
        is maintained.

        Note that this transform can be done in any way, and always
        involves the choice of a new grid. This method (arbitrarily)
        chooses the grid which results from skewing the along the
        z/qx direction for real/reciprocal space.

        The shifting is identical to doing a nearest neighbor
        interpolation, and it would not be difficult to use other
        interpolation orders by instead shifting an index array and
        using scipy.ndimage.interpolation.map_coordinates(). But then
        you have to decide how to interpolate complex numbers.

        Parameters
        ----------
        input_storage : The storage to operate on

        input_space: `real` or `reciprocal`

        input_system: `cartesian` or `natural`

        keep_dims : If True, maintain pixel size and number of pixels.
        If False, keeps all the data of the input storage, which means
        that the shape of the output storage will be larger than the
        input.

        """

        C_ = Container(data_type=input_storage.dtype, data_dims=3)
        S = input_storage

        # Four cases. In real and reciprocal space, these skewing
        # operations are done along different axes. For each space, the
        # direction of the transform is taken care of.

        if input_space == 'real':
            # create a padded copy of the data array
            shape = S.shape[1:]
            pad = int(np.ceil(self.sintheta *
                              shape[0] * S.psize[0] / S.psize[1]))
            d = np.pad(S.data[layer], pad_width=(
                (0, 0), (0, pad), (0, 0)), mode='constant')
            # walk along the r3/x axis and roll the r1/z axis. the
            # array is padded at the bottom (high indices) so the
            # actual shifts have to be positive.
            for i in range(shape[0]):
                if input_system == 'cartesian':
                    # roll the z axis in the negative direction for more
                    # positive x
                    shift = int(
                        round((shape[0] - i) * S.psize[0] * self.sintheta / S.psize[1]))
                elif input_system == 'natural':
                    # roll the r1 axis in the positive direction for more
                    # positive r3
                    shift = int(
                        round(i * S.psize[0] * self.sintheta / S.psize[1]))
                d_old = np.copy(d)
                d[i, :, :] = np.roll(d[i, :, :], shift, axis=0)

            # optionally crop the new array
            if keep_dims:
                d = d[:, pad // 2:shape[1] + pad // 2, :]
            # construct a new Storage
            if input_system == 'cartesian':
                new_psize = S.psize * np.array([1 / self.costheta, 1, 1])
            elif input_system == 'natural':
                new_psize = S.psize * np.array([self.costheta, 1, 1])
            old_center = S.origin + S.psize * shape / 2
            S_out = C_.new_storage(ID='S0', psize=new_psize,
                                   padonly=False, shape=None)
            V = View(container=C_, storageID='S0', coord=old_center,
                     shape=d.shape, psize=new_psize)
            S_out.reformat()
            # should use the view here, but there is a bug (#74)
            S_out.data[0] = d

        elif input_space == 'reciprocal':
            # create a padded copy of the data array
            shape = S.shape[1:]
            pad = int(np.ceil(self.sintheta * shape[1]))
            d = np.pad(S.data[layer], pad_width=(
                (0, pad), (0, 0), (0, 0)), mode='constant')
            # walk along the q1/qz axis and roll the q3/qx axis. the
            # array is padded at the right (high indices) so the
            # actual shifts have to be positive.
            for i in range(shape[1]):
                if input_system == 'cartesian':
                    # roll the qx axis in the positive direction for more
                    # positive qz
                    shift = int(round(i * self.sintheta))
                elif input_system == 'natural':
                    # roll the q3 axis in the positive direction for more
                    # negative q1
                    shift = int(round((shape[1] - i) * self.sintheta))
                d[:, i, :] = np.roll(d[:, i, :], shift, axis=0)
            # optionally crop the new array
            if keep_dims:
                d = d[pad // 2:shape[0] + pad // 2, :, :]
            # construct a new Storage
            if input_system == 'cartesian':
                new_psize = S.psize * np.array([1, 1 / self.costheta, 1])
            elif input_system == 'natural':
                new_psize = S.psize * np.array([1, self.costheta, 1])
            old_center = S.origin + S.psize * shape / 2
            S_out = C_.new_storage(ID='S0', psize=new_psize,
                                   padonly=False, shape=None)
            V = View(container=C_, storageID='S0', coord=old_center,
                     shape=d.shape, psize=new_psize)
            S_out.reformat()
            # should use the view here, but there is a bug (#74)
            S_out.data[0] = d

        return S_out

    def prepare_3d_probe(self, S_2d, auto_center=False, system='cartesian', layer=0):
        """

        Prepare a three-dimensional probe from a two-dimensional incident wavefront.

        Parameters
        ----------
        S_2d : Two-dimensional storage holding a (typically  complex)
        wavefront. The absolute positions are not important, only the
        pixel sizes. The first index is interpreted as the quasi-
        vertical axis zi (coincident with z at theta=0), and the second
        index as yi (coincident always with y and r2).

        center : If true, the input wavefront is centered at the
        intensity center of mass.

        system : `cartesian` or `natural`, the coordinate system of the
        returned object.

        layer : which layer of the 2d probe to use

        """

        if auto_center:
            raise NotImplementedError

        # storage in natural space to fill with the probe
        C = Container(data_type=S_2d.dtype, data_dims=3)
        if system == 'natural':
            View(C, storageID='S0000', psize=self.resolution, shape=self.shape)
            S_3d = C.storages['S0000']
        elif system == 'cartesian':
            View(C, storageID='S0000', psize=self.resolution * np.array([self.costheta, 1, 1]), shape=self.shape)
            S_3d = C.storages['S0000']

        # center both storages (meaning that the central pixel is the
        # physical origin)
        S_3d.center = np.array(S_3d.shape[1:]) // 2
        S_2d.center = np.array(S_2d.shape[1:]) // 2

        # find the physical coordinates (zi, yi) of each point in the 3d probe
        if system == 'natural':
            r3, r1, r2 = S_3d.grids()
            r3, r1, r2 = r3[0], r1[0], r2[0]  # layer 0
            x, z, y = self.transformed_grid((r3, r1, r2), input_space='real', input_system='natural')
            zi = x * self.sintheta + z * (1/self.costheta - self.sintheta * self.tantheta)
            yi = y
        elif system == 'cartesian':
            x, z, y = S_3d.grids()
            x, z, y = x[0], z[0], y[0]
            zi = x * self.sintheta + z * (1/self.costheta - self.sintheta * self.tantheta)
            yi = y

        # find the corresponding indices into S.data[layer]
        zi[:] = zi / S_2d.psize[1] + S_2d.center[1]
        yi[:] = yi / S_2d.psize[1] + S_2d.center[1]

        # interpolate
        if np.iscomplexobj(S_2d.data):
            S_3d.data[0][:] = map_coordinates(np.abs(S_2d.data[layer]), (zi, yi))
            S_3d.data[0][:] *= np.exp(1j * map_coordinates(np.angle(S_2d.data[layer]), (zi, yi)))
        else:
            S_3d.data[0][:] = map_coordinates(S_2d.data[layer], (zi, yi))

        #import ipdb; ipdb.set_trace()

        return S_3d

    def probe_extent_vs_fov(self):
        """
        Calculates the extent of the field of view as seen from the
        incoming beam. This is the size of the smallest probe (along its
        vertical direction zi and horizontal direction yi) which
        completely covers the field of view.

        Returns: zi_extent, yi_extent
        """
        g = self
        b, a, c = g.shape * g.resolution
        ap = a + b * g.sintheta
        bp = b * g.costheta
        y = np.sqrt(ap**2 + bp**2)
        gamma = np.arcsin(ap / y)
        phi = (np.pi / 2 - gamma - np.deg2rad(g.theta_bragg))
        zi_extent = np.cos(phi) * y
        yi_extent = c

        return zi_extent, yi_extent


class BasicBragg3dPropagator(object):
    """
    Just a wrapper for the n-dimensional FFT, no other Bragg-specific 
    magic applied here (at the moment).
    """

    def __init__(self, geo=None, ffttype='numpy'):
        self.geo = geo
        if ffttype == 'numpy':
            self.fft = np.fft.fftn
            self.ifft = np.fft.ifftn
        elif ffttype == 'fftw':
            import pyfftw
            self.fft = pyfftw.interfaces.numpy_fft.fftn
            self.ifft = pyfftw.interfaces.numpy_fft.ifftn

    def update(self):
        """
        Update any internal buffers.
        """
        return

    def fw(self, a):
        return np.fft.fftshift(self.fft(a))

    def bw(self, a):
        return self.ifft(np.fft.ifftshift(a))
