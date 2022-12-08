# -*- coding: utf-8 -*-
"""
Container management.

This module defines flexible containers for the various quantities needed
for ptychographic reconstructions.

**Container class**
    A high-level container that keeps track of sub-containers (Storage)
    and Views onto them. A container can copy itself to produce a buffer
    needed for calculations. Some basic Mathematical operations are
    implemented at this level as in place operations.
    In general, operations on arrays should be done using the Views, which
    simply return numpy arrays.

**Storage class**
    The sub-container, wrapping a numpy array buffer. A Storage defines a
    system of coordinate (for now only a scaled translation of the pixel
    coordinates, but more complicated affine transformation could be
    implemented if needed). The sub-class DynamicStorage can adapt the size
    of its buffer (cropping and/or padding) depending on the Views.

**View class**
    A low-weight class that contains all information to access a 2D piece
    of a Storage within a Container. The basic idea is that the View
    access is controlled by a physical position and its frame, such that
    one is not bothered by memory/array addresses when accessing data.

This file is part of the PTYPY package.

    :copyright: Copyright 2014 by the PTYPY team, see AUTHORS.
    :license: see LICENSE for details.

"""
import numpy as np
import weakref
from collections import OrderedDict

try:
    from pympler.asizeof import asizeof
    use_asizeof = True
except ImportError:
    use_asizeof = False

from .. import utils as u
from ..utils.verbose import logger
from ..utils.parameters import PARAM_PREFIX

__all__ = ['Container', 'Storage', 'View', 'POD', 'Base', 'DEFAULT_PSIZE',
           'DEFAULT_SHAPE']  # IDManager']

# Default pixel size
DEFAULT_PSIZE = 1.

# Default shape
DEFAULT_SHAPE = (1, 1, 1)

# Expected structure for Views initialization.
DEFAULT_ACCESSRULE = u.Param(
    # ID of storage (might not exist), (int)
    storageID=None,
    # Shape of the view in pixels, (2-tuple)
    shape=None,
    # Physical coordinates of the center of the view, (2-tuple)
    coord=None,
    # Pixel size (required for storage initialization), (float or None)
    psize=DEFAULT_PSIZE,
    # Index of the third dimension if applicable, (int)
    layer=0,
    active=True,
)

BASE_PREFIX = 'B'
VIEW_PREFIX = 'V'
CONTAINER_PREFIX = 'C'
STORAGE_PREFIX = 'S'
POD_PREFIX = 'P'
MODEL_PREFIX = 'mod'
PTYCHO_PREFIX = 'pty'
GEO_PREFIX = 'G'

# Hard-coded limit in array size
# TODO: make this dynamic from available memory.
MEGAPIXEL_LIMIT = 50


class Base(object):

    _CHILD_PREFIX = 'ID'
    _PREFIX = BASE_PREFIX
    
    __slots__ = ['ID','numID','owner','_pool','_recs','_record']
    _fields = [('ID','<S16')]
    
    def __init__(self, owner=None, ID=None, BeOwner=True):
        """
        Ptypy Base class to support some kind of hierarchy,
        conversion to and from dictionaries as well as a 'cross-node' ID
        management of python objects

        Parameters
        ----------
        owner : Other subclass of Base or Base
            Owner gives IDs to other ptypy objects that refer to it
            as owner. Owner also keeps a reference to these objects in
            its internal _pool where objects are key-sorted according
            to their ID prefix

        ID : None, str or int

        BeOwner : bool
            Set to `False` if this instance is not intended to own other
            ptypy objects.
        """
        self.owner = owner
        self.ID = ID
        self.numID = None
        
        # Try register yourself to your owner if it exists
        if isinstance(owner,Base):
            owner._new_ptypy_object(obj=self)
        else:
            self._record = None
            logger.debug(
                'Failed registering instance of %s with ID %s to object %s'
                % (type(self), self.ID, owner))
        
        # Make a pool for your own ptypy objects
        self._pool = {} if BeOwner else None
        self._recs = {} if BeOwner else None

    def _new_ptypy_object(self, obj):
        """
        Registers a new ptypy object into this object's pool and records.

        Parameters:
        -----------
        obj : [any object] or None
            The object to register.
        """
        try:
            prefix = obj._PREFIX
        except:
            prefix = self._CHILD_PREFIX

        if self._pool.get(prefix) is None:
            self._pool[prefix] = OrderedDict()
            self._recs[prefix] = np.zeros((8,),dtype=obj.__class__._fields)
            
        d = self._pool[prefix]
        # Check if ID is already taken and assign a new one
        ID = obj.ID
        if valid_ID(obj):
            if ID in d:
                logger.error('Overwriting ID %s in pool of %s'
                               % (ID, self.ID))
            else:
                nID = ID
        else:
            try:
                if str(ID) == ID:
                    nID = prefix + ID
                else:
                    nID = prefix + self._num_to_id(ID)
                if nID in d:
                    logger.error('Overwriting ID %s in pool of %s'
                                   % (nID, self.ID))
            except:
                idx = len(d)
                nID = prefix + self._num_to_id(idx)
                while nID in d:
                    idx += 1
                    nID = prefix + self._num_to_id(idx)
            
        d[nID] = obj
        obj.ID = nID
        idx = len(d)
        obj.numID = idx
        recs = self._recs[prefix]
        l = len(recs)
        if idx >= l:
            nl = l + 8192 if idx > 10000 else 2*l
            recs = np.resize(recs,(nl,))
            self._recs[prefix] = recs
        rec = recs[idx] 
        obj._record = rec
        rec['ID'] = nID
        
        return
        
    @staticmethod
    def _num_to_id(num):
        """
        maybe more sophisticated in future
        """
        return '%04d' % num

    @classmethod
    def _from_dict(cls, dct):
        """
        Create new instance from dictionary dct
        should be compatible with _to_dict()
        """
        inst = cls.__new__(cls)
        for k in inst.__slots__:
            if k not in dct:
                continue
            else:
                setattr(inst,k ,dct[k])
        if hasattr(inst,'__dict__'):
            inst.__dict__.update(dct)
        
        # Calling post dictionary import routine (empty in base)
        inst._post_dict_import()

        return inst

    def _post_dict_import(self):
        """
        Change here the specific behavior of child classes after
        being imported using _from_dict()
        """
        pass

    def _to_dict(self):
        """
        Extract all necessary information from object and store them in
        a dictionary. Overwrite in child for custom behavior.
        Default. Returns shallow copy of internal dict as default
        """
        res = OrderedDict()
        for k in self.__slots__:
            res[k] = getattr(self, k)
        if hasattr(self, '__dict__'):
            res.update(self.__dict__.copy())
        return res

    def calc_mem_usage(self):
        space = 64   # that is for the class itself
        pool_space = 0
        npy_space = 0
        if hasattr(self, '_pool'):
            if use_asizeof:
                space += asizeof(self._pool, limit=0)
            for k, v in self._pool.items():
                if use_asizeof:
                    space += asizeof(v, limit=0)
                for kk, vv in v.items():
                    pool_space += vv.calc_mem_usage()[0]
        
        if hasattr(self, '__dict__'):
            for k, v in self.__dict__.items():
                if issubclass(type(v), Base):
                    continue
                elif str(k) == '_pool' or str(k) == 'pods':
                    continue
                else:
                    if use_asizeof:
                        s = asizeof(v)
                        space += s
                    if type(v) is np.ndarray:
                        npy_space += v.nbytes

        return space + pool_space + npy_space, pool_space, npy_space



def get_class(ID):
    """
    Determine ptypy class from unique `ID`
    """
    if ID.startswith(VIEW_PREFIX):
        return View
    elif ID.startswith(PTYCHO_PREFIX):
        from .ptycho import Ptycho
        return Ptycho
    elif ID.startswith(STORAGE_PREFIX):
        return Storage
    elif ID.startswith(CONTAINER_PREFIX):
        return Container
    elif ID.startswith(BASE_PREFIX):
        return Base
    elif ID.startswith(POD_PREFIX):
        return POD
    elif ID.startswith(PARAM_PREFIX):
        return u.Param
    elif ID.startswith(MODEL_PREFIX):
        from .manager import ModelManager
        return ModelManager
    elif ID.startswith(GEO_PREFIX):
        from .geometry import Geo
        return Geo
    else:
        return None


def valid_ID(obj):
    """
    check if ID of object `obj` is compatible with the current format
    """
    valid = False
    try:
        cls = get_class(obj.ID)
        valid = (type(obj) is cls)
    except:
        pass

    return valid


class Storage(Base):
    """
    Inner container handling access to data arrays.

    This class essentially manages access to an internal numpy array
    buffer called :py:attr:`~Storage.data`

    * It returns a view to coordinates given (slicing)
    * It contains a physical coordinate grid

    """

    _PREFIX = STORAGE_PREFIX

    def __init__(self, container, ID=None, data=None, shape=DEFAULT_SHAPE, 
                 fill=0., psize=1., origin=None, layermap=None, padonly=False,
                 padding=0, **kwargs):
        """
        Parameters
        ----------
        container : Container
            The container instance

        ID : None, str or int
            A unique ID, managed by the parent, if None ID is generated
            by parent.

        data: ndarray or None
            A numpy array to use as initial buffer.
            *Deprecated* in v0.3, use fill method instead.

        shape : tuple, int, or None
            The shape of the buffer. If None or int, the dimensionality
            is found from the owning Container instance. Otherwise the
            dimensionality is `len(shape)-1`.

        fill : float or complex
            The default value to fill storage with, will be converted to
            data type of owner.

        psize : float or (ndim)-tuple of float
            The physical pixel size.

        origin : (ndim)-tuple of int
            The physical coordinates of the [0,..,0] pixel (upper-left
            corner of the storage buffer).

        layermap : list or None
            A list (or 1D numpy array) mapping input layer indices
            to the internal buffer. This may be useful if the buffer
            contains only a portion of a larger data set (as when using
            distributed data with MPI). If None, provide direct access.
            to the 3d internal data.

        padonly: bool
            If True, reformat() will enlarge the internal buffer if needed,
            but will not shrink it.

        padding: int
            Number of pixels (voxels) to add as padding around the area defined
            by the views.
        """
        super(Storage, self).__init__(container, ID)

        #: Default fill value
        self.fill_value = fill if fill is not None else 0.

        # For documentation
        #: Three/four or potentially N-dimensional array as data buffer
        self.data = None

        # Additional padding around tight field of view
        self.padding = padding

        # dimensionality suggestion from container
        ndim = container.ndim if container.ndim is not None else 2

        if shape is None:
            shape = (1,) + (1,) * ndim
            #shape = (1,) + (1 + 2*self.padding,) * ndim
        elif np.isscalar(shape):
            shape = (1,) + (int(shape),) * ndim
            #shape = (1,) + (int(shape+2*self.padding),) * ndim
        else:
            shape = tuple(shape)
            #shape = (shape[0],) + tuple(x+2*self.padding for x in shape[1:])

        if len(shape) not in [3, 4]:
            logger.warning('Storage view access dimension %d is not in regular '
                        'scope (2,3). Behavior is untested.' % len(shape[1:]))

        self.shape = shape
        self.data = np.empty(self.shape, self.dtype)
        self.data.fill(self.fill_value)

        """
        # Set data buffer
        if data is None:
            # Create data buffer
            self.shape = shape
            self.data = np.empty(self.shape, self.dtype)
            self.data.fill(self.fill_value)
        else:
            # Set initial buffer. Casting the type makes a copy
            data = np.asarray(data).astype(self.dtype)
            if data.ndim < self.ndim or data.ndim > (self.ndim + 1):
                raise ValueError(
                    'For `data_dims` = %d, initial buffer must be'
                    ' %dD or %dD, this one is %dD'
                    % (self.ndim, self.ndim, self.ndim+1, data.ndim))
            elif data.ndim == self.ndim:
                self.data = data.reshape((1,) + data.shape)
            else:
                self.data = data
            self.shape = self.data.shape
        """

        if layermap is None:
            layermap = list(range(len(self.data)))
        self.layermap = layermap

        # This is most often not accurate. Set this quantity from the outside
        self.nlayers = len(layermap)

        # Need to bootstrap the parameters.
        # We set the initial center to the middle of the array
        self._center = u.expectN(self.shape[-self.ndim:], self.ndim) // 2

        # Set pixel size (in physical units)
        self.psize = psize if psize is not None else DEFAULT_PSIZE

        # Set origin (in physical units - same as psize)
        if origin is not None:
            self.origin = origin

        # Used to check if data format is appropriate.
        self.DataTooSmall = False

        # Padding vs padding + cropping when reformatting.
        self.padonly = padonly

        # A flag
        self.model_initialized = False

        # MPI flag: is the storage distributed across nodes or are all nodes holding the same copy?
        self._is_scattered = container._is_scattered

        # Instance attributes
        # self._psize = None
        # SC: defining _psize here leads to failure of the code,
        # solution required
        # self._origin = None

    @property
    def ndim(self):
        """
        Number of dimensions for :any:`View` access
        """
        return len(self.shape[1:])

    @property
    def dtype(self):
        return self.owner.dtype if self.owner is not None else None

    def copy(self, owner=None, ID=None, fill=None):
        """
        Return a copy of this storage object.

        Note: the returned copy has the same container as self.

        Parameters
        ----------
        ID : str or int
             A unique ID, managed by the parent
        fill : scalar or None
               If float, set the content to this value. If None, copy the
               current content.
        """

        # if fill is None:
        #     # Return a new Storage or sub-class object with a copy of the data.
        #     return type(self)(owner,
        #                       ID,
        #                       data=self.data.copy(),
        #                       psize=self.psize,
        #                       origin=self.origin,
        #                       layermap=self.layermap)
        # else:
        #     # Return a new Storage or sub-class object with an empty buffer
        new_storage = type(self)(owner,
                                 ID,
                                 shape=self.shape,
                                 psize=self.psize,
                                 origin=self.origin,
                                 layermap=self.layermap,
                                 padding=self.padding)
        if fill is not None:
            new_storage.fill(fill)
        else:
            new_storage.fill(self.data.copy())
        return new_storage

    def fill(self, fill=None):
        """
        Fill managed buffer.

        Parameters
        ----------
        fill : scalar, numpy array or None.
               Fill value to use. If fill is a numpy array, it is cast
               as self.dtype and self.shape is updated to reflect the
               new buffer shape. If fill is None, use default value
               (self.fill_value).
        """
        if self.data is None:
            self.data = np.empty(self.shape)

        if fill is None:
            # Fill with default fill value
            self.data.fill(self.fill_value)
        elif np.isscalar(fill):
            # Fill with scalar value
            self.data.fill(fill)
            self.fill_value = fill
        elif type(fill) is np.ndarray:
            # Replace the buffer
            if fill.ndim < self.ndim or fill.ndim > (self.ndim + 1):
                raise ValueError(
                    'For `data_dims` = %d, initial buffer must be'
                    ' %dD or %dD, this one is %dD'
                    % (self.ndim, self.ndim, self.ndim+1, fill.ndim))
            elif fill.ndim == self.ndim:
                fill = np.resize(fill, (self.shape[0],) + fill.shape)
            self.data = fill.astype(self.dtype)
            self.shape = self.data.shape

    def update(self):
        """
        Update internal state, including all views on this storage to
        ensure consistency with the physical coordinate system.
        """
        # Update the access information for the views
        # (i.e. pcoord, dlow, dhigh and sp)
        # do this only for the original container 
        # to avoid iterating through all the views when creating copies
        if self.owner.original is self.owner:
            self.update_views()

    def update_views(self, v=None):
        """
        Update the access information for a given view.

        Parameters
        ----------
        v : View or None
            The view object to update. If None, loop through all views.
            Apart from that, no check is done, not even whether
            the view is actually on self. Use cautiously.
        """
        if v is None:
            for v in self.views:
                self.update_views(v)
            return

        if not self.ndim == v.ndim:
            raise ValueError(
                'Storage %s(ndim=%d) and View %s(ndim=%d) have conflicting '
                'data dimensions' % (self.ID, self.ndim, v.ID, v.ndim))

        # Synchronize pixel size
        v.psize = self.psize #.copy()

        # Convert the physical coordinates of the view to pixel coordinates
        pcoord = self._to_pix(v.coord)

        # Integer part (note that np.round is not stable for odd arrays)
        v.dcoord = np.round(pcoord + 0.00001).astype(int)

        # These are the important attributes used when accessing the data
        v.dlow = v.dcoord - v.shape // 2
        v.dhigh = v.dcoord + (v.shape + 1) // 2

        # Subpixel offset
        v.sp = pcoord - v.dcoord
        # if self.layermap is None:
        #     v.slayer = 0
        # else:
        #     v.slayer = self.layermap.index(v.layer)

    def reformat(self, newID=None, update=True):
        """
        Crop or pad if required.

        Parameters
        ----------
        newID : str or int
            If None (default) act on self. Otherwise create a copy
            of self before doing the cropping and padding.
            
        update : bool
            If True, updates all Views before reformatting. Not necessarily
            needed, if Views have been recently instantiated. Roughly doubles 
            execution time.

        Returns
        -------
        s : Storage
            returns new Storage instance in same :any:`Container` if
            `newId` is not None.
        """
        # If a new storage is requested, make a copy.
        if newID is not None:
            s = self.copy(newID)
            s.reformat()
            return s

        # Make sure all views are up to date
        # This call takes roughly half the time of .reformat()
        if update:
            self.update()

        # List of views on this storage
        views = self.views

        logger.debug('%s[%s] :: %d views for this storage'
                     % (self.owner.ID, self.ID, len(views)))

        sh = self.data.shape

        # Loop through all active views to get individual boundaries
        dlow_fov = [np.inf] * self.ndim
        dhigh_fov = [-np.inf] * self.ndim
        layers = []
        dims = list(range(self.ndim))
        for v in views:
            if not v.active:
                continue

            # Accumulate the regions of interest to
            # compute the full field of view
            for d in dims:
                dlow_fov[d] = min(dlow_fov[d], v.dlow[d])
                dhigh_fov[d] = max(dhigh_fov[d], v.dhigh[d])
                
            # Gather a (unique) list of layers
            if v.layer not in layers:
                layers.append(v.layer)

        # Check if storage is scattered
        # A storage is "scattered" if and only if layer maps are different across nodes.
        new_layermap = sorted(layers)

        # Update boundaries
        if not self._is_scattered and u.parallel.MPIenabled:
            dlow_fov[:]  = u.parallel.comm.allreduce(dlow_fov,  u.parallel.MPI.MIN)
            dhigh_fov[:] = u.parallel.comm.allreduce(dhigh_fov, u.parallel.MPI.MAX)

        # Return if no views, it is important that this only happens after self._is_scattered is updated 
        if not views:
            return self

        sh = self.data.shape

        # Compute Nd misfit (distance between the buffer boundaries and the
        # region required to fit all the views)
        misfit = self.padding + np.array([[-dlow_fov[d], dhigh_fov[d] - sh[d+1]] for d in dims])

        _misfit_str = ', '.join(['%s' % m for m in misfit])
        logger.debug('%s[%s] :: misfit = [%s]'
                     % (self.owner.ID, self.ID, _misfit_str))

        posmisfit = (misfit > 0)
        negmisfit = (misfit < 0)

        needtocrop_or_pad = (posmisfit.any() or
                             (negmisfit.any() and not self.padonly))

        if posmisfit.any() or negmisfit.any():
            logger.debug(
                'Storage %s of container %s has a misfit of [%s] between '
                'its data and its views'
                % (str(self.ID), str(self.owner.ID), _misfit_str))
        
        if needtocrop_or_pad:
            if self.padonly:
                misfit[negmisfit] = 0

            # Recompute center and shape
            new_center = self.center + misfit[:, 0]
            new_shape = (sh[0],)
            for d in dims:
                new_shape += (sh[d+1] + misfit[d].sum(),)

            logger.debug('%s[%s] :: center: %s -> %s'
                         % (self.owner.ID, self.ID, str(self.center),
                            str(new_center)))
            # logger.debug('%s[%s] :: shape: %s -> %s'
            #              % (self.owner.ID, self.ID, str(sh), str(new_shape)))

            megapixels = np.array(new_shape).astype(float).prod() / 1e6
            if megapixels > MEGAPIXEL_LIMIT:
                raise RuntimeError('Arrays larger than %dM not supported. You '
                                   'requested %.2fM pixels.' % (MEGAPIXEL_LIMIT, megapixels))

            # Apply Nd misfit
            if self.data is not None:
                new_data = u.crop_pad(
                    self.data,
                    misfit,
                    fillpar=self.fill_value).astype(self.dtype)
            else:
                new_data = np.empty(new_shape, self.dtype)
                new_data.fill(self.fill_value)
        else:
            # Nothing changes for now
            new_data = self.data
            new_shape = sh
            new_center = self.center
        
        # Deal with layermap
        if self.layermap != new_layermap:
            relaid_data = []
            for i in new_layermap:
                if i in self.layermap:
                    # This layer already exists
                    d = new_data[self.layermap.index(i)]
                else:
                    # A new layer
                    d = np.empty(new_shape[-self.ndim:], self.dtype)
                    d.fill(self.fill_value)
                relaid_data.append(d)
            new_data = np.array(relaid_data)
            new_shape = new_data.shape
            self.layermap = new_layermap

        self.nlayers = len(new_layermap)
        
        # set layer index in the view
        for v in views:
            v.dlayer = self.layermap.index(v.layer)

        logger.debug('%s[%s] :: shape: %s -> %s'
                     % (self.owner.ID, self.ID, str(sh), str(new_shape)))
        # Store new buffer
        self.data = new_data
        self.shape = new_shape
        self.center = new_center
                
    def _to_pix(self, coord):
        """
        Transforms physical coordinates `coord` to pixel coordinates.

        Parameters
        ----------
        coord : tuple or array-like
            A ``(N,2)``-array of the coordinates to be transformed
        """
        return (coord - self.origin) / self.psize

    def _to_phys(self, pix):
        """
        Transforms pixel coordinates `pix` to physical coordinates.

        Parameters
        ----------
        pix : tuple or array-like
            A ``(N,2)``-array of the coordinates to be transformed
        """
        return pix * self.psize + self.origin

    @property
    def psize(self):
        """
        The pixel size.
        """
        return self._psize

    @psize.setter
    def psize(self, v):
        """
        Set the pixel size, and update all the internal variables.
        """
        self._psize = u.expectN(v, self.ndim)
        self._origin = - self._center * self._psize
        self.update()

    @property
    def origin(self):
        """
        Return the physical position of the upper-left corner of the storage.
        """
        return self._origin

    @origin.setter
    def origin(self, v):
        """
        Set the origin and update all the internal variables.
        """
        self._origin = u.expectN(v, self.ndim)
        self._center = - self._origin / self._psize
        self.update()

    @property
    def center(self):
        """
        Return the position of the origin relative to the upper-left corner
        of the storage, in pixel coordinates
        """
        return self._center

    @center.setter
    def center(self, v):
        """
        Set the center and update all the internal variables.
        """
        self._center = u.expectN(v, self.ndim)
        self._origin = - self._center * self._psize
        self.update()

    @property
    def views(self):
        """
        Return all the views that refer to this storage.
        """
        if self.owner is not None:
            return self.owner.views_in_storage(self)
        else:
            return None

    def allreduce(self, op=None):
        """
        Performs MPI parallel ``allreduce`` with a default sum as
        reduction operation for internal data buffer ``self.data``.
        This method does nothing if the storage is distributed across
        nodes.

        :param op: Reduction operation. If ``None`` uses sum.

        See also
        --------
        ptypy.utils.parallel.allreduce
        Container.allreduce
        """
        if not self._is_scattered:
            u.parallel.allreduce(self.data, op=op)

    def zoom_to_psize(self, new_psize, **kwargs):
        """
        Changes pixel size and zooms the data buffer along last two axis
        accordingly, updates all attached views and reformats if necessary.
        **untested!!**

        Parameters
        ----------
        new_psize : scalar or array_like
                    new pixel size
        """
        new_psize = u.expectN(new_psize, self.ndim)
        sh = np.asarray(self.shape[1:])
        # psize is quantized
        new_sh = np.round(self.psize / new_psize * sh)
        new_psize = self.psize / new_sh * sh

        if (new_sh != sh).any():
            logger.info('Zooming from %s , %s to %s , %s'
                        % (self.psize, sh, new_psize, new_sh.astype(int)))

            # Zoom data buffer.
            # Could be that it is faster and cleaner to loop over first axis
            zoom = new_sh / sh
            self.fill(u.zoom(self.data, [1.0] + [z for z in zoom], **kwargs))

        self._psize = new_psize
        self.zoom_cycle += 1  # !!! BUG: Unresolved attribute reference
        # Update internal coordinate system, while zooming, the coordinate for
        # top left corner should remain the same
        origin = self.origin
        self.origin = origin # this call will also update the views' coordinates

        # reformat everything
        self.reformat()

    def grids(self):
        """
        Returns
        -------
        x, y : ndarray
            grids in the shape of internal buffer
        """
        sh = self.data.shape
        nm = np.indices(sh)[1:]
        flat = nm.reshape((self.ndim, self.data.size))
        c = self._to_phys(flat.T).T
        c = c.reshape((self.ndim,) + sh)
        return tuple(c)

    def get_view_coverage(self):
        """
        Creates an array in the shape of internal buffer where the value
        of each pixel represents the number of views that cover that pixel

        Returns
        -------
        ndarray
            view coverage in the shape of internal buffer
        """
        coverage = np.zeros_like(self.data)
        for v in self.views:
            coverage[v.slice] += 1

        return coverage

    def report(self):
        """
        Returns
        -------
        str
            a formatted string giving a report on this storage.
        """
        info = ["Shape: %s\n" % str(self.data.shape),
                "Pixel size (meters): %g x %g\n" % tuple(self.psize),
                "Dimensions (meters): %g x %g\n"
                % (self.psize[0] * self.data.shape[-2],
                   self.psize[1] * self.data.shape[-1]),
                "Number of views: %d\n" % len(self.views)]

        return ''.join(info)

    def formatted_report(self, table_format=None, offset=8, align='right',
                         separator=" : ", include_header=True):
        r"""
        Returns formatted string and a dict with the respective information

        Parameters
        ----------
        table_format : list, optional
            List of (*item*,*length*) pairs where item is name of the info
            to be listed in the report and length is the column width.
            The following items are allowed:

            - *memory*, for memory usage of the storages and total use
            - *shape*, for shape of internal storages
            - *dimensions*, is ``shape \* psize``
            - *psize*, for pixel size of storages
            - *views*, for number of views in each storage

        offset : int, optional
            First column width

        separator : str, optional
            Column separator.

        align : str, optional
            Column alignment, either ``'right'`` or ``'left'``.

        include_header : bool, optional
            Include a header if True.

        Returns
        -------
        fstring : str
            Formatted string

        dct : dict
            Dictionary containing with the respective info to the keys
            in `table_format`
        """
        fr = _Freport()
        if offset is not None:
            fr.offset = offset
        if table_format is not None:
            fr.table = table_format
        if separator is not None:
            fr.separator = separator
        dct = {}
        fstring = [self.ID.ljust(fr.offset)]

        for key, column in fr.table:
            if str(key) == 'shape':
                dct[key] = tuple(self.data.shape)
                info = ('%d' + ' * %d' * self.ndim) % dct[key]
            elif str(key) == 'psize':
                dec = np.floor(np.log10(self.psize).min())
                dct[key] = tuple(self.psize)
                info = '*'.join(['%.1f' % p for p in self.psize * 10**(-dec)])
                info += 'e%d' % dec
            elif str(key) == 'dimension':
                r = self.psize * self.data.shape[1:]
                dec = np.floor(np.log10(r).min())
                dct[key] = tuple(r)
                info = '*'.join(['%.1f' % p for p in r * 10**(-dec)])
                info += 'e%d' % dec
            elif str(key) == 'memory':
                dct[key] = float(self.data.nbytes) / 1e6
                info = '%.1f' % dct[key]
            elif str(key) == 'dtype':
                dct[key] = self.data.dtype
                info = dct[key].str
            elif str(key) == 'views':
                dct[key] = len(self.views)
                info = str(dct[key])
            else:
                dct[key] = None
                info = ""

            fstring.append(fr.separator)
            if str(align) == 'right':
                fstring.append(info.rjust(column)[-column:])
            else:
                fstring.append(info.ljust(column)[:column])

        if include_header:
            fstring.insert(0, fr.header())

        return ''.join(fstring), dct

    def __getitem__(self, v):
        """
        Storage[v]

        Returns
        -------
        ndarray
            The view to internal data buffer corresponding to View or layer `v`
        """

        # Here things could get complicated.
        # Coordinate transforms, 3D - 2D projection, ...
        # Current implementation: ROI + subpixel shift
        # return shift(self.datalist[v.layer][v.roi[0, 0]:v.roi[1, 0],
        #             v.roi[0, 1]:v.roi[1, 1]], v.sp)
        # return shift(self.data[v.slayer, v.roi[0, 0]:v.roi[1, 0],
        #             v.roi[0, 1]:v.roi[1, 1]], v.sp)
        if isinstance(v, View):
            if self.ndim == 2:
                return shift(self.data[
                             v.dlayer, v.dlow[0]:v.dhigh[0], v.dlow[1]:v.dhigh[1]],
                             v.sp)
            elif self.ndim == 3:
                return shift(self.data[
                             v.dlayer, v.dlow[0]:v.dhigh[0], v.dlow[1]:v.dhigh[1],
                             v.dlow[2]:v.dhigh[2]], v.sp)
        elif v in self.layermap:
            return self.data[self.layermap.index(v)]
        else:
            raise ValueError("View or layer '%s' is not present in storage %s"
                             % (v, self.ID))

    def __setitem__(self, v, newdata):
        """
        Storage[v] = newdata

        Set internal data buffer to `newdata` for the region of view `v`.

        Parameters
        ----------
        v : View
            A View for this storage

        newdata : ndarray
            Two-dimensional array that fits the view's shape
        """

        # Only ROI and shift for now.
        # This part must always be consistent with __getitem__!
        # self.datalist[v.layer][v.roi[0, 0]:v.roi[1, 0],
        #                       v.roi[0, 1]:v.roi[1, 1]] = shift(newdata, -v.sp)
        # self.data[v.slayer, v.roi[0, 0]:v.roi[1, 0],
        #          v.roi[0, 1]:v.roi[1, 1]] = shift(newdata, -v.sp)
        if isinstance(v, View):
            # there must be a nicer way to do this, numpy.take is nearly
            # right, but returns copies and not views.
            if self.ndim == 2:
                self.data[v.dlayer,
                          v.dlow[0]:v.dhigh[0],
                          v.dlow[1]:v.dhigh[1]] = (shift(newdata, -v.sp))
            elif self.ndim == 3:
                self.data[v.dlayer,
                          v.dlow[0]:v.dhigh[0],
                          v.dlow[1]:v.dhigh[1],
                          v.dlow[2]:v.dhigh[2]] = (shift(newdata, -v.sp))
            elif self.ndim == 4:
                self.data[v.dlayer,
                          v.dlow[0]:v.dhigh[0],
                          v.dlow[1]:v.dhigh[1],
                          v.dlow[2]:v.dhigh[2],
                          v.dlow[3]:v.dhigh[3]] = (shift(newdata, -v.sp))
            elif self.ndim == 5:
                self.data[v.dlayer,
                          v.dlow[0]:v.dhigh[0],
                          v.dlow[1]:v.dhigh[1],
                          v.dlow[2]:v.dhigh[2],
                          v.dlow[3]:v.dhigh[3],
                          v.dlow[4]:v.dhigh[4]] = (shift(newdata, -v.sp))
        elif v in self.layermap:
            self.data[self.layermap.index(v)] = newdata
        else:
            raise ValueError("View or layer '%s' is not present in storage %s"
                             % (v, self.ID))

    def __str__(self):
        info = '%15s : %7.2f MB :: ' % (self.ID, self.data.nbytes / 1e6)
        if self.data is not None:
            info += 'data=%s @%s' % (self.data.shape, self.data.dtype)
        else:
            info += 'empty=%s @%s' % (self.shape, self.dtype)
        return info + ' psize=%(_psize)s center=%(_center)s' % self.__dict__


def shift(v, sp):
    """
    Placeholder for future subpixel shifting method.
    """
    return v


class View(Base):
    """
    A 'window' on a Container.

    A view stores all the slicing information to extract a 2D piece
    of Container.

    Note
    ----
    The final structure of this class is yet up to debate
    and the constructor signature may change. Especially since
    'DEFAULT_ACCESSRULE' is yet so small, its contents could be
    incorporated in the constructor call.

    """
    _fields = Base._fields + \
               [('active', 'b1'),
                ('dlayer', '<i8'),
                ('layer', '<i8'), 
                ('dhigh', '(5,)i8'),
                ('dlow', '(5,)i8'),
                ('shape', '(5,)i8'),
                ('dcoord', '(5,)i8'),
                ('psize', '(5,)f8'),
                ('coord', '(5,)f8'),
                ('sp', '(5,)f8')]
    __slots__ = Base.__slots__ + ['_ndim', 'storage', 'storageID', '_pod', '_pods', 'error']
    ########
    # TODO #
    ########
    # - remove numpy array overhead by having only a few numpy arrays stored
    #   in view; access via properties
    # - get rid of self.pods dictionary also due to unnecessary overhead
    # - Don't instantiate the pods slot to save memory

    DEFAULT_ACCESSRULE = DEFAULT_ACCESSRULE
    _PREFIX = VIEW_PREFIX

    def __init__(self, container, ID=None, accessrule=None, **kwargs):
        """
        Parameters
        ----------
        container : Container
            The Container instance this view applies to.

        ID : str or int
            ID for this view. Automatically built from ID if None.

        accessrule : dict
            All the information necessary to access the wanted slice.
            Maybe subject to change as code evolve. See keyword arguments
            Almost all keys of accessrule will be available as attributes
            in the constructed View instance.

        Keyword Args
        ------------
        storageID : str
            ID of storage, If the Storage does not exist
            it will be created! (*default* is ``None``)

        shape : int or tuple of int
            Shape of the view in pixels (*default* is ``None``)

        coord : 2-tuple of float,
            Physical coordinates *(meter)* of the center of the view.

        psize : float or tuple of float
            Pixel size in *(meter)* . Required for storage initialization.

        layer : int
            Index of the third dimension if applicable.
            (*default* is ``0``)

        active : bool
            Whether this view is active (*default* is ``True``)
        """
        super(View, self).__init__(container, ID, False)

        # Prepare a dictionary for PODs (volatile!)
        self._pods = None 
        r""" Potential volatile dictionary for all :any:`POD`\ s that 
            connect to this view. Set by :any:`POD` """

        # A single pod lookup (weak reference), set by POD instance.
        self._pod = None

        self.active = True
        """ Active state. If False this view will be ignored when
            resizing the data buffer of the associated :any:`Storage`."""

        #: The :py:class:`Storage` instance that this view applies to by default.
        self.storage = None

        self.storageID = None
        """ The storage ID that this view will be forward to if applied
            to a :any:`Container`."""

        #: The "layer" i.e. first axis index in Storage data buffer
        self.dlayer = 0

        # The messy stuff
        if accessrule is not None or len(kwargs)>0:
            self._set(accessrule, **kwargs)

    def _set(self, accessrule, **kwargs):
        """
        Store internal info to get/set the 2D data in the container.
        """
        rule = u.Param(self.DEFAULT_ACCESSRULE)
        if accessrule is not None:
            rule.update(accessrule)
        rule.update(kwargs)

        self.active = True if rule.active else False

        self.storageID = rule.storageID

        # shape == None means "full frame"
        self.shape = rule.shape

        # Look for storage, create one if necessary
        s = self.owner.storages.get(self.storageID, None)
        if s is None:
            sh = (1,) + tuple(self.shape) if self.shape is not None else None
            s = self.owner.new_storage(ID=self.storageID,
                                       psize=rule.psize,
                                       origin=rule.coord,
                                       shape=sh)
        self.storage = s


        if self.shape is None:
            self._set_full_frame(s)

        # Information to access the slice within the storage buffer
        self.psize = rule.psize
        self.coord = rule.coord
        self.layer = rule.layer

        if (self.psize is not None
                and not np.allclose(self.storage.psize, self.psize)):
            logger.warning(
                'Inconsistent pixel size when creating view.\n (%s vs %s)'
                % (str(self.storage.psize), str(self.psize)))

        # This ensures self-consistency (sets pixel coordinate and ROI)
        if self.active:
            self.storage.update_views(self)

    def _set_full_frame(self, storage):
        self.shape = storage.shape[1:]
        pcoord = self.shape / 2.
        self.coord = storage._to_phys(pcoord)

    def __str__(self):
        first = ('%s -> %s[%s] : shape = %s layer = %s coord = %s'
                 % (self.owner.ID, self.storage.ID, self.ID,
                    self.shape, self.layer, self.coord))
        if not self.active:
            return first + '\n INACTIVE : slice = ...  '
        else:
            return first + '\n ACTIVE : slice = %s' % str(self.slice)

    def copy(self,ID=None, update = True):
        nView = View(self.owner, ID)
        nView._record = self._record.copy()
        nView._ndim = self._ndim
        nView.storage = self.storage
        nView.storageID = self.storageID
        if update:
            nView.storage.update_views(nView)
        return nView
        
    @property
    def active(self):
        return self._record['active'] 
        
    @active.setter
    def active(self, v):
        self._record['active'] = v
        
    @property
    def dlayer(self):
        return self._record['dlayer']
        
    @dlayer.setter
    def dlayer(self, v):
        self._record['dlayer'] = v
        
    @property
    def layer(self):
        return self._record['layer']
        
    @layer.setter
    def layer(self, v):
        self._record['layer'] = v

    @property
    def ndim(self):
        return self._ndim

    @property
    def slice(self):
        """
        Returns a slice-tuple according to ``self.layer``, ``self.dlow``
        and ``self.dhigh``.
        Please note, that this may not always makes sense
        """
        # if self.layer not in self.storage.layermap:
        #    slayer = None
        # else:
        #    slayer = self.storage.layermap.index(self.layer)

        res = (self.dlayer,)
        for d in range(self.ndim):
            res += (slice(self.dlow[d], self.dhigh[d]),)
        return res

    @property
    def pod(self):
        """
        Returns first :any:`POD` in the ``self.pods`` dict.
        This is a common call in the code and has therefore found
        its way here. May return ``None`` if there is no pod connected.
        """
        if isinstance(self._pod, weakref.ref):
            return self._pod()  # weak reference
        else:
            return self._pod
        
    @property
    def pods(self):
        r"""
        Returns all :any:`POD`\ s still connected to this view as a dict.
        """
        if self._pods is not None:
            return self._pods
        else:
            pod = self.pod
            return {} if pod is None else {pod.ID: pod}  

    @property
    def data(self):
        """
        The view content in data buffer of associated storage.
        """
        return self.storage[self]

    @data.setter
    def data(self, v):
        """
        Set the view content in data buffer of associated storage.
        """
        self.storage[self] = v

    @property
    def shape(self):
        """
        Two dimensional shape of View.
        """

        sh = self._record['shape']
        return None if (sh==0).all() else sh[:self._ndim]

    @shape.setter
    def shape(self, v):
        """
        Set two dimensional shape of View.
        """
        if v is None:
            self._record['shape'][:] = 0
        elif np.isscalar(v):
            sh = (int(v),) * self.owner.ndim
            self._ndim = len(sh)
            self._record['shape'][:len(sh)] = sh
        else:
            self._ndim = len(v)
            self._record['shape'][:len(v)] = v

    @property
    def dlow(self):
        """
        Low side of the View's data range.
        """
        return self._record['dlow'][:self._ndim]

    @dlow.setter
    def dlow(self, v):
        """
        Set low side of the View's data range.
        """
        self._record['dlow'][:self._ndim] = v

    @property
    def dhigh(self):
        """
        High side of the View's data range.
        """
        return self._record['dhigh'][:self._ndim]

    @dhigh.setter
    def dhigh(self, v):
        """
        Set high side of the View's data range.
        """
        self._record['dhigh'][:self._ndim] = v

    @property
    def dcoord(self):
        """
        Center coordinate (index) in data buffer.
        """
        return self._record['dcoord'][:self._ndim]

    @dcoord.setter
    def dcoord(self, v):
        """
        Set high side of the View's data range.
        """
        self._record['dcoord'][:self._ndim] = v

    @property
    def psize(self):
        """
        Pixel size of the View.
        """
        ps = self._record['psize'][:self._ndim]
        return ps if (ps > 0.).all() else None

    @psize.setter
    def psize(self, v):
        """
        Set pixel size.
        """
        if v is None:
            self._record['psize'][:] = 0.
        else:
            self._record['psize'][:self._ndim] = u.expectN(v, self._ndim)

    @property
    def coord(self):
        """
        The View's physical coordinate (meters)
        """
        return self._record['coord'][:self._ndim]

    @coord.setter
    def coord(self, v):
        """
        Set the View's physical coordinate (meters)
        """
        if v is None:
            self._record['coord'][:] = 0.
        elif type(v) is not np.ndarray:
            self._record['coord'][:self._ndim] = u.expectN(v, self._ndim)
        else:
            self._record['coord'][:self._ndim] = v

    @property
    def sp(self):
        """
        The subpixel difference (meters) between physical coordinate
        and data coordinate.
        """
        return self._record['sp'][:self._ndim]

    @sp.setter
    def sp(self, v):
        """
        Set the subpixel difference (meters) between physical coordinate
        and data coordinate.
        """
        if v is None:
            self._record['sp'][:] = 0.
        elif type(v) is not np.ndarray:
            self._record['sp'][:self._ndim] = u.expectN(v, self._ndim)
        else:
            self._record['sp'][:self._ndim] = v

    @property
    def pcoord(self):
        """ 
        The physical coordinate in pixel space
        """
        return self.dcoord + self.sp


class Container(Base):
    """
    High-level container class.

    Container can be seen as a "super-numpy-array" which can contain multiple
    sub-containers of type :any:`Storage`, potentially of different shape,
    along with all :any:`View` instances that act on these Storages to extract
    data from the internal data buffer :any:`Storage.data`.

    Typically there will be five such base containers in a :any:`Ptycho`
    reconstruction instance:

        - `Cprobe`, Storages for the illumination, i.e. **probe**,
        - `Cobj`, Storages for the sample transmission, i.e. **object**,
        - `Cexit`, Storages for the **exit waves**,
        - `Cdiff`, Storages for **diffraction data**, usually one per scan,
        - `Cmask`, Storages for **masks** (and weights), usually one per scan,

    A container can conveniently duplicate all its internal :any:`Storage`
    instances into a new Container using :py:meth:`copy`. This feature is
    intensively used in the reconstruction engines where buffer copies
    are needed to temporarily store results. These copies are referred
    by the "original" container through the property :py:meth:`copies` and
    a copy refers to its original through the attribute :py:attr:`original`
    In order to reduce the number of :any:`View` instances, Container copies
    do not hold views and use instead the Views held by the original container

    Attributes
    ----------
    original : Container
        If self is copy of a Container, this attribute refers to the original
        Container. Otherwise it is None.

    data_type : str
        Either "single" or "double"
    """
    _PREFIX = CONTAINER_PREFIX

    def __init__(self, owner=None, ID=None, data_type='complex', data_dims=2, distribution="cloned"):
        """
        Parameters
        ----------
        ID : str or int
            A unique ID, managed by the parent.

        owner : Base
            A possible subclass of :any:`Base` that holds this Container.

        data_type : str or numpy.dtype
            data type - either a numpy.dtype object or 'complex' or
            'real' (precision is taken from ptycho.FType or ptycho.CType)

        data_dims : int
            dimension of data, can be 2 or 3

        distribution : str
            Indicates if the data is "cloned" in all MPI processes or "scattered"

        """

        super(Container, self).__init__(owner, ID)

        self.data_type = data_type

        #: Default data dimensionality for Views and Storages.
        self.ndim = data_dims
        if self.ndim not in (2, 3):
            logger.warning('Container untested for `data_dim` other than 2 or 3')

        # Prepare for copy
        # self.original = original if original is not None else self
        self.original = self

        # boolean parameter for distributed containers
        self._is_scattered = (distribution == "scattered")

    @property
    def copies(self):
        """
        Property that returns list of all copies of this :any:`Container`
        """
        return [c for c in self.owner._pool[CONTAINER_PREFIX].values()
                if c.original is self and c is not self]

    def delete_copy(self, copy_IDs=None):
        """
        Delete a copy or all copies of this container from owner instance.

        Parameters
        ----------
        copy_IDs : str
            ID of copy to be deleted. If None, deletes *all* copies
        """
        if self.original is self:
            if copy_IDs is None:
                copy_IDs = [c.ID for c in self.copies]
            for cid in copy_IDs:
                del self.owner._pool[CONTAINER_PREFIX][cid]
        else:
            raise RuntimeError(
                'Container copy is not allowed to delete anything.')

    @property
    def dtype(self):
        """
        Property that returns numpy dtype of all internal data buffers
        """
        if self.data_type == 'complex':
            return self.owner.CType if self.owner is not None else np.complex128
        elif self.data_type == 'real':
            return self.owner.FType if self.owner is not None else np.float64
        else:
            return self.data_type

    @property
    def S(self):
        """
        A property that returns the internal dictionary of all
        :any:`Storage` instances in this :any:`Container`
        """
        return self._pool.get(STORAGE_PREFIX, {})

    @property
    def storages(self):
        """
        A property that returns the internal dictionary of all
        :any:`Storage` instances in this :any:`Container`
        """
        return self._pool.get(STORAGE_PREFIX, {})

    @property
    def Sp(self):
        """
        A property that returns the internal dictionary of all
        :any:`Storage` instances in this :any:`Container` as a :any:`Param`
        """
        return u.Param(self.storages)

    @property
    def V(self):
        """
        A property that returns the internal dictionary of all
        :any:`View` instances in this :any:`Container`
        """
        return self._pool.get(VIEW_PREFIX, {})

    @property
    def views(self):
        """
        A property that returns the internal dictionary of all
        :any:`View` instances in this :any:`Container`
        """
        return self._pool.get(VIEW_PREFIX, {})

    @property
    def Vp(self):
        """
        A property that returns the internal dictionary of all
        :any:`View` instances in this :any:`Container` as a :any:`Param`
        """
        return u.Param(self.V)

    @property
    def size(self):
        """
        Return total number of pixels in this container.
        """
        sz = 0
        for ID, s in self.storages.items():
            if s.data is not None:
                sz += s.data.size
        return sz

    @property
    def nbytes(self):
        """
        Return total number of bytes used by numpy array buffers
        in this container. This is not the actual size in memory of the
        whole container, as it does not include the views or dictionary
        overhead.
        """
        sz = 0
        for ID, s in self.storages.items():
            if s.data is not None:
                sz += s.data.nbytes
        return sz

    def views_in_storage(self, s, active_only=True):
        """
        Return a list of views on :any:`Storage` `s`.

        Parameters
        ----------
        s : Storage
            The storage to look for.
        active_only : True or False
                 If True (default), return only active views.
        """
        if active_only:
            return [v for v in self.original.V.values()
                    if v.active and (v.storageID == s.ID)]
        else:
            return [v for v in self.original.V.values()
                    if (v.storage.ID == s.ID)]

    def copy(self, ID=None, fill=None, dtype=None):
        """
        Create a new :any:`Container` matching self.

        The copy does not manage views.

        Parameters
        ----------
        fill : scalar or None
            If None (default), copy content. If scalar, initializes
            to this value
        dtype : valid data type or None
            If None, the copy will have the same data type as the 
            original, is specified the returned instance will contain
            empty buffers.
        """
        # Create an ID for this copy
        ID = self.ID + '_copy%d' % (len(self.copies)) if ID is None else ID

        # Create new container
        data_type = self.data_type if dtype is None else dtype
        new_cont = type(self)(self.owner,
                              ID=ID,
                              data_type=data_type)
        new_cont.original = self

        # If changing data type, avoid casting by producing empty buffers
        if (dtype is not None) and (fill is None):
                fill = 0

        # Copy storage objects
        for storageID, s in self.storages.items():
            news = s.copy(new_cont, storageID, fill)

        # We are done! Return the new container
        return new_cont

    def fill(self, fill=0.0):
        """
        Fill all storages with scalar value `fill`
        """
        if type(fill) is Container:
            self.fill(0.)
            self += fill
        for s in self.storages.values():
            s.fill(fill)

    def allreduce(self, op=None):
        """
        Performs MPI parallel ``allreduce`` with a sum as reduction
        for all :any:`Storage` instances held by *self*

        :param Container c: Input
        :param op: Reduction operation. If ``None`` uses sum.

        See also
        --------
        ptypy.utils.parallel.allreduce
        Storage.allreduce
        """
        for s in self.storages.values():
            s.allreduce(op=op)

    def clear(self):
        """
        Reduce / delete all data in attached storages
        """
        for s in self.storages.values():
            s.data = np.empty((s.data.shape[0], 1, 1), dtype=self.dtype)
            # s.datalist = [None]

    def new_storage(self, ID=None, **kwargs):
        """
        Create and register a storage object.

        Parameters
        ----------
        ID : str
             An ID for the storage. If None, a new ID is created. An
             error will be raised if the ID already exists.

        kwargs : ...
            Arguments for new storage creation. See doc for
            :any:`Storage`.

        """
        if self.storages is not None:
            if ID in self.storages:
                raise RuntimeError('Storage ID %s already exists.')

        # Create a new storage
        s = Storage(container=self, ID=ID, **kwargs)

        # Return new storage
        return s

    def reformat(self, also_in_copies=False):
        """
        Reformats all storages in this container.

        Parameters
        ----------
        also_in_copies : bool
            If True, also reformat associated copies of this container
        """
        for ID, s in self.storages.items():
            s.reformat()
            if also_in_copies:
                for c in self.copies:
                    c.S[ID].reformat()

    def report(self):
        """
        Returns a formatted report string on all storages in this container.
        """
        info = ["Containers ID: %s\n" % str(self.ID)]
        for ID, s in self.storages.items():
            info.extend(["Storage %s\n" % ID, s.report()])
        return ''.join(info)

    def formatted_report(self, table_format=None, offset=8, align='right',
                         separator=" : ", include_header=True):
        r"""
        Returns formatted string and a dict with the respective information

        Parameters
        ----------
        table_format : list
            List of (*item*, *length*) pairs where item is name of the info
            to be listed in the report and length is the column width.
            The following items are allowed:

            - *memory*, for memory usage of the storages and total use
            - *shape*, for shape of internal storages
            - *dimensions*, is ``shape \* psize``
            - *psize*, for pixel size of storages
            - *views*, for number of views in each storage

        offset : int, optional
            First column width

        separator : str, optional
            Column separator

        align : str, optional
            Column alignment, either ``'right'`` or ``'left'``

        include_header : bool
            Include a header if True

        Returns
        -------
        fstring : str
            Formatted string

        dct :dict
            Dictionary containing with the respective info to the keys
            in `table_format`

        See also
        --------
        Storage.formatted_report
        """
        fr = _Freport()
        if offset is not None:
            fr.offset = offset
        if table_format is not None:
            fr.table = table_format
        if separator is not None:
            fr.separator = separator

        # SC: method not returning any dict, feature still to be added?
        dct = {}
        mem = 0
        info = []
        for ID, s in self.storages.items():
            fstring, stats = s.formatted_report(fr.table,
                                                fr.offset,
                                                align,
                                                fr.separator,
                                                False)
            info.extend([fstring, '\n'])
            mem += stats.get('memory', 0)

        fstring = [str(self.ID).ljust(fr.offset) + fr.separator,
                   ('%.1f' % mem).rjust(fr.table[0][1]) + fr.separator]
        try:
            t = str(self.dtype).split("'")[1].split(".")[1]
        except:
            t = str(self.dtype)

        fstring.extend([t.rjust(fr.table[0][1]), '\n', ''.join(info)])

        if include_header:
            fstring.insert(0, fr.header())

        return ''.join(fstring)

    def __getitem__(self, view):
        """
        Access content through view.

        Parameters
        ----------
        view : View
               A valid :any:`View` object.
        """
        if not isinstance(view, View):
            raise ValueError

        # Access storage through its ID - this makes
        # the view applicable to a container copy.
        storage = self.storages.get(view.storage.ID, None)

        # This will raise an error is storage doesn't exist
        return storage[view]

    def __setitem__(self, view, newdata):
        """
        Set content given by view.

        Parameters
        ----------
        view : View
               A valid :any:`View` for this object

        newdata : array_like
                  The data to be stored 2D.
        """
        if not isinstance(view, View):
            raise ValueError

        # Access storage through its ID - this makes
        # the view applicable to a container copy.
        storage = self.storages.get(view.storage.ID, None)

        # This will raise an error is storage doesn't exist
        storage[view] = newdata

    def info(self):
        """
        Returns the container's total buffer space in bytes and storage info.

        Returns
        -------
        space : int
            Accumulated memory usage of all data buffers in this Container

        fstring : str
            Formatted string

        Note
        ----
        May get **deprecated** in future. Use formatted_report instead.

        See also
        --------
        report
        formatted_report
        """
        self.space = 0
        info_str = []
        for ID, s in self.storages.items():
            if s.data is not None:
                self.space += s.data.nbytes
            info_str.append(str(s) + '\n')

        return self.space, ''.join(info_str)

    def __iadd__(self, other):
        if isinstance(other, Container):
            for ID, s in self.storages.items():
                s2 = other.storages.get(ID)
                if s2 is not None:
                    s.data += s2.data
        else:
            for ID, s in self.storages.items():
                s.data += other

        return self

    def __isub__(self, other):
        if isinstance(other, Container):
            for ID, s in self.storages.items():
                s2 = other.storages.get(ID)
                if s2 is not None:
                    s.data -= s2.data
        else:
            for ID, s in self.storages.items():
                s.data -= other

        return self

    def __imul__(self, other):
        if isinstance(other, Container):
            for ID, s in self.storages.items():
                s2 = other.storages.get(ID)
                if s2 is not None:
                    s.data *= s2.data
        else:
            for ID, s in self.storages.items():
                s.data *= other

        return self

    def __truediv__(self, other):
        if isinstance(other, Container):
            for ID, s in self.storages.items():
                s2 = other.storages.get(ID)
                if s2 is not None:
                    s.data /= s2.data
        else:
            for ID, s in self.storages.items():
                s.data /= other
        return self

    def __lshift__(self, other):
        if isinstance(other, Container):
            for ID, s in self.storages.items():
                s2 = other.storages.get(ID)
                if s2 is not None:
                    s.data[:] = s2.data
        else:
            self.fill(other)

        return self


class POD(Base):
    """
    POD : Ptychographic Object Descriptor

    A POD brings together probe view, object view and diff view. It also
    gives access to "exit", a (coherent) exit wave, and to propagation
    objects to go from exit to diff space.
    """
    #: Default set of :any:`View`\ s used by a POD
    DEFAULT_VIEWS = {'probe': None,
                     'obj': None,
                     'exit': None,
                     'diff': None,
                     'mask': None}

    _PREFIX = POD_PREFIX

    def __init__(self, ptycho=None, model=None, ID=None, views=None,
                 geometry=None, **kwargs):
        """
        Parameters
        ----------
        ptycho : Ptycho
            The instance of Ptycho associated with this pod.

        model : ~ptypy.core.manager.ScanModel
            The instance of ScanModel (or it subclasses) which describes
            this pod.

        ID : str or int
            The pod ID, If None it is managed by the ptycho.

        views : dict or Param
            The views. See :py:attr:`DEFAULT_VIEWS`.

        geometry : Geo
            Geometry class instance and attached propagator

        """
        super(POD, self).__init__(ptycho, ID, False)
        self.model = model

        # other defaults:
        self.is_empty = False
        self.probe_weight = 1.
        self.object_weight = 1.

        # Store views in V and register this pod to the view
        self.V = u.Param(self.DEFAULT_VIEWS)
        if views is not None:
            self.V.update(views)
        for v in self.V.values():
            if v is None:
                continue
            if v._pod is None:
                # you are first
                v._pod = weakref.ref(self)
            else:
                # View has at least one POD connected
                if v._pods is None:
                    v._pods = weakref.WeakValueDictionary()
                    # register the older view
                    v._pods[v.pod.ID] = v.pod
                v._pods[self.ID] = self            

        #: :any:`Geo` instance with propagators
        self.geometry = geometry

        # Convenience access for all views. Note: assignment of the type
        # pod.ob_view = some_view should not be done because consistence with
        # self.V is not ensured. If this kind of assignment turns out to
        # be useful, we should consider declaring ??_view as @property.
        self.ob_view = self.V['obj']
        self.pr_view = self.V['probe']
        """ A reference to the (pr)obe-view. (ob)ject-, (ma)sk- and
        (di)ff-view are accessible in the same manner (``self.xx_view``). """

        self.di_view = self.V['diff']
        self.ex_view = self.V['exit']

        if self.ex_view is None:
            self.use_exit_container = False
            self._exit = np.ones_like(self.pr_view.shape,
                                      dtype=self.owner.CType)
        else:
            self.use_exit_container = True
            self._exit = None

        self.ma_view = self.V['mask']
        # Check whether this pod is active
        # it should maybe also have a check for an active mask view?
        # Maybe this should be tied to the diff views activeness through a
        # property

    @property
    def active(self):
        """
        Convenience property that describes whether this pod is active or not.
        Equivalent to ``self.di_view.active``
        """
        return self.di_view.active

    @property
    def fw(self):
        """
        Convenience property that returns forward propagator of attached
        Geometry instance. Equivalent to ``self.geometry.propagator.fw``.
        """
        return self.geometry.propagator.fw

    @property
    def bw(self):
        """
        Convenience property that returns backward propagator of attached
        Geometry instance. Equivalent to ``self.geometry.propagator.bw``.
        """
        return self.geometry.propagator.bw

    @property
    def upsample(self):
        """
        Convencience property that returns upsample function of attached
        Geometry instance. Equivalent to ``self.geometry.upsample``.
        """
        return self.geometry.upsample

    @property
    def downsample(self):
        """
        Convencience property that returns downsample function of attached
        Geometry instance. Equivalent to ``self.geometry.downsample``.
        """
        return self.geometry.downsample

    @property
    def object(self):
        """
        Convenience property that links to slice of object :any:`Storage`.
        Usually equivalent to ``self.ob_view.data``.
        """
        if not self.is_empty:
            return self.ob_view.data
        else:
            # Empty probe means no object (perfect transmission)
            return np.ones(self.ob_view.shape, dtype=self.owner.CType)

    @object.setter
    def object(self, v):
        if not self.is_empty:
            self.ob_view.data = v

    @property
    def probe(self):
        """
        Convenience property that links to slice of probe :any:`Storage`.
        Equivalent to ``self.pr_view.data``.
        """
        return self.pr_view.data

    @probe.setter
    def probe(self, v):
        self.pr_view.data = v

    @property
    def exit(self):
        """
        Convenience property that links to slice of exit wave
        :any:`Storage`. Equivalent to ``self.pr_view.data``.
        """
        if self.use_exit_container:
            return self.ex_view.data
        else:
            return self._exit

    @exit.setter
    def exit(self, v):
        if self.use_exit_container:
            self.ex_view.data = v
        else:
            self._exit = v

    @property
    def diff(self):
        """
        Convenience property that links to slice of diffraction
        :any:`Storage`. Equivalent to ``self.di_view.data``.
        """
        return self.di_view.data

    @diff.setter
    def diff(self, v):
        self.di_view.data = v

    @property
    def mask(self):
        """
        Convenience property that links to slice of masking
        :any:`Storage`. Equivalent to ``self.ma_view.data``.
        """
        return self.ma_view.data

    @mask.setter
    def mask(self, v):
        self.ma_view.data = v


class _Freport(object):
    """
    Class for creating headers for formatted reports.
    """

    def __init__(self):
        """
        Initializes class by setting default parameters.
        """
        self.offset = 8
        self.desc = dict([('memory', 'Memory'),
                          ('shape', 'Shape'),
                          ('psize', 'Pixel size'),
                          ('dimension', 'Dimensions'),
                          ('views', 'Views')])

        self.units = dict([('memory', '(MB)'),
                           ('shape', '(Pixel)'),
                           ('psize', '(meters)'),
                           ('dimension', '(meters)'),
                           ('views', 'act.')])

        self.table = [('memory', 6),
                      ('shape', 16),
                      ('psize', 15),
                      ('dimension', 15),
                      ('views', 5)]

        self.h1 = "(C)ontnr"
        self.h2 = "(S)torgs"
        self.separator = " : "
        self.headline = "-"

    def header(self, as_string=True):
        """
        Creates a header for formatted report method.

        :param as_string: bool
            Return header as a string if True

        :return: str or list
            Header for formatted report as string or list
        """
        header = [self.h1.ljust(self.offset), self.h2.ljust(self.offset)]

        for key, column in self.table:
            header[0] += self.separator + self.desc[key].ljust(column)
            header[1] += self.separator + self.units[key].ljust(column)

        header.append(self.headline * len(header[1]))

        if as_string:
            return '\n'.join(header) + '\n'
        else:
            return header
