# -*- coding: utf-8 -*-
"""
Container management.

This module defines flexible containers for the various quantities needed
for ptychographic reconstructions.

Container class:
    A high-level container that keeps track of sub-containers (Storage)
    and Views onto them. A container can copy itself to produce a buffer
    needed for calculations. Mathematical operations are not implemented at
    this level. Operations on arrays should be done using the Views, which
    simply return numpyarrays.

Storage class:
    The sub-container, wrapping a numpy array buffer. A Storage defines a
    system of coordinate (for now only a scaled translation of the pixel
    coordinates, but more complicated affine transformation could be
    implemented if needed). The sub-class DynamicStorage can adapt the size
    of its buffer (cropping and/or padding) depending on the Views.
    
View class:
    A low-weight class that contains all information to access a 2D piece
    of a Storage within a Container.

This file is part of the PTYPY package.

    :copyright: Copyright 2014 by the PTYPY team, see AUTHORS.
    :license: GPLv2, see LICENSE for details.

TODO: Rethink layer access (BE is not happy with current state)

"""

import numpy as np
import weakref

from .. import utils as u
from ..utils.parameters import PARAM_PREFIX
from ..utils.verbose import logger
#import ptypy

__all__=['Container','Storage','View','POD','Base']#IDManager']

# Default pixel size
DEFAULT_PSIZE = 1.

# Default shape
DEFAULT_SHAPE = (0,0,0)

# Expected structure for Views initialization.
DEFAULT_ACCESSRULE = u.Param(
        storageID = None, # (int) ID of storage, might not exist
        shape = None, # (2-tuple) shape of the view in pixels
        coord = None, # (2-tuple) physical coordinates of the center of the view
        psize = DEFAULT_PSIZE, # (float or None) pixel size (required for storage initialization)
        layer = 0 # (int) index of the third dimension if applicable.
)

BASE_PREFIX = 'B'
VIEW_PREFIX = 'V'
CONTAINER_PREFIX = 'C'
STORAGE_PREFIX = 'S'
POD_PREFIX = 'P'
MODEL_PREFIX = 'mod'
PTYCHO_PREFIX = 'pty'
GEO_PREFIX = 'G'

class Base(object):
    
    _CHILD_PREFIX = 'ID'
    _PREFIX = BASE_PREFIX                                                                                                                                                                                                                
    
    def __init__(self,owner=None, ID=None,BeOwner=True):
        """
        Ptypy Base class to support some kind of hierarchy,
        conversion to and from dictionarys as well as a 'cross-node' ID
        managament of python objects
        
        Parameters:
        -----------
        owner : Other subclass of Base or Base.
                Owner gives IDs to other ptypy objects that refer to him
                as owner. Owner also keeps a reference to these objects in
                its internal _pool where objects are key-sorted according 
                to their ID prfix
        
        ID : (None, string or scalar)
             
        BeOwner : (True) Set to False if this instance is not intended to own other other ptypy objects
        """
        self.owner = owner
        self.ID = ID
        
        # try register yourself to your owner if he exists
        try:
            owner._new_ptypy_object(obj=self)
        except:
            logger.debug('Failed registering instance of %s with ID %s to object %s' %(self.__class__,self.ID,owner))
            
        # make a pool for your own ptypy objects
        if BeOwner:
            self._pool = u.Param()  
        
    def _new_ptypy_object(self,obj):
        """
        Registers a new ptypy object into this object's pool.
        
        Parameters:
        -----------
        obj : [any object] or None
              The object to register.
        """
        try:
            prefix=obj._PREFIX
        except:
            prefix=self._CHILD_PREFIX
            
        if self._pool.get(prefix) is None:
            self._pool[prefix]={}
            
        d = self._pool[prefix]
        # Check if ID is already taken and assign a new one
        ID = obj.ID
        used = d.keys()
        if valid_ID(obj):
            if ID in used:
                logger.warning('Overwriting ID %s in pool of %s' %(ID,self.ID))
            d[ID]=obj
        else:
            try:
                if str(ID)==ID:
                    nID = prefix + ID
                else: 
                    nID = prefix + self._num_to_id(ID)
                if nID in used:
                    logger.warning('Overwriting ID %s in pool of %s' %(ID,self.ID))
            except:
                idx=len(d)
                nID = prefix+self._num_to_id(idx)
                while nID in used:
                    idx+=1
                    nID = prefix+self._num_to_id(idx)
                    
            d[nID]=obj
            obj.ID = nID
            
        return 
    
    def _num_to_id(self,num):
        """
        maybe more sophisticated in future
        """
        return '%04d' % num
        
    @classmethod
    def _from_dict(cls,dct):
        """
        create new instance from dictionary dct
        should be compatible with _to_dict()
        """
        ID = dct.pop('ID',None)
        owner = dct.pop('owner',None)
        #print cls,ID,owner
        inst = cls(owner,ID)
        inst.__dict__.update(dct)
        #calling post dictionary import routine (empty in base)
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
        a dict. Overwrite in child for custom behavior. 
        Default. Returns shallow copy of internal dict as default
        """
        return self.__dict__.copy()

def get_class(ID):
    """
    Determine ptypy class from unique ID
    """
    #typ,idx=ID[0]
    if ID.startswith(VIEW_PREFIX):
        return View
    elif ID.startswith(PTYCHO_PREFIX):
        from ptycho import Ptycho
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
        from scanmanager import ModelManager
        return ModelManager
    elif ID.startswith(GEO_PREFIX):
        from geometry import Geo
        return Geo
    else:
        return None
        
def valid_ID(obj):
    """
    check if ID of object is compatible with the current format
    """
    valid = False
    try:
        cls=get_class(obj.ID)
        valid = (obj.__class__ is cls)
    except:
        pass
        
    return valid

class Storage(Base):
    """
    Storage: Inner container handling acces to data arrays.

    * It returns view to coordinates given (slicing)
    * It contains a physical coordinate grid
    
    """

    _PREFIX = STORAGE_PREFIX

    def __init__(self, container, ID=None, **kwargs):
        """
        Storage: Inner container hangling access to arrays.

        This class essentially manages access to an internal numpy array 
        buffer.
                
        Parameters
        ----------
        ID : ...
             A unique ID, managed by the parent
        container : Container
                    The container instance
        name : str or None
               A name for this storage. If None, build a default name from ID
        data: numpy.ndarray or None
              A numpy array to use as initial buffer.
        shape : tuple or None
                The shape of the buffer (BE: CANNOT BE NONE)
        fill : float
               The default value to fill storage with.
        psize : float
                The physical pixel size.
        origin : 2-tuple
                 The physical coordinates of the [0,0] pixel (upper-left
                 corner of the storage buffer). 
        layermap : list or None
                   A list (or 1D numpy array) mapping input layer indices
                   to the internal buffer. This may be useful if the buffer
                   contains only a portion of a larger dataset (as when using
                   distributed data with MPI). If None, provide direct access.
                   to the 3d internal data.
        padonly: bool
                 If True, reformat() will enlarge the internal buffer if needed,
                 but will not shrink it.
        """
        super(Storage,self).__init__(container,ID)
        if len(kwargs)>0:
            self._initialize(**kwargs)

    def _initialize(self,data=None, shape=(1,1,1), fill=0., psize=None, origin=None, layermap=None, padonly=False):

        # Default fill value
        self.fill_value = fill
        
        #self.zoom_cycle = 0 

        if len(shape)==2:
            shape = (1,) + tuple(shape)
        # Set data buffer
        if data is None:
            # Create data buffer
            self.shape = shape
            self.data = np.empty(self.shape, self.dtype)
            self.data.fill(self.fill_value) 
        else:
            # Set initial buffer. Casting the type makes a copy
            self.data = data.astype(self.dtype)
            if self.data.ndim == 2:
                self.data.shape = (1,) + self.data.shape
            self.shape = data.shape
                
        if layermap is None:
            layermap = range(len(self.data))
        self.layermap = layermap
        self.nlayers = max(layermap)+1 # this is most often not accurate. set this quantity from the outside
        self._make_datalist()
              
        # Need to bootstrap the parameters. We set the initial center
        # in the middle of the array
        self._center = u.expect2(self.shape[-2:])//2

        # Set pixel size (in physical units)
        if psize is not None: self.psize = psize

        # Set origin (in physical units - same as psize)
        if origin is not None: self.origin = origin

        # Used to check if data format is appropriate.
        self.DataTooSmall = False

        # Padding vs padding+cropping when reformatting.
        self.padonly = padonly
        
        # A flag
        self.model_initialized = False
        
    def _to_dict(self):
        """
        We will have to recompute the datalist here
        """
        cp = self.__dict__.copy()
        # delete datalist reference
        try:
            del cp['_datalist']
        except:
            pass
            
        return cp
        #self._make_datalist()
        
    def _make_datalist(self):
        """
        Helper to build self.datalist, providing access to the data buffer
        in a transparent way.
        """
        # BE does not give the same result on all nodes
        #self._datalist = [None] * (max(self.layermap)+1)
        #u.parallel.barrier()
        #print u.parallel.rank
        self._datalist = [None] * max(self.nlayers,max(self.layermap)+1)
        for k,i in enumerate(self.layermap):
            self._datalist[i] = self.data[k]

    @property
    def datalist(self):
        
        if not hasattr(self,'datalist'):
            self._make_datalist()
        
        return self._datalist
        
    @property
    def dtype(self):
        return self.owner.dtype
        
    def copy(self,owner=None, ID=None, fill=None):
        """
        Return a copy of this storage object.
        
        Note: the returned copy has the same container as self.
        
        Parameters
        ----------
        ID : ...
             A unique ID, managed by the parent
        fill : number or None
               If float, set the content to this value. If None, copy the
               current content. 
        """
        if fill is None:
            # Return a new Storage or sub-class object with a copy of the data.
            return self.__class__(owner, ID, data=self.data.copy(), psize=self.psize, origin=self.origin, layermap=self.layermap)
        else:
            # Return a new Storage or sub-class object with an empty buffer
            newStorage = self.__class__(owner, ID, shape=self.shape, psize=self.psize, origin=self.origin, layermap=self.layermap)
            newStorage.fill(fill)
            return newStorage
                    
    def fill(self, fill=None):
        """
        Fill managed buffer. 
        
        Parameters
        ----------
        fill : float, numpy array or None.
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
        else:
            # Replace the buffer
            self.data = fill.astype(self.dtype)
            self.shape = self.data.shape

    def update(self):
        """
        Update internal state, including all views on this storage to 
        ensure consistency with the physical coordinate system.
        """
        # Update the access information for the views (i.e. pixcoord, roi and sp)
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
            for v in self.views: self.update_views(v)
            return

        # Synchronize pixel size
        v.psize = self.psize.copy()

        # v.shape can be None upon initialization - this means "full frame"
        if v.shape is None:
            v.shape = u.expect2(self.shape[-2:])
            v.pixcoord = v.shape/2.
            v.physcoord = self._to_phys(v.pixcoord)
        else:
            # Convert the physical coordinates of the view to pixel coordinates
            v.pixcoord = self._to_pix(v.physcoord)
        
        # Integer part (note that np.round is not stable for odd arrays)
        pix = np.round(v.pixcoord).astype(int)
        
        # These are the important attributes used when accessing the data
        v.roi = np.array([pix - v.shape/2, pix + (v.shape+1)/2])
        v.sp = v.pixcoord - pix
        #v.slayer = 0 if self.layermap is None else self.layermap.index(v.layer)

    def reformat(self, newID=None):
        """
        Crop or pad if required.
        
        Parameters:
        -----------
        newID : int
                If None (default) act on self. Otherwise create a copy of self
                before doing the cropping and padding.
                
        return the cropped storage (a new one or self)
        """

        # If a new storage is requested, make a copy.
        if newID is not None:
            s = self.copy(newID)
            s.reformat(newID=None)
            return s

        # Make sure all views are up to date
        self.update()
               
        # List of views on this storage
        views = self.views
        if not views: 
            return self

        logger.debug('%s[%s] :: %d views for this storage' % (self.owner.ID, self.ID,len(views)))

        # Loop through all active views to get individual boundaries
        rows = []
        cols = []
        layers = []
        for v in views:
            if not v.active: continue

            # Accumulate the regions of interest to compute the full field of view
            rows+=[v.roi[0,0],v.roi[1,0]]
            cols+=[v.roi[0,1],v.roi[1,1]]

            # Gather a (unique) list of layers
            if v.layer not in layers: layers.append(v.layer)

        sh = self.data.shape

        # Compute 2d misfit (distance between the buffer boundaries and the
        # region required to fit all the views)   
        misfit=np.array([[-np.min(rows), np.max(rows)-sh[-2]],\
                         [-np.min(cols), np.max(cols)-sh[-1]]])

        logger.debug('%s[%s] :: misfit = [%s,%s]' % (self.owner.ID, self.ID, misfit[0],misfit[1]))

        posmisfit = (misfit > 0)
        negmisfit = (misfit < 0)

        needtocrop_or_pad = posmisfit.any() or (negmisfit.any() and not self.padonly)
        
        if posmisfit.any() or negmisfit.any():
            logger.debug('Storage %s of container %s has a misfit of [%s,%s] between its data and its views' % (str(self.ID),str(self.owner.ID), misfit[0],misfit[1]))
        if needtocrop_or_pad:
            if self.padonly:
                misfit[negmisfit] = 0

            # Recompute center and shape
            new_center = self.center + misfit[:,0]
            new_shape = (sh[0], sh[1]+misfit[0].sum(), sh[2]+misfit[1].sum())
            logger.debug('%s[%s] :: center: %s -> %s' % (self.owner.ID, self.ID, str(self.center), str(new_center)))
            #logger.debug('%s[%s] :: shape: %s -> %s' % (self.owner.ID, self.ID, str(sh), str(new_shape)))
     
            # Apply 2d misfit
            if self.data is not None:
                new_data = u.crop_pad(self.data, misfit, fillpar=self.fill_value).astype(self.dtype)
            else:
                new_data = np.empty(new_shape, self.dtype)
                new_data.fill(self.fill_value)
        else:
            # Nothing changes for now
            new_data = self.data
            new_shape = sh
            new_center = self.center

        # Deal with layermap
        new_layermap = sorted(layers)
        if self.layermap != new_layermap:
            relayered_data = []
            for i in new_layermap:
                if i in self.layermap:
                    # This layer already exists
                    d = new_data[self.layermap.index(i)]
                else:
                    # A new layer
                    d = np.empty(new_shape[-2:], self.dtype)
                    d.fill(self.fill_value)
                relayered_data.append(d)
            new_data = np.array(relayered_data)
            new_shape = new_data.shape
            self.layermap = new_layermap 

        # BE: set a layer index in the view the datalist access has proven to be too slow.
        for v in views:
            v.slayer = self.layermap.index(v.layer)
            
        logger.debug('%s[%s] :: shape: %s -> %s' % (self.owner.ID, self.ID,str(sh), str(new_shape)))
        # store new buffer
        self.data = new_data
        self.shape = new_shape
        self.center = new_center

        # make datalist
        self._make_datalist()

    def _to_pix(self, coord):
        """
        Transforms physical coordinates 'coord' to pixel coordinates.
        
        Parameters
        ----------
        coord : (N,2) numpy array
            the coordinate
        """
        return (coord - self.origin)/self.psize

    def _to_phys(self, pix):
        """
        Transforms pixcel coordinates 'pix' to physical coordinates.
        
        Parameters
        ----------
        pix : (N,2) numpy array
            the coordinate
        """
        return pix*self.psize + self.origin
        
    @property
    def psize(self):
        """
        Return the pixel size.
        """
        return self._psize
    
    @psize.setter
    def psize(self,v):
        """
        Set the pixel size, and update all the internal variables.
        """
        self._psize = u.expect2(v)
        self._origin = -self._center * self._psize
        self.update()
            
    @property
    def origin(self):
        """
        Return the physical position of the upper-left corner of the storage.
        """
        return self._origin
    
    @origin.setter
    def origin(self,v):
        """
        Set the origin and update all the internal variables.
        """
        self._origin = u.expect2(v) 
        self._center = -self._origin / self._psize
        self.update()

    @property
    def center(self):
        """
        Return the position of the origin relative to the upper-left corner
        of the storage, in pixel coordinates
        """
        return self._center
    
    @center.setter
    def center(self,v):
        """
        Set the center and update all the internal variables.
        """
        self._center = u.expect2(v) 
        self._origin = -self._center * self._psize
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

    def zoom_to_psize(self,new_psize,**kwargs):
        """
        ---- untested!!! ----
        changes pixel size and zooms the data buffer along last two axis accordingly
        updates all attached views and reformats if neccessary
        
        Parameters:
        -----------
        new_psize : scalar, 2-tuple or (1,2)-array
                    new pixel size 
        """
        new_psize = u.expect2(new_psize)
        sh = np.asarray(self.shape[-2:])
        # psize is quantized
        new_sh = np.round(self.psize / new_psize * sh)
        new_psize = self.psize/ new_sh *sh
        
        if (new_sh!=sh).any():
            logger.info('Zooming from %s , %s to %s , %s'  %(self.psize,sh,new_psize,new_sh.astype(int)))
            
            # Zoom data buffer. 
            # Could be that it is faster and cleaner to loop over first axis
            zoom = new_sh / sh
            self.fill(u.zoom(self.data,[1.0,zoom[0],zoom[1]],**kwargs))
            
        self._psize = new_psize
        self.zoom_cycle+=1
        # update internal coordinate system, while zooming, the coordinate for top left corner should remain the same
        origin = self.origin
        self.origin = origin  #this call will also update the views' coordinates
        
        self.reformat()
        # reformat everything

        
    def grids(self):
        """
        Returns x and y grids in the shape of internal buffer
        """
        sh = self.data.shape 
        nm = np.indices(sh)[-2:]
        flat = nm.reshape((2,self.data.size))
        c = self._to_phys(flat.T).T
        c = c.reshape((2,)+sh)
        return c[0],c[1]
    
    def get_view_coverage(self):
        """
        Returns an array of the shape of internal buffer
        showing the view coverage of this storage 
        """
        coverage = np.zeros_like(self.data)
        for v in self.views:
            coverage[v.slice]+=1
        
        return coverage
        
        
    def __getitem__(self, v):
        """
        Storage[v]
        
        Return the numpy array corresponding to the view on this storage.
        """
        if not isinstance(v,View):
            raise ValueError
 
        # Here things could get complicated. Coordinate transforms, 3D - 2D projection, ... 
        # Current implementation: ROI + subpixel shift
        #return shift(self.datalist[v.layer][v.roi[0,0]:v.roi[1,0],v.roi[0,1]:v.roi[1,1]], v.sp)
        return shift(self.data[v.slayer,v.roi[0,0]:v.roi[1,0],v.roi[0,1]:v.roi[1,1]], v.sp)
        
    def __setitem__(self, v, newdata):
        """
        Storage[v] = newdata

        Set the data to newdata for view v.
        """
        if not isinstance(v,View):
            raise ValueError
        
        # Only ROI and shift for now. This part must always be consistent with __getitem__!
        #self.datalist[v.layer][v.roi[0,0]:v.roi[1,0],v.roi[0,1]:v.roi[1,1]] = shift(newdata, -v.sp) 
        self.data[v.slayer,v.roi[0,0]:v.roi[1,0],v.roi[0,1]:v.roi[1,1]] = shift(newdata, -v.sp)
        
    def __str__(self):
        info = '%15s : %7.2f MB :: '  % (self.ID,self.data.nbytes /1e6)
        if self.data is not None:
            info += 'data=%s @%s' % (self.data.shape,self.data.dtype)
        else:
            info += 'empty=%s @%s' % (self.shape,self.dtype)
        return info+' psize=%(_psize)s center=%(_center)s' % self.__dict__
         
def shift(v,sp):
    """
    Placeholder for future subpixel shifting method. 
    """
    return v
                    
class View(Base):
    """
    A "window" on a Container.
    
    A view stores all the slicing information to extract a 2D piece
    of Container. 
    """
    DEFAULT_ACCESSRULE = DEFAULT_ACCESSRULE
    _PREFIX = VIEW_PREFIX
      
    def __init__(self, container,ID=None, **kwargs):
        """
        A "window" on a container.
        
        A view stores all the slicing information to extract a 2D piece
        of Container. 
        
        Parameters
        ----------
        container : Container
                    The Container instance this view applies to.
        accessrule : dict
                   All the information necessary to access the wanted slice.
                   Maybe subject to change as code evolve. See DEFAULT_ACCESSRULE
        name : str or None
               name for this view. Automatically built from ID if None.
        active : bool
                 Whether this view is active (default to True) 
        """
        super(View,self).__init__(container,ID,False)
        if len(kwargs) >0 :
            self._initialize(**kwargs)
            
    def _initialize(self,accessrule=None, active=True):

        # Prepare a dictionary for PODs (volatile!)
        self.pods = weakref.WeakValueDictionary()

        # Set active state
        self.active = active

        # The messy stuff
        self._set_accessrule(accessrule)
        
    def _set_accessrule(self, accessrule):
        """
        Store internal info to get/set the 2D data in the container. 
        """
        rule = u.Param(self.DEFAULT_ACCESSRULE)
        rule.update(accessrule)

        # The storage ID this view will apply to
        self.storageID = rule.storageID

        # Information to access the slice within the storage buffer
        self.psize = u.expect2(rule.psize)
        
        # shape == None means "full frame"
        if rule.shape is not None:
            self.shape = u.expect2(rule.shape)
        else:
            self.shape = None
        self.physcoord = u.expect2(rule.coord)
        self.layer = rule.layer

        # Look for storage, create one if necessary
        s = self.owner.S.get(self.storageID, None)
        if s is None:
            s = self.owner.new_storage(ID=self.storageID, psize=rule.psize, shape=rule.shape)
        self.storage = s
            
        if (self.storage.psize != rule.psize).any():
            raise RuntimeError('Inconsistent pixel size when creating view')

        # This ensures self-consistency (sets pixel coordinate and ROI)
        if self.active: self.storage.update_views(self)

    def __str__(self):
        first = '%s -> %s[%s] : shape = %s layer = %s physcoord = %s' % (self.owner.ID, self.storage.ID, self.ID, self.shape, self.layer, self.physcoord)
        if not self.active:
            return first+'\n INACTIVE : slice = ...  '
        else:
            return first+'\n ACTIVE : slice = %s' % str(self.slice)
        
    @property
    def slice(self):
        """
        returns a slice according to layer and roi
        Please note, that this not always makes sense
        """
        slayer = None if self.layer not in self.storage.layermap else self.storage.layermap.index(self.layer)
        return (slayer,slice(self.roi[0,0],self.roi[1,0]),slice(self.roi[0,1],self.roi[1,1]))
        
    @property
    def pod(self):
        """
        returns first pod in the pod dict. This is a common call in the code 
        and has therefore found its way here
        """
        return self.pods.values()[0]
        
    @property
    def data(self):
        """
        Return the view content
        """
        return self.storage[self]
        
    @data.setter
    def data(self,v):
        """
        Set the view content
        """
        self.storage[self]=v
        
class Container(Base):
    """
    High-level container class.
    
    Container can be seen as a "super-numpy-array" which can contain multiple
    sub-containers, potentially of different shape. 
    Typically there will be only 5 such containers in a reconstruction:
    "probe", "object", "exit", "diff" and "mask"
    A container can duplicate its internal storage and apply views on them.
    """
    _PREFIX = CONTAINER_PREFIX
    
    def __init__(self, ptycho=None,ID=None, **kwargs):
        """
        High-level container class.
        Typically there will be only 4 such containers in a reconstruction:
        "probe", "object", "exit" and "diff"
        A container knows how to duplicate its internal storage and apply views on them.

        Parameters
        ----------
        ID : ...
             A unique ID, managed by the parent
        name : str or None
               The name of this container. For documentation and debugging
               purposes only. Defaults to str(ID)
        ptycho : Ptycho
                 The instance of Ptycho associated with this pod. If None,
                 defaults to ptypy.currentPtycho
        dtype : str or numpy.dtype
                data type - either a numpy.dtype object or 'complex' or 
                'real' (precision is taken from ptycho.FType or ptycho.CType) 
        """
        #if ptycho is None:
        #    ptycho = ptypy.currentPtycho
    
        super(Container,self).__init__(ptycho,ID)
        if len(kwargs) > 0:
            self._initialize(**kwargs)
        
    def _initialize(self,original=None, data_type='complex'):

        self.data_type = data_type
             
        # Prepare for copy
        self.original = original if original is not None else self
        
        
    @property
    def copies(self):
        return [c for c in self.owner.containers.itervalues() if c.original is self and c is not self]
        
    @property
    def dtype(self):
        # get datatype
        if self.data_type == 'complex':
            return self.owner.CType
        elif self.data_type == 'real':
            return self.owner.FType
        else:
            return self.data_type
            
    @property
    def S(self):
        return self._pool.get(STORAGE_PREFIX,{})

    @property
    def size(self):
        """
        Return total number of pixels in this container.
        """
        sz = 0
        for ID,s in self.S.iteritems():
            if s.data is not None:
                sz += s.data.size
        return sz
        
    @property
    def V(self):
        return self._pool.get(VIEW_PREFIX,{})
        
    def views_in_storage(self, s, active=True):
        """
        Return a list of views on Storage s.
        
        Parameters
        ----------
        s : Storage
            The storage to look for.
        active : True or False
                 If True (default), return only active views.
        """
        if active:
            return [v for v in self.original.V.values() if  v.active and (v.storageID == s.ID)]
        else:
            return [v for v in self.original.V.values() if (v.storage.ID == s.ID)]
           
    def copy(self, ID=None, fill=None):
        """
        Create a new container matching self. 
        
        The copy does not manage views. 
        
        Parameters
        ----------
        fill : ...
               If None (default), copy content. If a float, initialize to this value
        """
        # Create an ID for this copy
        ID = self.ID + '_copy%d' % (len(self.copies)) if ID is None else ID

        # Create new container
        newCont = self.__class__(ptycho=self.owner,ID=ID, original = self,data_type=self.data_type)

        # Copy storage objects
        for storageID, s in self.S.iteritems():
            news = s.copy(newCont,storageID, fill)

        # We are done! Return the new container
        return newCont
        
    def fill(self, fill=0.0):
        """
        Fill all storages.
        """
        for s in self.S.itervalues(): 
            s.fill(fill)
            s._make_datalist() 
        
    def clear(self):
        """
        reduce / delete all data in attached storages
        """
        for s in self.S.itervalues():
            s.data = np.empty((s.data.shape[0],1,1),dtype=self.dtype)
            #s.datalist = [None]
            
    def new_storage(self, ID=None, **kwargs):
        """
        Create and register a storage object.
        
        Assign the provided ID is assigned automatically.
        
        Parameters
        ----------
        ID : str
             An ID for the storage. If None, a new ID is created. An
             error will be raised if the ID already exists.
        **kwargs : ...
                   Arguments for new storage creation. See doc for
                   Storage.
        
        """
        if self.S is not None:
            if self.S.has_key(ID):
                raise RuntimeError('Storage ID %s already exists.')

        # Create a new storage
        s = Storage(container=self, ID=ID, **kwargs)

        # Return new storage
        return s

    def reformat(self,AlsoInCopies=False):
        """
        Reformat all storages in this container.
        """
        for ID,s in self.S.iteritems():
            s.reformat()
            if AlsoInCopies:
                for c in self.copies:
                    c.S[ID].reformat()

    def __getitem__(self,view):
        """
        Access content through view.
        
        Parameters
        ----------
        view : View
               A valid view object.
        """
        if not isinstance(view,View):
            raise ValueError
            
        # Access storage through its ID - this makes the view applicable
        # to a container copy.
        storage = self.S.get(view.storage.ID, None)

        # This will raise an error is storage doesn't exist
        return storage[view]
            
    def __setitem__(self,view,newdata):
        """
        Set content given by view.
        
        Parameters
        ----------
        view : View
               A valid view for this object
        newdata : array_like, 2D
                  The data to be stored.
        """
        if not isinstance(view,View):
            raise ValueError

        # Access storage through its ID - this makes the view applicable
        # to a container copy.
        storage = self.S.get(view.storage.ID, None)

        # This will raise an error is storage doesn't exist
        storage[view] = newdata

    def info(self):
        """
        Return the total buffer space for this container in bytes and storage info
        """
        self.space=0
        info_str =''
        for ID,s in self.S.iteritems():
            if s.data is not None:
                self.space+=s.data.nbytes
            info_str+=str(s)+'\n'
        return self.space,info_str
                
    def __iadd__(self,other):
        if isinstance(other,Container):
            for ID,s in self.S.iteritems():
                s2 = other.S.get(ID)
                if s2 is not None:
                    s.data += s2.data
        else:
            for ID,s in self.S.iteritems():
                s.data += other
        return self
                
    def __isub__(self,other):
        if isinstance(other,Container):
            for ID,s in self.S.iteritems():
                s2 = other.S.get(ID)
                if s2 is not None:
                    s.data -= s2.data
        else:
            for ID,s in self.S.iteritems():
                s.data -= other
        return self 
        
    def __imul__(self,other):
        if isinstance(other,Container):
            for ID,s in self.S.iteritems():
                s2 = other.S.get(ID)
                if s2 is not None:
                    s.data *= s2.data
        else:
            for ID,s in self.S.iteritems():
                s.data *= other
        return self
        
    def __idiv__(self,other):
        if isinstance(other,Container):
            for ID,s in self.S.iteritems():
                s2 = other.S.get(ID)
                if s2 is not None:
                    s.data /= s2.data
        else:
            for ID,s in self.S.iteritems():
                s.data /= other
        return self
        
    def __lshift__(self,other):
        if isinstance(other,Container):
            for ID,s in self.S.iteritems():
                s2 = other.S.get(ID)
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
    DEFAULT_VIEWS={'probe':None,'obj':None,'exit':None,'diff':None,'mask':None}
    _PREFIX = POD_PREFIX
    
    def __init__(self,ptycho=None,ID=None,**kwargs):
        """
        POD : Ptychographic Object Descriptor
    
        A POD brings together probe view, object view and diff view. It also
        gives access to "exit", a (coherent) exit wave, and to propagation
        objects to go from exit to diff space. 
        
        Parameters
        ----------
        ID : ...
            The view ID, managed by the parent.
        info : dict or Param
                This dict is needed for the modelmanager to figure out to
                which storage and layer the views should point to
        views : dict or Param
                The views. See POD.DEFAULT_VIEWS.
        geometry : geometry class
                it also handles propagation
        ptycho : Ptycho
                 The instance of Ptycho associated with this pod. If None,
                 defaults to ptypy.currentPtycho
        """
        super(POD,self).__init__(ptycho,ID,False)
        if len(kwargs) > 0:
            self._initialize(**kwargs)
            
    def _initialize(self,views=None,geometry=None,meta=None):
        # store meta data 
        # this maybe not so clever if meta_data is large
        self.meta = meta
        
        # other defaults:
        self.is_empty=False
        self.probe_weight = 1.
        self.object_weight = 1.
        
        # Store views in V and register this pod to the view
        self.V = u.Param(self.DEFAULT_VIEWS)
        if views is not None: self.V.update(views)           
        for v in self.V.values():
            if v is None:
                continue
            v.pods[self.ID]=self
        # Get geometry with propagators
        self.geometry = geometry
        
        # Convenience access for all views. Note: assignement of the type
        # pod.ob_view = some_view should not be done because consistence with
        # self.V is not ensured. If this kind of assignment turns out to 
        # be useful, we should consider declaring ??_view as @property.
        self.ob_view = self.V['obj']
        self.pr_view = self.V['probe']
        self.di_view = self.V['diff']
        self.ex_view = self.V['exit']
        self.ma_view = self.V['mask']
        self.exit = np.ones_like(self.geometry.N,dtype=self.owner.CType)
        # Check whether this pod is active it should maybe also have a check for an active mask view?
        # Maybe this should be tight to to the diff views activeness through a property
    @property
    def active(self):    
        return self.di_view.active
        
    @property
    def fw(self):
        return self.geometry.propagator.fw
    
    @property
    def bw(self):
        return self.geometry.propagator.bw
    
    @property
    def object(self):
        if not self.is_empty:
            return self.ob_view.data
        else:
            return np.ones(self.geometry.N, dtype = self.owner.CType)
            
    @object.setter
    def object(self,v):
        self.ob_view.data=v

    @property
    def probe(self):
        return self.pr_view.data
        
    @probe.setter
    def probe(self,v):
        self.pr_view.data=v

    #@property
    #def exit(self):
    #    return self.ex_view.data
        
    #@exit.setter
    #def exit(self,v):
    #    self.ex_view.data=v

    @property
    def diff(self):
        return self.di_view.data
        
    @diff.setter
    def diff(self,v):
        self.di_view.data=v
        
    @property
    def mask(self):
        return self.ma_view.data
        
    @mask.setter
    def mask(self,v):
        self.ma_view.data=v
