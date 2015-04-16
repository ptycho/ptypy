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
try: 
    from pympler.asizeof import asizeof
    use_asizeof=True
except ImportError:
    use_asizeof=False
    
from .. import utils as u
from ..utils.parameters import PARAM_PREFIX
from ..utils.verbose import logger
#import ptypy

__all__=['Container','Storage','View','POD','Base','DEFAULT_PSIZE','DEFAULT_SHAPE' ]#IDManager']

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
        layer = 0, # (int) index of the third dimension if applicable.
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
        
        Parameters
        ----------
        owner : Other subclass of Base or Base
            Owner gives IDs to other ptypy objects that refer to him
            as owner. Owner also keeps a reference to these objects in
            its internal _pool where objects are key-sorted according 
            to their ID prfix
        
        ID : None, str or int
            
        BeOwner : bool
            Set to `False` if this instance is not intended to own other
            ptypy objects.
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

    def calc_mem_usage(self):
        space = 64 # that is for the class itself
        pool_space = 0
        npy_space = 0
        if hasattr(self,'_pool'):
            if use_asizeof:
                space+=asizeof(self._pool,limit=0) 
            for k, v in self._pool.iteritems():
                if use_asizeof:
                    space+=asizeof(v,limit=0)
                for kk,vv in v.iteritems():
                    pool_space += vv.calc_mem_usage()[0]
                
        for k,v in self.__dict__.iteritems():
            if issubclass(v.__class__,Base):
                #print 'jump ' + str(v.__class__)
                continue
            elif str(k)=='_pool' or str(k)=='pods':
                continue
            else:
                if use_asizeof:
                    s= asizeof(v)
                    space += s
                if type(v) is np.ndarray:
                    npy_space+=v.nbytes
            #print str(k)+":"+str(s)       
        return space+pool_space+npy_space, pool_space, npy_space
        
def get_class(ID):
    """
    Determine ptypy class from unique `ID`
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
        from manager import ModelManager
        return ModelManager
    elif ID.startswith(GEO_PREFIX):
        from geometry import Geo
        return Geo
    else:
        return None
        
def valid_ID(obj):
    """
    check if ID of object `obj` is compatible with the current format
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
    Inner container handling acces to data arrays.

    This class essentially manages access to an internal numpy array buffer.
                
    * It returns a view to coordinates given (slicing)
    * It contains a physical coordinate grid
    
    """

    _PREFIX = STORAGE_PREFIX

    def __init__(self, container, ID=None, **kwargs):
        """
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
        ID : str or int
             A unique ID, managed by the parent
        fill : scalar or None
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
        
        Parameters
        ----------
        newID : str or int
            If None (default) act on self. Otherwise create a copy 
            of self before doing the cropping and padding.
            
        Returns
        -------
        s : Storage
            returns new Storage instance in same :any:`Container` if
            `newId` is not None.
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
            
            Mpixels = np.array(new_shape).astype(float).prod()/1e6
            if Mpixels > 20:
                raise RuntimeError('Arrays larger than 20M not supported. You requested %.2fM pixels, Bitch' % (Mpixels))

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
        Transforms physical coordinates `coord` to pixel coordinates.
        
        Parameters
        ----------
        coord : tuple or array-like
            A ``(N,2)``-array of the coordinates to be trasnformed
        """
        return (coord - self.origin)/self.psize

    def _to_phys(self, pix):
        """
        Transforms pixcel coordinates `pix` to physical coordinates.
        
        Parameters
        ----------
        pix : tuple or array-like
            A ``(N,2)``-array of the coordinates to be transformed
        """
        return pix*self.psize + self.origin
        
    @property
    def psize(self):
        """
        :returns: The pixel size.
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
        Changes pixel size and zooms the data buffer along last two axis 
        accordingly, updates all attached views and reformats if neccessary.
        **untested!!**
        
        Parameters
        ----------
        new_psize : scalar or array_like
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
        Returns
        ------- 
        x, y: ndarray 
            grids in the shape of internal buffer
        """
        sh = self.data.shape 
        nm = np.indices(sh)[-2:]
        flat = nm.reshape((2,self.data.size))
        c = self._to_phys(flat.T).T
        c = c.reshape((2,)+sh)
        return c[0],c[1]
    
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
            coverage[v.slice]+=1
        
        return coverage

    def report(self):
        """
        Returns
        -------
        str
            a formatted string giving a report on this storage.
        """
        info = "Shape: %s\n" % str(self.data.shape)
        info += "Pixel size (meters): %g x %g\n" % tuple(self.psize)
        info += "Dimensions (meters): %g x %g\n" % (self.psize[0] * self.data.shape[-2], self.psize[1] * self.data.shape[-1])
        info += "Number of views: %d\n" % len(self.views)
        return info
    
    def formatted_report(self,table_format=None,align='right',seperator=" : "):
        """
        Returns formatted string and a dict with the respective information
        
        Parameters
        ----------
        table_format : list 
            list of key(`str`),value(`str`) pairs where key is the item 
            to be listed in the report and value is a string formatter
        
        Returns
        -------
        fstring : str
            formatted string 
            
        dct :dict
            dictionary containing with the respective info to the keys
            in `table_format`
        """
        dct ={}
        fstring = ""
        _table = [('memory',8),('shape',17),('psize',15),('dimension',15),('views',5),('dtype',8)]
        table_format = _table if table_format is None else table_format
        for key,column in table_format:
            if str(key)=='shape':
                dct[key] = tuple(self.data.shape)
                info = '%d*%d*%d' % dct[key]
            elif str(key)=='psize':
                dct[key] = tuple(self.psize)
                info = '%.2e*%.2e' % tuple(dct[key])
                info = info.split('e',1)[0]+info.split('e',1)[1][3:]
            elif str(key)=='dimension':
                dct[key] = (self.psize[0] * self.data.shape[-2], self.psize[1] * self.data.shape[-1])
                info = '%.2e*%.2e' % tuple(dct[key])
                info = info.split('e',1)[0]+info.split('e',1)[1][3:]
            elif str(key)=='memory':
                dct[key] = float(self.data.nbytes) /1e6
                info = '%.1f' % dct[key]
            elif str(key)=='dtype':
                dct[key] = self.data.dtype
                info = dct[key].str
            elif str(key)=='views':
                dct[key] = len(self.views)
                info = str(dct[key])
            else:
                dct[key] = None
                info = ""
                
            fstring += seperator
            if str(align)=='right':
                fstring += info.rjust(column)[-column:]
            else:
                fstring += info.ljust(column)[:column]
                
        return fstring, dct
    
            
    def __getitem__(self, v):
        """
        Storage[v]
        
        Returns
        -------
        ndarray
            The view to internal data buffe correspondieng to View `v`
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

        Set internal data buffer to `newdata` for the region of view `v`.
        
        Parameters
        ----------
        v : View
            A View for this storage
        
        newdata : ndarray
            Two-dimensional array that fits the view's shape
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
    
    Note
    ----
    The final structure of this class is yet up to debate
    and the constructor signature may change. Especially since
    "DEFAULT_ACCESSRULE" is yet so small, its contents could be
    incorporated in the constructor call.
    
    """
    DEFAULT_ACCESSRULE = DEFAULT_ACCESSRULE
    _PREFIX = VIEW_PREFIX
      
    def __init__(self, container,ID=None, **kwargs):
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
        
        Keyword Args
        ------------
        active : bool
            Whether this view is active (*default* is ``True``) 
            
        storageID : str
            ID of storage, If the Storage does not exist 
            it will be created! *(default = None)*
            
        shape : int or tuple of int
            Shape of the view in pixels (*default* is ``None``)
            
        coord : 2-tuple of float, 
            Physical coordinates [meter] of the center of the view.
            
        psize : float or tuple of float
            Pixel size (required for storage initialization)
            See :any:`DEFAULT_PSIZE`
            
        layer : int
            Index of the third dimension if applicable.
            (*default* is ``0``)
        """
        super(View,self).__init__(container,ID,False)
        if len(kwargs) >0 :
            self._initialize(**kwargs)
            
    def _initialize(self,accessrule=None, active=True, **kwargs):

        # Prepare a dictionary for PODs (volatile!)
        if not self.__dict__.has_key('pods'):
            self.pods = weakref.WeakValueDictionary()

        # Set active state
        self.active = active

        # The messy stuff
        self._set_accessrule(accessrule,**kwargs )
        
    def _set_accessrule(self, accessrule,**kwargs ):
        """
        Store internal info to get/set the 2D data in the container. 
        """
        rule = u.Param(self.DEFAULT_ACCESSRULE)
        if accessrule is not None:
            rule.update(accessrule)
        rule.update(kwargs)
        
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
    sub-containers of type :any:`Storage`, potentially of different shape,
    along with all :any:`View` instances that act on these Storages to extract
    data from the internal data buffer :any:`Storage.data`. 
    
    Typically there will be five such base containers in a :any:`Ptycho`
    reconstruction instance:
    
        - `Cprobe`, Storages for the illumination, i.e. **probe**, 
        - `Cobj`, Storages for the sample transmission, i.e. **object**, 
        - `Cexit`, Storages for the **exit waves**, 
        - `Cdiff`, Storages for **diffraction data**, usually one per scan,
        - `Cmask`, Strorages for **masks** (and weights), usually one per scan,
        
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
    
    def __init__(self, ptycho=None,ID=None, **kwargs):
        """
        Parameters
        ----------
        ID : str or int
             A unique ID, managed by the parent
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
        """
        Property that returns list of all copies of this :any:`Container`
        """
        return [c for c in self.owner.containers.itervalues() if c.original is self and c is not self]
        
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
        return self._pool.get(STORAGE_PREFIX,{})

    @property
    def Sp(self):
        """
        A property that returns the internal dictionary of all 
        :any:`Storage` instances in this :any:`Container` as a :any:`Param`
        """
        return u.Param(self.S)

    @property
    def V(self):
        """
        A property that returns the internal dictionary of all 
        :any:`View` instances in this :any:`Container`
        """
        return self._pool.get(VIEW_PREFIX,{})
        
    @property
    def Vp(self):
        """
        A property that returns the internal dictionary of all 
        :any:`View` instances in this :any:`Container` as a :any:`Param`
        """
        return self._pool.get(VIEW_PREFIX,{})
        
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
    def nbytes(self):
        """
        Return total number of bytes used by numpy array buffers
        in this container. This is not the actual size in memory of the
        whole contianer, as it does not include the views nor dictionary
        overhead.
        """
        sz = 0
        for ID,s in self.S.iteritems():
            if s.data is not None:
                sz += s.data.nbytes
        return sz
        
    def views_in_storage(self, s, active=True):
        """
        Return a list of views on :any:`Storage` `s`.
        
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
        Create a new :any:`Container` matching self. 
        
        The copy does not manage views. 
        
        Parameters
        ----------
        fill : scalar or None
            If None (default), copy content. If scalar, initializes 
            to this value
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
        Fill all storages with scalar value `fill`
        """
        for s in self.S.itervalues(): 
            s.fill(fill)
            s._make_datalist() 
        
    def clear(self):
        """
        Reduce / delete all data in attached storages
        """
        for s in self.S.itervalues():
            s.data = np.empty((s.data.shape[0],1,1),dtype=self.dtype)
            #s.datalist = [None]
            
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
        if self.S is not None:
            if self.S.has_key(ID):
                raise RuntimeError('Storage ID %s already exists.')

        # Create a new storage
        s = Storage(container=self, ID=ID, **kwargs)

        # Return new storage
        return s

    def reformat(self,AlsoInCopies=False):
        """
        Reformats all storages in this container.
        
        Parameters
        ----------
        AlsoInCopies : bool
            If True, also reformat associated copies of this container 
        """
        for ID,s in self.S.iteritems():
            s.reformat()
            if AlsoInCopies:
                for c in self.copies:
                    c.S[ID].reformat()

    def report(self):
        """
        Returns a formatted string giving a report on all storages in this container.
        """
        info = "Containers ID: %s\n" % str(self.ID)
        for ID,s in self.S.iteritems():
            info += "Storage %s\n" % ID
            info += s.report()
        return info

    def formatted_report(self,offset=8,table_format=None,align='right',seperator=" : "):
        """
        Returns formatted string and a dict with the respective information
        
        Parameters
        ----------
        table_format : list 
            list of key(`str`),value(`str`) pairs where key is the item 
            to be listed in the report and value is a string formatter
        
        Returns
        -------
        fstring : str
            formatted string 
            
        dct :dict
            dictionary containing with the respective info to the keys
            in `table_format`
            
        See also
        --------
        Storage.formatted_report
        """
        dct ={}
        id_length = offset
        _table = [('memory',6),('shape',15),('psize',15),('dimension',15),('views',4)]
        table_format = _table if table_format is None else table_format
        mem = 0
        info = ""
        for ID,s in self.S.iteritems():
            fstring, stats = s.formatted_report(table_format,align,seperator)
            info += str(ID).ljust(id_length) 
            info += fstring
            info += '\n'
            mem += stats.get('memory',0)
            
        fstring = str(self.ID).ljust(id_length)+seperator  
        fstring += ('%.1f' % mem).rjust(table_format[0][1]) + seperator
        try:
            t = str(self.dtype).split("'")[1].split(".")[1]
        except:
            t = str(self.dtype)
        fstring += t.rjust(table_format[0][1])
        fstring += '\n'
        fstring += info
        return fstring

    def __getitem__(self,view):
        """
        Access content through view.
        
        Parameters
        ----------
        view : View
               A valid :any:`View` object.
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
               A valid :any:`View` for this object
               
        newdata : array_like
                  The data to be stored 2D.
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
        Parameters
        ----------
        ptycho : Ptycho
            The instance of Ptycho associated with this pod. 
            
        ID : str or int
            The pod ID, If None it is managed by the ptycho.
            
        Keyword Args
        ------------
        views : dict or Param
            The views. See POD.DEFAULT_VIEWS.
            
        geometry : Geo
            Geometry class instance and attached propagator

        """
        super(POD,self).__init__(ptycho,ID,False)
        if len(kwargs) > 0:
            self._initialize(**kwargs)
            
    def _initialize(self,views=None,geometry=None):#,meta=None):
       
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
        
        if self.ex_view is None:
            self.use_exit_container = False
            self._exit = np.ones_like(self.geometry.shape,dtype=self.owner.CType)
        else:
            self.use_exit_container = True
            self._exit = None
            
        self.ma_view = self.V['mask']
        # Check whether this pod is active it should maybe also have a check for an active mask view?
        # Maybe this should be tight to to the diff views activeness through a property

    @property
    def active(self):
        """
        Convenience property that describes where this pod is active or not.
        Equivalent to ``self.di_view.active``
        """    
        return self.di_view.active
        
    @property
    def fw(self):
        """
        Convenience property that returns forward propagator of attached 
        Geometry instance. Equivalent to ``self.geometry.propagator.fw``
        """  
        return self.geometry.propagator.fw
    
    @property
    def bw(self):
        """
        Convenience property that returns backward propagator of attached 
        Geometry instance. Equivalent to ``self.geometry.propagator.bw``
        """
        return self.geometry.propagator.bw
    
    @property
    def object(self):
        """
        Convenience property that links to slice of object :any:`Storage`
        Usually equivalent to ``self.ob_view.data``
        """
        if not self.is_empty:
            return self.ob_view.data
        else:
            return np.ones(self.geometry.N, dtype = self.owner.CType)
            
    @object.setter
    def object(self,v):
        self.ob_view.data=v

    @property
    def probe(self):
        """
        Convenience property that links to slice of probe :any:`Storage`
        Usually equivalent to ``self.pr_view.data``
        """
        return self.pr_view.data
        
    @probe.setter
    def probe(self,v):
        self.pr_view.data=v

    @property
    def exit(self):
        """
        Convenience property that links to slice of exit wave 
        :any:`Storage`. Equivalent to ``self.pr_view.data``
        """
        if self.use_exit_container:
            return self.ex_view.data
        else:
            return self._exit
            
    @exit.setter
    def exit(self,v):
        if self.use_exit_container:
            self.ex_view.data = v
        else:
            self._exit=v

    @property
    def diff(self):
        """
        Convenience property that links to slice of diffraction 
        :any:`Storage`. Equivalent to ``self.di_view.data``
        """
        return self.di_view.data
        
    @diff.setter
    def diff(self,v):
        self.di_view.data=v
        
    @property
    def mask(self):
        """
        Convenience property that links to slice of masking 
        :any:`Storage`. Equivalent to ``self.ma_view.data``
        """
        return self.ma_view.data
        
    @mask.setter
    def mask(self,v):
        self.ma_view.data=v
