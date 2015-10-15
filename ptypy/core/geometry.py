"""
Geometry manager

This file is part of the PTYPY package.

    :copyright: Copyright 2014 by the PTYPY team, see AUTHORS.
    :license: GPLv2, see LICENSE for details.
"""
# for solo use ##########
if __name__ == "__main__":
    from ptypy import utils as u
    from ptypy.utils.verbose import logger
    from ptypy.core import Base
    GEO_PREFIX='G'
else:
# for in package use #####
    from .. import utils as u
    from ..utils.verbose import logger
    from classes import Base,GEO_PREFIX

from scipy import fftpack

import numpy as np


__all__=['DEFAULT','Geo','BasicNearfieldPropagator','BasicFarfieldPropagator']

DEFAULT=u.Param(
    energy = 6.2,                # Incident photon energy (in keV)
    lam = None,                 # Wavelength (in meters)
    distance = 7.0,                    # Distance from object to screen 
    psize = 172e-6,  # Pixel size (in meters) at detector plane
    resolution = None,               # Pixel sixe (in meters) at sample plane  
    shape = 256,                     # Number of detector pixels
    propagation = 'farfield',      # propagation type 
    misfit = 0,
    center = 'fftshift',
    origin = 'fftshift',
    precedence = None,
)
""" Default geometry parameters. See also :py:data:`.scan.geometry` and an unflattened representation below """

_old2new = u.Param(
    z = 'distance',                    # Distance from object to screen 
    psize_det = 'psize',  # Pixel size (in meters) at detector plane
    psize_sam = 'resolution',               # Pixel sixe (in meters) at sample plane  
    N = 'shape',                     # Number of detector pixels
    prop_type='propagation',
    origin_det= 'center', 
    origin_sam= 'origin', 
)


class Geo(Base):
    """
    Hold and keep consistant the information about experimental parameters.  
    
    Keeps also reference to the Propagator and updates this reference
    when resolution, pixel size, etc. is changed in `Geo`. Reference to
    :py:data:`.io.paths.recons`.          
    
    Attributes
    ----------
    interact : bool (True)
        If set to True, changes to properties like :py:meth:`energy`,
        :py:meth:`lam`, :py:meth:`shape` or :py:meth:`psize` will cause 
        a call to :py:meth:`update`
    
    """
    
    DEFAULT=DEFAULT
    _keV2m=1.240597288e-09
    _PREFIX = GEO_PREFIX
    
    def __init__(self,owner=None,ID=None,pars=None,**kwargs):
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
        super(Geo,self).__init__(owner,ID)
        #if len(kwargs)>0:
            #self._initialize(**kwargs)
    
    #def _initialize(self,pars=None,**kwargs):
        # Starting parameters
        p=u.Param(DEFAULT)
        if pars is not None:
            p.update(pars)
            for k,v in p.items():
                if k in _old2new.keys():
                    p[_old2new[k]] = v
        for k,v in kwargs.iteritems():
            if p.has_key(k): p[k] = v
        
        self.p = p
        self.interact = False
        
        # set distance
        if self.p.distance is None or self.p.distance==0:
            raise ValueError('Distance (geometry.distance) must not be None or 0')
        
        # set frame shape
        if self.p.shape is None or (np.array(self.p.shape)==0).any():
            raise ValueError('Frame size (geometry.shape) must not be None or 0')
        else:
            self.p.shape = u.expect2(p.shape)
            
        # Set energy and wavelength
        if p.energy is None:
            if p.lam is None:
                raise ValueError('Wavelength (geometry.lam) and energy (geometry.energy)\n must not both be None')
            else:
                self.lam = p.lam # also sets energy
        else:
            if p.lam is not None:
                logger.debug('Energy and wavelength both specified. Energy takes precedence over wavelength')
            
            self.energy = p.energy
            
        # set initial geometrical misfit to 0
        self.p.misfit = u.expect2(0.)
        
        # Pixel size
        self.p.psize_is_fix = p.psize is not None
        self.p.resolution_is_fix = p.resolution is not None
        
        if not self.p.psize_is_fix and not self.p.resolution_is_fix:
            raise ValueError('Pixel size in sample plane (geometry.resolution) and detector plane \n(geometry.psize) must not both be None')
        
        # fill pixel sizes
        self.p.resolution = u.expect2(p.resolution) if self.p.resolution_is_fix else u.expect2(1.0)
        self.p.psize = u.expect2(p.psize) if self.p.psize_is_fix else u.expect2(1.0)
        
        # update other values
        self.update(False)
        
        # attach propagator
        self._propagator = self._get_propagator()
        self.interact=True

    def update(self,update_propagator=True):
        """
        Update the internal pixel sizes, giving precedence to the sample
        pixel size (resolution) if self.psize_is_fixed is True.
        """
        # 4 cases
        if not self.p.resolution_is_fix and not self.p.psize_is_fix:
            # this is a rare case
            logger.debug('No pixel size is marked as constant. Setting detector pixel size as fix')
            self.psize_is_fix = True
            self.update()
            return
        elif not self.p.resolution_is_fix and self.p.psize_is_fix:
            if self.p.propagation=='farfield':
                self.p.resolution[:] = self.lz / self.p.psize / self.p.shape
            else:
                self.p.resolution[:] = self.p.psize
                
        elif self.p.resolution_is_fix and not self.p.psize_is_fix:
            if self.p.propagation=='farfield':
                self.p.psize[:] = self.lz / self.p.resolution / self.p.shape
            else:
                self.p.psize[:] = self.p.resolution
        else:
            # both psizes are fix
            if self.p.propagation=='farfield':
                # frame misfit that would make it work
                self.p.misfit[:] = self.lz / self.p.resolution / self.p.psize  - self.p.shape 
            else: 
                self.p.misfit[:] = self.p.resolution - self.p.psize
            
        # Update the propagator too (optionally pass the dictionary,
        # but Geometry & Propagator should share a dict
        
        if update_propagator: self.propagator.update(self.p)
           
    @property
    def energy(self):
        """
        Property to get and set the energy
        """
        return self.p.energy
        
    @energy.setter
    def energy(self,v):
        self.p.energy = v
        # actively change inner variables
        self.p.lam = self._keV2m / v
        if self.interact: self.update()
        
    @property
    def lam(self):
        """
        Property to get and set the wavelength
        """
        return self.p.lam
        
    @lam.setter
    def lam(self,v):
        # changing wavelengths never changes N, only psize
        # for changing N, please do so manually
        self.p.lam = v
        self.p.energy = self._keV2m / v 
        if self.interact: self.update()
            
    @property
    def resolution(self):
        """
        Property to get and set the pixel size in source plane
        """
        return self.p.resolution
    
    @resolution.setter
    def resolution(self,v):
        """
        changing source space pixel size 
        """
        self.p.resolution[:] = u.expect2(v)
        if self.interact: self.update()
        
    @property
    def psize(self):
        """
        Property to get and set the pixel size in the propagated plane
        """
        return self.p.psize
    
    @psize.setter
    def psize(self,v):
        self.p.psize[:] = u.expect2(v)
        if self.interact: self.update()
        
    @property
    def lz(self):
        """
        Retrieves product of wavelength and propagation distance
        """
        return self.p.lam * self.p.distance
        
    @property
    def shape(self):
        """
        Property to get and set the *shape* i.e. the frame dimensions
        """
        return self.p.shape
    
    @shape.setter
    def shape(self,v):
        self.p.shape[:] = u.expect2(v).astype(int)
        if self.interact: self.update()
    
    @property
    def distance(self):
        """
        Propagation distance in meters
        """
        return self.p.distance
        
    @property
    def propagator(self):
        """
        Retrieves propagator, createas propagator instance if necessary.
        """
        if not hasattr(self,'_propagator'):
            self._propagator = self._get_propagator()
        
        return self._propagator
        
    def __str__(self):
        keys = self.p.keys()
        keys.sort()
        start =""
        for key in keys:
            start += "%25s : %s\n" % (str(key),str(self.p[key]))
        return start
        
    def _to_dict(self):
        # delete propagator reference
        del self._propagator
        # return internal dicts
        return self.__dict__.copy()
    
    #def _post_dict_import(self):
    #    self.propagator = get_propagator(self.p)
    #    self.interact = True
        
    def _get_propagator(self):
        # attach desired datatype for propagator
        try:
            dt = self.owner.CType
        except:
            dt = np.complex64
        
        return get_propagator(self.p,dtype=dt)
            
        
def get_propagator(geo_dct,**kwargs):
    """
    Helper function to determine which propagator should be attached to Geometry class
    """
    if geo_dct['propagation']=='farfield':
        return BasicFarfieldPropagator(geo_dct,**kwargs)
    else:
        return BasicNearfieldPropagator(geo_dct,**kwargs)
    
class BasicFarfieldPropagator(object):
    """
    Basic single step Farfield Propagator.
    
    Includes quadratic phase factors and arbitrary origin in array.
    
    Be aware though, that if the origin is not in the center of the frame, 
    coordinates are rolled peridically, just like in the conventional fft case.
    """
    DEFAULT = DEFAULT
    
    def __init__(self,geo_pars=None,ffttype='numpy',**kwargs):
        """
        Parameters
        ----------
        geo_pars : Param or dict
            Parameter dictionary as in :py:attr:`DEFAULT`.
        
        ffttype : str or other
            Type of CPU-based FFT implementation. One of
            
            - 'numpy' for numpy.fft.fft2 
            - 'scipy' for scipy.fft.fft2
            - 2 or 4-tupel of (forward_fft2(),invese_fft2(),
              [scaling,inverse_scaling])
        """
        self.p=u.Param(DEFAULT)
        self.dtype = kwargs['dtype'] if kwargs.has_key('dtype') else np.complex128
        self.ffttype = ffttype
        self.update(geo_pars,**kwargs)
        
    
    def update(self,geo_pars=None,**kwargs):
        """
        update internal p dictionary. Recompute all internal array buffers
        """
        # local reference to avoid excessive self. use
        p = self.p
        if geo_pars is not None:
            p.update(geo_pars)
        for k,v in kwargs.iteritems():
            if p.has_key(k): p[k] = v
               
        # wavelength * distance factor
        lz= p.lam * p.distance
        
        #calculate real space pixel size. 
        resolution = p.resolution if p.resolution is not None else lz / p.shape / p.psize
        
        # calculate array shape from misfit 
        mis = u.expect2(p.misfit)
        self.crop_pad = np.round(mis /2.0).astype(int) * 2
        self.sh = p.shape + self.crop_pad

        # undo rounding error
        lz /= (self.sh[0] + mis[0] - self.crop_pad[0])/self.sh[0]
        
        # calculate the grids
        c_sam = p.origin if str(p.origin)==p.origin else p.origin + self.crop_pad/2.
        c_det = p.center if str(p.center)==p.center else p.center + self.crop_pad/2.

        [X,Y] = u.grids(self.sh,resolution,c_sam)
        [V,W] = u.grids(self.sh,p.psize,c_det)
        
        # maybe useful later. delete this references if space is short
        self.grids_sam = [X,Y]  
        self.grids_det = [V,W]
        
        # quadratic phase + shift factor before fft
        self.pre_curve = np.exp(1j * np.pi * (X**2+Y**2) / lz ).astype(self.dtype)
        #self.pre_check = np.exp(-2.0*np.pi*1j*((X-X[0,0])*V[0,0]+(Y-Y[0,0])*W[0,0])/ lz).astype(self.dtype)
        self.pre_fft = self.pre_curve*np.exp(-2.0*np.pi*1j*((X-X[0,0])*V[0,0]+(Y-Y[0,0])*W[0,0])/ lz).astype(self.dtype)
        
        # quadratic phase + shift factor before fft
        self.post_curve = np.exp(1j * np.pi * (V**2+W**2) / lz ).astype(self.dtype)
        #self.post_check = np.exp(-2.0*np.pi*1j*(X[0,0]*V+Y[0,0]*W)/ lz).astype(self.dtype)
        self.post_fft = self.post_curve*np.exp(-2.0*np.pi*1j*(X[0,0]*V+Y[0,0]*W)/ lz).astype(self.dtype)

        # factors for inverse operation
        self.pre_ifft = self.post_fft.conj()
        self.post_ifft = self.pre_fft.conj()
        
        self._assign_fft(self.ffttype)
        
    def fw(self,W):
        """
        computes forward propagated wavefront of input wavefront W
        """
        # check for cropping
        if (self.crop_pad != 0).any() : 
            w = u.crop_pad(W,self.crop_pad)
        else:
            w = W
        
        w = self.post_fft * self.sc * self.fft(self.pre_fft * w) 
        
        # cropping again
        if (self.crop_pad != 0).any() : 
            return u.crop_pad(w,-self.crop_pad)
        else:
            return w
        
    def bw(self,W):
        """
        computes backward propagated wavefront of input wavefront W
        """
        # check for cropping
        if (self.crop_pad != 0).any() : 
            w = u.crop_pad(W,self.crop_pad)
        else:
            w = W
            
        # compute transform
        w = self.ifft(self.pre_ifft * w) * self.isc * self.post_ifft
        
        # cropping again
        if (self.crop_pad != 0).any() : 
            return u.crop_pad(w,-self.crop_pad)
        else:
            return w
            
    def _assign_fft(self,ffttype='std'):
        self.sc = 1./np.sqrt(np.prod(self.sh))
        #~ self.sc = 1./np.sqrt(np.prod(self.sh)/np.prod(self.sh)*np.prod(self.p.shape))
        self.isc = 1./self.sc #np.sqrt(np.prod(self.sh))
        if str(ffttype)=='scipy':
            self.fft = lambda x: fftpack.fft2(x).astype(x.dtype)
            self.ifft = lambda x: fftpack.ifft2(x).astype(x.dtype)          
        elif str(ffttype)=='numpy':
            self.fft = lambda x: np.fft.fft2(x).astype(x.dtype)
            self.ifft = lambda x: np.fft.ifft2(x).astype(x.dtype)
        else:
            self.fft = ffttype[0]
            self.ifft = ffttype[1]
            if len(ffttype) > 2:
                self.sc = ffttype[2]
                self.isc = ffttype[3]

def translate_to_pix(sh,center):
    """\
    Take arbitrary input and translate it to a pixel position with respect to sh.
    """
    sh=np.array(sh)
    if center=='fftshift':
        cen=sh//2.0
    elif center=='geometric':
        cen=sh/2.0-0.5
    elif center=='fft':
        cen=sh*0.0
    elif center is not None:
        #cen=sh*np.asarray(center) % sh - 0.5
        cen = np.asarray(center) % sh
        
    return cen

class BasicNearfieldPropagator(object):
    """
    Basic two step (i.e. two ffts) Nearfield Propagator.
    """
    DEFAULT = DEFAULT
    
    def __init__(self,geo_dct=None,ffttype='scipy',**kwargs):
        """
        Parameters
        ----------
        geo_pars : Param or dict
            Parameter dictionary as in :py:data:`DEFAULT`.
        
        ffttype : str or other
            Type of CPU-based FFT implementation. One of
            
            - 'numpy' for numpy.fft.fft2 
            - 'scipy' for scipy.fft.fft2
            - 2 or 4-tupel of (forward_fft2(),invese_fft2(),
              [scaling,inverse_scaling])
        """
        self.p=u.Param(DEFAULT)
        self.dtype = kwargs['dtype'] if kwargs.has_key('dtype') else np.complex128
        self.update(geo_dct,**kwargs)
        self._assign_fft(ffttype=ffttype)
    
    def update(self,geo_pars=None,**kwargs):
        """
        update internal p dictionary. Recompute all internal array buffers
        """
        # local reference to avoid excessive self. use
        p = self.p
        if geo_pars is not None:
            p.update(geo_pars)
        for k,v in kwargs.iteritems():
            if p.has_key(k): p[k] = v
               

        self.sh = p.shape
        
        # calculate the grids
        [X,Y] = u.grids(self.sh,p.resolution,p.origin)
                
        # maybe useful later. delete this references if space is short
        self.grids_sam = [X,Y]  
        self.grids_det = [X,Y] 
        
        # calculating kernel
        # psize_fspace = p.lam * p.distance / p.shape / p.resolution
        # [V,W] = u.grids(self.sh,psize_fspace,'fft')
        # a2 = (V**2+W**2) /p.distance**2
        psize_fspace = p.lam / p.shape / p.resolution
        [V,W] = u.grids(self.sh,psize_fspace,'fft')
        a2 = (V**2+W**2)
        
        self.kernel = np.exp(2j * np.pi * (p.distance / p.lam) * (np.sqrt(1 - a2) - 1))
        #self.kernel = np.fft.fftshift(self.kernel)
        self.ikernel = self.kernel.conj()
        
    def fw(self,W):
        """
        computes forward propagated wavefront of input wavefront W
        """
        return self.ifft(self.fft(W) * self.kernel)
        
    def bw(self,W):
        """
        computes backward propagated wavefront of input wavefront W
        """
        return self.ifft(self.fft(W) * self.ikernel)
            
    def _assign_fft(self,ffttype='std'):
        self.sc = 1./np.sqrt(np.prod(self.sh))
        self.isc = np.sqrt(np.prod(self.sh))
        if str(ffttype)=='scipy':
            self.fft = lambda x: fftpack.fft2(x).astype(x.dtype)
            self.ifft = lambda x: fftpack.ifft2(x).astype(x.dtype)          
        elif str(ffttype)=='numpy':
            self.fft = lambda x: np.fft.fft2(x).astype(x.dtype)
            self.ifft = lambda x: np.fft.ifft2(x).astype(x.dtype)
        else:
            self.fft = ffttype[0]
            self.ifft = ffttype[1]
            if len(ffttype) > 2:
                self.sc = ffttype[2]
                self.isc = ffttype[3]


############
# TESTING ##
############

if __name__ == "__main__":
    G = Geo()
    #G._initialize()
    GD = G._to_dict()
    G2 = Geo._from_dict(GD)
