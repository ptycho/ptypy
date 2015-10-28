

# -*- coding: utf-8 -*-
"""
This module generates a sample

@author: Bjoern Enders

This file is part of the PTYPY package.

    :copyright: Copyright 2014 by the PTYPY team, see AUTHORS.
    :license: GPLv2, see LICENSE for details.
"""
import numpy as np
#import os
#from matplotlib import pyplot as plt

if __name__ == '__main__':
    from ptypy import utils as u
    from ptypy import resources
else:
    from .. import utils as u
    from .. import resources
    
logger = u.verbose.logger

TEMPLATES = dict()

DEFAULT_process = u.Param(
    offset = 0,               # (float,tuple) Offset between center of object array and scan pattern (in pixel)
    zoom = None,                # (float,tuple) Zoom value for object simulation.
    formula = None,             # (str) Chemical formula
    density = None,             # (None,float) Density in [g/ccm]
    thickness = None,            # (float) Maximum thickness of sample in meter
    ref_index = 0.5+0.j,         # (complex float) Assigned refractive index (maximum) relative to air
    smoothing = 2,            # (float,tuple) Gaussian filter smoothing with this FWHM (pixel)
)

DEFAULT_diversity = u.Param(
    noise = None,
    shift = None,
    power = 1.0,
)

DEFAULT = u.Param(
    override = None,               # 'scripting option to override'
    model= None,                  # (None,'sim','stxm','recon')Type of object model
    fill = 1.0,
    recon = u.Param(
        ID = None,
        layer = None,
        rfile = '*.ptyr',
    ),
    stxm = u.Param(
        label=None,                 # label of the scan of whose diffraction data to initialize stxm. If None, use own scan_label
    ),                # STXM analysis parameters
    process = DEFAULT_process,
    diversity = DEFAULT_diversity , # see other for noise
)
"""Default sample parameters. See :py:data:`.scan.sample` and a short listing below"""

__all__=['DEFAULT','init_storage','simulate']


def init_storage(storage,sample_pars = None, energy = None):
    """
    Initializes a storage as sample transmission.
    
    Parameters
    ----------
    storage : Storage
        The object :any:`Storage` to initialize
        
    sample_pars : Param
        Parameter structure that defines how the sample is created.
        See :any:`DEFAULT` for the parameters.
    
    energy : float, optional
        Energy associated in the experiment for this sample object.
        If None, the ptypy structure is searched for the appropriate
        :any:`Geo` instance:  ``storage.views[0].pod.geometry.energy``
    """
    s = storage
    prefix = "[Object %s] " % s.ID
    
    sam = sample_pars
    p = DEFAULT.copy(depth=3)
    model = None
    if hasattr(sam,'items') or hasattr(sam,'iteritems'):
        # this is a dict
        p.update(sam, in_place_depth = 3)    
    
    # first we check for scripting shortcuts. This is only convenience
    elif str(sam)==sam:
        # this maybe a template now or a file
        
        # deactivate further processing
        p.process = None
        
        if sam.endswith('.ptyr'):
            recon = u.Param(rfile=sam,layer=None,ID=s.ID)
            p.recon = recon
            p.model = 'recon'
            p.process = None
            init_storage(s,p)
        elif sam in TEMPLATES.keys():
            init_storage(s,TEMPLATES[sam])
        elif resources.objects.has_key(sam) or sam =='stxm':
            p.model = sam
            init_storage(s,p)            
        else:
            raise RuntimeError(prefix+'Shortcut string `%s` for object creation is not understood' %sam)
    elif type(sam) is np.ndarray:
        p.model=sam
        p.process = None
        init_storage(s,p)
    else:
        ValueError(prefix+'Shortcut for object creation is not understood')
        
    if p.model is None or str(p.model)=='sim':
        model = np.ones(s.shape,s.dtype)*p.fill
    elif type(p.model) is np.ndarray:
        model = p.model
    elif resources.objects.has_key(p.model):
        model = resources.objects[p.model](A.shape)
    elif str(p.model) == 'recon':
        # loading from a reconstruction file
        layer = p.recon.get('layer')
        ID = p.recon.get('ID')
        logger.info(prefix+'Attempt to load object storage with ID %s from %s' %(str(ID),p.recon.rfile))
        model = u.load_from_ptyr(p.recon.rfile,'obj',ID,layer)
        # this could be more sophisticated, i.e. matching the real spac grids etc.
        
    elif str(p.model) == 'stxm':
        logger.info(prefix+'STXM initialization using diffraction data')
        #print s.data.shape, s.shape
        trans, dpc_row, dpc_col = u.stxm_analysis(s)
        model = trans * np.exp(1j * u.phase_from_dpc(dpc_row, dpc_col))
    else:
        raise ValueError(prefix+'Value to `model` key not understood in object creation')
        
    assert type(model) is np.ndarray, "".join([prefix,"Internal model should be numpy array now but it is %s" %str(type(model))])
    
    # expand model to the right length filling with copies
    sh = model.shape[-2:]
    model = np.resize(model,(s.shape[0],sh[0],sh[1]))

    # try to retrieve the energy
    if energy is None:
        try:
            energy  =  s.views[0].pod.geometry.energy
        except:
            logger.info(prefix+'Could not retrieve energy from pod network... Maybe there are no pods yet created?')
    s._energy = energy
            
    # process the given model
    if str(p.model)=='sim' or p.process is not None:

        # make this a single call in future
        model = simulate(model,p.process,energy,p.fill,prefix )
    
    # symmetrically cut to shape of data
    model = u.crop_pad_symmetric_2d(model,s.shape)[0]
    
    # add diversity
    if p.diversity is not None:
        u.diversify(model,**(p.diversity))
    # return back to storage
    s.fill(model)
    
def simulate(A,pars, energy, fill =1.0,prefix="", **kwargs):
    """
    Simulates a sample object into model numpy array `A`
    
    Parameters
    ----------
    A : ndarray
        Numpy array as buffer. Must be at least twodimensional
        
    pars : Param
        Simulation parameters. See :any:`DEFAULT` .simulate
    """
    lam = u.keV2m(energy)
    p = DEFAULT_process.copy()
    p.update(pars)
    p.update(kwargs)
        
    """
    res = p.resource 
    if res is None:
        raise RuntimeError("Resource for simulation cannot be None. Please specify one of ptypy's resources or load numpy array into this key")
    elif str(res)==res:
        # check if we have a ptypy resource for this

        else:
            # try loading as if it was an image
            try:
                res = u.imload(res).astype(float)
                if res.ndmim > 2:
                    res = res.mean(-1)
            except:
                raise RuntimeError("Loading resource %s as image has failed")
        
    assert type(res) is np.ndarray, "Resource should be a numpy array now"
    """
    
    # Resize along first index
    #newsh = (A.shape[0], res.shape[-2],res.shape[-1])
    #obj = np.resize(res,newsh)
    obj = A.copy()
    
    if p.zoom is not None:
        zoom = u.expect3(p.zoom)
        zoom[0] = 1
        obj = u.zoom(obj,zoom)
        
    if p.smoothing is not None:
        obj = u.gf_2d(obj,p.smoothing / 2.35)
       
    off = u.expect2(p.offset)
    k = 2 * np.pi / lam
    ri = p.ref_index
    d = p.thickness
    # Check what we got for an array
    if np.iscomplexobj(obj) and d is None:
        logger.info(prefix+"Simulation resource is object transmission")
    elif np.iscomplexobj(obj) and d is not None:
        logger.info(prefix+"Simulation resource is integrated refractive index")
        obj = np.exp(1.j*obj*k*d)
    else:
        logger.info(prefix+"Simulation resource is a thickness profile")
        # enforce floats
        ob = obj.astype(np.float)
        ob -= ob.min()
        if d is not None:
            logger.info(prefix+"Rescaling to maximum thickness")
            ob /= ob.max()/d
            
        if p.formula is not None or ri is not None:
            # use only magnitude of obj and scale to [0 1]
            if ri is None:
                en = u.keV2m(1e-3)/lam
                if u.parallel.master:
                    logger.info(prefix+"Quering cxro database for refractive index \
                     in object creation with paramters:\n Formula=%s Energy=%d Density=%.2f" % (p.formula,en,p.density))
                    result = np.array(u.cxro_iref(p.formula,en,density=p.density))
                else:
                    result = None
                result = u.parallel.bcast(result)
                energy, delta,beta = result
                ri = - delta +1j*beta
                
            else:
                logger.info(prefix+"Using given refractive index in object creation")

        obj = np.exp(1.j*ob*k*ri)
    #if p.diffuser is not None:
    #    obj*=u.parallel.MPInoise2d(obj.shape,*p.diffuser)
    
    # get obj back in original shape and apply a possible offset
    shape = u.expect2(A.shape[-2:])
    crops = list(-np.array(obj.shape[-2:]) + shape + 2*np.abs(off))
    obj = u.crop_pad(obj,crops,fillpar=fill)
    off +=np.abs(off)
    
    return np.array(obj[...,off[0]:off[0]+shape[0],off[1]:off[1]+shape[1]])


DEFAULT_old=u.Param(
    source = None,      # None,path to a previous recon, or nd-array
                        
    offset = (0,0),      # offset= offset_list(int(par[0]))       
                          # (offsetx,offsety) move scan pattern relative to center in pixel 

    zoom = None,          # None, scalar or 2-tupel. If None, the pixel is assumed to be right 
                          # otherwise the image will be resized using ndimage.zoom            
    formula = None ,      # chemical formula (string) 
    density = 1.0 ,       # density in [g/ccm] only used if formula is not None
    ref_index = None,  # If None, treat projection as projection of refractive index/
                          # If a refractive index is provided the object's absolut value will be
                          # used to scale the refractive index.              
    thickness = 1e-6,      # max thickness of sample if None, the absolute values of loaded src array will be used

    smoothing_mfs = None, # Smooth with minimum feature size (in pixel units) if not None
    
    noise_rms = None,  # noise applied, relative to 2*pi in phase and relative to 1 in amplitude
    noise_mfs = None,
    fill = 1.0,        # if object is smaller than the objectframe, fill with fill:
    obj = None,     # override
    mode_diversity = 'noise',
    mode_weights = [1.,0.1]         # first weight is main mode, last weight will be copied if
                                    # more modes requested than weight given
)



def from_pars_old(shape,lam,pars=None,dtype=np.complex):
    """
    *DEPRECATED*
    """
    p=u.Param(DEFAULT)
    if pars is not None and (isinstance(pars,dict) or isinstance(pars,u.Param)):
        p.update(pars)
    if p.obj is not None:
        #abort her if object is set
        return p
    else:
        if isinstance(p.source,np.ndarray):
            logger.info('Found nd-array')
            obj=p.source
        else:
            logger.info('Fill with ones!')
            obj = np.ones(shape)
                
        obj=obj.astype(dtype)
        
        off = u.expect2(p.offset)
        
        if p.zoom is not None:
            obj = u.zoom(obj,p.zoom)
            
        if p.smoothing_mfs is not None:
            obj = u.gf(obj,p.smoothing_mfs / 2.35)
        
        k = 2 * np.pi / lam
        ri = p.ref_index
        if p.formula is not None or ri is not None:
            # use only magnitude of obj and scale to [0 1]
            if ri is None:
                en = u.keV2m(1e-3)/lam
                if u.parallel.master:
                    logger.info("Quering cxro database for refractive index \
                     in object creation with paramters:\n Formula=%s Energy=%d Density=%.2f" % (p.formula,en,p.density))
                    result = np.array(iofr(p.formula,en,density=p.density))
                else:
                    result = None
                result = u.parallel.bcast(result)
                energy, delta,beta = result
                ri = - delta +1j*beta
                
            else:
                logger.info("using given refractive index in object creation")
            
            ob = np.abs(obj).astype(np.float)
            ob -= ob.min()
            if p.thickness is not None:
                ob /= ob.max()/p.thickness

            obj = np.exp(1.j*ob*k*ri)

        shape = u.expect2(shape)
        crops = list(-np.array(obj.shape) + shape + 2*np.abs(off))
        obj = u.crop_pad(obj,crops,fillpar=p.fill)
        
        if p.noise_rms is not None:
            n = u.expect2(p.noise_rms)
            noise = np.random.normal(1.0,n[0]+1e-10,obj.shape)*np.exp(2j*np.pi*np.random.normal(0.0,n[1]+1e-10,obj.shape))
            if p.noise_mfs is not None:
                noise=u.gf(noise,p.noise_mfs / 2.35)
            obj*=noise
            

            
        off +=np.abs(off)
        p.obj = obj[off[0]:off[0]+shape[0],off[1]:off[1]+shape[1]]
        
        return p

def _create_modes(layers,pars):
    """
    **DEPRECATED**
    """
    p=u.Param(pars)
    pr=p.obj
    sh_old = pr.shape
    if pr.ndim==2:
        ppr=np.zeros((1,) + pr.shape).astype(pr.dtype) 
        ppr[0] = pr
        pr=ppr
    elif pr.ndim==4:
        pr=pr[0]
    w = p.mode_weights 
    # press w into 1d flattened array:
    w=np.atleast_1d(w).flatten()
    w=u.crop_pad(w,[[0,layers-w.shape[0]]],filltype='project')
    w/=w.sum()
    # make it an array now: flattens
    pr = u.crop_pad(pr, [[0,layers-pr.shape[0]]] ,axes=[0],filltype='project')
    #if p.mode_diversity=='noise'
    p.mode_weights = w
    p.obj = pr * w.reshape((layers,1,1))
    return p

