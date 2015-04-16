# -*- coding: utf-8 -*-
"""
This module generates the probe

@author: Bjoern Enders

This file is part of the PTYPY package.

    :copyright: Copyright 2014 by the PTYPY team, see AUTHORS.
    :license: GPLv2, see LICENSE for details.

"""
import numpy as np
#import os

from .. import utils as u
from ..core import geometry
from ..utils.verbose import logger
from .. import resources

TEMPLATES = dict()

DEFAULT_aperture = u.Param(
    form='circ',        # (str) One of None, 'rect' or 'circ'
                        #  One of:
                        # - None: no aperture, this may be useful for nearfield
                        # - 'rect': rectangular aperture
                        # - 'circ': circular aperture
    diffuser=None,      # (float) Static Noise in the transparent part of the aperture
                        # Can act like a diffuser but has no wavelength dependency
                        # Can be either:
                        # - None : no noise
                        # - 2-tuple: noise in phase (amplitude (rms), minimum feature size)
                        # - 4-tuple: noise in phase & modulus (rms, mfs, rms_mod, mfs_mod)
    size=None,          # (float) Aperture width or diameter
                        # May also be a tuple (vertical,horizontal) for size in case of an asymmetric aperture
    edge=2,             # (int) Edge width of aperture in pixel to suppress aliasing
    central_stop=None,  # (float) size of central stop as a fraction of aperture.size
                        # If not None: places a central beam stop in aperture.
                        #  The value given here is the fraction of the stop compared to size
    offset=0.,          # (float) offset between center of aperture and optical axes
                        # May also be a tuple (vertical,horizontal) for size in case of an asymmetric offset
    rotate=0.,          # (float) rotate aperture by this value
)

DEFAULT_propagation = u.Param(    # Parameters for propagation after aperture plane
                        # Propagation to focus takes precedence to parallel propagation if focused is not None
    parallel=None,      # (float) Parallel propagation distance
                        # If None or 0 : No parallel propagation
    focussed=None,      # (float) Propagation distance from aperture to focus
                        # If None or 0 : No focus propagation
    spot_size=None,     # (float) Focal spot diameter 
    antialiasing=None,  # (float) antialiasing factor [not implemented]
                        # Antialiasing factor used when generating the probe.
                        # (numbers larger than 2 or 3 are memory hungry)
)

DEFAULT_diversity = u.Param(
    noise = None,       # (float) Noise added on top add the end of initialisation
                        # Can be either:
                        # - None : no noise
                        # - 2-tuple: noise in phase (amplitude (rms), minimum feature size)
                        # - 4-tuple: noise in phase & modulus (rms, mfs, rms_mod, mfs_mod)
    shift = None,
    power = 1.0,
)

DEFAULT = u.Param(
    ovveride = None,
    model=None,             # (str) User-defined probe (if type is None)
                            # `None`, path to a *.ptyr file or any python evaluable statement yielding a 3d numpy array.
                            #  If `None` illumination is modeled.
    photons=None,           # (int,float,None) Number of photons in the incident illumination
    recon = u.Param(
        rfile = '*.ptyr',
        ID = None,
        layer = None,
    ),
    stxm = u.Param(
        label=None,                 # label of the scan of whose diffraction data to initialize stxm. If None, use own scan_label
    ), 
    diversity=DEFAULT_diversity,       # diversity parameters, can be None = no diversity
    aperture=DEFAULT_aperture,         # aperture parameters, can be None = no aperture
    propagation=DEFAULT_propagation,   # propagation parameters, can be None = no propagation
)
"""See unflattened DEFAULTS below"""

__all__ =['DEFAULT','init_storage','aperture','process']

def rectangle(grids, dims=None, ew=2):
    if dims is None:
        dims = (grids.shape[-2]/2., grids.shape[-1]/2.)
    v, h = dims
    V, H = grids
    return u.smooth_step(-np.abs(V)+v/2,ew) * u.smooth_step(-np.abs(H)+h/2,ew)

def ellipsis(grids, dims=None, ew=2):
    if dims is None:
        dims = (grids.shape[-2]/2., grids.shape[-1]/2.)
    v, h = dims
    V, H = grids
    return u.smooth_step(0.5 - np.sqrt(V**2 / v**2 + H**2 / h**2),ew/np.sqrt(v*h))


def aperture(A,grids=None,pars=None, **kwargs):
    """
    Creates an aperture in the shape and dtype of `A` according 
    to x,y-grids `grids`. Keyword Arguments may be any of 
    :any:`DEFAULT`.aperture.
    
    Parameters
    ----------
    A : ndarray
        Model array (at least 2-dimensional) to place aperture on top.
        
    pars: dict or ptypy.utils.Param
        Parameters, see :any:`DEFAULT`.aperture
        
    grids : ndarray
        Cartesion coordinate grids, if None, they will be created with
        ``grids = u.grids(sh[-2:],psize=(1.0,1.0))``
        
    Returns
    -------
    ap : ndarray
        Aperture array (complex) in shape of A
    """
    p=u.Param(DEFAULT_aperture.copy())
    if pars is not None:
        p.update(pars)
        p.update(**kwargs)
    
    sh = A.shape
    if grids is not None:
        grids = np.array(grids).copy()
        psize = np.array((grids[0,1,0]-grids[0,0,0],grids[1,0,1]-grids[1,0,0]))
        assert ((np.array(grids.shape[-2:])-np.array(sh[-2:]))==0).any(), 'Grid and Input dimensions do not match'
    else:
        psize = u.expect2(1.0)
        grids = u.grids(sh[-2:],psize=psize)
        
    ap = np.ones(sh[-2:],dtype=A.dtype)

    if p.diffuser is not None:
        ap *= u.parallel.MPInoise2d(sh[-2:],*p.diffuser)
        
    if p.form is not None:
        off = u.expect2(p.offset)
        cgrid = grids[0].astype(complex)+1j*grids[1]
        cgrid -= np.complex(off[0],off[1])
        cgrid *= np.exp(1j*p.rotate)
        grids[0] = cgrid.real / psize[0]
        grids[1] = cgrid.imag / psize[1]

        if str(p.form)=='circ':
            apert = lambda x: ellipsis(grids,x,p.edge)
        elif str(p.form)=='rect':
            apert = lambda x: rectangle(grids,x,p.edge)
        else:
            raise NotImplementedError('Only elliptical `circ` or rectangular `rect` apertures supported for now')

        dims = u.expect2(p.size)/ psize if p.size is not None else np.array(cgrid.shape)/3.
        print dims
        ap *= apert(dims)
        if p.central_stop is not None:
            dims *= u.expect2(p.central_stop)
            ap *= 1-apert(dims)

    return np.resize(ap,sh)
        
def init_storage(storage, pars, energy =None,**kwargs):
    """
    Initializes :any:`Storage` `storage` with parameters from `pars`
    
    Parameters
    ----------
    storage : ptypy.core.Storage
        A :any:`Storage` instance in the *probe* container of :any:`Ptycho`
        
    pars : Param
        Parameter structure for creating a probe / illumination. 
        See :any:`DEFAULT`
        
    energy : float, optional
        Energy associated with this storage. If None, tries to retrieve
        the energy from the already initialized ptypy network. 
    """
    
    s =storage
    p = DEFAULT.copy(depth=3)
    model = None
    if hasattr(pars,'items') or hasattr(pars,'iteritems'):
        # this is a dict
        p.update(pars, in_place_depth=3)    
    
    # first we check for scripting shortcuts. This is only convenience
    elif str(pars)==pars:
        # this maybe a template now or a file
        
        # deactivate further processing
        p.aperture = None
        p.propagation = None
        p.diversity = None
        
        if pars.endswith('.ptyr'):
            recon = u.Param(rfile=pars,layer=None,ID=s.ID)
            p.recon = recon
            p.model = 'recon'
            try:
                init_storage(s,p,energy = 1.0)
            except KeyError:
                logger.warning('Loading of probe storage `%s` failed. Trying any storage.' % s.ID)
                p.recon.ID = None
                init_storage(s,p,energy = 1.0)
            return
        elif pars in TEMPLATES.keys():
            init_storage(s,TEMPLATES[pars])
            return 
        elif resources.probes.has_key(pars) or pars =='stxm':
            p.model = pars
            init_storage(s,p)
            return
        else:
            raise RuntimeError('Shortcut string `%s` for probe creation is not understood' %pars)
    elif type(pars) is np.ndarray:
        p.model=pars
        p.aperture = None
        p.propagation = None
        p.diversity = None
        init_storage(s,p)
        return
    else:
        ValueError('Shortcut for probe creation is not understood')
        
    if p.model is None:
        model = np.ones(s.shape,s.dtype)
        if p.photons is not None:
            model*=np.sqrt(p.photons)/np.prod(s.shape) 
    elif type(p.model) is np.ndarray:
        model = p.model
    elif resources.probes.has_key(p.model):
        model = resources.probes[p.model](A.shape)
    elif str(p.model) == 'recon':
        # loading from a reconstruction file
        layer = p.recon.get('layer')
        ID = p.recon.get('ID')
        logger.info('Attempt to load layer `%s` of probe storage with ID `%s` from `%s`' %(str(layer),str(ID),p.recon.rfile))
        model = u.load_from_ptyr(p.recon.rfile,'probe',ID,layer)
        # this could be more sophisticated, i.e. matching the real spac grids etc.
        
    elif str(p.model) == 'stxm':
        logger.info('Probe initialization using averaged diffraction data')
        # pick a pod that no matter if active or not
        pod = s.views[0].pod
           
        alldiff = np.zeros(pod.geometry.shape)
        n = 0
        for v in s.views:
            if not v.pod.active : continue
            alldiff += u.abs2(v.pod.diff)
            n+=1
        if n> 0:
            alldiff /= n
        # communicate result
        u.parallel.allreduce(alldiff)
        #pick a propagator and a pod
        # in far field we will have to know the wavefront curvature
        try:
            curve = pod.geometry.propagator.post_curve
        except:
            # ok this is nearfield
            curve = 1.0 
            
        model = pod.bw(curve * np.sqrt(alldiff))
    else:
        raise ValueError('Value to `model` key not understood in probe creation')
        
    assert type(model) is np.ndarray, "Internal model should be numpy array now but it is %s" %str(type(model))
    # expand model to the right length filling with copies
    sh = model.shape[-2:]
    model = np.resize(model,(s.shape[0],sh[0],sh[1]))

    # find out about energy if not set
    if energy is None:
        energy  =  s.views[0].pod.geometry.energy

    # perform aperture multiplication, propagation etc.
    model = _process(model,p.aperture,p.propagation, p.photons, energy, s.psize)
    
    # apply diversity
    if p.diversity is not None:
        u.diversify(model,**(p.diversity))
    
    # fill storage array
    s.fill(model)
    
def _process(model,aperture_pars=None,prop_pars=None, photons = 1e7, energy=6.,resolution=7e-8,**kwargs):
    """
    Processes 3d stack of incoming wavefronts `model`. Applies aperture
    accoridng to `aperture_pars` and propagates according to `prop_pars`
    and other keywords arguments. 
    """
    # create the propagator
    ap_size, grids, prop  = _propagation(prop_pars,model.shape[-2:],resolution,energy)
        
    # form the aperture on top of the model
    if type(aperture_pars) is np.ndarray:
        model *= np.resize(aperture_pars, model.shape)
    elif aperture_pars is not None:
        if ap_size is not None:
            aperture_pars.size = ap_size
        model *= aperture(model, grids, aperture_pars)
    else:
        logger.warning('No aperture defined in probe creation. This may lead to artifacts if the probe model is not choosen carefully.')
    # propagate
    model = prop(model)
    
    # apply photon count
    if photons is not None:
        model *= np.sqrt( photons / u.norm2(model))
    
    
    return model

def _propagation(prop_pars,shape=256,resolution=7e-8,energy=6.):
    p = prop_pars
    grids = None
    if p is not None and len(p)>0:
        ap_size = p.spot_size if p.spot_size is not None else None 
        ffGeo = None
        nfGeo = None
        fdist = p.focussed
        if fdist is not None:
            geodct = u.Param(
                energy=energy,
                shape=shape,
                psize=resolution,
                resolution=None,
                distance=fdist,
                propagation='farfield'
                )
            ffGeo = geometry.Geo()
            ffGeo._initialize(geodct)
            #print ffGeo
            if p.spot_size is not None:
                ap_size = ffGeo.lam*fdist/np.array(p.spot_size)*2*np.sqrt(np.sqrt(2))
            else:
                ap_size = None
            grids = ffGeo.propagator.grids_sam
            phase=np.exp(-1j*np.pi/ffGeo.lam/fdist*(grids[0]**2+grids[1]**2))
            #from matplotlib import pyplot as plt
            #plt.figure(100);plt.imshow(u.imsave(ffGeo.propagator.post_fft))
        if p.parallel is not None:
            geodct = u.Param(
                energy=energy,
                shape=shape,
                resolution=resolution,
                psize=None,
                distance=p.parallel,
                propagation='nearfield'
                )
            nfGeo = geometry.Geo()
            nfGeo._initialize(geodct)
            grids = nfGeo.propagator.grids_sam if grids is None else grids
            
        if ffGeo is not None and nfGeo is not None:
            prop = lambda x: nfGeo.propagator.fw(ffGeo.propagator.fw(x*phase))
        elif ffGeo is not None and nfGeo is None:
            prop = lambda x: ffGeo.propagator.fw(x*phase)
        elif ffGeo is None and nfGeo is not None:
            prop = lambda x: nfGeo.propagator.fw(x)
        else:
            grids = u.grids(u.expect2(shape),psize=u.expect2(resolution))
            prop = lambda x: x
    else:
        grids = u.grids(u.expect2(shape),psize=u.expect2(resolution))
        prop = lambda x: x
        ap_size = None
        
    return ap_size, grids, prop

DEFAULT_old = u.Param(
    probe_type='parallel',		# 'focus','parallel','path_to_file'
    aperture_type = 'circ',			# 'rect','circ','path_to_file'
    aperture_size = None,           # aperture diameter
    aperture_edge = 1,              # edge smoothing width of aperture in pixel
    focal_dist = None,              # distance from prefocus aperture to focus
    prop_dist = 0.001,              # propagation distance from focus (or from aperture if parallel)
    UseConjugate = False,           # use the conjugate of the probe instef of the probe
    antialiasing = 2.0,             # antialiasing factor used when generating the probe
    spot_size = None,                # if aperture_size = None this parameter is used instead.
                                    # Gives the desired probe size in sample plane
                                    # if spot_size = None, a 50 pixel aperture will be used
    incoming = None,                # incomin
    probe = None,
    photons = 1e8,                  # photons in the probe
    mode_diversity = 'noise',
    mode_weights = [1.,0.1],         # first weight is main mode, last weight will be copied if
                                    # more modes requested than weight given
    spectrum = None,                # energy spectrum of source, choose None, 'Gauss' or 'Box'
    bandwidth = 0.1,              # bandwidth of source
    phase_noise_rms = None,
    phase_noise_mfs = 0.0,
)



if __name__=='__main__':
    energy=6.
    shape=512
    resolution=8e-8
    p = u.Param()
    p.aperture = u.Param()
    p.aperture.form='circ'
    p.aperture.diffuser=(10.0,5,0.1,20.0)
    p.aperture.size=100e-6
    p.aperture.edge=2             # (int) Edge width of aperture in pixel to suppress aliasing
    p.aperture.central_stop=0.3
    p.aperture.offset=0. 
    p.aperture.rotate=0.          # (float) rotate aperture by this value
    p.propagation=u.Param()    # Parameters for propagation after aperture plane
    p.propagation.parallel=0.015      # (float) Parallel propagation distance
    p.propagation.focussed=0.1     # (float) Propagation distance from aperture to focus
    p.propagation.spot_size=None     # (float) Focal spot diameter
    p.propagation.antialiasing=None  # (float) antialiasing factor
    p.probe=None             # (str) User-defined probe (if type is None)
    p.photons=None           # (int,float,None) Number of photons in the incident illumination
    p.noise=None             # (float) Noise added on top add the end of initialisation

    probe = from_pars_no_storage(pars=p, energy=energy,shape=shape,resolution=resolution)
    from matplotlib import pyplot as plt
    plt.imshow(u.imsave(abs(probe[0])))
