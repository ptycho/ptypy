# -*- coding: utf-8 -*-
"""
This module generates the probe

@author: Bjoern Enders

This file is part of the PTYPY package.

    :copyright: Copyright 2014 by the PTYPY team, see AUTHORS.
    :license: GPLv2, see LICENSE for details.

TODO:
- loading a probe from file

"""
import numpy as np
import os

from .. import utils as u
from ..utils import prop 
from ..utils.verbose import logger
from ..utils import validator

DEFAULT=u.Param(
    probe_type = 'parallel',		# 'focus','parallel','path_to_file'
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
)



def from_pars(sh,psize,lam,off=0.,pars=None,dtype=np.complex128):
    p = u.Param(DEFAULT)
    if pars is not None:
        p.update(pars)

    #validator.validate(p, '.scan.illumination')

    if p.probe is not None: return p
    #print p.paramdict

    # FIXME: implement loading probe from data.
    if os.path.isfile(p.probe_type):
        logger.info('File for probe found, attempt to load')
        try:
            # io.load.....
            # p.probe =
            # return p 
            pass
        except:
            pass
            
    if p.incoming is None:
        p.incoming=np.ones(sh)
    else:
        shorg = u.expect2(p.incoming.shape).astype(int)
        shtar = u.expect2(sh).astype(int)        
        if (shtar != shorg).any() : 
            p.incoming = czoom(p.incoming, shtar.astype(float)/shorg) 
    
    if p.phase_noise_rms is not None:
        noise = u.parallel.MPIrand_normal(0.0,p.phase_noise_rms,sh)
        noise = u.gf(noise,u.expect2(p.phase_noise_mfs/ 2.35))
        p.incoming = p.incoming.astype(np.complex) *np.exp(1j * noise)
        
    if p.spot_size is None:
        p.spot_size = 50*psize
    #print p
    if p.probe_type == 'focus':
        p.probe = simple_probe(p.incoming,lam,psize,p.focal_dist,p.prop_dist,p.aperture_type,p.aperture_size,p.spot_size,p.antialiasing,p.aperture_edge)
    elif p.probe_type == 'parallel':
        p.probe = simple_probe(p.incoming,lam,psize,None,p.prop_dist,p.aperture_type,p.aperture_size,p.spot_size,p.antialiasing,p.aperture_edge)
    else:
        try:
            p.probe = np.load(probe_type)
        except:
            pass

    # FIXME: warn user because most of the time this is deduced from the data.
    if p.photons is not None:
        p.probe *= np.sqrt(p.photons / np.sum(np.abs(p.probe)**2)) 

    if str(p.spectrum) == 'Gauss':
        p.probe *= u.gauss_fwhm(off,p.bandwidth)*p.bandwidth
  
    return p

def create_modes(layers,pars):
    p=u.Param(pars)
    pr=p.probe
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
    # FIXME: mode initialization has to be diverse!
    #if p.mode_diversity=='noise'
    p.mode_weights = w
    p.probe = pr * w.reshape((layers,1,1))
    if p.mode_diversity=='noise':
        noise = np.exp(1j*u.parallel.MPIrand_normal(0.0,np.pi,pr.shape))
        noise[0] = 1.0+0.j
        p.probe*= noise
    return p
##### other Functions ###########

def simple_probe(w,l,psize,fdist=None,defocus=0.0,pupil='rect',pupildims=None,focusdims=None,antialiasing=1.0,edgewidth=1):
    """\
    Generates a focussed probe from standard pupils (rectangle,ellipsoid)
    
    PARAMETERS
    -----------------------------------
    w : input illumination. use np.zeros() if uniform illumination is wanted. 
        l : wavelength.
    psize : pixelsize at focal area
    fdist (None)   : distance to focal plane, if None then we assume a parallel beam and pupil is in focus
    defocus (0.0)  : free propagation distance of prop close to focus (note that it is nearfield propagation):
    pupil ('rect')     : ['rect','circ','other'] 
                         entrance pupil type prior focussing, if nor 'rect' or 'circ' it interprets it as file and tries to load it.
    pupildims (None)   : chractetiscic size(s) of pupil. corresponds to length (rect) or diameter (circ)
    focusdims (None    : if pupildims are not set they can be guessed from a desired focal size
    antialiasing (1.0) : set higher than 1.0, pads array in focal plane to suppress aliasing.
    """    
    if w.ndim != 2:
        raise RunTimeError("A 2-dimensional array 'w' was expected")
    sh=u.expect2(w.shape)
    psize= u.expect2(psize)
    ew=edgewidth
    #print ew
    if focusdims==None:
        focusdims=0.1*np.array(sh)*u.expect2(psize)
    
    if antialiasing>1.0:
        lines=np.round(sh*(antialiasing-1.0))
        sh+=lines
        w = u.crop_pad(w,lines,fillpar=0.0)

    if fdist is not None:
        Pfocus=prop.Propagator(sh,[None,psize],l,fdist, ffttype='std',org = ['fftshift','fftshift'])
        fgrid=Pfocus.get_grids()[0]
        ew=ew*Pfocus.psize[0]
               
        if pupil=='rect':
            if pupildims==None:
                pupildims=u.expect2(l*fdist/np.array(focusdims)*2)
            pdims = u.expect2(pupildims) 
            apert=u.smooth_step(-np.abs(fgrid[0])+pdims[0]/2,ew[0]) * u.smooth_step(-np.abs(fgrid[1])+pdims[1]/2,ew[1])

        elif pupil=='circ':
            if pupildims==None:
                pupildims=l*fdist/np.array(focusdims)*2*np.sqrt(np.sqrt(2))
            pdims = u.expect2(pupildims)
            apert=u.smooth_step(0.5 - np.sqrt(fgrid[0]**2 / pdims[0]**2 + fgrid[1]**2 / pdims[1]**2),ew[0]/np.sqrt(pdims[0]*pdims[1]))
        else:
            try:
                apert=np.load(pupil)
            except:
                print('Expected path to .npy dexcribing the aperture if not specified as "circ" or "rect"')
                print('Using no pupil... I hope your incoming illumination is structured...')
                apert=np.ones(sh)

        phase=np.exp(-1j*np.pi/l/fdist*(fgrid[0]**2+fgrid[1]**2))
        field=Pfocus.ff(apert*phase*w)
        if defocus!=0.:
            Pdefoc = prop.Propagator(sh,[psize,None],l,defocus, ffttype='std',org = ['fftshift','fftshift'])
            field = Pdefoc.nf(field)
             
    else:
        if defocus!=0. and defocus is not None:
            Pdefoc = prop.Propagator(sh,[psize,None],l,defocus, ffttype='std',org = ['fftshift','fftshift'])
            rgrid=Pdefoc.get_grids()[0]
            ew=ew*Pdefoc.psize[0]
        else:
            rgrid = u.grids(sh,psize,center='fftshift')
            ew = u.expect2(ew*psize)
            
        if pupil=='rect':
            if pupildims==None:
                pupildims=focusdims
            pdims = u.expect2(pupildims)
            apert=u.smooth_step(-np.abs(rgrid[0])+pdims[0]/2,ew[0]) * u.smooth_step(-np.abs(rgrid[1])+pdims[1]/2,ew[1])
        elif pupil=='circ':
            if pupildims==None:
                pupildims=focusdims
            pdims = u.expect2(pupildims)
            apert=u.smooth_step(-np.sqrt(rgrid[0]**2 / pdims[0]**2 + rgrid[1]**2 / pdims[1]**2) + 0.5,ew[0]/np.sqrt(pdims[0]*pdims[1]))
        else:
            try:
                apert=np.load(pupil)
            except:
                print('Expected path to .npy dexcribing the aperture if not specified as "circ" or "rect"')
                print('Using no pupil... I hope your incoming illumination is structured...')
                apert=np.ones(sh)

        field=apert*w
        
        if defocus!=0.:
            field = Pdefoc.nf(field)                  
    
    if antialiasing>1.0:
        field=u.crop_pad(field,-lines,fillpar=0.0)
        
    return field

