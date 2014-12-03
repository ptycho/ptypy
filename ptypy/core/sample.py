# -*- coding: utf-8 -*-
"""
This module generates a sample

@author: Bjoern Enders

This file is part of the PTYPY package.

    :copyright: Copyright 2014 by the PTYPY team, see AUTHORS.
    :license: GPLv2, see LICENSE for details.
"""
import numpy as np
import os
from matplotlib import pyplot as plt

from .. import utils as u
from ..utils import prop 
from ..utils.verbose import logger

DEFAULT=u.Param(
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

def object_loader(source,args):
    """
    place holder  for file loading
    """
    pass


def from_pars(shape,lam,pars=None,dtype=np.complex):
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
            obj = u.c_zoom(obj,p.zoom)
            
        if p.smoothing_mfs is not None:
            obj = u.c_gf(obj,p.smoothing_mfs / 2.35)
        
        k = 2 * np.pi / lam
        ri = p.ref_index
        if p.formula is not None or ri is not None:
            # use only magnitude of obj and scale to [0 1]
            if ri is None:
                en = u.keV2m(1e-3)/lam
                logger.info("Quering cxro database for refractive index \
                 in object creation with paramters:\n Formula=%s Energy=%d Density=%.2f" % (p.formula,en,p.density))
                energy, delta,beta = iofr(p.formula,en,density=p.density)
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
                noise=u.c_gf(noise,p.noise_mfs / 2.35)
            obj*=noise
            

            
        off +=np.abs(off)
        p.obj = obj[off[0]:off[0]+shape[0],off[1]:off[1]+shape[1]]
        
        return p


def create_modes(layers,pars):
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


cxro_server = 'http://henke.lbl.gov'

cxro_POST_query = ('Material=Enter+Formula' +
             '&Formula=%(formula)s&Density=%(density)s&Scan=Energy' +
             '&Min=%(emin)s&Max=%(emax)s&Npts=%(npts)s&Output=Text+File')

def iofr(formula, energy,density=-1, npts=100):
    """\
    Query CXRO database for index of refraction values.

    Parameters:
    ----------
    formula: str
        String representation of the Formula to use.
    energy: float or (float,float)
        Either a single energy (in keV) or the minimum/maximum bounds
    npts: int [optional]
        Number of points between the min and max energies. 

    Returns:
        (energy, delta, beta), either scalars or vectors.
    """
    import urllib
    import urllib2
    import numpy as np
    
    if np.isscalar(energy):
        emin = energy
        emax = energy
        npts = 1
    else:
        emin,emax = energy

    data = cxro_POST_query % {'formula':formula,
                     'emin':emin,
                     'emax':emax,
                     'npts':npts,
                     'density':density}

    url = cxro_server+'/cgi-bin/getdb.pl'
    #u.logger.info('Querying CRXO database...')
    req = urllib2.Request(url, data)
    response = urllib2.urlopen(req)
    t = response.read()
    datafile = t[t.find('/tmp/'):].split('"')[0]

    url = cxro_server + datafile
    req = urllib2.Request(url)
    response = urllib2.urlopen(req)
    data = response.read()

    d = data.split('\n')
    #print d
    dt = np.array([[float(x) for x in dd.split()] for dd in d[2:] if dd])

    #u.logger.info('done, retrieved: ' +  d[0].strip())
    #print d[0].strip()
    if npts==1:
        return dt[-1,0], dt[-1,1], dt[-1,2]
    else:
        return dt[:,0], dt[:,1], dt[:,2]
