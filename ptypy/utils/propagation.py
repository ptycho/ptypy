# -*- coding: utf-8 -*-
"""
propagation module.

@author: Bjoern Enders

This file is part of the PTYPY package.

    :copyright: Copyright 2014 by the PTYPY team, see AUTHORS.
    :license: GPLv2, see LICENSE for details.
"""
import numpy as np

from misc import *
from array_utils import *
from parameters import *
from utils_BE import *
from .. import *
#import ..utils as u

def from_geometry(geometry,shift=False):
    """\
    The idea of this function is to provide propagators alongside with
    the modes. Each mode can have its own propagator. also if probe array
    sizes change, this can be dealt with by casting new propagators.
    """

    geo=Param(geometry)
    sh = geo.N
    pxsz = [geo.detector_psize,None]
    if geo.prop_type=='nearfield':
        org=['fftshift', 'fftshift']
    elif shift:
        org=['fftshift', shift/sh]
    else:
        org=['fftshift','fftshift']
    
    P=Propagator(sh,pxsz,geo.lam,geo.z,org=org,prop_type=geo.prop_type)
   
    return P


class Propagator:

    def __init__(self,sh,psize,l=0.1, z=1e2, ffttype='std',org = ['fftshift','fftshift'], prop_type='farfield'):
        self.z = z
        self.l = l
        self.sh = np.asarray(sh)
        self.org = org
        self.prop_type=prop_type
        self.psize = [None]*len(sh)
        self._update_psize(psize)       
        self._assign_fft(ffttype)
        
    def _update_all(self):
        self._update_grids()
        self._update_fft_factors()
        
    def _update_grids(self):
        self.grids = self.get_grids()
        #self.grid_src = self.get_grid_src()
   
    def _update_fft_factors(self):
        #if propagation paramaters change a lot better compute differently
    
        self.kernel = self.get_nf_kernel()
        self.ikernel = self.kernel.conj()
    
        self.pre_fft = self.get_pre_fft_factor()
        self.post_fft = self.get_post_fft_factor()
        self.pre_ifft = self.get_pre_ifft_factor()
        self.post_ifft = self.get_post_ifft_factor()
                
    def ff(self,w):
        return self.sc * self.post_fft * self.fft(self.pre_fft * w)
    
    def iff(self,w):
        return self.isc * self.post_ifft * self.ifft(self.pre_ifft * w)
    
    def nf(self,w):
        return self.ifft(self.fft(w) * np.fft.fftshift(self.kernel))
    
    def inf(self,w):
        return self.ifft(self.fft(w) * np.fft.fftshift(self.ikernel))
        
    def props(self):
        if self.prop_type=='nearfield':
            return self.nf,self.inf
        else:
            return self.ff,self.iff
            
    def fw(self,w):
        if self.prop_type=='nearfield':
            return self.nf(w)
        else:
            return self.ff(w)
    
    def bw(self,w):
        if self.prop_type=='nearfield':
            return self.inf(w)
        else:
            return self.iff(w)
    
#    def nf_8thr(self,w):
#        w2=w.copy()
#        prop.propagate_wave_angular_spectrum(w2,self.z,2*np.pi/self.l,self.psize[0][0],self.psize[0][0])
#        return w2
#    
#    def inf_8thr(self,w):
#        w2=w.copy()
#        prop.propagate_wave_angular_spectrum(w2,-self.z,2*np.pi/self.l,self.psize[0][0],self.psize[0][0])
#        return w2
          
    def get_grids(self):
        return grids(self.sh,self.psize[0],self.org[0]),grids(self.sh,self.psize[1],self.org[1]) 

    def _update_psize(self,psize):
        #print psize
        if psize[0] is not None:
            if psize[1] is not None:
                raise RuntimeError(" a or b of psize=[a,b] has to be None-type")
            else:
                self._update_psize_rplane(psize[0])
        else:
            if psize[1] is None:
                raise RuntimeError(" a and b of psize=[a,b] cannot both be None-type")
            else:
                self._update_psize_fplane(psize[1])
                     
    def _update_psize_rplane(self,new_psize):
        new_psize = np.array(new_psize)
        if new_psize.size == 1:
            new_psize = new_psize *np.ones((len(self.sh),))
        #print new_psize    
        self.psize[0] = new_psize
        self.psize[1] = self.get_ffpsize(self.psize[0])
        self._update_all()
   
    def _update_psize_fplane(self,new_psize):
        new_psize = np.array(new_psize)
        if new_psize.size == 1:
            new_psize = new_psize *np.ones((len(self.sh),))          
        self.psize[1] = new_psize
        self.psize[0] = self.get_ffpsize(self.psize[1])
        self._update_all()
          
    def get_ffpsize(self,psize):
        return self.l*np.abs(self.z)/self.sh/ np.asarray(psize)
        
    def get_nf_kernel(self):
        [V,W] = self.grids[1]
        a2 = (V**2+W**2) /self.z**2          
        return np.exp(2j * np.pi * (self.z / self.l) * (np.sqrt(1 - a2) - 1))

    def get_pre_fft_factor(self):
        [X,Y],[V,W] = self.grids
        fac=self.l * self.z
        pre=np.exp(1j * np.pi * (X**2+Y**2) / fac )
        pre=pre*np.exp(-2.0*np.pi*1j*((X-X[0,0])*V[0,0]+(Y-Y[0,0])*W[0,0])/ fac)     
        return pre
        
    def get_post_fft_factor(self):
        [X,Y],[V,W] = self.grids
        fac=self.l * self.z
        post=np.exp(1j * np.pi * (V**2+W**2) / fac )
        post=post*np.exp(-2.0*np.pi*1j*(X[0,0]*V+Y[0,0]*W)/ fac)           
        return post
        
    def get_pre_ifft_factor(self):
        return self.get_post_fft_factor().conj()
        
    def get_post_ifft_factor(self):
        return self.get_pre_fft_factor().conj()
        
    def _assign_fft(self,ffttype='std'):
        self.sc = 1./np.sqrt(np.prod(self.sh))
        self.isc = np.sqrt(np.prod(self.sh))
        if ffttype!='std':
            self.fft = ffttype[0]
            self.ifft = ffttype[1]
            if len(ffttype) > 2:
                self.sc = ffttype[2]
                self.isc = ffttype[3]
        else:
            self.fft = np.fft.fft2
            self.ifft = np.fft.ifft2
         

def prop_to_nf(w, l, z, pixelsize=1.,grid='fftshift'):
    """\
    Free-space propagation (near field) of the wavefield of a distance z.
    l is the wavelength.

    For grid='fft' the object should be centered at the top left corner.
    For grid='subpixel' the object should be in the geometric center of the matrix.
    For grid='fftshift' the object should be centered at the fftshift origin.
    """    
    if w.ndim != 2:
        raise RunTimeError("A 2-dimensional wave front 'w' was expected")

    psize = expect2(pixelsize)
    sh = np.asarray(w.shape)

    # Evaluate if aliasing could be a problem
    if min(sh)/np.sqrt(2.) < z*l:
        print "Warning: z > N/(sqrt(2)*lamda) = %.6g: this calculation could fail." % (min(sh)/(l*np.sqrt(2.)))
        print "(consider padding your array, or try a far field method)"
    
    if grid=='subpixel':
        anglex,angley=cen_grid(sh,l/(sh*psize))
        a2=anglex**2+angley**2
        kernel=np.exp(2j * np.pi * (z / l) * (np.sqrt(1 - a2) - 1))
        out = cifftn(cfftn(w) * kernel)
    elif grid=='fft':
        a2=U.fvec2(sh,l/(sh*psize))
        kernel=np.exp(2j * np.pi * (z / l) * (np.sqrt(1 - a2) - 1))
        out = np.fft.ifftn(np.fft.fftn(w) * kernel)
    else:
        a2=U.fvec2(sh,l/(sh*psize))
        kernel=np.exp(2j * np.pi * (z / l) * (np.sqrt(1 - a2) - 1))
        out = np.fft.fftshift(np.fft.ifftn(np.fft.fftn(np.fft.fftshift(w)) * kernel))
    
    return out,kernel

def pixelsize_from_grids(grids):
    #only works 2d
    return np.array([grids[0][1,1]-grids[0][0,0],grids[1][1,1]-grids[1][0,0]])

def wave_prop_ff(w, l, z, grid_src, org_tar='fftshift',**kwargs):
    return farfield(w,l=l,z=z,grid_src=grid_src,org_tar=org_tar,returnmode='wave',**kwargs)

def farfield(w, psize=1., lz=1e1, org_src='fftshift', org_tar='fftshift', returnmode='pre+post', fft_type='std', pre=None, post=None, grid_src=None, grid_tar=None):
    """\
    z<0 is inverse transform.
    Free-space propagation (far field) of the wavefield  w.
    lz is the product of wavelength & propagation distance.

    With newpsize=True the function returns the pixel size of the propagated wave 
    psize'=lz/(sh*psize) 
    with psize=[dx,dy] being the pixel size of the original wave w.

    For grid='fft' the object should be centered at the top left corner.
    For grid='subpixel' the object should be in the geometric center of the matrix.
    For grid='fftshift' the object should be centered at the fftshift origin.
    
    
    """
    scale=np.sqrt(np.prod(w.shape))
    if fft_type!='std':
        fft=fft_type
        sc=1.0
    elif lz>0:
        sc=1./scale
        fft=np.fft.fftn
    elif lz<0:
        sc=scale
        fft=np.fft.ifftn
    #print z,p
    if pre!=None and post!=None:
        if (w.ndim != 2) and (pre.ndim != 2) and (post.ndim != 2):
            raise RunTimeError("2-dimensional arrays for 'w', 'pre' and 'post' were expected")
        else:
            if returnmode=='all' or returnmode=='pre+post':
                return post*fft(pre*w), pre,post
            else:
                return post*fft(pre*w)

    elif pre==None or post==None:
        sh=np.array(w.shape)
        psize = expect2(psize)
        if grid_src==None:
            #print "making grid_src"
            grid_src=grids(sh,psize,org_src)
        else:
            psize=pixelsize_from_grids(grid_src)    
        if grid_tar==None:     
            psize_neu=np.abs(lz)/sh/psize
            grid_tar=grids(sh,psize_neu,org_tar)
        else:
            psize_neu=pixelsize_from_grids(grid_tar)
          
        [V,W]=grid_tar;
        [X,Y]=grid_src;
        pre=np.exp(1j * np.pi * (X**2+Y**2) / (lz) )
        pre=pre*np.exp(-2.0*np.pi*1j*((X-X[0,0])*V[0,0]+(Y-Y[0,0])*W[0,0])/ (lz))
        post=sc*np.exp(1j * np.pi * (V**2+W**2) / (lz) )
        post=post*np.exp(-2.0*np.pi*1j*(X[0,0]*V+Y[0,0]*W)/ (lz))
       
    if returnmode=='all':
        return post*fft(pre*w), pre,post,grid_src,grid_tar,psize_neu
    elif returnmode=='pre+post':
        return post*fft(pre*w), pre,post
    elif returnmode=='ppp':
        return post*fft(pre*w), pre,post, psize_neu
    elif returnmode=='wave':
        return post*fft(pre*w), grid_tar, psize_neu
    else:
        return post*fft(pre*w)

def prop_to_ff_vargrids(w,l,z,mag=1.,psize=1.,grid='fftshift',backwards=True):
    """\
    propagates with variable grid spacing the distance z
    see NSOWP page 93 ff
    """
    if backwards:
        z1=z /(1-mag)
    else:
        z1=z /(1+mag)
    
    z2=z-z1
    wt,psizet=prop_to_ff(w,l,z1,psize,grid)
    return prop_to_ff(wt,l,z2,psizet,grid)    

def fprop_pink(w, psize,lz_central,lines,
        fillpar=0.0, filltype='project',
        org_src='fftshift',org_tar='fftshift',
        **kwargs):
    """        
    if org_src=='fft':
        w=np.fft.fftshift(w)
    elif org_src!='fftshift' and org_src!='geometric':
        raise RunTimeError('currently only a limited number of grids supported')
    if org_tar=='fft':
        Swap=True
        org_tar='fftshift'
    else:
        Swap=False
    """
    lines=2*lines
    lz = (w.shape[0]+lines)*(lz_central/w.shape[0])    #That is incompatible to asymmetric grids
    x = U.crop_pad(w,[lines,lines],cen=org_src,fillpar=fillpar,filltype=filltype)
    f,pre,post,p = farfield(x,psize,lz, org_src=org_src, org_tar=org_tar, returnmode='ppp')
    out = U.crop_pad(f,[-lines,-lines],cen=org_tar,fillpar=fillpar,filltype=filltype)        
    mask = U.crop_pad(np.ones_like(f),[-lines,-lines],cen=org_tar,fillpar=0.0,filltype='scalar')
    """
    if Swap:
        out=np.fft.fftshift(out)
        mask=np.fft.fftshift(mask)
    """
    return out,pre,post,mask,p,lz
    
