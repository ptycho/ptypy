# -*- coding: utf-8 -*-
"""
Detector module

Note that "dark current" and consequently exposure time is not used.
Modern detectors are insensitive to dark current.
Scale the incoming Intensity accordingly to match the number of photons
in your acquisition.

This file is part of the PTYPY package.

    :copyright: Copyright 2014 by the PTYPY team, see AUTHORS.
    :license: see LICENSE for details.
"""
import numpy as np
from scipy import ndimage as ndi

__all__=['shot','Detector','conv','fill2D']

DEFAULT= dict(
    sci_psf = None,     # (None or float, 2-tuple, array) Parameters for gaussian convolution or convolution kernel after exposure of scintillator
    sci_qe = 1,         # (float) how many optical photons per x-ray photon
    psf = None,         # (None or float, 2-tuple, array) Parameters for gaussian convolution or convolution kernel after exposure
    qe = 1.,            # (float) detector quantum efficiency for converting a photon to a well count
    shot_noise = 0,     # (float) noise rms (counts) for well counts to digital units conversion
    adu = 1,            # (float) count-to-digital units (counts) conversion factor
    full_well = 2**16-1, # (int) per pixel capacity of well counts
    shape = 2048,       # (int,tuple) total detector area of one module (in pixel)
    gaps = 0,           # (int, tuple) gap between adjacent modules (in pixel)
    modules = (1,1),    # (tuple) number of modules for each dimension
    center = 1024,      # (int,tuple) frame center of the exposure within the first module
    dtype = np.uint16,  # (numpy integer dtype) data type for storing (can also be the type char)
    on_limit = 'clip',  #
    psize = 10e-6,      # pixel size (for documentation only)
    #beamstop = None,    # (None,'rect','circ') beamstop
    #bs_size = 0.0,      # (float, tuple) beamstop size in
    #bs_trans = 0.0,     #
    #bs_edgewith = 0.5,
)

TEMPLATES = {}
TEMPLATES['GenericCCD16bit'] = DEFAULT.copy()

TEMPLATES['GenericCCD32bit'] = DEFAULT.copy()
TEMPLATES['GenericCCD32bit']['full_well']= 2**32-1
TEMPLATES['GenericCCD32bit']['dtype']=np.uint32

TEMPLATES['FLI_PL1001'] = dict(
    shape = 1024,
    full_well = 500000,
    shot_noise = 9,
    adu = 10,
    center = 512,
    psize = 24e-6
)

TEMPLATES['FRELON_TAPER'] = dict(
    shape = 2048,
    full_well = 550000,
    shot_noise = 15,
    adu = 9,
    center = 1024,
    psize = 51e-6,
    qe = 0.60,
    sci_qe = 10,   # needed for having ca 1ADU/X-ray
    sci_psf = 1.2, #that one fits with observation
)

TEMPLATES['PILATUS_1M'] = dict(
    shape = (197,486),
    full_well = 2**20,
    shot_noise = 0,
    adu = 1,
    center = None,
    psize = 172e-6,
    dtype = np.uint32,
    gaps = (17,7),
    modules = (5,2),
)

TEMPLATES['PILATUS_2M'] = dict(TEMPLATES['PILATUS_1M'])
TEMPLATES['PILATUS_2M']['modules'] = (8,3)

TEMPLATES['PILATUS_6M'] = dict(TEMPLATES['PILATUS_1M'])
TEMPLATES['PILATUS_6M']['modules'] = (12,5)

TEMPLATES['PILATUS_300K'] = dict(TEMPLATES['PILATUS_1M'])
TEMPLATES['PILATUS_300K']['modules'] = (3,1)


class Detector(object):

    def __init__(self,pars=None):
        self._update(DEFAULT)
        if str(pars)==pars:
            t = TEMPLATES.get(pars,{})
            self._update(t)
        elif pars is not None:
            self._update(pars)

        self.shape = expect2(self.shape)
        self._make_mask()
        if self.center is None:
            self.center = expect2(self._mask.shape)//2

    def _update(self,pars=None):
        if pars is not None:
            self.__dict__.update(pars)

    def _make_mask(self):
        gaps = expect2(self.gaps)
        module = np.ones(self.shape).astype(np.bool)
        start = module.copy()
        for i in range(self.modules[0]-1):
            gap = np.zeros((gaps[0],module.shape[1])).astype(np.bool)
            start = np.concatenate([start,np.concatenate([gap,module],axis=0)],axis=0)
        module = start.copy()
        for i in range(self.modules[1]-1):
            gap = np.zeros((module.shape[0],gaps[1])).astype(np.bool)
            start = np.concatenate([start,np.concatenate([gap,module],axis=1)],axis=1)
        self._mask = start

    def _get_mask(self,sh):
        msh =  expect2(sh[-2:])
        mask = np.zeros(msh).astype(np.bool)
        offset = msh//2 - expect2(self.center)
        mask = fill2D(mask,self._mask,-offset)
        return np.resize(mask,sh)

    def filter(self,intensity_stack,convert_dtype=False):
        I= intensity_stack
        I_dtype = I.dtype if not convert_dtype else self.dtype

        mask = self._get_mask(I.shape)

        I = np.abs(np.array(I).astype(float))
        if self.sci_psf is not None:
            I = self.sci_qe*conv(np.random.poisson(I).astype(float),self.sci_psf)
        if self.psf is not None:
            I = conv(I,self.psf)

        # convert to well counts
        Iel = np.random.poisson(I* self.qe).astype(float)
        # add shot noise
        Iel += np.abs(np.random.standard_normal(I.shape)*self.shot_noise)
        overexposed = Iel>=self.full_well
        Iel[overexposed]=self.full_well
        Iel[Iel<0.]=0.
        DU = Iel // self.adu
        dt = self.dtype
        if self.on_limit=='clip':
            mx = np.iinfo(dt).max
            DU[DU>mx]=mx

        DU[np.invert(mask)]=0.0
        mask &= np.invert(overexposed)

        return DU.astype(I_dtype), mask

def conv(A,inp,**kwargs):
    dims = A.ndim
    assert dims in [2,3], "Filtered array has to be 2D or 3D."
    if inp is None:
        return A
    elif np.size(inp)<=2:
        inp = expect2(inp)
        if dims==3:
            inp=[0,inp[0],inp[1]]
        return ndi.gaussian_filter(A,inp,**kwargs)
    else:
        inp = np.array(inp)
        assert inp.ndim == 2, "Convolution kernel must be 2D"
        if dims ==3:
            inp = inp.reshape((1,inp.shape[0],inp.shape[1]))
        return ndi.convolve(A,inp,**kwargs)

def shot(I,exp=0.1,flux=1e5,sensitivity=1.0,dark_c=None,io_noise=0.,full_well=2**10-1,el_per_ADU=1.0,offset=50.):
    """\
    I : intensity distribution
    flux : overall photon photons per seconds coming in
    exp : exposition time in sec
    io_noise : readout noise rms
    full_well : electron capacity of each pixel
    el_per_ADU : conversion effficiency of electrons to digitally counted units
    dark_curr : electrons per second per pixel on average
    """

    I=np.asarray(I).astype(float)

    if I.sum() != 0. :
        I=I/I.sum()

    photo_el = np.floor(sensitivity*np.random.poisson(exp*flux*I))
    if dark_c is not None:
        therm_el = np.floor(np.random.poisson(dark_c*exp*np.ones_like(I)))
    else:
        therm_el = np.zeros_like(I)

    el = photo_el + therm_el
    el[el > full_well] = full_well
    out = offset + io_noise*np.random.standard_normal(I.shape)+ el / el_per_ADU
    out[out<0.]=0.
    return out.astype(int)

def fill2D(imA,imB,offset):
    """
    fill array imA with imB
    """
    shA= expect2(imA.shape[-2:])
    offset=expect2(offset)
    shB=expect2(imB.shape[-2:])
    #print shA,shB,offset
    def vmin(A):
        return np.min(np.vstack(A),axis=0)
    def vmax(A):
        return np.max(np.vstack(A),axis=0)
    minA=vmax([expect2(0),vmin([-offset,shA])])
    maxA=vmin([shA,vmax([shB-offset,expect2(0)])])
    minB=vmax([expect2(0),vmin([+offset,shB])])
    maxB=vmin([shB,vmax([shA+offset,expect2(0)])])
    #print minA,maxA,minB,maxB
    imA[...,minA[0]:maxA[0],minA[1]:maxA[1]] = imB[...,minB[0]:maxB[0],minB[1]:maxB[1]]
    return imA

def expect2(a):
    """\
    generates 1d numpy array with 2 entries generated from multiple inputs
    (tuples, arrays, scalars). main puprose of this function is to circumvent
    debugging of input.

    expect2( 3.0 ) -> np.array([3.0,3.0])
    expect2( (3.0,4.0) ) -> np.array([3.0,4.0])

    even higher order inputs possible, though not tested much
    """
    a=np.atleast_1d(a)
    if len(a)==1:
        b=np.array([a.flat[0],a.flat[0]])
    else: #len(psize)!=2:
        b=np.array([a.flat[0],a.flat[1]])
    return b

def smooth_step(x,mfs):
    return 0.5*erf(x*2.35/mfs) +0.5

if __name__ == "__main__":
    import sys
    import numpy as np
    from scipy.misc import lena
    from scipy.ndimage import gaussian_filter as gf
    from matplotlib import pyplot as plt
    from scipy.special import erf
    if len(sys.argv) > 1:
        det = sys.argv[1]
    else:
        det = None
    if len(sys.argv) > 1:
        phot = float(sys.argv[2])
    else:
        phot = 1e7
    D=Detector(det)
    l= lena()
    size=1024
    x,y=np.indices((size,size))-512
    A = smooth_step(x**2+y**2-40**2,400)
    I=np.abs(np.fft.fftshift(np.fft.fft2(A.astype(np.complex128)))).astype(float)
    I/=I.sum()
    I*= phot
    I=np.resize(I,(2,)+I.shape)
    plt.ion()
    plt.figure()
    plt.imshow(A)
    plt.figure()

    plt.imshow(np.log10(D.expose(I[1])[0]+1))
