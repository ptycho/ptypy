# -*- coding: utf-8 -*-
"""
This module generates a sample.

@author: Bjoern Enders

This file is part of the PTYPY package.

    :copyright: Copyright 2014 by the PTYPY team, see AUTHORS.
    :license: see LICENSE for details.
"""
import numpy as np

from .. import utils as u
from .. import resources
from ..utils.descriptor import EvalDescriptor

logger = u.verbose.logger

TEMPLATES = dict()

# Local, module-level defaults. These can be appended to the defaults of
# other classes.
sample_desc = EvalDescriptor('sample')
sample_desc.from_string(r"""
    [model]
    default = None
    help = Type of initial object model
    doc = One of:
       - ``None`` : model initialitziation defaults to flat array filled `fill`
       - ``'recon'`` : load model from STXM analysis of diffraction data
       - ``'stxm'`` : Estimate model from autocorrelation of mean diffraction data
       - *<resource>* : one of ptypys internal model resource strings
       - *<template>* : one of the templates in sample module
      In script, you may pass a numpy.array here directly as the model. This array will be
      processed according to `process` in order to *simulate* a sample from e.g. a thickness
      profile.
    type = str, array
    userlevel = 0

    [fill]
    default = 1
    help = Default fill value
    doc = 
    type = float, complex
    userlevel = 

    [recon]
    default = 
    help = Parameters to load from previous reconstruction
    doc = 
    type = Param
    userlevel = 

    [recon.rfile]
    default = \*.ptyr
    help = Path to a ``.ptyr`` compatible file
    doc = 
    type = file
    userlevel = 0

    [stxm]
    default = 
    help = STXM analysis parameters
    doc = 
    type = Param
    userlevel = 1

    [stxm.label]
    default = None
    help = Scan label of diffraction that is to be used for probe estimate
    doc = ``None``, own scan label is used
    type = str
    userlevel = 1

    [process]
    default = None
    help = Model processing parameters
    doc = Can be ``None``, i.e. no processing
    type = Param, None
    userlevel = 

    [process.offset]
    default = (0, 0)
    help = Offset between center of object array and scan pattern
    doc = 
    type = tuple, list
    userlevel = 2
    lowlim = 0

    [process.zoom]
    default = None
    help = Zoom value for object simulation.
    doc = If ``None``, leave the array untouched. Otherwise the modeled or loaded image will be
      resized using :py:func:`zoom`.
    type = list, tuple, float
    userlevel = 2
    lowlim = 0

    [process.formula]
    default = None
    help = Chemical formula
    doc = A Formula compatible with a cxro database query,e.g. ``'Au'`` or ``'NaCl'`` or ``'H2O'``
    type = str
    userlevel = 2

    [process.density]
    default = 1
    help = Density in [g/ccm]
    doc = Only used if `formula` is not None
    type = float
    userlevel = 2

    [process.thickness]
    default = 1.00E-06
    help = Maximum thickness of sample
    doc = If ``None``, the absolute values of loaded source array will be used
    type = float
    userlevel = 2

    [process.ref_index]
    default = (0.5, 0.0)
    help = Assigned refractive index, tuple of format (real, complex)
    doc = If ``None``, treat source array as projection of refractive index a+bj for (a, b). If a refractive index
      is provided the array's absolute value will be used to scale the refractive index.
    type = list, tuple
    userlevel = 2
    lowlim = 0

    [process.smoothing]
    default = 2
    help = Smoothing scale
    doc = Smooth the projection with gaussian kernel of width given by `smoothing_mfs`
    type = int
    userlevel = 2
    lowlim = 0

    [diversity]
    default = 
    help = Probe mode(s) diversity parameters
    doc = Can be ``None`` i.e. no diversity
    type = Param
    userlevel = 

    [diversity.noise]
    default = None
    help = Noise in the generated modes of the illumination
    doc = Can be either:
       - ``None`` : no noise
       - ``2-tuple`` : noise in phase (amplitude (rms), minimum feature size)
       - ``4-tuple`` : noise in phase & modulus (rms, mfs, rms_mod, mfs_mod)
    type = tuple
    userlevel = 1

    [diversity.power]
    default = 0.1
    help = Power of modes relative to main mode (zero-layer)
    doc = 
    type = tuple, float
    userlevel = 1

    [diversity.shift]
    default = None
    help = Lateral shift of modes relative to main mode
    doc = **[not implemented]**
    type = float
    userlevel = 2
    """)

# Strings are also supported as input parameters
sample_desc.options['type'] = 'Param, str'

DEFAULT = sample_desc.make_default(99)

DEFAULT_process = DEFAULT.process


def init_storage(storage, sample_pars=None, energy=None):
    """
    Initializes a storage as sample transmission.

    Parameters
    ----------
    storage : Storage
        The object :any:`Storage` to initialize

    sample_pars : Param
        Parameter structure that defines how the sample is created.
        *FIXME* Link to parameters

    energy : float, optional
        Energy associated in the experiment for this sample object.
        If None, the ptypy structure is searched for the appropriate
        :py:class:`Geo` instance:  ``storage.views[0].pod.geometry.energy``
    """
    s = storage
    prefix = "[Object %s] " % s.ID

    sam = sample_pars
    p = DEFAULT.copy(depth=3)
    model = None
    if hasattr(sam, 'items') or hasattr(sam, 'items'):
        # This is a dict
        p.update(sam, in_place_depth=3)

    # First we check for scripting shortcuts. This is only convenience.
    elif str(sam) == sam:
        # This maybe a template now or a file

        # Deactivate further processing
        p.process = None

        if sam.endswith('.ptyr'):
            recon = u.Param(rfile=sam, layer=None, ID=s.ID)
            p.recon = recon
            p.model = 'recon'
            p.process = None
            init_storage(s, p)
        elif sam in TEMPLATES.keys():
            init_storage(s, TEMPLATES[sam])
        elif sam in resources.objects or sam == 'stxm':
            p.model = sam
            init_storage(s, p)
        else:
            raise RuntimeError(
                prefix +
                'Shortcut string `%s` for object creation is not understood.'
                % sam)
    elif type(sam) is np.ndarray:
        p.model = sam
        p.process = None
        init_storage(s, p)
    else:
        ValueError(prefix + 'Shortcut for object creation is not understood.')

    if p.model is None or str(p.model) == 'sim':
        model = np.ones(s.shape, s.dtype) * p.fill
    elif type(p.model) is np.ndarray:
        model = p.model
    elif p.model in resources.objects:
        model = resources.objects[p.model](s.shape)
    elif str(p.model) == 'recon':
        # Loading from a reconstruction file
        layer = p.recon.get('layer')
        ID = p.recon.get('ID')
        logger.info(prefix +
                    'Attempt to load object storage with ID %s from %s.'
                    % (str(ID), p.recon.rfile))
        model = u.load_from_ptyr(p.recon.rfile, 'obj', ID, layer)
        # This could be more sophisticated,
        # i.e. matching the real space grids etc.

    elif str(p.model) == 'stxm':
        logger.info(prefix + 'STXM initialization using diffraction data.')
        trans, dpc_row, dpc_col = u.stxm_analysis(s)
        model = trans * np.exp(1j * u.phase_from_dpc(dpc_row, dpc_col))
    else:
        raise ValueError(
            prefix + 'Value to `model` key not understood in object creation.')

    assert type(model) is np.ndarray, "".join(
        [prefix, "Internal model should be numpy array now but it is %s."
         % str(type(model))])

    # Expand model to the right length filling with copies
    sh = model.shape[-2:]
    model = np.resize(model, (s.shape[0], sh[0], sh[1]))

    # Try to retrieve the energy
    if energy is None:
        try:
            energy = s.views[0].pod.geometry.energy
        except:
            logger.info(prefix +
                        'Could not retrieve energy from pod network... '
                        'Maybe there are no pods yet created?')
    s._energy = energy

    # Process the given model
    if str(p.model) == 'sim' or p.process is not None:

        # Make this a single call in future
        model = simulate(model, p.process, energy, p.fill, prefix)

    # Symmetrically cut to shape of data
    model = u.crop_pad_symmetric_2d(model, s.shape)[0]

    # Add diversity
    if p.diversity is not None:
        u.diversify(model, **p.diversity)
    # Return back to storage
    s.fill(model)
    # avoids sharp edges on resize
    s.fill_value = model.mean()


def simulate(A, pars, energy, fill=1.0, prefix="", **kwargs):
    """
    Simulates a sample object into model numpy array `A`

    Parameters
    ----------
    A : ndarray
        Numpy array as buffer. Must be at least two-dimensional

    pars : Param
        Simulation parameters. *FIXME* link to paramaters.
    """
    lam = u.keV2m(energy)
    p = DEFAULT_process.copy()
    p.update(pars)
    p.update(kwargs)

    """
    res = p.resource
    if res is None:
        raise RuntimeError(
            "Resource for simulation cannot be None. Please specify one of "
            "ptypy's resources or load numpy array into this key")
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
    # newsh = (A.shape[0], res.shape[-2], res.shape[-1])
    # obj = np.resize(res, newsh)
    obj = A.copy()

    if p.zoom is not None:
        zoom = u.expect3(p.zoom)
        zoom[0] = 1
        obj = u.zoom(obj, zoom)

    if p.smoothing is not None:
        obj = u.gf_2d(obj, p.smoothing / 2.35)

    off = u.expect2(p.offset)
    k = 2 * np.pi / lam
    ri = p.ref_index
    d = p.thickness
    # Check what we got for an array
    if np.iscomplexobj(obj) and d is None:
        logger.info(prefix +
                    "Simulation resource is object transmission")
    elif np.iscomplexobj(obj) and d is not None:
        logger.info(prefix +
                    "Simulation resource is integrated refractive index")
        obj = np.exp(1.j * obj * k * d)
    else:
        logger.info(prefix +
                    "Simulation resource is a thickness profile")
        # Enforce floats
        ob = obj.astype(np.float)
        ob -= ob.min()
        if d is not None:
            logger.info(prefix + "Rescaling to maximum thickness")
            ob /= ob.max() / d

        if p.formula is not None or ri is not None:
            # Use only magnitude of obj and scale to [0 1]
            if ri is None:
                en = u.keV2m(1e-3) / lam
                if u.parallel.master:
                    logger.info(
                        prefix +
                        'Queuing cxro database for refractive index in object '
                        'creation with parameters:\n'
                        'Formula=%s Energy=%d Density=%.2f'
                        % (p.formula, en, p.density))
                    result = np.array(u.cxro_iref(p.formula,
                                                  en,
                                                  density=p.density))
                else:
                    result = None
                result = u.parallel.bcast(result)
                energy, delta, beta = result
                ri = - delta + 1j*beta

            else:
                logger.info(prefix +
                            "Using given refractive index in object creation")

        obj = np.exp(1.j * ob * k * ri)
    # if p.diffuser is not None:
    #    obj *= u.parallel.MPInoise2d(obj.shape, *p.diffuser)

    # Get obj back in original shape and apply a possible offset
    shape = u.expect2(A.shape[-2:])
    crops = list(-np.array(obj.shape[-2:]) + shape + 2*np.abs(off))
    obj = u.crop_pad(obj, crops, fillpar=fill)
    off += np.abs(off)

    return np.array(obj[..., off[0]:off[0]+shape[0], off[1]:off[1]+shape[1]])


DEFAULT_old = u.Param(
    # None, path to a previous recon, or nd-array
    source=None,
    # offset = offset_list(int(par[0]))
    # (offsetx, offsety) move scan pattern relative to center in pixel
    offset=(0, 0),
    # None, scalar or 2-tuple. If None, the pixel is assumed to be right
    # otherwise the image will be resized using ndimage.zoom
    zoom=None,
    # chemical formula (string)
    formula=None,
    # density in [g/ccm] only used if formula is not None
    density=1.0,
    # If None, treat projection as projection of refractive index/
    # If a refractive index is provided the object's absolute value will be
    # used to scale the refractive index.
    ref_index=None,
    # max thickness of sample if None,
    # the absolute values of loaded src array will be used
    thickness=1e-6,
    # Smooth with minimum feature size (in pixel units) if not None
    smoothing_mfs=None,
    # noise applied, relative to 2*pi in phase and relative to 1 in amplitude
    noise_rms=None,
    noise_mfs=None,
    # if object is smaller than the object frame, fill with fill:
    fill=1.0,
    # override
    obj=None,
    mode_diversity='noise',
    # first weight is main mode, last weight will be copied if
    # more modes requested than weight given
    mode_weights=[1., 0.1]
)
