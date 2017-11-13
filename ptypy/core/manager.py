# -*- coding: utf-8 -*-
"""
Scan management.

The main task of this module is to prepare the data structure for
reconstruction, taking a data feed and connecting individual diffraction
measurements to the other containers. The way this connection is done
is defined by ScanModel and its subclasses. The connections are
described by the POD objects.

This file is part of the PTYPY package.

    :copyright: Copyright 2014 by the PTYPY team, see AUTHORS.
    :license: GPLv2, see LICENSE for details.
"""
import numpy as np
import illumination
import sample
import geometry
import xy
import data

from collections import OrderedDict
from .. import utils as u
from ..utils.verbose import logger, headerline, log
from classes import *
from classes import DEFAULT_ACCESSRULE
from classes import MODEL_PREFIX
from ..utils import parallel
from ..utils.descriptor import defaults_tree

# Please set these globally later
FType = np.float64
CType = np.complex128

__all__ = ['ModelManager', 'ScanModel', 'Full', 'Vanilla', 'Bragg3dModel']


@defaults_tree.parse_doc('scan.ScanModel')
class ScanModel(object):
    """
    Abstract base class for models. Override at least these methods:
    _create_pods(self)
    _initialize_geo(self, common)
    _initialize_probe(self, probe_ids)
    _initialize_object(self, object_ids)

    Defaults:

    [tags]
    default = ['dummy']
    help = Comma seperated string tags describing the data input
    doc = [deprecated?]
    type = list
    userlevel = 2

    [propagation]
    type = str
    default = farfield
    help = Propagation type
    doc = Either "farfield" or "nearfield"
    userlevel = 1

    [data]
    default =
    type = @scandata.*
    help = Link to container for data preparation
    doc =

    [data.name]
    default =
    type = str
    help = Name of the PtyScan subclass to use

    [illumination]
    type = Param
    default =
    help = Container for probe initialization model

    [sample]
    type = Param
    default =
    help = Container for sample initialization model

    """
    def __init__(self, ptycho=None, pars=None, label=None):
        """
        Create scan model object.

        Parameters
        ----------
        pars : dict or Param
            Input parameter tree.

        ptycho : Ptycho instance
            Ptycho instance to which this scan belongs

        label : str
            Unique label
        """
        # Update parameter structure
        # Load default parameter structure
        p = self.DEFAULT.copy(99)
        p.update(pars, in_place_depth=4)
        self.p = p
        self.label = label
        self.ptycho = ptycho

        # Create Associated PtyScan object
        self.ptyscan = self.makePtyScan(self.p.data)

        # Initialize instance attributes
        self.mask = None
        self.diff = None
        self.positions = []
        self.mask_views = []
        self.diff_views = []
        self.new_positions = None
        self.new_diff_views = None
        self.new_mask_views = None

        self.geometries = []
        self.shape = None
        self.psize = None

        # Object flags and constants
        self.containers_initialized = False
        self.data_available = True
        self.CType = CType
        self.FType = FType
        self.frames_per_call = 100000

    @classmethod
    def makePtyScan(cls, pars):
        """
        Factory for PtyScan object. Return an instance of the appropriate PtyScan subclass based on the
        input parameters.

        Parameters
        ----------
        pars: dict or Param
            Input parameters according to :py:data:`.scan.data`.
        """

        # Extract information on the type of object to build
        name = pars.name

        from .. import experiment

        if name in u.all_subclasses(data.PtyScan, names=True):
            ps_class = eval('experiment.' + name)
            logger.info('Scan will be prepared with the PtyScan subclass "%s"' % name)
            ps_instance = ps_class(pars)
        else:
            raise RuntimeError('Could not manage source "%s"' % str(name))

        return ps_instance

    def _extra_analysis(self):
        return True

    def new_data(self):
        """
        Feed data from ptyscan object.
        :return: None if no data is available, True otherwise.
        """

        # Initialize if that has not been done yet
        if not self.ptyscan.is_initialized:
            self.ptyscan.initialize()

        # Get data
        dp = self.ptyscan.auto(self.frames_per_call)

        self.data_available = (dp != data.EOS)
        logger.debug(u.verbose.report(dp))

        if dp == data.WAIT or not self.data_available:
            return None

        label = self.label
        logger.info('Importing data from scan %s.' % label)

        # Prepare the scan geometry if not already done.
        if not self.geometries:
            self._initialize_geo(dp['common'])

        # Create containers if not already done
        if not self.containers_initialized:
            self._initialize_containers()

        # Generalized shape which works for 2d and 3d cases
        sh = (1,) + tuple(self.shape)

        # Storage generation if not already existing
        if self.diff is None:
            # This scan is brand new so we create storages for it
            self.diff = self.Cdiff.new_storage(shape=sh, psize=self.psize, padonly=True,
                                                     layermap=None)
            old_diff_views = []
            old_diff_layers = []
        else:
            # ok storage exists already. Views most likely also. We store them so we can update their status later.
            old_diff_views = self.Cdiff.views_in_storage(self.diff, active_only=False)
            old_diff_layers = []
            for v in old_diff_views:
                old_diff_layers.append(v.layer)

        # Same for mask
        if self.mask is None:
            self.mask = self.Cmask.new_storage(shape=sh, psize=self.psize, padonly=True,
                                                     layermap=None)
            old_mask_views = []
            old_mask_layers = []
        else:
            old_mask_views = self.Cmask.views_in_storage(self.mask, active_only=False)
            old_mask_layers = []
            for v in old_mask_views:
                old_mask_layers.append(v.layer)

        # this is a hack for now
        dp = self._new_data_extra_analysis(dp)
        if dp is None:
            return None

        # Prepare for View generation
        AR_diff_base = DEFAULT_ACCESSRULE.copy()
        AR_diff_base.shape = self.shape
        AR_diff_base.coord = 0.0
        AR_diff_base.psize = self.psize
        AR_mask_base = AR_diff_base.copy()
        AR_diff_base.storageID = self.diff.ID
        AR_mask_base.storageID = self.mask.ID

        diff_views = []
        mask_views = []
        positions = []

        # First pass: create or update views and reformat corresponding storage
        for dct in dp['iterable']:

            index = dct['index']
            active = dct['data'] is not None

            pos = dct.get('position')

            if pos is None:
                logger.warning('No position set to scan point %d of scan %s' % (index, label))

            AR_diff = AR_diff_base
            AR_mask = AR_mask_base
            AR_diff.layer = index
            AR_mask.layer = index
            AR_diff.active = active
            AR_mask.active = active

            # check here: is there already a view to this layer? Is it active?
            try:
                old_view = old_diff_views[old_diff_layers.index(index)]
                old_active = old_view.active
                old_view.active = active

                logger.debug(
                    'Diff view with layer/index %s of scan %s exists. \nSetting view active state from %s to %s' % (
                        index, label, old_active, active))
            except ValueError:
                v = View(self.Cdiff, accessrule=AR_diff)
                diff_views.append(v)
                logger.debug(
                    'Diff view with layer/index %s of scan %s does not exist. \nCreating view with ID %s and set active state to %s' % (
                        index, label, v.ID, active))
                # append position also
                positions.append(pos)

            try:
                old_view = old_mask_views[old_mask_layers.index(index)]
                old_view.active = active
            except ValueError:
                v = View(self.Cmask, accessrule=AR_mask)
                mask_views.append(v)

        # so now we should have the right views to this storages. Let them reformat()
        # that will create the right sizes and the datalist access
        self.diff.reformat()
        self.mask.reformat()

        # Second pass: copy the data
        for dct in dp['iterable']:
            parallel.barrier()
            if dct['data'] is None:
                continue
            diff_data = dct['data']
            idx = dct['index']

            # FIXME: Find a more transparent way than this.
            self.diff.data[self.diff.layermap.index(idx)][:] = diff_data
            self.mask.data[self.mask.layermap.index(idx)][:] = dct.get('mask', np.ones_like(diff_data))

        self.diff.nlayers = parallel.MPImax(self.diff.layermap) + 1
        self.mask.nlayers = parallel.MPImax(self.mask.layermap) + 1

        self.new_positions = positions
        self.new_diff_views = diff_views
        self.new_mask_views = mask_views
        self.positions += positions
        self.diff_views += diff_views
        self.mask_views += mask_views

        self._update_stats()

        # Create new views on object, probe, and exit wave, and connect
        # these through new pods.
        new_pods, new_probe_ids, new_object_ids = self._create_pods()
        for pod_ in new_pods:
            if pod_.model is not None:
                continue
            pod_.model = self
        logger.info('Process %d created %d new PODs, %d new probes and %d new objects.' % (
            parallel.rank, len(new_pods), len(new_probe_ids), len(new_object_ids)), extra={'allprocesses': True})

        # Adjust storages
        self.ptycho.probe.reformat(True)
        self.ptycho.obj.reformat(True)
        self.ptycho.exit.reformat()

        self._initialize_probe(new_probe_ids)
        self._initialize_object(new_object_ids)
        self._initialize_exit(new_pods)

        return True

    def _initialize_containers(self):
        """
        Initialize containers appropriate for the model. This 
        implementation works for 2d models, override if necessary.
        """
        if self.ptycho is None:
            # Stand-alone use
            self.Cdiff = Container(ptycho=self, ID='Cdiff', data_type='real')
            self.Cmask = Container(ptycho=self, ID='Cmask', data_type='bool')
        else:
            # Use with a Ptycho instance
            self.ptycho.probe = Container(ptycho=self.ptycho, ID='Cprobe', data_type='complex')
            self.ptycho.obj = Container(ptycho=self.ptycho, ID='Cobj', data_type='complex')
            self.ptycho.exit = Container(ptycho=self.ptycho, ID='Cexit', data_type='complex')
            self.ptycho.diff = Container(ptycho=self.ptycho, ID='Cdiff', data_type='real')
            self.ptycho.mask = Container(ptycho=self.ptycho, ID='Cmask', data_type='bool')
            self.Cdiff = self.ptycho.diff
            self.Cmask = self.ptycho.mask
        self.containers_initialized = True

    @staticmethod
    def _initialize_exit(pods):
        """
        Initializes exit waves using the pods.
        """
        logger.info('\n' + headerline('Creating exit waves', 'l'))
        for pod in pods:
            if not pod.active:
                continue
            pod.exit = pod.probe * pod.object

    def _update_stats(self):
        """
        (Re)compute the statistics for the data stored in the scan.
        These statistics are:
         * Itotal: The integrated power per frame
         * max/min/mean_frame: pixel-by-pixel maximum, minimum and
           average among all frames.
        """
        mask_views = self.mask_views
        diff_views = self.diff_views

        # Nothing to do if no view exist
        if not self.diff: return

        # Reinitialize containers
        Itotal = []
        max_frame = np.zeros(self.diff_views[0].shape)
        min_frame = np.zeros_like(max_frame)
        mean_frame = np.zeros_like(max_frame)
        norm = np.zeros_like(max_frame)

        for maview, diview in zip(mask_views, diff_views):
            if not diview.active:
                continue
            dv = diview.data
            m = maview.data
            v = m * dv
            Itotal.append(np.sum(v))
            max_frame[max_frame < v] = v[max_frame < v]
            min_frame[min_frame > v] = v[min_frame > v]
            mean_frame += v
            norm += m

        parallel.allreduce(mean_frame)
        parallel.allreduce(norm)
        parallel.allreduce(max_frame, parallel.MPI.MAX)
        parallel.allreduce(max_frame, parallel.MPI.MIN)
        mean_frame /= (norm + (norm == 0))

        self.diff.norm = norm
        self.diff.max_power = parallel.MPImax(Itotal)
        self.diff.tot_power = parallel.MPIsum(Itotal)
        self.diff.pbound_stub = self.diff.max_power / mean_frame.shape[-1]**2
        self.diff.mean = mean_frame
        self.diff.max = max_frame
        self.diff.min = min_frame

        info = {'label': self.label, 'max': self.diff.max_power, 'tot': self.diff.tot_power, 'mean': mean_frame.sum()}
        logger.info(
            '\n--- Scan %(label)s photon report ---\nTotal photons   : %(tot).2e \nAverage photons : %(mean).2e\nMaximum photons : %(max).2e\n' % info + '-' * 29)

    def _create_pods(self):
        """
        Create all new pods as specified in the new_positions,
        new_diff_views and new_mask_views object attributes. Also create
        all necessary views on object, probe, and exit wave.

        Return the list of new pods, and dicts of new probe and object
        ids (to allow for initialization).
        """
        raise NotImplementedError

    def _initialize_geo(self, common):
        """
        Initialize the geometry/geometries based on input data package
        Parameters
        ----------
        common: dict
                metadata part of the data package passed into new_data.

        """
        raise NotImplementedError

    def _initialize_probe(self, probe_ids):
        raise NotImplementedError

    def _initialize_object(self, object_ids):
        raise NotImplementedError


@defaults_tree.parse_doc('scan.Vanilla')
class Vanilla(ScanModel):
    """
    Dummy for testing, there must be more than one for validate to react
    to invalid names.

    Defaults:

    [name]
    default = Vanilla
    type = str
    help =

    [illumination.size]
    default = None
    type = float
    help = Initial probe size
    doc = The probe is initialized as a flat circle.

    [sample.fill]
    default = 1
    type = float, complex
    help = Initial sample value
    doc = The sample is initialized with this value everywhere.

    """

    def _create_pods(self):
        """
        Create all new pods as specified in the new_positions,
        new_diff_views and new_mask_views object attributes.
        """
        logger.info('\n' + headerline('Creating PODS', 'l'))
        new_pods = []
        new_probe_ids = {}
        new_object_ids = {}

        # We can just decide what the storage ID:s will be
        ID ='S00G00'

        # We need to return info on what storages are created
        if not ID in self.ptycho.probe.storages.keys():
            new_probe_ids[ID] = True
        if not ID in self.ptycho.obj.storages.keys():
            new_object_ids[ID] = True

        geometry = self.geometries[0]

        # Loop through diffraction patterns
        for i in range(len(self.new_diff_views)):
            dv, mv = self.new_diff_views.pop(0), self.new_mask_views.pop(0)

            # Create views
            ndim = self.Cdiff.ndim
            pv = View(container=self.ptycho.probe,
                      accessrule={'shape': geometry.shape,
                                  'psize': geometry.resolution,
                                  'coord': u.expectN(0.0, ndim),
                                  'storageID': ID,
                                  'layer': 0,
                                  'active': True})

            ov = View(container=self.ptycho.obj,
                      accessrule={'shape': geometry.shape,
                                  'psize': geometry.resolution,
                                  'coord': self.new_positions[i],
                                  'storageID': ID,
                                  'layer': 0,
                                  'active': True})

            ev = View(container=self.ptycho.exit,
                      accessrule={'shape': geometry.shape,
                                  'psize': geometry.resolution,
                                  'coord': u.expectN(0.0, ndim),
                                  'storageID': ID,
                                  'layer': dv.layer,
                                  'active': dv.active})

            views = {'probe': pv,
                     'obj': ov,
                     'diff': dv,
                     'mask': mv,
                     'exit': ev}

            pod = POD(ptycho=self.ptycho,
                      ID=None,
                      views=views,
                      geometry=geometry)
            pod.probe_weight = 1
            pod.object_weight = 1

            new_pods.append(pod)

        return new_pods, new_probe_ids, new_object_ids

    def _initialize_geo(self, common):
        """
        Initialize the geometry based on input data package
        Parameters.
        """

        # Collect geometry parameters
        get_keys = ['distance', 'center', 'energy', 'psize', 'shape']
        geo_pars = u.Param({key: common[key] for key in get_keys})
        geo_pars.propagation = self.p.propagation

        # make a Geo instance and fix resolution
        g = geometry.Geo(owner=self.ptycho, pars=geo_pars)
        g.p.resolution_is_fix = True

        # save the geometry
        self.geometries = [g]

        # Store frame shape
        self.shape = np.array(common.get('shape', g.shape))
        self.psize = g.psize

        return

    def _initialize_probe(self, probe_ids):
        """
        Initialize the probe storage referred to by probe_ids.keys()[0]
        """
        logger.info('\n'+headerline('Probe initialization', 'l'))

        # pick storage from container, there's only one probe
        pid = probe_ids.keys()[0]
        s = self.ptycho.probe.S.get(pid)
        logger.info('Initializing probe storage %s' % pid)

        # use the illumination module as a utility
        logger.info('Initializing as circle of size ' + str(self.p.illumination.size))
        illu_pars = u.Param({'aperture':
            {'form': 'circ', 'size': self.p.illumination.size}})
        illumination.init_storage(s, illu_pars)

        s.model_initialized = True

    def _initialize_object(self, object_ids):
        """
        Initializes the probe storage referred to by object_ids.keys()[0]
        """
        logger.info('\n'+headerline('Object initialization', 'l'))

        # pick storage from container, there's only one object
        oid = object_ids.keys()[0]
        s = self.ptycho.obj.S.get(oid)
        logger.info('Initializing probe storage %s' % oid)

        # simple fill, no need to use the sample module for this
        s.fill(self.p.sample.fill)

        s.model_initialized = True


@defaults_tree.parse_doc('scan.Full')
class Full(ScanModel):
    """
    Manage a single scan model (sharing, coherence, propagation, ...)

    Defaults:

    [name]
    default = Full
    type = str
    help =
    doc =

    [coherence]
    default = 
    help = Coherence parameters
    doc = 
    type = Param
    userlevel = 
    lowlim = 0

    [coherence.num_probe_modes]
    default = 1
    help = Number of probe modes
    doc = 
    type = int
    userlevel = 0
    lowlim = 0

    [coherence.num_object_modes]
    default = 1
    help = Number of object modes
    doc = 
    type = int
    userlevel = 0
    lowlim = 0

    [coherence.energies]
    default = [1.0]
    type = list
    help = ?
    doc = ?

    [coherence.spectrum]
    default = [1.0]
    help = Amplitude of relative energy bins if the probe modes have a different energy
    doc = 
    type = list
    userlevel = 2
    lowlim = 0

    [coherence.object_dispersion]
    default = None
    help = Energy dispersive response of the object
    doc = One of:
       - ``None`` or ``'achromatic'``: no dispersion
       - ``'linear'``: linear response model
       - ``'irregular'``: no assumption
      **[not implemented]**
    type = str
    userlevel = 2

    [coherence.probe_dispersion]
    default = None
    help = Energy dispersive response of the probe
    doc = One of:
       - ``None`` or ``'achromatic'``: no dispersion
       - ``'linear'``: linear response model
       - ``'irregular'``: no assumption
      **[not implemented]**
    type = str
    userlevel = 2

    [illumination.aperture]
    type = Param
    default =
    help = Beam aperture parameters

    [illumination.aperture.rotate]
    type = float
    default = 0.
    help = Rotate aperture by this value
    doc =

    [illumination.aperture.central_stop]
    help = size of central stop as a fraction of aperture.size
    default = None
    doc = If not None: places a central beam stop in aperture. The value given here is the fraction of the beam stop compared to `size`
    lowlim = 0.
    uplim = 1.
    userlevel = 1
    type = float

    [illumination.aperture.diffuser]
    help = Noise in the transparen part of the aperture
    default = None
    doc = Can be either:
    	 - ``None`` : no noise
    	 - ``2-tuple`` : noise in phase (amplitude (rms), minimum feature size)
    	 - ``4-tuple`` : noise in phase & modulus (rms, mfs, rms_mod, mfs_mod)
    userlevel = 2
    type = tuple

    [illumination.aperture.edge]
    help = Edge width of aperture (in pixels!)
    type = float
    default = 2.0
    userlevel = 2

    [illumination.aperture.form]
    default = circ
    type = None, str
    help = One of None, 'rect' or 'circ'
    doc = One of:
    	 - ``None`` : no aperture, this may be useful for nearfield
    	 - ``'rect'`` : rectangular aperture
    	 - ``'circ'`` : circular aperture
    choices = None,'rect','circ'
    userlevel = 2

    [illumination.aperture.offset]
    default = 0.
    type = float, tuple
    help = Offset between center of aperture and optical axes
    doc = May also be a tuple (vertical,horizontal) for size in case of an asymmetric offset
    userlevel = 2

    [illumination.aperture.size]
    default = None
    type = float
    help = Aperture width or diameter
    doc = May also be a tuple *(vertical,horizontal)* in case of an asymmetric aperture
    lowlim = 0.
    userlevel = 0

    [illumination.diversity]
    default = None
    type = Param, None
    help = Probe mode(s) diversity parameters
    doc = Can be ``None`` i.e. no diversity
    userlevel = 1

    [illumination.diversity.noise]
    default = None
    type = tuple
    help = Noise in the generated modes of the illumination
    doc = Can be either:
    	 - ``None`` : no noise
    	 - ``2-tuple`` : noise in phase (amplitude (rms), minimum feature size)
    	 - ``4-tuple`` : noise in phase & modulus (rms, mfs, rms_mod, mfs_mod)
    userlevel = 1

    [illumination.diversity.power]
    default = 0.1
    type = tuple, float
    help = Power of modes relative to main mode (zero-layer)
    uplim = 1.0
    lowlim = 0.0
    userlevel = 1

    [illumination.diversity.shift]
    default = None
    type = float
    help = Lateral shift of modes relative to main mode
    doc = **[not implemented]**
    userlevel = 2

    [illumination.model]
    default = None
    type = str
    help = Type of illumination model
    doc = One of:
    	 - ``None`` : model initialitziation defaults to flat array filled with the specified number of photons
    	 - ``'recon'`` : load model from previous reconstruction, see `recon` Parameters
    	 - ``'stxm'`` : Estimate model from autocorrelation of mean diffraction data
    	 - *<resource>* : one of ptypys internal image resource strings
    	 - *<template>* : one of the templates inillumination module

    	In script, you may pass a numpy.ndarray here directly as the model. It is considered as incoming wavefront and will be propagated according to `propagation` with an optional `aperture` applied before.
    userlevel = 0

    [illumination.photons]
    type = int, None
    default = None
    help = Number of photons in the incident illumination
    doc = A value specified here will take precedence over calculated statistics from the loaded data.
    lowlim = 0
    userlevel = 2

    [illumination.propagation]
    type = Param
    default =
    help = Parameters for propagation after aperture plane
    doc = Propagation to focus takes precedence to parallel propagation if `foccused` is not ``None``

    [illumination.propagation.antialiasing]
    default = 1
    type = float
    help = Antialiasing factor
    doc = Antialiasing factor used when generating the probe. (numbers larger than 2 or 3 are memory hungry)
    	**[Untested]**
    userlevel = 2

    [illumination.propagation.focussed]
    default = None
    type = None, float
    lowlim =
    help = Propagation distance from aperture to focus
    doc = If ``None`` or ``0`` : No focus propagation
    userlevel = 0

    [illumination.propagation.parallel]
    default = None
    type = None, float
    help = Parallel propagation distance
    doc = If ``None`` or ``0`` : No parallel propagation
    userlevel = 0

    [illumination.propagation.spot_size]
    default = None
    type = None, float
    help = Focal spot diameter
    doc = If not ``None``, this parameter is used to generate the appropriate aperture size instead of :py:data:`size`
    lowlim = 0
    userlevel = 1

    [illumination.recon]
    default =
    type = Param
    help = Parameters to load from previous reconstruction

    [illumination.recon.label]
    default = None
    type = None, str
    help = Scan label of diffraction that is to be used for probe estimate
    doc = If ``None``, own scan label is used
    userlevel = 1

    [illumination.recon.rfile]
    default = \*.ptyr
    type = str
    help = Path to a ``.ptyr`` compatible file
    userlevel = 0

    [sample.model]
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
    type = str
    userlevel = 0

    [sample.fill]
    default = 1
    help = Default fill value
    doc =
    type = float, complex
    userlevel =

    [sample.recon]
    default =
    help = Parameters to load from previous reconstruction
    doc =
    type = Param
    userlevel =

    [sample.recon.rfile]
    default = \*.ptyr
    help = Path to a ``.ptyr`` compatible file
    doc =
    type = file
    userlevel = 0

    [sample.stxm]
    default =
    help = STXM analysis parameters
    doc =
    type = Param
    userlevel = 1

    [sample.stxm.label]
    default = None
    help = Scan label of diffraction that is to be used for probe estimate
    doc = ``None``, own scan label is used
    type = str
    userlevel = 1

    [sample.process]
    default = None
    help = Model processing parameters
    doc = Can be ``None``, i.e. no processing
    type = Param
    userlevel =

    [sample.process.offset]
    default = (0,0)
    help = Offset between center of object array and scan pattern
    doc =
    type = tuple
    userlevel = 2
    lowlim = 0

    [sample.process.zoom]
    default = None
    help = Zoom value for object simulation.
    doc = If ``None``, leave the array untouched. Otherwise the modeled or loaded image will be
      resized using :py:func:`zoom`.
    type = tuple
    userlevel = 2
    lowlim = 0

    [sample.process.formula]
    default = None
    help = Chemical formula
    doc = A Formula compatible with a cxro database query,e.g. ``'Au'`` or ``'NaCl'`` or ``'H2O'``
    type = str
    userlevel = 2

    [sample.process.density]
    default = 1
    help = Density in [g/ccm]
    doc = Only used if `formula` is not None
    type = float
    userlevel = 2

    [sample.process.thickness]
    default = 1.00E-06
    help = Maximum thickness of sample
    doc = If ``None``, the absolute values of loaded source array will be used
    type = float
    userlevel = 2

    [sample.process.ref_index]
    default = 0.5+0.j
    help = Assigned refractive index
    doc = If ``None``, treat source array as projection of refractive index. If a refractive index
      is provided the array's absolute value will be used to scale the refractive index.
    type = complex
    userlevel = 2
    lowlim = 0

    [sample.process.smoothing]
    default = 2
    help = Smoothing scale
    doc = Smooth the projection with gaussian kernel of width given by `smoothing_mfs`
    type = int
    userlevel = 2
    lowlim = 0

    [sample.diversity]
    default =
    help = Probe mode(s) diversity parameters
    doc = Can be ``None`` i.e. no diversity
    type = Param
    userlevel =

    [sample.diversity.noise]
    default = None
    help = Noise in the generated modes of the illumination
    doc = Can be either:
       - ``None`` : no noise
       - ``2-tuple`` : noise in phase (amplitude (rms), minimum feature size)
       - ``4-tuple`` : noise in phase & modulus (rms, mfs, rms_mod, mfs_mod)
    type = tuple
    userlevel = 1

    [sample.diversity.power]
    default = 0.1
    help = Power of modes relative to main mode (zero-layer)
    doc =
    type = tuple, float
    userlevel = 1

    [sample.diversity.shift]
    default = None
    help = Lateral shift of modes relative to main mode
    doc = **[not implemented]**
    type = float
    userlevel = 2

    """

    _PREFIX = MODEL_PREFIX

    def _create_pods(self):
        """
        Create all new pods as specified in the new_positions,
        new_diff_views and new_mask_views object attributes.
        """
        logger.info('\n' + headerline('Creating PODS', 'l'))
        new_pods = []
        new_probe_ids = {}
        new_object_ids = {}

        label = self.label

        # Get a list of probe and object that already exist
        existing_probes = self.ptycho.probe.storages.keys()
        existing_objects = self.ptycho.obj.storages.keys()
        logger.info('Found these probes : ' + ', '.join(existing_probes))
        logger.info('Found these objects: ' + ', '.join(existing_objects))

        object_id = 'S00'
        probe_id = 'S00'

        positions = self.new_positions
        di_views = self.new_diff_views
        ma_views = self.new_mask_views

        # Loop through diffraction patterns
        for i in range(len(di_views)):
            dv, mv = di_views.pop(0), ma_views.pop(0)

            index = dv.layer

            # Object and probe position
            pos_pr = u.expect2(0.0)
            pos_obj = positions[i] if 'empty' not in self.p.tags else 0.0

            # For multiwavelength reconstructions: loop here over
            # geometries, and modify probe_id and object_id.
            for ii, geometry in enumerate(self.geometries):
                # Make new IDs and keep them in record
                # sharing_rules is not aware of IDs with suffix
                
                pdis = self.p.coherence.probe_dispersion

                if pdis is None or str(pdis) == 'achromatic':
                    gind = 0
                else:
                    gind = ii

                probe_id_suf = probe_id + 'G%02d' % gind
                if (probe_id_suf not in new_probe_ids.keys()
                        and probe_id_suf not in existing_probes):
                    new_probe_ids[probe_id_suf] = True

                odis = self.p.coherence.object_dispersion

                if odis is None or str(odis) == 'achromatic':
                    gind = 0
                else:
                    gind = ii

                object_id_suf = object_id + 'G%02d' % gind
                if (object_id_suf not in new_object_ids.keys()
                        and object_id_suf not in existing_objects):
                    new_object_ids[object_id_suf] = True

                # Loop through modes
                for pm in range(self.p.coherence.num_probe_modes):
                    for om in range(self.p.coherence.num_object_modes):
                        # Make a unique layer index for exit view
                        # The actual number does not matter due to the
                        # layermap access
                        exit_index = index * 10000 + pm * 100 + om

                        # Create views
                        # Please note that mostly references are passed,
                        # i.e. the views do mostly not own the accessrule
                        # contents
                        pv = View(container=self.ptycho.probe,
                                  accessrule={'shape': geometry.shape,
                                              'psize': geometry.resolution,
                                              'coord': pos_pr,
                                              'storageID': probe_id_suf,
                                              'layer': pm,
                                              'active': True})

                        ov = View(container=self.ptycho.obj,
                                  accessrule={'shape': geometry.shape,
                                              'psize': geometry.resolution,
                                              'coord': pos_obj,
                                              'storageID': object_id_suf,
                                              'layer': om,
                                              'active': True})

                        ev = View(container=self.ptycho.exit,
                                  accessrule={'shape': geometry.shape,
                                              'psize': geometry.resolution,
                                              'coord': pos_pr,
                                              'storageID': (probe_id +
                                                            object_id[1:] +
                                                            'G%02d' % ii),
                                              'layer': exit_index,
                                              'active': dv.active})

                        views = {'probe': pv,
                                 'obj': ov,
                                 'diff': dv,
                                 'mask': mv,
                                 'exit': ev}

                        pod = POD(ptycho=self.ptycho,
                                  ID=None,
                                  views=views,
                                  geometry=geometry)  # , meta=meta)

                        new_pods.append(pod)

                        pod.probe_weight = 1
                        pod.object_weight = 1

        return new_pods, new_probe_ids, new_object_ids

    def _initialize_geo(self, common):
        """
        Initialize the geometry/geometries.
        """
        # Extract necessary info from the received data package
        get_keys = ['distance', 'center', 'energy', 'psize', 'shape']
        geo_pars = u.Param({key: common[key] for key in get_keys})

        # Add propagation info from this scan model
        geo_pars.propagation = self.p.propagation

        # The multispectral case will have multiple geometries
        for ii, fac in enumerate(self.p.coherence.energies):
            geoID = geometry.Geo._PREFIX + '%02d' % ii + self.label
            g = geometry.Geo(self.ptycho, geoID, pars=geo_pars)
            # now we fix the sample pixel size, This will make the frame size adapt
            g.p.resolution_is_fix = True
            # save old energy value:
            g.p.energy_orig = g.energy
            # change energy
            g.energy *= fac
            # append the geometry
            self.geometries.append(g)

        # Store frame shape
        self.shape = np.array(common.get('shape', self.geometries[0].shape))
        self.psize = self.geometries[0].psize

        return

    def _initialize_probe(self, probe_ids):
        """
        Initialize the probe storages referred to by the probe_ids.

        For this case the parameter interface of the illumination module
        matches the illumination parameters of this class, so they are
        just fed in directly.
        """
        logger.info('\n'+headerline('Probe initialization', 'l'))

        # Loop through probe ids
        for pid, labels in probe_ids.items():

            illu_pars = self.p.illumination

            # pick storage from container
            s = self.ptycho.probe.S.get(pid)

            if s is None:
                continue
            else:
                logger.info('Initializing probe storage %s using scan %s.'
                            % (pid, self.label))


            # if photon count is None, assign a number from the stats.
            phot = illu_pars.get('photons')
            phot_max = self.diff.max_power

            if phot is None:
                logger.info('Found no photon count for probe in parameters.\nUsing photon count %.2e from photon report' % phot_max)
                illu_pars['photons'] = phot_max
            elif np.abs(np.log10(phot)-np.log10(phot_max)) > 1:
                logger.warn('Photon count from input parameters (%.2e) differs from statistics (%.2e) by more than a magnitude' % (phot, phot_max))

            illumination.init_storage(s, illu_pars)

            s.reformat()  # Maybe not needed
            s.model_initialized = True

    def _initialize_object(self, object_ids):
        """
        Initializes the probe storages referred to by the object_ids.
        """

        logger.info('\n'+headerline('Object initialization', 'l'))

        # Loop through object IDs
        for oid, labels in object_ids.items():

            sample_pars = self.p.sample

            # pick storage from container
            s = self.ptycho.obj.S.get(oid)

            if s is None or s.model_initialized:
                continue
            else:
                logger.info('Initializing object storage %s using scan %s.'
                            % (oid, self.label))

            sample_pars = self.p.sample

            if type(sample_pars) is u.Param:
                # Deep copy
                sample_pars = sample_pars.copy(depth=10)

                # Quickfix spectral contribution.
                if (self.p.coherence.object_dispersion
                        not in [None, 'achromatic']
                        and self.p.coherence.probe_dispersion
                        in [None, 'achromatic']):
                    logger.info(
                        'Applying spectral distribution input to object fill.')
                    sample_pars['fill'] *= s.views[0].pod.geometry.p.spectral


            sample.init_storage(s, sample_pars)
            s.reformat()  # maybe not needed

            s.model_initialized = True


import geometry_bragg
@defaults_tree.parse_doc('scan.Bragg3dModel')
class Bragg3dModel(Vanilla):
    """
    Model for 3D Bragg ptycho data, where a set of rocking angles are
    measured for each scanning position. The result is pods carrying
    3D diffraction patterns and 3D Views into a 3D object.

    Inherits from Vanilla because _create_pods and the probe/object
    initializations are identical.

    Frames for each position are assembled according to the actual
    xyz data, so it will not work if two acquisitions are done at the
    same position.

    Defaults:

    [name]
    default = Bragg3dModel
    type = str
    help =

    [illumination.size]
    default = None
    type = float
    help = Initial probe size
    doc = The probe is initialized as a flat circle.

    [sample.fill]
    default = 1
    type = float, complex
    help = Initial sample value
    doc = The sample is initialized with this value everywhere.
    """

    def __init__(self, ptycho=None, pars=None, label=None):
        super(Bragg3dModel, self).__init__(ptycho, pars, label)
        # This model holds on to incoming frames until a complete 3d
        # diffraction pattern can be built for that position.
        self.buffered_frames = {}
        self.buffered_positions = []
        #self.frames_per_call = 216 # just for testing

    def _new_data_extra_analysis(self, dp):
        """
        The heavy override is new_data. I've inserted an extra method
        for now, so as not to duplicate all the new_data code.

        The PtyScans give 2d diff images at 4d (angle, x, z, y)
        positions in the sample frame. These need to be assembled into
        3d (q3, q1, q2) at 3d positions. This means receiving images,
        holding on to them, and only calling _create_pods once a
        complete 3d diff View has been created.

        The xyz axes are those specified in Geo_Bragg, and the angle
        parameter defined such that a more positive angle corresponds to
        a more positive q3. That is, it is the angle between the xy
        plane of the sample with respect to the incident beam.
        """

        # go through and buffer the new 2d frames
        for dct in dp['iterable']:
            pos = dct['position'][1:]
            try:
                # index into the frame buffer where this frame belongs
                idx = np.where(np.prod(np.isclose(pos, self.buffered_positions), axis=1))[0][0]
                logger.debug('Frame %d belongs in frame buffer %d'
                    % (dct['index'], idx))
            except:
                # this position hasn't been encountered before, so create a buffer entry
                idx = len(self.buffered_positions)
                logger.debug('Frame %d doesn\'t belong in an existing frame buffer, creating buffer %d' % (dct['index'], idx))
                self.buffered_positions.append(pos)
                self.buffered_frames[idx] = {
                    'position': pos,
                    'frames': [],
                    'masks': [],
                    'angles': [],
                }

            # buffer the frame, mask, and angle
            self.buffered_frames[idx]['frames'].append(dct['data'])
            self.buffered_frames[idx]['masks'].append(dct['mask'])
            self.buffered_frames[idx]['angles'].append(dct['position'][0])

        # go through the buffer to see if any positions have all their
        # 2d frames, and create a new dp-compatible structure with
        # complete positions.
        dp_new = {'iterable': []}
        for idx, dct in self.buffered_frames.iteritems():
            if len(dct['angles']) == self.geometries[0].shape[0]:
                # this one is ready to go
                logger.debug('3d diffraction data for position %d ready, will create POD' % idx)

                # first sort the frames in increasing angle (increasing q3) order
                order = [i[0] for i in sorted(enumerate(dct['angles']), key=lambda x:x[1])]
                dct['frames'] = [dct['frames'][i] for i in order]
                dct['masks'] = [dct['masks'][i] for i in order]

                # then assemble the data and masks
                dp_new['iterable'].append({
                    'index': idx,
                    'position': dct['position'],
                    'data': np.array(dct['frames'], dtype=self.ptycho.FType),
                    'mask': np.array(dct['masks'], dtype=bool),
                    })
            else:
                logger.debug('3d diffraction data for position %d isn\'t ready, have %d out of %d frames'
                    % (idx, len(dct['angles']), self.geometries[0].shape[0]))

        # delete complete entries from the buffer
        for dct in dp_new['iterable']:
            del self.buffered_frames[dct['index']]

        # continue to pod creation if there is data for it
        if len(dp_new['iterable']):
            logger.debug('Will continue with POD creation for %d complete positions'
                % len(dp_new['iterable']))
            return dp_new
        else:
            return None

    def _initialize_containers(self):
        """
        Override to get 3D containers.
        """
        self.ptycho.probe = Container(ptycho=self.ptycho, ID='Cprobe', data_type='complex', data_dims=3)
        self.ptycho.obj = Container(ptycho=self.ptycho, ID='Cobj', data_type='complex', data_dims=3)
        self.ptycho.exit = Container(ptycho=self.ptycho, ID='Cexit', data_type='complex', data_dims=3)
        self.ptycho.diff = Container(ptycho=self.ptycho, ID='Cdiff', data_type='real', data_dims=3)
        self.ptycho.mask = Container(ptycho=self.ptycho, ID='Cmask', data_type='bool', data_dims=3)
        self.Cdiff = self.ptycho.diff
        self.Cmask = self.ptycho.mask
        self.containers_initialized = True

    def _initialize_geo(self, common):
        """
        Initialize the geometry based on parameters from a PtyScan.auto
        data package. Now psize and shape change meanings: from referring
        to raw data frames, they now refer to 3-dimensional diffraction
        patterns as specified by Geo_Bragg.
        """

        # Collect and assemble geometric parameters
        get_keys = ['distance', 'center', 'energy']
        geo_pars = u.Param({key: common[key] for key in get_keys})
        geo_pars.propagation = self.p.propagation
        # take extra Bragg information into account
        psize = tuple(common['psize'])
        geo_pars.psize = (self.ptyscan.common.rocking_step,) + psize
        sh = tuple(common['shape'])
        geo_pars.shape = (self.ptyscan.common.n_rocking_positions,) + sh
        geo_pars.theta_bragg = self.ptyscan.common.theta_bragg

        # make a Geo instance and fix resolution
        g = geometry_bragg.Geo_Bragg(owner=self.ptycho, pars=geo_pars)
        g.p.resolution_is_fix = True

        # save the geometry
        self.geometries = [g]

        # Store frame shape
        self.shape = g.shape
        self.psize = g.psize


class ModelManager(object):
    """
    Thin wrapper class which now just interfaces Ptycho with ScanModel. 
    This should probably all be done directly in Ptycho.
    """

    def __init__(self, ptycho, pars):
        """

        Parameters
        ----------
        ptycho: Ptycho
            The parent Ptycho object

        pars : dict or Param
            The .scans tree of the :any:`Ptycho` parameters.
        """
        assert ptycho is not None
        self.ptycho = ptycho

        # Create scan model objects
        self.scans = OrderedDict()
        for label, scan_pars in pars.iteritems():
            # find out which scan model class to instantiate
            if scan_pars.name in u.all_subclasses(ScanModel, names=True):
                cls = eval(scan_pars.name)
            else:
                raise RuntimeError('Could not manage model %s' % scan_pars.name)
            # instantiate!
            self.scans[label] = cls(ptycho=self.ptycho, pars=scan_pars, label=label)

    def _to_dict(self):
        return self.__dict__.copy()

    @classmethod
    def _from_dict(cls, dct):
        # create instance
        inst = cls(None, None)
        # overwrite internal dictionary
        inst.__dict__ = dct
        return inst

    @property
    def data_available(self):
        return any(s.data_available for s in self.scans.values())

    def new_data(self):
        """
        Get all new diffraction patterns and create all views and pods
        accordingly.
        """
        parallel.barrier()

        # Nothing to do if there are no new data.
        if not self.data_available:
            return 'No data'

        logger.info('Processing new data.')

        # Attempt to get new data
        for label, scan in self.scans.iteritems():
            new_data = scan.new_data()
