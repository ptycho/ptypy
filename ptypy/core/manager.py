# -*- coding: utf-8 -*-
"""
Scan management.

The main task of this module is to prepare the data structure for
reconstruction, taking a data feed and connecting individual diffraction
measurements to the other containers. The way this connection is done
as defined by ScanModel and its subclasses. The connections are
described by the POD objects.

This file is part of the PTYPY package.

    :copyright: Copyright 2014 by the PTYPY team, see AUTHORS.
    :license: see LICENSE for details.
"""
import numpy as np
import time
from collections import OrderedDict
from . import illumination
from . import sample
from . import geometry
from . import data

from .. import utils as u
from ..utils.verbose import logger, headerline, log
from ..utils.verbose import ilog_message, ilog_streamer, ilog_newline
from .classes import *
from .classes import DEFAULT_ACCESSRULE
from .classes import MODEL_PREFIX
from ..utils import parallel
from ..utils.descriptor import EvalDescriptor
from .. import defaults_tree

# Please set these globally later
FType = np.float64
CType = np.complex128

__all__ = ['ModelManager', 'ScanModel', 'Full', 'Vanilla', 'Bragg3dModel', 'OPRModel', 'BlockScanModel',
           'BlockVanilla', 'BlockFull', 'BlockOPRModel']

class _LogTime(object):

    def __init__(self):
        self._t = time.time()

    def __call__(self, msg=None):
        logger.info('Duration %.2f for ' % (time.time() - self._t) + str(msg))
        self._t = time.time()


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

    [ffttype]
    type = str
    default = scipy
    help = FFT library
    doc = Choose from "numpy", "scipy" or "fftw"
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
    type = Param, str
    default =
    help = Container for probe initialization model

    [sample]
    type = Param, str
    default =
    help = Container for sample initialization model

    [resample]
    type = int, None
    default = 1
    help = Resampling fraction of the image frames w.r.t. diffraction frames
    doc = A resampling of 2 means that the image frame is to be sampled (in the detector plane) twice
          as densely as the raw diffraction data.
    """
    _PREFIX = MODEL_PREFIX

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
        self.diff_shape = None    # Shape of diffaction frames
        self.probe_shape = None   # Shape of probe views (may be different if resampling or padding)
        self.object_shape = None  # Currently same as probe_shape
        self.exit_shape = None    # Currently same as object_shape
        self.psize = None         # Pixel size in the detector plane

        # Object flags and constants
        self.containers_initialized = False
        self.data_available = True
        self.CType = CType
        self.FType = FType

        # Keep track of the maximum frames in a block
        # For the ScanModel this will be equivalent to the total nr. of frames in the scan
        # For the BlockScanModel this is defined by the user (frames_per_block) and the MPI settings
        self.max_frames_per_block = 0

        # By default we create a new exit buffer for each view
        self._single_exit_buffer_for_all_views = False

    @classmethod
    def makePtyScan(cls, pars):
        """
        Factory for PtyScan object. Return an instance of the appropriate PtyScan 
        subclass based on the input parameters.

        Parameters
        ----------
        pars: dict or Param
            Input parameters according to :py:data:`.scan.data`.
        """

        # Extract information on the type of object to build
        name = pars.name

        from .. import experiment

        if name in (u.all_subclasses(data.PtyScan, names=True)) \
                or name == 'PtyScan':
            ps_class = experiment.PTYSCANS.get(name, None)
            if ps_class is None:
                raise RuntimeError('Unknown PtyScan subclass: "%s". Did you import it?' % name)
            logger.info('Scan will be prepared with the PtyScan subclass "%s"' % name)
            ps_instance = ps_class(pars)
        else:
            raise RuntimeError('Could not manage source "%s"' % str(name))

        return ps_instance

    def new_data(self, max_frames):
        """
        Feed data from ptyscan object.
        :return: None if no data is available, True otherwise.
        """
        report_time = _LogTime()

        # Initialize if that has not been done yet
        if not self.ptyscan.is_initialized:
            self.ptyscan.initialize()

        report_time('ptyscan init')

        # Get data
        logger.info('Importing data from scan %s.' % self.label)

        dp = self.ptyscan.auto(max_frames)
        #dp = self.ptyscan.auto(self.frames_per_call)

        self.data_available = (dp != data.EOS)
        logger.debug(u.verbose.report(dp))

        if dp == data.WAIT or not self.data_available:
            return None

        label = self.label

        report_time('read data')
        logger.info('Creating views and storages.')

        # Prepare the scan geometry if not already done.
        if not self.geometries:
            self._initialize_geo(dp['common'])

        # Create containers if not already done
        if not self.containers_initialized:
            self._initialize_containers()

        # Generalized shape which works for 2d and 3d cases
        sh = (1,) + tuple(self.diff_shape)

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
        AR_diff = DEFAULT_ACCESSRULE.copy()
        AR_diff.shape = self.diff_shape
        AR_diff.coord = 0.0
        AR_diff.psize = self.psize
        AR_mask = AR_diff.copy()
        AR_diff.storageID = self.diff.ID
        AR_mask.storageID = self.mask.ID

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
        report_time('creating views and storages')
        logger.info('Inserting data in diff and mask storages')

        # Second pass: copy the data 
        # Benchmark: scales quadratic (!!) with number of frames per node.
        for dct in dp['iterable']:
            if dct['data'] is None:
                continue
            diff_data = dct['data']
            idx = dct['index']

            # FIXME: Find a more transparent way than this.
            self.diff.data[self.diff.layermap.index(idx)][:] = diff_data
            self.mask.data[self.mask.layermap.index(idx)][:] = dct.get('mask', np.ones_like(diff_data))

        # Update maximum nr. of frames in a block
        self.max_frames_per_block = self.diff.nlayers

        self.diff.nlayers = parallel.MPImax(self.diff.layermap) + 1
        self.mask.nlayers = parallel.MPImax(self.mask.layermap) + 1

        self.new_positions = positions
        self.new_diff_views = diff_views
        self.new_mask_views = mask_views
        self.positions += positions
        self.diff_views += diff_views
        self.mask_views += mask_views
        report_time('inserting data')
        logger.info('Data organization complete, updating stats')

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

        u.parallel.barrier()
        report_time('creating pods')
        logger.info('Process %d completed new_data.' % parallel.rank, extra={'allprocesses': True})

        return self.diff, new_probe_ids, new_object_ids, new_pods

    def _new_data_extra_analysis(self, dp):
        """
        This is a hack for 3d Bragg. Extra analysis on the incoming
        data package. Returns modified dp, or None if no completa data
        is available for pod creation.
        """
        return dp

    def _initialize_containers(self):
        """
        Initialize containers appropriate for the model. This 
        implementation works for 2d models, override if necessary.
        """
        if self.ptycho is None:
            # Stand-alone use
            self.Cdiff = Container(self, ID='Cdiff', data_type='real')
            self.Cmask = Container(self, ID='Cmask', data_type='bool')
        else:
            # Use with a Ptycho instance
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
        max_frame = np.zeros(diff_views[0].shape)
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
        if parallel.MPIenabled:
            parallel.allreduce(max_frame, parallel.MPI.MAX)
            parallel.allreduce(min_frame, parallel.MPI.MIN)
        mean_frame /= (norm + (norm == 0))
        self.diff.norm = norm
        self.diff.max_power = parallel.MPImax(Itotal)
        self.diff.tot_power = parallel.MPIsum(Itotal)
        self.diff.mean_power = self.diff.tot_power / (len(diff_views) * np.prod(self.diff_shape))
        self.diff.pbound_stub = self.diff.max_power / np.prod(self.diff_shape)
        self.diff.mean = mean_frame
        self.diff.max = max_frame
        self.diff.min = min_frame
        self.diff.label = self.label

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

    def _get_data(self, max_frames):
        # Get data
        logger.info('Importing data from scan %s.' % self.label)
        dp = self.ptyscan.auto(max_frames)

        self.data_available = (dp != data.EOS)

        # TODO remove reports if not needed
        #logger.debug(u.verbose.report(dp))

        if dp == data.WAIT or not self.data_available:
            return None
        else:
            return dp


@defaults_tree.parse_doc('scan.BlockScanModel')
class BlockScanModel(ScanModel):

    def new_data(self, max_frames):
        """
        Feed data from ptyscan object.
        :return: None if no data is available, Diffraction storage otherwise.
        """
        report_time = _LogTime()

        # Initialize if that has not been done yet
        if not self.ptyscan.is_initialized:
            self.ptyscan.initialize()

        report_time('ptyscan init')

        dp = self._get_data(max_frames)
        if dp is None:
            return None

        report_time('read data')

        logger.info('Creating views and storages.')
        # Prepare the scan geometry if not already done.
        if not self.geometries:
            self._initialize_geo(dp['common'])

        # Create containers if not already done
        if not self.containers_initialized:
            self._initialize_containers()

        sh = (1,) + tuple(self.diff_shape)

        # this is a hack for now
        dp = self._new_data_extra_analysis(dp)
        if dp is None:
            return None
        else:
            common = dp['common']
            chunk = dp['chunk']

        # Generalized shape which works for 2d and 3d cases
        sh = (max(len(chunk.indices_node),1),) + tuple(self.diff_shape)

        indices_node = chunk['indices_node']

        diff = self.Cdiff.new_storage(shape=sh, psize=self.psize, padonly=True,
                                      fill=0.0, layermap=indices_node)
        mask = self.Cmask.new_storage(shape=sh, psize=self.psize, padonly=True,
                                      fill=1.0, layermap=indices_node)

        # Prepare for View generation
        AR_diff = DEFAULT_ACCESSRULE.copy()
        AR_diff.shape = self.diff_shape # this is None due to init
        AR_diff.coord = 0.0
        AR_diff.psize = self.psize
        AR_mask = AR_diff.copy()
        AR_diff.storageID = diff.ID
        AR_mask.storageID = mask.ID

        diff_views = []
        mask_views = []
        positions = []

        dv = None
        mv = None

        data = chunk['data']
        weights = chunk['weights']

        # First pass: create or update views and reformat corresponding storage
        for index in chunk['indices']:

            if dv is None:
                dv = View(self.Cdiff, accessrule=AR_diff)  # maybe use index here
                mv = View(self.Cmask, accessrule=AR_mask)
            else:
                dv = dv.copy()
                mv = mv.copy()

            maybe_data = data.get(index)
            active = maybe_data is not None

            dv.active = active
            mv.active = active
            dv.layer = index
            mv.layer = index

            diff_views.append(dv)
            mask_views.append(mv)

            if active:
                l = indices_node.index(index)
                dv.dlayer = l
                mv.dlayer = l
                dv.data[:] = maybe_data
                mv.data[:] = weights.get(index, np.ones_like(maybe_data))

                # positions
        positions = chunk.positions

        ## warning message for empty postions?

        # Update maximum nr. of frames in a block
        self.max_frames_per_block = max(diff.nlayers, self.max_frames_per_block)

        # this is not absolutely necessary
        # diff.update_views()
        # mask.update_views()
        diff.nlayers = parallel.MPImax(diff.layermap) + 1
        mask.nlayers = parallel.MPImax(mask.layermap) + 1
        # save state / could be replaced by handing of arguments to methods
        self.diff = diff
        self.mask = mask
        self.new_positions = positions
        self.new_diff_views = diff_views
        self.new_mask_views = mask_views
        self.positions += list(positions)
        self.diff_views += diff_views
        self.mask_views += mask_views
        report_time('creating views and storages')
        logger.info('Data organization complete, updating stats')

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

        report_time('creating pods')

        return diff, new_probe_ids, new_object_ids, new_pods


class _Vanilla(object):
    """
    Vanilla model, one probe and one object per scan.
    Probe has only size as parameter, object only the initial fill value.

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

        # One probe / object storage per scan.
        ID = 'S' + self.label

        # We need to return info on what storages are created
        if not ID in self.ptycho.probe.storages.keys():
            new_probe_ids[ID] = True
        if not ID in self.ptycho.obj.storages.keys():
            new_object_ids[ID] = True

        geometry = self.geometries[0]

        pv = None
        ev = None
        ov = None
        ndim = self.Cdiff.ndim

        # Loop through diffraction patterns
        for i in range(len(self.new_diff_views)):
            dv, mv = self.new_diff_views.pop(0), self.new_mask_views.pop(0)

            # Create views
            # if True:
            if pv is None:
                pv = View(container=self.ptycho.probe,
                      accessrule={'shape': self.probe_shape,
                                  'psize': geometry.resolution,
                                  'coord': u.expectN(0.0, ndim),
                                  'storageID': ID,
                                  'layer': 0,
                                  'active': True})
            else:
                pv = pv.copy(update=False)
                pv.coord = 0.0

            # if True:
            if ov is None:
                ov = View(container=self.ptycho.obj,
                      accessrule={'shape': self.object_shape,
                                  'psize': geometry.resolution,
                                  'coord': self.new_positions[i],
                                  'storageID': ID,
                                  'layer': 0,
                                  'active': True})
            else:
                ov = ov.copy(update=False)
                ov.coord = self.new_positions[i]

            # if True:
            if ev is None:
                ev = View(container=self.ptycho.exit,
                      accessrule={'shape': self.exit_shape,
                                  'psize': geometry.resolution,
                                  'coord': u.expectN(0.0, ndim),
                                  'storageID': dv.storageID,
                                  'layer': dv.layer,
                                  'active': dv.active})
            else: 
                ev = ev.copy(update=False)
                ev.storageID = dv.storageID
                ev.layer = dv.layer
                ev.active = dv.active
                ev.coord = 0.0

            views = {'probe': pv,
                     'obj': ov,
                     'diff': dv,
                     'mask': mv,
                     'exit': ev}

            pod = POD(ptycho=self.ptycho,
                      ID=None,
                      views=views,
                      geometry=geometry)
            pod.probe_weight = 1.0
            pod.object_weight = 1.0

            new_pods.append(pod)

        return new_pods, new_probe_ids, new_object_ids

    def _initialize_geo(self, common):
        """
        Initialize the geometry based on input data package
        Parameters.
        """
        probe_shape = common['shape']
        center = common['center']
        psize = common['psize']        
        
        # Adjust geometry parameters for resampling
        self.resample = self.p.resample
        probe_shape = tuple(np.ceil(self.resample * np.array(probe_shape)).astype(int))
        center = tuple(np.ceil(self.resample * np.array(center)).astype(int))
        psize = np.array(psize) / self.resample

        # Collect geometry parameters
        get_keys = ['distance', 'center', 'energy', 'psize']
        geo_pars = u.Param({key: common[key] for key in get_keys})
        geo_pars.shape = probe_shape
        geo_pars.center = center
        geo_pars.propagation = self.p.propagation
        geo_pars.ffttype = self.p.ffttype
        geo_pars.psize = psize

        # make a Geo instance and fix resolution
        g = geometry.Geo(owner=self.ptycho, pars=geo_pars)
        g.p.resolution_is_fix = True
        g.resample = self.resample

        # save the geometry
        self.geometries = [g]

        # Store frame shapes
        self.diff_shape = np.array(common.get('shape', g.shape))
        self.probe_shape = probe_shape
        self.object_shape = probe_shape
        self.exit_shape = probe_shape
        self.psize = g.psize
        return

    def _initialize_probe(self, probe_ids):
        """
        Initialize the probe storage referred to by probe_ids.keys()[0]
        """
        if not probe_ids:
            return

        logger.info('\n' + headerline('Probe initialization', 'l'))

        # pick storage from container, there's only one probe
        pid = list(probe_ids.keys())[0]
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
        if not object_ids:
            return

        logger.info('\n' + headerline('Object initialization', 'l'))

        # pick storage from container, there's only one object
        oid = list(object_ids.keys())[0]
        s = self.ptycho.obj.S.get(oid)
        logger.info('Initializing probe storage %s' % oid)

        # simple fill, no need to use the sample module for this
        s.fill(self.p.sample.fill)

        s.model_initialized = True


class _Full(object):
    """
    Manage a single scan model (sharing, coherence, propagation, ...)

    Defaults:

    # note: this class also imports the module-level defaults for sample
    # and illumination, below.

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

    [resolution]
    default = None
    help = Will force the reconstruction to adapt to the given resolution, this might lead to cropping/padding in diffraction space which could reduce performance.
    doc = Half-period resolution given in [m] 
    type = None, float
    userlevel = 0
    lowlim = 0

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

        label = self.label

        # Get a list of probe and object that already exist
        existing_probes = list(self.ptycho.probe.storages.keys())
        existing_objects = list(self.ptycho.obj.storages.keys())
        logger.info('Found these probes : ' + ', '.join(existing_probes))
        logger.info('Found these objects: ' + ', '.join(existing_objects))

        object_id = 'S' + self.label
        probe_id = 'S' + self.label

        # Loop through diffraction patterns
        for i in range(len(self.new_diff_views)):
            dv, mv = self.new_diff_views.pop(0), self.new_mask_views.pop(0)

            # For stochastic engines (e.g. ePIE) we only need one exit buffer
            if self._single_exit_buffer_for_all_views:
                index = 0
            else:
                index = dv.layer

            # Object and probe position
            pos_pr = u.expect2(0.0)
            pos_obj = self.new_positions[i] if 'empty' not in self.p.tags else 0.0

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
                                  accessrule={'shape': self.probe_shape,
                                              'psize': geometry.resolution,
                                              'coord': pos_pr,
                                              'storageID': probe_id_suf,
                                              'layer': pm,
                                              'active': True})

                        ov = View(container=self.ptycho.obj,
                                  accessrule={'shape': self.object_shape,
                                              'psize': geometry.resolution,
                                              'coord': pos_obj,
                                              'storageID': object_id_suf,
                                              'layer': om,
                                              'active': True})

                        ev = View(container=self.ptycho.exit,
                                  accessrule={'shape': self.exit_shape,
                                              'psize': geometry.resolution,
                                              'coord': pos_pr,
                                              'storageID': (dv.storageID +
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

                        pod.probe_weight = 1.0
                        pod.object_weight = 1.0

        return new_pods, new_probe_ids, new_object_ids

    def _initialize_geo(self, common):
        """
        Initialize the geometry/geometries.
        """
        probe_shape = common['shape']
        center = common['center']
        psize = common['psize']        
        
        # Adjust geometry parameters for resampling
        self.resample = self.p.resample
        probe_shape = tuple(np.ceil(self.resample * np.array(probe_shape)).astype(int))
        center = tuple(np.ceil(self.resample * np.array(center)).astype(int))
        psize = np.array(psize) / self.resample

        # Extract necessary info from the received data package
        get_keys = ['distance', 'center', 'energy', 'psize']
        geo_pars = u.Param({key: common[key] for key in get_keys})
        geo_pars.shape = probe_shape
        geo_pars.center = center
        geo_pars.psize = psize
        geo_pars.resolution = self.p.resolution

        # Add propagation info from this scan model
        geo_pars.propagation = self.p.propagation
        geo_pars.ffttype = self.p.ffttype

        # The multispectral case will have multiple geometries
        for ii, fac in enumerate(self.p.coherence.energies):
            geoID = geometry.Geo._PREFIX + '%02d' % ii + self.label
            g = geometry.Geo(self.ptycho, geoID, pars=geo_pars)
            # now we fix the sample pixel size, This will make the frame size adapt
            g.p.resolution_is_fix = True
            # save old energy value
            g.p.energy_orig = g.energy
            # change energy
            g.energy *= fac
            # resampling
            g.resample = self.resample
            # append the geometry
            self.geometries.append(g)

        # Store frame shape
        self.diff_shape = np.array(common.get('shape', self.geometries[0].shape))
        self.probe_shape = probe_shape
        self.object_shape = probe_shape
        self.exit_shape = probe_shape
        self.psize = self.geometries[0].psize

        return

    def _initialize_probe(self, probe_ids):
        """
        Initialize the probe storages referred to by the probe_ids.

        For this case the parameter interface of the illumination module
        matches the illumination parameters of this class, so they are
        just fed in directly.
        """
        logger.info('\n' + headerline('Probe initialization', 'l'))

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

            # Bypass additional tests if input is a string (previous reconstruction)
            if illu_pars != str(illu_pars):

                # if photon count is None, assign a number from the stats.
                phot = illu_pars.get('photons')
                phot_max = self.diff.max_power

                if phot is None:
                    logger.info(
                        'Found no photon count for probe in parameters.\nUsing photon count %.2e from photon report' % phot_max)
                    illu_pars['photons'] = phot_max
                elif np.abs(np.log10(phot) - np.log10(phot_max)) > 1:
                    logger.warning(
                        'Photon count from input parameters (%.2e) differs from statistics (%.2e) by more than a magnitude' % (
                        phot, phot_max))

                if (self.p.coherence.num_probe_modes > 1) and (type(illu_pars) is not np.ndarray):

                    if (illu_pars.diversity is None) or (
                            None in [illu_pars.diversity.noise, illu_pars.diversity.power]):
                        log(2,
                            "You are doing a multimodal reconstruction with none/ not much diversity between the modes! \n"
                            "This will likely not reconstruct. You should set .scan.illumination.diversity.power and "
                            ".scan.illumination.diversity.noise to something for the best results.")

            illumination.init_storage(s, illu_pars)

            s.reformat()  # Maybe not needed
            s.model_initialized = True

    def _initialize_object(self, object_ids):
        """
        Initializes the probe storages referred to by the object_ids.
        """

        logger.info('\n' + headerline('Object initialization', 'l'))

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


class _OPRModel(object):
    """
    Scan for Orthogonal Probe Relaxation (OPR) ptychography, where each has its own probe. 

    Defaults:

    [name]
    default = OPRModel
    type = str
    help =
    doc =
    """

    def __init__(self, ptycho=None, pars=None, label=None):
        super(_OPRModel, self).__init__(ptycho, pars, label)
        self.p.illumination['diversity'] = None

    def _create_pods(self):
        new_pods, new_probe_ids, new_object_ids = super(_OPRModel, self)._create_pods()

        for vID, v in self.ptycho.probe.views.items():
            # Get the associated diffraction frame
            di_view = v.pod.di_view
            # Reformat the layer
            v.layer = di_view.layer*self.p.coherence.num_probe_modes + v.layer
            # Deactivate if the associate di_view is inactive (to spread the probe across nodes consistently with diff)
            v.active = di_view.active

        # Create dictionaries to store OPR modes
        self.OPR_modes = {}
        self.OPR_coeffs = {}
        self.OPR_allprobes = {}

        return new_pods, new_probe_ids, new_object_ids

@defaults_tree.parse_doc('scan.Vanilla')
class Vanilla(_Vanilla, ScanModel):
    pass


@defaults_tree.parse_doc('scan.BlockVanilla')
class BlockVanilla(_Vanilla, BlockScanModel):
    pass


@defaults_tree.parse_doc('scan.Full')
class Full(_Full, ScanModel):
    pass


@defaults_tree.parse_doc('scan.BlockFull')
class BlockFull(_Full, BlockScanModel):
    pass

@defaults_tree.parse_doc('scan.OPRModel')
class OPRModel(_OPRModel, Full):
    pass

@defaults_tree.parse_doc('scan.BlockOPRModel')
class BlockOPRModel(_OPRModel, BlockFull):
    pass

@defaults_tree.parse_doc('scan.GradFull')
class GradFull(Full):
    def __init__(self, ptycho=None, pars=None, label=None):
        super(GradFull, self).__init__(ptycho, pars, label)
        self._single_exit_buffer_for_all_views = True

@defaults_tree.parse_doc('scan.BlockGradFull')
class BlockGradFull(BlockFull):
    def __init__(self, ptycho=None, pars=None, label=None):
        super(BlockGradFull, self).__init__(ptycho, pars, label)
        self._single_exit_buffer_for_all_views = True

# Append illumination and sample defaults
defaults_tree['scan.Full'].add_child(illumination.illumination_desc)
defaults_tree['scan.BlockFull'].add_child(illumination.illumination_desc)
defaults_tree['scan.Full'].add_child(sample.sample_desc)
defaults_tree['scan.BlockFull'].add_child(sample.sample_desc)

# Update defaults
Full.DEFAULT = defaults_tree['scan.Full'].make_default(99)

from . import geometry_bragg

defaults_tree['scan'].add_child(EvalDescriptor('Bragg3dModel'))
defaults_tree['scan.Bragg3dModel'].add_child(illumination.illumination_desc, copy=True)
defaults_tree['scan.Bragg3dModel.illumination'].prune_child('diversity')


@defaults_tree.parse_doc('scan.Bragg3dModel')
class Bragg3dModel(Vanilla):
    """
    Model for 3D Bragg ptycho data, where a set of rocking angles are
    measured for each scanning position. The result is pods carrying
    3D diffraction patterns and 3D Views into a 3D object.

    Inherits from Vanilla because _create_pods and the object init
    is identical.

    Frames for each position are assembled according to the actual
    xyz data, so it will not work if two acquisitions are done at the
    same position.

    Defaults:

    [name]
    default = Bragg3dModel
    type = str
    help =

    [resample]
    type = int, None
    default = 1
    help = Diffraction resampling *CURRENTLY NOT SUPPORTED FOR BRAGG CASE*
    """

    def __init__(self, ptycho=None, pars=None, label=None):
        super(Bragg3dModel, self).__init__(ptycho, pars, label)
        # This model holds on to incoming frames until a complete 3d
        # diffraction pattern can be built for that position.
        self.buffered_frames = {}
        self.buffered_positions = []
        # self.frames_per_call = 216 # just for testing

    def _new_data_extra_analysis(self, dp):
        """
        The heavy override is new_data. I've inserted this extra method
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

        logger.info(
            'Redistributing incoming frames so that all data from each scanning position is on the same node.')
        dp = self._mpi_redistribute_raw_frames(dp)

        logger.info(
            'Buffering incoming frames and binning these by scanning position.')
        self._buffer_incoming_frames(dp)

        logger.info(
            'Repackaging complete buffered rocking curves as 3d data packages.')
        dp_new = self._make_3d_data_package()

        # continue to pod creation if there is data for it
        if len(dp_new['iterable']):
            logger.info('Will continue with POD creation for %d complete positions.'
                        % len(dp_new['iterable']))
            return dp_new
        else:
            return None

    def _mpi_redistribute_raw_frames(self, dp):
        """
        Linear decomposition of incoming frames based on the scan
        dimension that varies the most. Modifies a data package in-place
        so that the angles of each unique scanning position end up on
        the same node.
        """

        # work out the xyz range, we have all positions here
        pos = []
        for dct in dp['iterable']:
            pos.append(dct['position'][1:])
        pos = np.array(pos)
        xmin, xmax = pos[:, 0].min(), pos[:, 0].max()
        ymin, ymax = pos[:, 2].min(), pos[:, 2].max()
        zmin, zmax = pos[:, 1].min(), pos[:, 1].max()
        diffs = [xmax - xmin, zmax - zmin, ymax - ymin]

        # the axis along which to slice
        axis = diffs.index(max(diffs))
        logger.info(
            'Will slice incoming frames along axis %d (%s).' % (axis, ['x', 'z', 'y'][axis]))

        # pick the relevant limits and expand slightly to avoid edge effects
        lims = {0: [xmin, xmax], 1: [zmin, zmax], 2: [ymin, ymax]}[axis]
        lims = np.array(lims) + np.array([-1, 1]) * np.diff(lims) * .01
        domain_width = np.diff(lims) / parallel.size

        # now we can work out which node should own a certain position
        def __node(pos):
            return (pos - lims[0]) // domain_width

        # work out which node should have each of my buffered frames
        N = parallel.size
        senditems = {}
        for idx in range(len(dp['iterable'])):
            if dp['iterable'][idx]['data'] is None:
                continue
            pos_ = pos[idx][axis]
            if not __node(pos_) == parallel.rank:
                senditems[idx] = __node(pos_)

        # transfer data as a list corresponding to dp['iterable']
        for sending_node in range(parallel.size):
            if sending_node == parallel.rank:
                # My turn to send
                for receiver in range(parallel.size):
                    if receiver == parallel.rank:
                        continue
                    lst = []
                    for idx, rec in senditems.items():
                        if rec == receiver:
                            lst.append(dp['iterable'][idx])
                    parallel.send(lst, dest=receiver)
            else:
                received = parallel.receive(source=sending_node)
                for frame in received:
                    idx = frame['index']
                    dp['iterable'][idx] = frame.copy()

        # mark sent frames disabled, would be nice to do in the loop but
        # you can't trust communication will be blocking.
        for idx in senditems.keys():
            dp['iterable'][idx]['data'] = None
            dp['iterable'][idx]['mask'] = None

        return dp

    def _buffer_incoming_frames(self, dp):
        """
        Store incoming frames in an internal buffer, binned by scanning
        position.
        """
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
                logger.debug(
                    'Frame %d doesn\'t belong in an existing frame buffer, creating buffer %d' % (dct['index'], idx))
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

    def _make_3d_data_package(self):
        """
        Go through the internal buffer to see if any positions have all
        their 2d frames, and create a new dp-compatible structure with
        complete 3d positions.
        """
        dp_new = {'iterable': []}
        for idx, dct in self.buffered_frames.items():
            if len(dct['angles']) == self.geometries[0].shape[0]:
                # this one is ready to go
                logger.debug('3d diffraction data for position %d ready, will create POD' % idx)

                if dct['frames'][0] is not None:
                    # First sort the frames in increasing angle (increasing
                    # q3) order. Also assume the images came in as (-q1, q2)
                    # from PtyScan. We want (q3, q1, q2) as required by
                    # Geo_Bragg, so flip the q1 dimension.
                    order = [i[0] for i in sorted(enumerate(dct['angles']), key=lambda x: x[1])]
                    dct['frames'] = [dct['frames'][i][::-1, :] for i in order]
                    dct['masks'] = [dct['masks'][i][::-1, :] for i in order]
                    diffdata = np.array(dct['frames'], dtype=self.ptycho.FType)
                    maskdata = np.array(dct['masks'], dtype=bool)
                else:
                    # this buffer belongs to another node
                    diffdata = None
                    maskdata = None

                # then assemble the data and masks
                dp_new['iterable'].append({
                    'index': idx,
                    'position': dct['position'],
                    'data': diffdata,
                    'mask': maskdata,
                })

            else:
                logger.debug('3d diffraction data for position %d isn\'t ready, have %d out of %d frames'
                             % (idx, len(dct['angles']), self.geometries[0].shape[0]))

        # delete complete entries from the buffer
        for dct in dp_new['iterable']:
            del self.buffered_frames[dct['index']]

        # make the indices on the 3d dp contiguous and unique
        cnt = {parallel.rank: sum([(d['data'] is not None) for d in dp_new['iterable']])}
        parallel.allgather_dict(cnt)
        offset = sum([cnt[i] for i in range(parallel.rank)])
        idx = 0
        for dct in dp_new['iterable']:
            dct['index'] = offset + idx
            idx += 1
        return dp_new

    def _initialize_containers(self):
        """
        Override to get 3D containers.
        """
        self.ptycho._pool['C'].pop('Cprobe')
        self.ptycho.probe = Container(self.ptycho, ID='Cprobe', data_type='complex', data_dims=3)
        self.ptycho._pool['C'].pop('Cobj')
        self.ptycho.obj = Container(self.ptycho, ID='Cobj', data_type='complex', data_dims=3)
        self.ptycho._pool['C'].pop('Cexit')
        self.ptycho.exit = Container(self.ptycho, ID='Cexit', data_type='complex', data_dims=3)
        self.ptycho._pool['C'].pop('Cdiff')
        self.ptycho.diff = Container(self.ptycho, ID='Cdiff', data_type='real', data_dims=3)
        self.ptycho._pool['C'].pop('Cmask')
        self.ptycho.mask = Container(self.ptycho, ID='Cmask', data_type='bool', data_dims=3)
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
        if self.p.resample != 1:
            raise NotImplementedError('Diffraction pattern resampling is not supported by Bragg Scan Model')
        self.resample = 1

        # Collect and assemble geometric parameters
        get_keys = ['distance', 'center', 'energy']
        geo_pars = u.Param({key: common[key] for key in get_keys})
        geo_pars.propagation = self.p.propagation
        geo_pars.ffttype = self.p.ffttype
        # take extra Bragg information into account
        psize = tuple(common['psize'])
        geo_pars.psize = (self.ptyscan.common.rocking_step,) + psize
        sh = tuple(common['shape'])
        geo_pars.shape = (self.ptyscan.common.n_rocking_positions,) + sh
        geo_pars.theta_bragg = self.ptyscan.common.theta_bragg

        # make a Geo instance and fix resolution
        g = geometry_bragg.Geo_Bragg(owner=self.ptycho, pars=geo_pars)
        logger.info('Reconstruction will use these geometric parameters:')
        logger.info(g)
        g.p.resolution_is_fix = True
        g.resample = self.resample

        # save the geometry
        self.geometries = [g]

        # Store frame shape
        self.diff_shape = g.shape
        self.probe_shape = self.diff_shape
        self.object_shape = self.diff_shape
        self.exit_shape = self.diff_shape
        self.psize = g.psize

    def _initialize_probe(self, probe_ids):
        """
        Initialize the probe storage referred to by probe_ids.keys()[0]
        """
        logger.info('\n' + headerline('Probe initialization', 'l'))

        # pick storage from container, there's only one probe
        pid = list(probe_ids.keys())[0]
        s = self.ptycho.probe.S.get(pid)
        logger.info('Initializing probe storage %s' % pid)

        # create an oversampled probe perpendicular to its incoming
        # direction, using the illumination module as a utility.
        geo = self.geometries[0]
        extent = max(geo.probe_extent_vs_fov())
        psize = min(geo.resolution) / 5
        shape = int(np.ceil(extent / psize))
        logger.info('Generating incoming probe %d x %d (%.3e x %.3e) with psize %.3e...'
                    % (shape, shape, extent, extent, psize))
        t0 = time.time()

        Cprobe = Container(data_dims=2, data_type='complex')
        Sprobe = Cprobe.new_storage(psize=psize, shape=shape)

        # fill the incoming probe
        illumination.init_storage(Sprobe, self.p.illumination, energy=geo.energy)
        logger.info('...done in %.3f seconds' % (time.time() - t0))

        # Extrude the incoming probe in the right direction and frame
        s.data[:] = geo.prepare_3d_probe(Sprobe, system='natural').data

        s.model_initialized = True


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
        for label, scan_pars in pars.items():
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
        return any(s.data_available for s in list(self.scans.values()))

    def new_data(self):
        """
        Get all new diffraction patterns and create all views and pods
        accordingly.s
        """
        parallel.barrier()

        # Nothing to do if there are no new data.
        if not self.data_available:
            return None


        logger.info('Processing new data.')

        # making sure frames_per_block is defined per rank
        _nframes = self.ptycho.frames_per_block * parallel.size

        # Attempt to get new data
        new_data = []
        for label, scan in self.scans.items():
            if not scan.data_available:
                continue
            else:
                ilog_streamer('%s: loading data for scan %s' %(type(scan).__name__,label))
                prb_ids, obj_ids, pod_ids = dict(), dict(), set()
                nd = scan.new_data(_nframes)
                while nd:
                    new_data.append((label, nd[0]))
                    prb_ids.update(nd[1])
                    obj_ids.update(nd[2])
                    pod_ids = pod_ids.union(nd[3])
                    ilog_streamer('%s: loading data for scan %s (%d diffraction frames, %d PODs, %d probe(s) and %d object(s))' 
                                   %(type(scan).__name__,label, sum([d.shape[0] if l==label else 0 for l,d in new_data]), len(pod_ids), len(prb_ids), len(obj_ids)))
                    nd = scan.new_data(_nframes)
                ilog_newline()

                # Reformatting
                ilog_message('%s: loading data for scan %s (reformatting probe/obj/exit)'  %(type(scan).__name__,label))
                self.ptycho.probe.reformat(True)
                self.ptycho.obj.reformat(True)
                self.ptycho.exit.reformat(True)

                # Initialize probe/object/exit
                ilog_message('%s: loading data for scan %s (initializing probe/obj/exit)'  %(type(scan).__name__,label))
                scan._initialize_probe(prb_ids)
                scan._initialize_object(obj_ids)
                scan._initialize_exit(list(pod_ids))

        return new_data
