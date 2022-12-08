"""
data - Diffraction data access.

This module defines a PtyScan, a container to hold the experimental
data of a ptychography scan. Instrument-specific reduction routines should
inherit PtyScan to prepare data for the Ptycho Instance in a uniform format.

The experiment specific child class only needs to overwrite 2 functions
of the base class:

For the moment the module contains two main objects:
PtyScan, which holds a single ptychography scan, and DataSource, which
holds a collection of datascans and feeds the data as required.

This file is part of the PTYPY package.

    :copyright: Copyright 2014 by the PTYPY team, see AUTHORS.
    :license: see LICENSE for details.
"""
import numpy as np
import os
import h5py
from . import geometry
from . import xy
from .. import utils as u
from .. import io
from .. import resources
from ..utils import parallel
from ..utils.verbose import logger, log, headerline
from .. import defaults_tree

PTYD = dict(
    # frames, positions
    chunks={},
    # important to understand data. loaded by every process
    meta={},
    # this dictionary is not loaded from a ptyd. Mainly for documentation
    info={},
)
""" Basic Structure of a .ptyd datafile """


WAIT = 'msg1'
EOS = 'msgEOS'
CODES = {WAIT: 'Scan unfinished. More frames available after a pause',
         EOS: 'End of scan reached'}

__all__ = ['PtyScan', 'PTYD', 'PtydScan',
           'MoonFlowerScan']


@defaults_tree.parse_doc('scandata.PtyScan')
class PtyScan(object):
    """
    PtyScan: A single ptychography scan, created on the fly or read from file.

    *BASECLASS*

    Objectives:
     - Stand alone functionality
     - Can produce .ptyd data formats
     - Child instances should be able to prepare from raw data
     - On-the-fly support in form of chunked data.
     - mpi capable, child classes should not worry about mpi

    Default data parameters. See :py:data:`.scan.data`

    Defaults:

    [name]
    default = PtyScan
    type = str
    help =

    [dfile]
    type = file
    default = None
    help = File path where prepared data will be saved in the ``ptyd`` format.
    userlevel = 0

    [chunk_format]
    type = str
    default = .chunk%02d
    help = Appendix to saved files if save == 'link'
    doc =
    userlevel = 2

    [save]
    type = str
    default = None
    help = Saving mode
    doc = Mode to use to save data to file.
        <newline>
        - ``None``: No saving
        - ``'merge'``: attemts to merge data in single chunk **[not implemented]**
        - ``'append'``: appends each chunk in master \\*.ptyd file
        - ``'link'``: appends external links in master \\*.ptyd file and stores chunks separately
        <newline>
        in the path given by the link. Links file paths are relative to master file.
    userlevel = 1

    [auto_center]
    type = bool
    default = None
    help = Determine if center in data is calculated automatically
    doc =  
       - ``False``, no automatic centering
       - ``None``, only if :py:data:`center` is ``None``
       - ``True``, it will be enforced
    userlevel = 0

    [load_parallel]
    type = str
    default = data
    help = Determines what will be loaded in parallel
    doc = Choose from ``None``, ``'data'``, ``'common'``, ``'all'``

    [rebin]
    type = int
    default = None
    help = Rebinning factor
    doc = Rebinning factor for the raw data frames. ``'None'`` or ``1`` both mean *no binning*
    userlevel = 1
    lowlim = 1
    uplim = 8

    [orientation]
    type = int, tuple, list
    default = None
    help = Data frame orientation
    doc = Choose
       <newline> 
       - ``None`` or ``0``: correct orientation
       - ``1``: invert columns (numpy.flip_lr)
       - ``2``: invert rows  (numpy.flip_ud)
       - ``3``: invert columns, invert rows
       - ``4``: transpose (numpy.transpose)
       - ``4+i``: tranpose + other operations from above
       <newline>
       Alternatively, a 3-tuple of booleans may be provided ``(do_transpose, 
       do_flipud, do_fliplr)``
    userlevel = 1

    [min_frames]
    type = int
    default = 1
    help = Minimum number of frames loaded by each node
    doc =
    userlevel = 2
    lowlim = 1

    [positions_theory]
    type = ndarray
    default = None
    help = Theoretical positions for this scan
    doc = If provided, experimental positions from :py:class:`PtyScan` subclass will be ignored. If data
      preparation is called from Ptycho instance, the calculated positions from the
      :py:func:`ptypy.core.xy.from_pars` dict will be inserted here
    userlevel = 2

    [num_frames]
    type = int
    default = None
    help = Maximum number of frames to be prepared
    doc = If `positions_theory` are provided, num_frames will be ovverriden with the number of
      positions available
    userlevel = 1

    [label]
    type = str
    default = None
    help = The scan label
    doc = Unique string identifying the scan
    userlevel = 1

    [experimentID]
    type = str
    default = None
    help = Name of the experiment
    doc = If None, a default value will be provided by the recipe. **unused**
    userlevel = 2

    [version]
    type = float
    default = 0.1
    help = TODO: Explain this and decide if it is a user parameter.
    doc =
    userlevel = 2

    [shape]
    type = int, tuple
    default = 256
    help = Shape of the region of interest cropped from the raw data.
    doc = Cropping dimension of the diffraction frame
      Can be None, (dimx, dimy), or dim. In the latter case shape will be (dim, dim).
    userlevel = 1

    [center]
    type = list, tuple, str
    default = 'fftshift'
    help = Center (pixel) of the optical axes in raw data
    doc = If ``None``, this parameter will be set by :py:data:`~.scan.data.auto_center` or elsewhere
    userlevel = 1

    [psize]
    type = float, tuple
    default = 0.000172
    help = Detector pixel size
    doc = Dimensions of the detector pixels (in meters)
    userlevel = 0
    lowlim = 0

    [distance]
    type = float
    default = 7.19
    help = Sample to detector distance
    doc = In meters.
    userlevel = 0
    lowlim = 0

    [energy]
    type = float
    default = 7.2
    help = Photon energy of the incident radiation in keV
    doc =
    userlevel = 0
    lowlim = 0

    [add_poisson_noise]
    default = False
    type = bool
    help = Decides whether the scan should have poisson noise or not
    """

    WAIT = WAIT
    EOS = EOS
    CODES = CODES

    METAKEYS = ['version', 'num_frames', 'label', 'shape', 'psize', 'energy', 'center', 'distance']
    """ Keys to store in meta param """

    def __init__(self, pars=None, **kwargs):
        # filename='./foo.ptyd', shape=None, save=True):
        """
        Class creation with minimum set of parameters, see :py:data:`PtyScan.DEFAULT`
        Please note that class creation is not meant to load data.

        Call :py:data:`initialize` to begin loading and data file creation.
        """
        # Load default parameter structure
        p = self.DEFAULT.copy(99)
        p.update(pars)
        p.update(kwargs)

        # Attempt to get number of frames.
        self.num_frames = p.num_frames
        """ Total number of frames to prepare / load.
            Set by :py:data:`~.scan.data.num_frames` """

        self.min_frames = p.min_frames * parallel.size
        """ Minimum number of frames to prepare / load
            with call of :py:meth:`auto` """

        if p.positions_theory is not None:
            num = len(p.positions_theory)
            logger.info('Theoretical positions are available. '
                        'There will be %d frames.' % num)
            logger.info(
                'Any experimental position information will be ignored.')
            logger.info(
                'Former input value of frame number `num_frames` %s is '
                'overridden to %d.' % (str(self.num_frames), num))
            self.num_frames = num

        # None for rebin should be allowed, as in "don't rebin".
        if p.rebin is None:
            p.rebin = 1

        self.info = p
        """:py:class:`Param` container that stores all input parameters."""

        # Print a report
        log(4, 'Ptypy Scan instance got the following parameters:')
        log(4, u.verbose.report(p))

        # Check MPI settings
        lp = str(self.info.load_parallel)
        self.load_common_in_parallel = (lp == 'all' or lp == 'common')
        self.load_in_parallel = (lp == 'all' or lp == 'data')

        # Set data chunk and frame counters to zero
        self.framestart = 0
        self.chunknum = 0
        self.start = 0
        self.chunk = None

        # Initialize other instance attributes
        self.common = {}
        self.has_weight2d = None
        self.weight2d = None
        self.has_positions = None
        self.dfile = None
        self.save = self.info.save

        # Construct meta
        self.meta = u.Param({k: self.info[k] for k in self.METAKEYS})

        self.orientation = self.info.orientation
        self.rebin = self.info.rebin

        # Initialize flags
        self._flags = np.array([0, 0, 0], dtype=int)
        self.is_initialized = False

    def initialize(self):
        """
        Begins the Data preparation and intended as the first method
        that does read-write access on (large) data. Does the following:

        * Creates a \\*.ptyd data file at location specified by
          :py:data:`dfile` (master node only)
        * Calls :py:meth:`load_weight`, :py:meth:`load_positions`
          :py:meth:`load_common` (master node only for
          ``load_parallel==None`` or ``load_parallel=='data'``)
        * Sets :py:attr:`num_frames` if needed
        * Calls :py:meth:`post_initialize`
        """
        logger.info(headerline('Enter %s.initialize()'
                               % self.__class__.__name__, 'l'))

        # Prepare writing to file
        if self.info.save is not None:
            # We will create a .ptyd
            self.dfile = self.info.dfile
            if parallel.master:
                if os.path.exists(self.dfile):
                    backup = self.dfile + '.old'
                    logger.warning('File %s already exist. Renamed to %s.'
                                   % (self.dfile, backup))
                    try:
                        # on windows, os.rename doesn't work if target exists
                        os.remove(backup)
                    except:
                        pass
                    os.rename(self.dfile, backup)
                # Prepare an empty file with the appropriate structure
                io.h5write(self.dfile, PTYD.copy())
            # Wait for master
            parallel.barrier()

        if parallel.master or self.load_common_in_parallel:
            positions = self.load_positions()
            self.weight2d = self.load_weight()
            self.common = self.load_common()
        else:
            positions = None

        # Broadcast
        if not self.load_common_in_parallel:
            positions = parallel.bcast(positions)
            self.weight2d = parallel.bcast(self.weight2d)
            self.common = parallel.bcast(self.common)

        parallel.barrier()
        self.common = u.Param(self.common)

        # Check for weight2d and positions
        self.has_weight2d = (self.weight2d is not None and
                             len(self.weight2d) > 0)
        self.has_positions = positions is not None and len(positions) > 0

        # Output information on weight and positions
        logger.info('Common weight : '.rjust(29) + str(self.has_weight2d))
        if self.has_weight2d:
            logger.info('shape = '.rjust(29) + str(self.weight2d.shape))

            # FIXME: Saving weight to info. This is not ideal, info is optional
            self.info.weight2d = self.has_weight2d

        logger.info('All experimental positions : ' + str(self.has_positions))
        if self.has_positions:
            logger.info('shape = '.rjust(29) + str(positions.shape))

        if self.info.positions_theory is not None:
            logger.info('Skipping experimental positions `positions_scan`')
        elif self.has_positions:
            # Store positions in the info dictionary
            self.info.positions_scan = positions
            num_pos = len(positions)
            if self.num_frames is None:
                # Frame number was not known. We just set it now.
                logger.info('Scanning positions found. There will be %d frames.'
                            % num_pos)
                self.num_frames = num_pos
            else:
                # Frame number was already specified.
                # Maybe we didn't want to use everything?
                if num_pos > self.num_frames:
                    # logger.info('Scanning positions have the same number of'
                    #             'points as the theoretical ones (%d).'
                    #             % num_pos)
                    logger.info('Scanning positions (%d) exceed the desired '
                                'number of scan points (%d).'
                                % (num_pos, self.num_frames))
                    logger.info('Set `num_frames` to None or to a larger value '
                                'for more scan points.')
                elif num_pos < self.num_frames:
                    logger.info('Scanning positions (%d) are fewer than the '
                                'desired number of scan points (%d).'
                                % (num_pos, self.num_frames))
                    logger.info('Resetting `num_frames` to lower value.')
                    self.num_frames = num_pos
                    # raise RuntimeError(
                    #     'Scanning positions have a number of points (%d) '
                    #     'inconsistent with what was previously deduced (%d).'
                    #     % (num_pos, self.info.num_frames))
        else:
            logger.info(
                'No scanning position have been provided at this stage.')

        # Warn that the total number of frames is unknown here.
        # The .check() method must now determine the end of the scan.
        if self.num_frames is None:
            logger.warning(
                'Number of frames `num_frames` not specified at this stage.')

        # A note about how much this scan class knows about the number
        # of frames expected. PtydScan uses this information.
        self.meta.num_frames = self.num_frames
        parallel.barrier()
        """
        #logger.info('#######  MPI Report: ########\n')
        log(4,u.verbose.report(self.common), True)
        parallel.barrier()
        logger.info(headerline('Analysis done',' l') + '\n')
        """

        if self.info.save is not None and parallel.master:
            logger.info('Appending info dict to file %s\n' % self.info.dfile)
            io.h5append(self.info.dfile, info=dict(self.info))
        # Wait for master
        parallel.barrier()

        self.is_initialized = True
        self.post_initialize()
        logger.info(headerline('Leaving PtyScan.initialize()', 'l'))

    def _finalize(self):
        """
        Last actions when Eon-of-Scan is reached
        """
        # Maybe do this at end of everything
        if self.info.save is not None and parallel.master:
            io.h5append(self.info.dfile, info=dict(self.info))

    def load_weight(self):
        """
        **Override in subclass for custom implementation**

        *Called in* :py:meth:`initialize`

        Loads a common (2d)-weight for all diffraction patterns. The weight
        loaded here will be available by all processes through the
        attribute ``self.weight2d``. If a *per-frame-weight* is specified
        in :py:meth:`load` , this function has no effect.

        The purpose of this function is to avoid reloading and parallel
        reads. If that is not critical to the implementation,
        reimplementing this function in a subclass can be ignored.

        If `load_parallel` is set to `all` or common`, this function is
        executed by all nodes, otherwise the master node executes this
        function and broadcasts the results to other nodes.

        Returns
        -------
        weight2d : ndarray
            A two-dimensional array with a shape compatible to the raw
            diffraction data frames

        Note
        ----
        For now, weights will be converted to a mask,
        ``mask = weight2d > 0`` for use in reconstruction algorithms.
        It is planned to use a general weight instead of a mask in future
        releases.
        """
        if self.info.shape is None:
            return None
        else:
            return np.ones(u.expect2(self.info.shape), dtype='bool')

    def load_positions(self):
        """
        **Override in subclass for custom implementation**

        *Called in* :py:meth:`initialize`

        Loads all positions for all diffraction patterns in this scan.
        The positions loaded here will be available by all processes
        through the attribute ``self.positions``. If you specify position
        on a per frame basis in :py:meth:`load` , this function has no
        effect.

        If theoretical positions :py:data:`positions_theory` are
        provided in the initial parameter set :py:data:`DEFAULT`,
        specifying positions here has NO effect and will be ignored.

        The purpose of this function is to avoid reloading and parallel
        reads on files that may require intense parsing to retrieve the
        information, e.g. long SPEC log files. If parallel reads or
        log file parsing for each set of frames is not a time critical
        issue of the subclass, reimplementing this function can be ignored
        and it is recommended to only reimplement the :py:meth:`load`
        method.

        If `load_parallel` is set to `all` or common`, this function is
        executed by all nodes, otherwise the master node executes this
        function and broadcasts the results to other nodes.

        Returns
        -------
        positions : ndarray
            A (N,2)-array where *N* is the number of positions.

        Note
        ----
        Be aware that this method sets attribute :py:attr:`num_frames`
        in the following manner.

        * If ``num_frames == None`` : ``num_frames = N``.
        * If ``num_frames < N`` , no effect.
        * If ``num_frames > N`` : ``num_frames = N``.

        """
        if self.num_frames is None:
            return None
        else:
            return np.indices((self.num_frames, 2)).sum(0)

    def load_common(self):
        """
        **Override in subclass for custom implementation**

        *Called in* :py:meth:`initialize`

        Loads anything and stores that in a dict. This dict will be
        available to all processes after :py:meth:`initialize` through
        the attribute :py:attr:`common`

        The purpose of this method is the same as :py:meth:`load_weight`
        and :py:meth:`load_positions` except for that the contents
        of :py:attr:`common` have no built-in effect of the behavior in
        the processing other than the user specifies it in py:meth:`load`

        If `load_parallel` is set to `all` or common`, this function is
        executed by all nodes, otherwise the master node executes this
        function and broadcasts the results to other nodes.

        Returns
        -------
        common : dict

        """
        return {}

    def post_initialize(self):
        """
        Placeholder. Called at the end of :py:meth:`initialize` by all
        processes.

        Use this method to benefit from 'hard-to-retrieve but now available'
        information after initialize.
        """
        pass

    def _mpi_check(self, chunksize, start=None):
        """
        Executes the check() function on master node and communicates
        the result with the other nodes.
        This function determines if the end of the scan is reached
        or if there is more data after a pause.

        returns:
            - codes WAIT or EOS
            - or (start, frames) if data can be loaded
        """
        # Take internal counter if not specified
        s = self.framestart if start is None else int(start)

        # Check may contain data system access, so we keep it to one process
        if parallel.master:
            self.frames_accessible, eos = self.check(chunksize, start=s)
            if self.num_frames is None and eos is None:
                logger.warning('Number of frames not specified and .check()'
                               'cannot determine end-of-scan. Aborting..')
                self.abort = True

            if eos is None:
                self.end_of_scan = (s + self.frames_accessible
                                    >= self.num_frames)
            else:
                self.end_of_scan = eos

        # Wait for master
        parallel.barrier()
        # Communicate result
        self._flags = parallel.bcast(self._flags)

        # Abort here if the flag was set
        if self.abort:
            raise RuntimeError(
                'Load routine incapable to determine the end-of-scan.')

        frames_accessible = self.frames_accessible
        # Wait if too few frames are available and we are not at the end
        if frames_accessible < self.min_frames and not self.end_of_scan:
            return WAIT
        elif self.end_of_scan and frames_accessible <= 0:
            return EOS
        else:
            # Move forward, set new starting point
            self.framestart += frames_accessible
            return s, frames_accessible

    def _mpi_indices(self, start, step):
        """
        Function to generate the diffraction data index lists that
        determine which node contains which data.
        """
        indices = u.Param()

        # All indices in this chunk of data
        indices.chunk = list(range(start, start + step))

        # Let parallel.loadmanager take care of assigning indices to nodes
        indices.lm = parallel.loadmanager.assign(indices.chunk)
        # This one contains now a list of indices listed after rank

        # Index list (node specific)
        indices.node = [indices.chunk[k] for k in indices.lm[parallel.rank]]

        # Store internally
        self.indices = indices

        return indices

    def get_data_chunk(self, chunksize, start=None):
        """
        This function prepares a container that is compatible to data package.

        This function is called from the auto() function.
        """
        msg = self._mpi_check(chunksize, start)
        if msg in [EOS, WAIT]:
            logger.info(CODES[msg])
            return msg

        start, step = msg

        # Get scan point index distribution
        indices = self._mpi_indices(start, step)

        # What did we get?
        data, positions, weights = self._mpi_pipeline_with_dictionaries(
            indices)
        # All these dictionaries could be empty
        # Fill weights dictionary with references to the weights in common

        has_data = (len(data) > 0)
        has_weights = (len(weights) > 0) and len(list(weights.values())[0]) > 0

        if has_data:
            dsh = np.array(list(data.values())[0].shape[-2:])
        else:
            dsh = np.array([0, 0])

        # Communicate result
        dsh[0] = parallel.MPImax([dsh[0]])
        dsh[1] = parallel.MPImax([dsh[1]])

        if not has_weights:
            # Peak at first item
            if self.has_weight2d:
                altweight = self.weight2d
            else:
                try:
                    altweight = self.info.weight2d
                except:
                    altweight = np.ones(dsh)
            weights = dict.fromkeys(data.keys(), altweight)

        assert len(weights) == len(data), (
            'Data and Weight frames unbalanced %d vs %d'
            % (len(data), len(weights)))

        sh = self.info.shape
        # Adapt roi if not set
        if sh is None:
            logger.info('ROI not set. Using full frame shape of (%d, %d).'
                        % tuple(dsh))
            sh = dsh
        else:
            sh = u.expect2(sh)

        # Only allow square slices in data
        if sh[0] != sh[1]:
            roi = u.expect2(sh.min())
            logger.warning('Asymmetric data ROI not allowed. Setting ROI '
                           'from (%d, %d) to (%d, %d).'
                           % (sh[0], sh[1], roi[0], roi[1]))
            sh = roi

        self.info.shape = sh

        cen = self.info.center
        if isinstance(cen, str):
            cen = geometry.translate_to_pix(dsh, cen)

        auto = self.info.auto_center
        # Get center in diffraction image
        if auto is None or auto is True:
            auto_cen = self._mpi_autocenter(data, weights)
        else:
            auto_cen = None

        if cen is None and auto_cen is not None:
            logger.info('Setting center for ROI from %s to %s.'
                        % (str(cen), str(auto_cen)))
            cen = auto_cen
        elif cen is None and auto is False:
            cen = dsh // 2
        else:
            # Center is number or tuple
            cen = u.expect2(cen[-2:])
            if auto_cen is not None:
                logger.info('ROI center is %s, automatic guess is %s.'
                            % (str(cen), str(auto_cen)))

        # It is important to set center again in order to NOT have
        # different centers for each chunk, the downside is that the center
        # may be off if only a few diffraction patterns are used for the
        # analysis. In such case it is beneficial to set the center in the
        # parameters
        self.info.center = cen

        # Make sure center is in the image frame
        assert (cen > 0).all() and (dsh - cen > 0).all(), (
            'Optical axes (center = (%.1f, %.1f) outside diffraction image '
            'frame (%d, %d).' % (tuple(cen) + tuple(dsh)))

        # Determine if the arrays require further processing
        do_flip = (self.orientation is not None
                   and np.array(self.orientation).any())
        do_crop = (np.abs(sh - dsh) > 0.5).any()
        do_rebin = self.rebin is not None and (self.rebin != 1)

        if do_flip or do_crop or do_rebin:
            logger.info(
                'Enter preprocessing '
                '(crop/pad %s, rebin %s, flip/rotate %s) ... \n'
                % (str(do_crop), str(do_rebin), str(do_flip)))

            # We proceed with numpy arrays.That is probably now more memory
            # intensive but shorter in writing
            if has_data:
                d = np.array([data[ind] for ind in indices.node])
                w = np.array([weights[ind] for ind in indices.node])
            else:
                d = np.ones((1,) + tuple(dsh))
                w = np.ones((1,) + tuple(dsh))

            # Crop data
            d, tmp = u.crop_pad_symmetric_2d(d, sh, cen)

            # Check if provided mask has the same shape as data, if not,
            # use the mask's center for cropping the mask. The latter is
            # also needed if no mask is provided, as weights is then
            # created using the requested cropping shape and thus might
            # have a different center than the raw data.
            # NOTE: Maybe distinguish between no mask provided and mask
            # with wrong size in warning

            if (dsh == np.array(w[0].shape)).all():
                w, cen = u.crop_pad_symmetric_2d(w, sh, cen)
            else:
                logger.warning('Mask does not have the same shape as data. '
                               'Will use mask center for cropping mask.')
                cen = np.array(w[0].shape) // 2
                w, cen = u.crop_pad_symmetric_2d(w, sh, cen)

            # Flip, rotate etc.
            d, tmp = u.switch_orientation(d, self.orientation, cen)
            w, cen = u.switch_orientation(w, self.orientation, cen)

            # Rebin, check if rebinning is neither to strong nor impossible
            rebin = self.rebin
            if rebin <= 1:
                pass
            elif (rebin in range(2, 6)
                  and (((sh / float(rebin)) % 1) == 0.0).all()):
                mask = w > 0
                d = u.rebin_2d(d, rebin)
                w = u.rebin_2d(w, rebin)
                mask = u.rebin_2d(mask, rebin)
                # We keep only the pixels that do not include a masked pixel
                # w[mask < mask.max()] = 0
                # TODO: apply this operation when weights actually are weights
                w = (mask == mask.max())
            else:
                raise RuntimeError(
                    'Binning (%d) is to large or incompatible with array '
                    'shape (%s).' % (rebin, str(tuple(sh))))

            # restore contiguity of the cropped/padded/rotated/flipped array
            d = np.ascontiguousarray(d)
            w = np.ascontiguousarray(w)

            if has_data:
                # Translate back to dictionaries
                data = dict(zip(indices.node, d))
                weights = dict(zip(indices.node, w))

        # Adapt geometric info
        self.meta.center = cen / float(self.rebin)
        self.meta.shape = u.expect2(sh) // self.rebin

        if self.info.psize is not None:
            self.meta.psize = u.expect2(self.info.psize) * self.rebin

        # Prepare chunk of data
        chunk = u.Param()
        chunk.indices = indices.chunk
        chunk.indices_node = indices.node
        chunk.num = self.chunknum
        chunk.data = data
        
        # chunk now always has weights
        chunk.weights = weights
        
        # If there are weights we add them to chunk,
        # otherwise we push it into meta
        #if has_weights:
        #    chunk.weights = weights
        #elif has_data:
        #    chunk.weights = {}
        #    self.weight2d = list(weights.values())[0]

        # Slice positions from common if they are empty too
        if positions is None or len(positions) == 0:
            pt = self.info.positions_theory
            if pt is not None:
                chunk.positions = pt[indices.chunk]
            else:
                try:
                    chunk.positions = (
                        self.info.positions_scan[indices.chunk])
                except:
                    logger.info('Unable to slice position information from '
                                'experimental or theoretical resource.')
                    chunk.positions = [None] * len(indices.chunk)
        else:
            # A dict : sort positions to indices.chunk
            # This may fail if there are less positions than scan points
            # (unlikely)
            chunk.positions = np.asarray(
                [positions[k] for k in indices.chunk])
            # Positions complete

        # With first chunk we update info
        if self.chunknum < 1:
            if self.info.save is not None and parallel.master:
                io.h5append(self.dfile, meta=dict(self.meta))

            parallel.barrier()

        self.chunk = chunk
        self.chunknum += 1

        return chunk

    def auto(self, frames):
        """
        Repeated calls to this function will process the data.

        Parameters
        ----------
        frames : int
            Number of frames to process.

        Returns
        -------
        variable
            one of the following
              - WAIT, if scan's end is not reached,
                but no data could be prepared yet
              - EOS, if scan's end is reached
              - a data package otherwise
        """
        # attempt to get data:
        msg = self.get_data_chunk(frames)
        if msg == WAIT:
            return msg
        elif msg == EOS:
            # Cleanup maybe?
            self._finalize()
            # del self.common
            # del self.chunk
            return msg
        else:
            out = self._make_data_package(msg)
            # save chunk
            if self.info.save is not None:
                self._mpi_save_chunk(self.info.save, msg)
            # delete chunk
            del self.chunk
            return out

    def _make_data_package(self, chunk):
        """
        Returns the loaded data chunk `chunk` as a data package.
        """

        # The "common" part
        out = {'common': self.meta}

        # The "raw" part. Might replace the iterable in future.
        out['chunk'] = chunk
        
        # The "iterable" part
        iterables = []
        for pos, index in zip(chunk.positions, chunk.indices):
            frame = {'index': index,
                     'data': chunk.data.get(index),
                     'position': pos}

            if frame['data'] is None:
                frame['mask'] = None
            else:
                # Ok, we now know that we need a mask since data is not None
                # First look in chunk for a weight to this index, then
                # look for a 2d-weight in meta, then arbitrarily set
                # weight to ones.
                try:
                    fallback = self.weight2d
                except AttributeError:
                    fallback = np.ones_like(frame['data'])
                w = chunk.weights.get(index, fallback)
                frame['mask'] = (w > 0)

            iterables.append(frame)

        out['iterable'] = iterables

        return out

    def _mpi_pipeline_with_dictionaries(self, indices):
        """
        Example processing pipeline using dictionaries.

        return :
            positions, data, weights
             -- Dictionaries. Keys are the respective scan point indices
                `positions` and `weights` may be empty. If so, the information
                is taken from the self.common dictionary

        """
        if self.load_in_parallel:
            # All nodes load raw_data and slice according to indices
            raw, pos, weights = self.load(indices=indices.node)

            # Gather position information as every node needs it later
            pos = parallel.gather_dict(pos)
        else:
            if parallel.master:
                raw, pos, weights = self.load(indices=indices.chunk)
            else:
                raw = {}
                pos = {}
                weights = {}
            # Distribute raw data across nodes according to indices
            raw = parallel.bcast_dict(raw, indices.node)
            weights = parallel.bcast_dict(weights, indices.node)

        # (re)distribute position information - every node should now be
        # aware of all positions
        pos = parallel.bcast_dict(pos)

        # Prepare data across nodes
        data, weights = self.correct(raw, weights, self.common)

        return data, pos, weights

    def check(self, frames=None, start=None):
        """
        **Override in subclass for custom implementation**

        This method checks how many frames the preparation routine may
        process, starting from frame `start` at a request of `frames`.

        This method is supposed to return the number of accessible frames
        for preparation and should determine if data acquisition for this
        scan is finished. Its main purpose is to allow for a data
        acquisition scheme, where the number of frames is not known
        when :py:class:`PtyScan` is constructed, i.e. a data stream or an
        on-the-fly reconstructions.

        Note
        ----
        If :py:data:`num_frames` is set on ``__init__()`` of the subclass,
        this method can be left as it is.

        Parameters
        ----------
        frames : int or None
            Number of frames requested.
        start : int or None
            Scanpoint index to start checking from.

        Returns
        -------
        frames_accessible : int
            Number of frames readable.

        end_of_scan : int or None
            is one of the following,
            - 0, end of the scan is not reached
            - 1, end of scan will be reached or is
            - None, can't say

        """
        if start is None:
            start = self.framestart

        if frames is None:
            frames = self.min_frames

        frames_accessible = min((frames, self.num_frames - start))

        return frames_accessible, None

    @property
    def end_of_scan(self):
        return not (self._flags[1] == 0)

    @end_of_scan.setter
    def end_of_scan(self, eos):
        self._flags[1] = int(eos)

    @property
    def frames_accessible(self):
        return self._flags[0]

    @frames_accessible.setter
    def frames_accessible(self, frames):
        self._flags[0] = frames

    @property
    def abort(self):
        return not (self._flags[2] == 0)

    @abort.setter
    def abort(self, abort):
        self._flags[2] = int(abort)

    def load(self, indices):
        """
        **Override in subclass for custom implementation**

        Loads data according to node specific scanpoint indices that have
        been determined by :py:class:`LoadManager` or otherwise.

        Returns
        -------
        raw, positions, weight : dict
            Dictionaries whose keys are the given scan point `indices`
            and whose values are the respective frame / position according
            to the scan point index. `weight` and `positions` may be empty

        Note
        ----
        This is the *most* important method to change when subclassing
        :py:class:`PtyScan`. Most often it suffices to override the constructor
        and this method to create a subclass suited for a specific
        experiment.
        """
        # Dummy fill
        raw = {}
        for k in indices:
            raw[k] = k * np.ones(u.expect2(self.info.shape))

        return raw, {}, {}

    def correct(self, raw, weights, common):
        """
        **Override in subclass for custom implementation**

        Place holder for dark and flatfield correction. If :any:`load`
        already provides data in the form of photon counts, and no frame
        specific weight is needed, this method may be left as it is.

        May get *merged* with :any:`load` in future.

        Returns
        -------
        data, weights : dict
            Flat and dark-corrected data dictionaries. These dictionaries
            must have the same keys as the input `raw` and contain
            corrected frames (`data`) and statistical weights (`weights`)
            which are zero for invalid or masked pixel other the number
            of detector counts that correspond to one photon count
        """
        # c = dict(indices=None, data=None, weight=None)
        data = raw
        return data, weights

    def _mpi_autocenter(self, data, weights):
        """
        Calculates the frame center across all nodes.

        Data and weights are dicts of the same length and different on each
        node.
        """
        cen = {}
        for k, d in data.items():
            cen[k] = u.mass_center(d * (weights[k] > 0))

        # For some nodes, cen may still be empty.
        # Therefore we use gather_dict to be safe
        cen = parallel.gather_dict(cen)
        parallel.barrier()

        # Now master possesses all calculated centers
        if parallel.master:
            cen = np.array(list(cen.values())).mean(0)
        cen = parallel.bcast(cen)

        return cen

    def report(self, what=None, shout=True):
        """
        Make a report on internal structure.
        """
        what = what if what is not None else self.__dict__
        msg = u.verbose.report(what)

        if shout:
            logger.info(msg, extra={'allprocesses': True})
        else:
            return msg

    def _mpi_save_chunk(self, kind='link', chunk=None):
        """
        Saves data chunk to hdf5 file specified with `dfile`.

        It works by gathering weights and data to the master node.
        Master node then writes to disk.

        In case you support parallel hdf5 writing, please modify this
        function to suit your installation.

        2 out of 3 modes currently supported

        kind : 'merge','append','link'

            'append' : appends chunks of data in same file
            'link' : saves chunks in separate files and adds ExternalLinks

        TODO:
            * For the 'link case, saving still requires access to
              main file `dfile` even so for just adding the link.
              This may result in conflict if main file is polled often
              by separate read process.
              Workaround would be to write the links on startup in
              initialise()

        """
        # Gather all distributed dictionary data.
        c = chunk if chunk is not None else self.chunk

        # Shallow copy
        todisk = dict(c)
        num = todisk.pop('num')
        ind = todisk.pop('indices_node')

        for k in ['data', 'weights']:
            if k in c.keys():
                if hasattr(c[k], 'items'):
                    v = c[k]
                else:
                    v = dict(zip(ind, np.asarray(c[k])))

                parallel.barrier()
                # Gather the content
                newv = parallel.gather_dict(v)
                todisk[k] = np.asarray([newv[j] for j in sorted(newv.keys())])

        parallel.barrier()

        # All information is at master node.
        if parallel.master:
            # Form a dictionary
            if str(kind) == 'append':
                h5address = 'chunks/%d' % num
                io.h5append(self.dfile, {h5address: todisk})
            elif str(kind) == 'link':
                h5address = 'chunks/%d' % num
                hddaddress = self.dfile + '.part%03d' % num
                io.h5write(hddaddress, todisk)

                with h5py.File(self.dfile, 'a') as f:
                    f[h5address] = h5py.ExternalLink(hddaddress, '/')
                    f.close()

            elif str(kind) == 'merge':
                raise NotImplementedError('Merging all data into single chunk '
                                          'is not yet implemented.')
        parallel.barrier()


@defaults_tree.parse_doc('scandata.PtydScan')
class PtydScan(PtyScan):
    """
    PtyScan provided by native "ptyd" file format.

    Defaults:

    [name]
    default = PtydScan
    type = str
    help =
    doc =

    [dfile]
    type = file
    default = None
    help = Prepared data file path
    doc = If source is ``None`` or ``'file'``, data will be loaded from this file and processing as
      well as saving is deactivated. If source is the path to a file, data will be saved to this file.
    userlevel = 0

    [source]
    default = 'file'
    type = str, None
    help = Alternate source file path if data is meant to be reprocessed.
    doc = `None` for input shall be deprecated in future

    """

    def __init__(self, pars=None, **kwargs):
        """
        PtyScan provided by native "ptyd" file format.
        """
        # Create parameter set
        p = self.DEFAULT.copy(99)
        p.update(pars)
        p.update(kwargs)
        source = p.source

        if source is None or str(source) == 'file':
            # This is the case of absolutely no additional work
            logger.info('No source file was given. '
                        'Using dfile as read-only source.')
            source = pars['dfile']
            manipulate = False
        elif pars is None or len(pars) == 0:
            logger.info('No parameters provided. '
                        'Saving / modification disabled.')
            manipulate = False
        else:
            logger.info('Explicit source file given. '
                        'Modification is possible.\n')
            dfile = pars.get('dfile')

            # Check for conflict
            if dfile and (str(u.unique_path(source)) == str(u.unique_path(dfile))):
                logger.info('Source and Sink files are the same.')
                dfile = os.path.splitext(dfile)
                dfile = dfile[0] + '_n.' + dfile[1]
                logger.info('Will instead save to %s if necessary.'
                            % os.path.split(dfile)[1])

            pars['dfile'] = dfile
            manipulate = True
            p.update(pars)

        # Make sure the source exists.
        assert os.path.exists(u.unique_path(source)), (
            'Source File (%s) not found' % source)

        self.source = source

        # At least ONE chunk must exist to ensure everything works
        with h5py.File(source, 'r') as f:
            check = f.get('chunks/0')
            f.close()
            # Get number of frames supposedly in the file
            # FIXME: try/except clause only for backward compatibilty
            # for .ptyd files created priot to commit 2e626ff
            #try:
            #    source_frames = f.get('info/num_frames_actual')[...].item()
            #except TypeError:
            #    source_frames = len(f.get('info/positions_scan')[...])
            #f.close()

        if check is None:
            raise IOError('Ptyd source %s contains no data. Load aborted'
                          % source)

        """
        if source_frames is None:
            logger.warning('Ptyd source is not aware of the total'
                           'number of diffraction frames expected')
        """

        # Get meta information
        meta = u.Param(io.h5read(self.source, 'meta')['meta'])

        if meta.get('num_frames') is None:
            logger.warning('Ptyd source is not aware of the total'
                           'number of diffraction frames expected')

        if len(meta) == 0:
            logger.warning('There should be meta information in '
                           '%s. Something is odd here.' % source)

        # Apply parameters from ptyd file.
        p.update(meta)

        if manipulate:
            # Override parameters that are explicitly passed as input arguments
            for k in self.METAKEYS:
                if k in pars:
                    p[k] = pars[k]

        super(PtydScan, self).__init__(p)

        """
        if source_frames is not None:
            if self.num_frames is None:
                self.num_frames = source_frames
            elif self.num_frames > source_frames:
                self.num_frames = source_frames
        else:
            # Ptyd source doesn't know its total number of frames
            # but we cannot do anything about it. This should be dealt
            # with with a flag in the meta package probably.
            pass
        """

        # Other instance attributes
        self._checked = {}
        self._ch_frame_ind = None

    def check(self, frames=None, start=None):
        """
        Implementation of the check routine for a .ptyd file format.

        See also
        --------
        PtyScan.check
        """
        if start is None:
            start = self.framestart

        if frames is None:
            frames = self.min_frames

        # Get info about size of currently available chunks.
        # Dead external links will produce None and are excluded
        with h5py.File(self.source, 'r') as f:
            d = {}
            ch_items = []
            for k, v in f['chunks'].items():
                if v is not None:
                    ch_items.append((int(k), v))

            ch_items = sorted(ch_items, key=lambda t: t[0])

            for ch_key in ch_items[0][1].keys():
                d[ch_key] = np.array([(int(k),) + v[ch_key].shape
                                      for k, v in ch_items if v is not None])

            f.close()

        self._checked = d
        all_frames = int(sum([ch[1] for ch in d['data']]))
        ch_frame_ind = []
        for dd in d['data']:
            for frame in range(dd[1]):
                ch_frame_ind.append((dd[0], frame))

        self._ch_frame_ind = np.array(ch_frame_ind)

        # Accessible frames
        frames_accessible = min((frames, all_frames - start))
        # end_of_scan = source_frames <= start + frames_accessible
        return frames_accessible, None

    def _coord_to_h5_calls(self, key, coord):
        return 'chunks/%d/%s' % (coord[0], key), slice(coord[1], coord[1] + 1)

    def load_weight(self):
        if 'weight2d' in self.info:
            return self.info.weight2d
        else:
            return None

    def load_positions(self):
        return None

    def load(self, indices):
        """
        Load from ptyd.

        Due to possible chunked data, slicing frames is non-trivial.
        """
        # Ok we need to communicate the some internal info
        parallel.barrier()
        self._ch_frame_ind = parallel.bcast(self._ch_frame_ind)
        parallel.barrier()
        self._checked = parallel.bcast_dict(self._checked)

        # Get the coordinates in the chunks
        coords = self._ch_frame_ind[indices]
        calls = {}

        for key in self._checked.keys():
            calls[key] = [self._coord_to_h5_calls(key, c) for c in coords]

        # Get our data from the ptyd file
        out = {}
        with h5py.File(self.source, 'r') as f:
            for array, call in calls.items():
                out[array] = [np.squeeze(f[path][slce]) for path, slce in call]

            f.close()

        # If the chunk provided indices, we use those instead of our own
        # Dangerous and not yet implemented
        # indices = out.get('indices', indices)

        # Wrap in a dict
        for k, v in out.items():
            out[k] = dict(zip(indices, v))

        return (out.get(key, {}) for key in ['data', 'positions', 'weights'])


@defaults_tree.parse_doc('scandata.MoonFlowerScan')
class MoonFlowerScan(PtyScan):
    """
    Test PtyScan class producing a romantic ptychographic data set of a moon
    illuminating flowers.

    Override parent class default:

    Defaults:

    [name]
    default = MoonFlowerScan
    type = str
    help =
    doc =

    [num_frames]
    default = 100
    type = int
    help = Number of frames to simulate
    doc =

    [shape]
    type = int, tuple
    default = 128
    help = Shape of the region of interest cropped from the raw data.
    doc = Cropping dimension of the diffraction frame
      Can be None, (dimx, dimy), or dim. In the latter case shape will be (dim, dim).
    userlevel = 1

    [density]
    default = 0.2
    type = float
    help = Position distance in fraction of illumination frame

    [model]
    default = 'round'
    type = str
    help = The scan pattern

    [photons]
    default = 1e8
    type = float
    help = Total number of photons for Poisson noise

    [psf]
    default = 0.
    type = float
    help = Point spread function of the detector
    
    [add_poisson_noise]
    default = True
    type = bool
    help = Decides whether the scan should have poisson noise or not

    [block_wait_count]
    default = 0
    type = int
    help = Signals a WAIT to the model after this many blocks.

    """

    def __init__(self, pars=None, **kwargs):
        """
        Parent pars are for the
        """

        p = self.DEFAULT.copy(depth=99)
        p.update(pars)

        # Initialize parent class
        super(MoonFlowerScan, self).__init__(p, **kwargs)

        # Derive geometry from input
        geo = geometry.Geo(pars=self.meta)

        # Derive scan pattern
        if p.model == 'raster':
            pos = u.Param()
            pos.spacing = geo.resolution * geo.shape * p.density
            pos.steps = int(np.round(np.sqrt(self.num_frames))) + 1
            pos.extent = pos.steps * pos.spacing
            pos.model = p.model
            self.num_frames = pos.steps**2
            pos.count = self.num_frames
            self.pos = xy.raster_scan(pos.spacing[0], pos.spacing[1], pos.steps, pos.steps)
        else:
            pos = u.Param()
            pos.spacing = geo.resolution * geo.shape * p.density
            pos.steps = int(np.round(np.sqrt(self.num_frames) + 1))
            pos.extent = pos.steps * pos.spacing
            pos.model = p.model
            pos.count = self.num_frames
            self.pos = xy.from_pars(pos)

        # Calculate pixel positions
        pixel = self.pos / geo.resolution
        pixel -= pixel.min(0)
        self.pixel = np.round(pixel).astype(int) + 10
        frame = self.pixel.max(0) + 10 + geo.shape
        self.geo = geo
        
        try:
            self.obj = resources.flower_obj(frame)
        except:
            # matplotlib failsafe
            self.obj = u.gf_2d(u.parallel.MPInoise2d(frame),1.)
            
        # Get probe
        try:
            moon = resources.moon_pr(self.geo.shape)
        except:
            # matplotlib failsafe
            moon = u.ellipsis(u.grids(self.geo.shape)).astype(complex)
        
        moon /= np.sqrt(u.abs2(moon).sum() / p.photons)
        self.pr = moon
        self.load_common_in_parallel = True

        self._check_called = 0
        self.p = p

    def check(self, frames=None, start=None):
        frames_accessible, eos = super().check(frames, start)
        self._check_called += 1

        bwc = self.p.block_wait_count
        if bwc >=1 and self._check_called % (bwc+1) == 0:
            frames_accessible = 0

        return frames_accessible, eos

    def load_positions(self):
        return self.pos

    def load_weight(self):
        return np.ones(self.pr.shape)

    def load(self, indices):
        p = self.pixel
        s = self.geo.shape
        raw = {}

        if self.p.add_poisson_noise:
            logger.info("Generating data with poisson noise.")
        else:
            logger.info("Generating data without poisson noise.")

        for k in indices:
            intensity_j = u.abs2(self.geo.propagator.fw(
                self.pr * self.obj[p[k][0]:p[k][0] + s[0],
                                   p[k][1]:p[k][1] + s[1]]))

            if self.p.psf > 0.:
                intensity_j = u.gf(intensity_j, self.p.psf)

            if self.p.add_poisson_noise:
                raw[k] = np.random.poisson(intensity_j).astype(np.int32)
            else:
                raw[k] = intensity_j.astype(np.int32)


        return raw, {}, {}


@defaults_tree.parse_doc('scandata.QuickScan')
class QuickScan(PtyScan):
    """
    Test PtyScan to benchmark graph creation further down the line.

    Override parent class default:

    Defaults:

    [name]
    default = MoonFlowerScan
    type = str
    help =
    doc =

    [num_frames]
    default = 100
    type = int
    help = Number of frames to simulate
    doc =

    [shape]
    type = int, tuple
    default = 64
    help = Shape of the region of interest cropped from the raw data.
    doc = Cropping dimension of the diffraction frame
      Can be None, (dimx, dimy), or dim. In the latter case shape will be (dim, dim).
    userlevel = 1

    [density]
    default = 0.05
    type = float
    help = Position distance in fraction of illumination frame
    """

    def __init__(self, pars=None, **kwargs):
        """
        Parent pars are for the
        """

        p = self.DEFAULT.copy(depth=99)
        p.update(pars)

        # Initialize parent class
        super(QuickScan, self).__init__(p, **kwargs)

        # Derive geometry from input
        geo = geometry.Geo(pars=self.meta)

        # Derive scan pattern
        pos = u.Param()
        pos.spacing = geo.resolution * geo.shape * p.density
        pos.steps = int(np.round(np.sqrt(self.num_frames))) + 1
        pos.extent = pos.steps * pos.spacing
        pos.model = 'round'
        pos.count = self.num_frames
        self.pos = xy.from_pars(pos)
        self.geo = geo
        # dumm diff pattern
        self.diff = np.zeros(geo.shape)
        self.diff[0,0] = 1e7
        self.diff = np.fft.fftshift(self.diff)
        
        self.p = p

    def load_positions(self):
        return self.pos

    def load_weight(self):
        return np.ones(self.diff.shape)

    def load(self, indices):
        raw = {}
        for k in indices:
            raw[k] = self.diff.copy().astype(np.int32)
        return raw, {}, {}


if __name__ == "__main__":
    u.verbose.set_level(3)
    MS = MoonFlowerScan(num_frames=100)
    MS.initialize()
    for i in range(50):
        msg = MS.auto(10)
        logger.info(u.verbose.report(msg), extra={'allprocesses': True})
        parallel.barrier()
