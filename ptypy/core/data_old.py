"""
data - Diffraction data access

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
    :license: GPLv2, see LICENSE for details.
"""
if __name__ == "__main__":
    from ptypy import utils as u
    from ptypy import io
    from ptypy import resources
    from ptypy.core import geometry
    from ptypy.utils.verbose import logger, log, headerline
    import numpy as np
    import os
    import h5py
else:
    from .. import utils as u
    from .. import io
    from .. import resources
    from ..utils.verbose import logger, log, headerline
    import geometry
    import numpy as np
    import os
    import h5py

parallel = u.parallel


PTYD = dict(
    chunks={},  # frames, positions
    meta={},  # important to understand data. loaded by every process
    info={},  # this dictionary is not loaded from a ptyd. Mainly for documentation
    common={},
)
""" Basic Structure of a .ptyd datafile """

META = dict(
    label=None,   # label will be set internally 
    experimentID=None, # a unique label of user choice
    version='0.1',
    shape=None,
    psize=None,
    #lam=None,
    energy=None,
    center=None,
    distance = None,
)
GENERIC = dict(
    dfile = None,  # filename (e.g. 'foo.ptyd')
    chunk_format='.chunk%02d',  # Format for chunk file appendix.
    #roi = None,  # 2-tuple or int for the desired fina frame size
    save = None,  # None, 'merge', 'append', 'extlink'
    auto_center = None,  # False: no automatic center,None only  if center is None, True it will be enforced   
    load_parallel = 'data',  # None, 'data', 'common', 'all'
    rebin = None,  # rebin diffraction data
    orientation = None,  # None,int or 3-tuple switch, actions are (transpose, invert rows, invert cols)
    min_frames = 1,  # minimum number of frames of one chunk if not at end of scan
    positions_theory = None,  # Theoretical position list (This input parameter may get deprecated)
    num_frames = None, # Total number of frames to be prepared
    recipe = {},
)
"""Default data parameters. See :py:data:`.scan.data` and a short listing below"""
GENERIC.update(META)

WAIT = 'msg1'
EOS = 'msgEOS'
CODES = {WAIT: 'Scan unfinished. More frames available after a pause',
         EOS: 'End of scan reached'}

__all__ = ['GENERIC','PtyScan','PTYD','PtydScan','MoonFlowerScan']

import warnings
warnings.simplefilter('always', DeprecationWarning)
warnings.warn('This module is deprecated and will be removed from the package on 30/11/16',DeprecationWarning)

class PtyScan(object):
    """\
    PtyScan: A single ptychography scan, created on the fly or read from file.
    BASECLASS
    
    Objectives:
     - Stand alone functionality
     - Can produce .ptyd data formats
     - Child instances should be able to prepare from raw data
     - On-the-fly support in form of chunked data.
     - mpi capable, child classes should not worry about mpi
     
    """

    DEFAULTS = GENERIC

    def __init__(self, pars=None, **kwargs):  # filename='./foo.ptyd',shape=None, save=True):
        """
        Class creation with minimum set of parameters, see DEFAULT dict in core/data.py
        Please note the the class creation does not necessarily load data.
        Call <cls_instance>.initialize() to begin
        """

        # Load default parameter structure
        info = u.Param(self.DEFAULTS.copy())
        info.update(pars)
        info.update(kwargs)

        # validate(pars, '.scan.preparation')

        # Prepare meta data
        self.meta = u.Param(META.copy())

        # Attempt to get number of frames.
        self.num_frames = info.num_frames
        self.min_frames = info.min_frames * parallel.size
        #logger.info('Looking for position information input parameter structure ....\n')
        #if (info.positions_theory is None) and (info.xy is not None):
        #    from ptypy.core import xy
        #    info.positions_theory = xy.from_pars(info.xy)

        if info.positions_theory is not None:
            num = len(info.positions_theory )
            logger.info('Theoretical positions are available. There will be %d frames.' % num)
            logger.info('Any experimental position information will be ignored.')
            logger.info('Former input value of frame number `num_frames` %s is overriden to %d' %(str(self.num_frames),num))
            self.num_frames = num
        """
        # check if we got information on geometry from ptycho
        if info.geometry is not None:
            for k, v in info.geometry.items():
                # FIXME: This is a bit ugly - some parameters are added to info without documentation.
                info[k] = v if info.get(k) is None else None
            # FIXME: This should probably be done more transparently: it is not clear for the user that info.roi has precedence over geometry.N
            if info.roi is None:
                info.roi = u.expect2(info.geometry.N)
        """
        # None for rebin should be allowed, as in "don't rebin".
        if info.rebin is None:
            info.rebin = 1
        
        self.info = info

        # Print a report
        log(4,'Ptypy Scan instance got the following parameters:')
        log(4,u.verbose.report(info))

        # Dump all input parameters as class attributes.
        # FIXME: This duplication of parameters can lead to much confusion...
        # self.__dict__.update(info)

        # Check MPI settings
        lp = str(self.info.load_parallel)
        self.load_common_in_parallel = (lp == 'all' or lp == 'common')
        self.load_in_parallel = (lp == 'all' or lp == 'data')

        # set data chunk and frame counters to zero
        self.framestart = 0
        self.chunknum = 0
        self.start = 0
        self.chunk = None

        # Initialize other instance attributes
        self.common = {}
        self.has_weight2d = None
        self.has_positions = None
        self.dfile = None
        self.save = self.info.save
        
        # copy all values for meta
        for k in self.meta.keys():
            self.meta[k] = self.info[k]
        #self.center = None  # Center will be set later
        #self.roi = self.info.roi #None  # ROI will be set later
        #self.shape = None
        self.orientation = self.info.orientation
        self.rebin = self.info.rebin

        # initialize flags
        self._flags = np.array([0, 0, 0], dtype=int)
        self.is_initialized = False

    def initialize(self):
        """
        Time for some read /write access
        """

        # Prepare writing to file
        if self.info.save is not None:
            # We will create a .ptyd
            self.dfile = self.info.dfile
            if parallel.master:
                if os.path.exists(self.dfile):
                    backup = self.dfile + '.old'
                    logger.warning('File %s already exist. Renamed to %s' % (self.dfile, backup))
                    os.rename(self.dfile, backup)
                # Prepare an empty file with the appropriate structure
                io.h5write(self.dfile, PTYD.copy())
            # Wait for master
            parallel.barrier()

        if parallel.master or self.load_common_in_parallel:
            self.common = self.load_common()
            # FIXME
            # Replace Nones, because we cannot broadcast them later
            # I don't get it, why didn't we allow for a missing key?
            # Also I disagree that we should enforce a 'weight2d' or 
            # positions. The user can very well add them later. 
            self.common = dict([(k,v) if v is not None else (k,np.array([])) for k,v in self.common.items() ])

        # broadcast
        if not self.load_common_in_parallel:
            parallel.bcast_dict(self.common)

        self.common = u.Param(self.common)
        assert 'weight2d' in self.common and 'positions_scan' in self.common
            
        logger.info('\n'+headerline('Analysis of the "common" arrays','l'))
        # Check if weights (or mask) have been loaded by load_common.
        weight2d = self.common.weight2d
        self.has_weight2d = weight2d is not None and len(weight2d)>0
        logger.info('Check for weight or mask,  "weight2d"  .... %s : shape = %s' % (str(self.has_weight2d),str(weight2d.shape)))

            
        # Check if positions have been loaded by load_common
        positions = self.common.positions_scan
        self.has_positions = positions is not None and len(positions)>0
        logger.info('Check for positions, "positions_scan" .... %s : shape = %s' % (str(self.has_positions),str(positions.shape)))


        if self.info.positions_theory  is not None:
            logger.info('Skipping experimental positions `positions_scan`')
        elif self.has_positions:
            # Store positions in the info dictionary
            self.info.positions_scan = positions
            num_pos = len(positions)
            if self.num_frames is None:
                # Frame number was not known. We just set it now.
                logger.info('Scanning positions found. There will be %d frames' % num_pos)
                self.num_frames = num_pos
            else:
                # Frame number was already specified. Maybe we didn't want to use everything?
                if num_pos > self.num_frames:
                    #logger.info('Scanning positions have the same number of points as the theoretical ones (%d).' % num_pos)
                    logger.info('Scanning positions (%d) exceed the desired number of scan points (%d).' % (num_pos,self.num_frames))
                    logger.info('Set `num_frames` to None or to a larger value for more scan points')
                elif num_pos < self.num_frames:
                    logger.info('Scanning positions (%d) are fewer than the desired number of scan points (%d).' % (num_pos,self.num_frames))
                    logger.info('Resetting `num_frames` to lower value')
                    self.num_frames = num_pos
                    #raise RuntimeError('Scanning positions have a number of points (%d) inconsistent with what was previously deduced (%d).'
                    #                    % (num_pos, self.info.num_frames))
        else:
            logger.info('No scanning position have been provided at this stage.')
        
        if self.num_frames is None:
            logger.warning('Number of frames `num_frames` not specified at this stage\n.')
        
        parallel.barrier()
        
        #logger.info('#######  MPI Report: ########\n')
        log(4,u.verbose.report(self.common),True)
        parallel.barrier()
        logger.info(headerline('Analysis done','l')+'\n')

        if self.info.save is not None and parallel.master:
            logger.info('Appending common dict to file %s\n' % self.info.dfile)
            io.h5append(self.info.dfile, common=dict(self.common), info=dict(self.info))
        # wait for master
        parallel.barrier()

        self.is_initialized = True

    def _finalize(self):
        """
        Last actions when Eon-of-Scan is reached
        """
        # maybe do this at end of everything
        if self.info.save is not None and parallel.master:
            io.h5append(self.info.dfile, common=dict(self.common), info=dict(self.info))

    def load_common(self):
        """
        **Overwrite in child class**
        
        Loads arrays that are common and needed for preparation of all 
        other data frames coming in. Any array loaded here and returned 
        in a dict will be distributed afterwards through a broadcoast. 
        It is not intended to load diffraction data in this step.
        
        The main purpose is that there may be common data to all processes, that
        is slow to retrieve or large files, such that if all processes
        attempt to get the data, perfomance will decline or RAM usage 
        may be too large.
                       
        It is a good idea to load dark field, flat field, etc
        Also positions may be handy to load here
        
        If `load_parallel` is set to `all` or common`, this function is 
        executed by all nodes, otherwise the master node executes this
        function and braodcasts the results to other nodes. 
        
        Returns
        -------
        common : dict of numpy arrays
            At least two keys, `weight2d` and `positions_scan` must be 
            given (they can be None). The return dictionary is available
            throught
                    
        Note
        ----
        The return signature of this function is not yet fixed and may 
        get altered in near future. One Option would be to include
        `weight2d` and `positions_scan` as part of the return signature
        and thus fixing them in the Base class.
        """
        weight2d = None if self.info.shape is None else np.ones(u.expect2(self.info.shape), dtype='bool') 
        positions_scan = None if self.num_frames is None else np.indices((self.num_frames, 2)).sum(0)
        return {'weight2d': weight2d,
                'positions_scan': positions_scan}

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
        # take internal counter if not specified
        s = self.framestart if start is None else int(start)

        # check may contain data system access, so we keep it to one process
        if parallel.master:
            self.frames_accessible, eos = self.check(chunksize, start=s)
            if self.num_frames is None and eos is None:
                logger.warning('Number of frames not specified and .check() cannot determine end-of-scan. Aborting..')
                self.abort = True
            self.end_of_scan = (s + self.frames_accessible >= self.num_frames) if eos is None else eos
        # wait for master
        parallel.barrier()
        # communicate result
        self._flags = parallel.bcast(self._flags)
        
        # abort here if the flag was set
        if self.abort:
            raise RuntimeError('Load routine incapable to determine end-of-scan.')
            
        N = self.frames_accessible
        # wait if too few frames are available and we are not at the end
        if N < self.min_frames and not self.end_of_scan:
            return WAIT
        elif self.end_of_scan and N <= 0:
            return EOS
        else:
            # move forward,set new starting point
            self.framestart += N
            return s, N

    def _mpi_indices(self, start, step):
        """
        Funtion to generate the diffraction data index lists that
        determine which node contains which data.
        """
        indices = u.Param()

        # all indices in this chunk of data
        indices.chunk = range(start, start + step)

        # let parallel.loadmanager take care of assigning these indices to nodes
        indices.lm = parallel.loadmanager.assign(indices.chunk)
        # this one contains now a list of indices listed after ranke

        # index list (node specific)
        indices.node = [indices.chunk[i] for i in indices.lm[parallel.rank]]

        # store internally
        self.indices = indices
        return indices

    def get_data_chunk(self, chunksize, start=None):
        """
        This function prepares a container that is compatible to data package
        This function is called from the auto() function.
        """
        msg = self._mpi_check(chunksize, start)
        if msg in [EOS, WAIT]:
            logger.info(CODES[msg])
            return msg
        else:
            start, step = msg

            # get scan point index distribution
            indices = self._mpi_indices(start, step)

            # what did we get?
            data, positions, weights = self._mpi_pipeline_with_dictionaries(indices)
            # all these dictionaries could be empty
            # fill weights dictionary with references to the weights in common
            
            has_data = (len(data) > 0) 
            has_weights = (len(weights) > 0) and len(weights.values()[0])>0

            if has_data:
                dsh = np.array(data.values()[0].shape[-2:])
            else:
                dsh = np.zeros([0, 0])

            # communicate result
            parallel.MPImax(dsh)

            if not has_weights:
                # peak at first item
                if self.has_weight2d:
                    altweight = self.common.weight2d
                else:
                    try:
                        altweight = self.meta.weight2d
                    except:
                        altweight = np.ones(dsh)
                weights = dict.fromkeys(data.keys(), altweight)

            assert len(weights) == len(data), 'Data and Weight frames unbalanced %d vs %d' % (len(data), len(weights))
                       
            sh = self.info.shape
            # adapt roi if not set
            if sh is None:
                logger.info('ROI not set. Using full frame shape of (%d,%d).' % tuple(dsh))
                sh = dsh
            else:
                sh = u.expect2(sh)
                
            # only allow square slices in data
            if sh[0] != sh[1]:
                roi = u.expect2(sh.min())
                logger.warning('Asymmetric data ROI not allowed. Setting ROI from (%d,%d) to (%d,%d)' % (sh[0],sh[1],roi[0],roi[1]))
                sh = roi
            
            self.info.shape = sh

            cen = self.info.center
            if str(cen)==cen:
                cen=geometry.translate_to_pix(sh,cen)
                
            auto = self.info.auto_center 
            # get center in diffraction image
            if auto is None or auto is True:
                auto_cen = self._mpi_autocenter(data, weights)
            else:
                auto_cen = None
                
            if cen is None and auto_cen is not None:
                logger.info('Setting center for ROI from %s to %s.' %(str(cen),str(auto_cen)))
                cen = auto_cen
            elif cen is None and auto is False:
                cen = dsh // 2
            else:
                # center is number or tuple
                cen = u.expect2(cen[-2:])
                if auto_cen is not None:
                    logger.info('ROI center is %s, automatic guess is %s.' %(str(cen),str(auto_cen)))
                    
            # It is important to set center again in order to NOT have different centers for each chunk
            # the downside is that the center may be off if only a few diffraction patterns were
            # used for the analysis. In such case it is beneficial to set the center in the parameters
            self.info.center = cen  

            # make sure center is in the image frame
            assert (cen > 0).all() and (
                dsh - cen > 0).all(), 'Optical axes (center = (%.1f,%.1f) outside diffraction image frame (%d,%d)' % tuple(cen)+tuple(dsh)

            # determine if the arrays require further processing
            do_flip = self.orientation is not None and np.array(self.orientation).any()
            do_crop = (np.abs(sh - dsh) > 0.5).any()
            do_rebin = self.rebin is not None and (self.rebin != 1)

            if do_flip or do_crop or do_rebin:
                logger.info('Enter preprocessing (crop/pad %s, rebin %s, flip/rotate %s) ... \n' %
                            (str(do_crop), str(do_flip), str(do_rebin)))
                # we proceed with numpy arrays. That is probably now
                # more memory intensive but shorter in writing
                if has_data:                       
                    d = np.array([data[ind] for ind in indices.node])
                    w = np.array([weights[ind] for ind in indices.node])
                else:
                    d = np.ones((1,)+tuple(dsh))
                    w = np.ones((1,)+tuple(dsh))    

                # flip, rotate etc.
                d, tmp = u.switch_orientation(d, self.orientation, cen)
                w, cen = u.switch_orientation(w, self.orientation, cen)

                # crop
                d, tmp = u.crop_pad_symmetric_2d(d, sh, cen)
                w, cen = u.crop_pad_symmetric_2d(w, sh, cen)

                # rebin, check if rebinning is neither to strong nor impossible
                rebin = self.rebin
                if rebin<=1:
                    pass
                elif rebin in range(2, 6) and (((self.roi / float(rebin)) % 1) == 0.0).all():
                    mask = w > 0
                    d = u.rebin_2d(d, rebin)
                    w = u.rebin_2d(w, rebin)
                    mask = u.rebin_2d(mask, rebin)
                    w[mask < 1] = 0
                else:
                    raise RuntimeError('Binning (%d) is to large or incompatible with array shape (%s)' % (rebin,str(tuple(sh))))
                                    
                if has_data:
                    # translate back to dictionaries
                    data = dict(zip(indices.node, d))
                    weights = dict(zip(indices.node, w))

            # adapt meta info
            self.meta.center = cen / float(self.rebin)
            self.meta.shape = u.expect2(sh) / self.rebin
            self.meta.psize = u.expect2(self.info.psize) * self.rebin if self.info.psize is not None else None
            
            # prepare chunk of data
            chunk = u.Param()
            chunk.indices = indices.chunk
            chunk.indices_node = indices.node
            chunk.num = self.chunknum
            chunk.data = data

            # if there were weights we add them to chunk, 
            # otherwise we push it into meta
            if has_weights:
                chunk.weights = weights
            elif has_data:
                chunk.weights = {}
                self.meta.weight2d = weights.values()[0]

            # slice positions from common if they are empty too
            if positions is None or len(positions) == 0:
                pt =  self.info.positions_theory
                if pt is not None:
                    chunk.positions = pt[indices.chunk]
                else:
                    try:
                        chunk.positions = self.info.positions_scan[indices.chunk]
                    except:
                        logger.info('Unable to slice position information from experimental or theoretical ressource')
                        chunk.positions = [None]*len(indices.chunk)
            else:
                # a dict : sort positions to indices.chunk
                # this may fail if there a less positions than scan points (unlikely)
                chunk.positions = np.asarray([positions[i] for i in indices.chunk])
                # positions complete

            # with first chunk we update meta
            if self.chunknum < 1:
                """
                for k, v in self.meta.items():
                    # FIXME: I would like to avoid this "blind copying"
                    # BE: This is not a blind copy as only keys in META above are used
                    self.meta[k] = self.__dict__.get(k, self.info.get(k)) if v is None else v
                self.meta['center'] = cen
                """
                
                if self.info.save is not None and parallel.master:
                    io.h5append(self.dfile, meta=dict(self.meta))
                parallel.barrier()

            self.chunk = chunk
            self.chunknum += 1

            return chunk

    def auto(self, frames, chunk_form='dp'):
        """
        Repeated calls to this function will process the data
        
        Parameters
        ----------
        frames : int
            Number of frames to process.
            
        Returns
        -------
        variable
            one of the following
              - None, if scan's end is not reached, but no data could be prepared yet
              - False, if scan's end is reached
              - a data package otherwise
        """
        # attempt to get data:
        msg = self.get_data_chunk(frames)
        if msg == WAIT:
            return msg
        elif msg == EOS:
            # cleanup maybe?
            self._finalize()
            # del self.common
            # del self.chunk
            return msg
        else:
            out = self.return_chunk_as(msg, chunk_form)
            # save chunk
            if self.info.save is not None:
                self._mpi_save_chunk(self.info.save,msg)
            # delete chunk
            del self.chunk
            return out

    def return_chunk_as(self, chunk, kind='dp'):
        """
        Returns the loaded data chunk `chunk` in the format `kind`
        For now only kind=='dp' (data package) is valid.
        """
        # this is a bit ugly now
        if kind != 'dp':
            raise RuntimeError('Unknown kind of chunck format: %s' % str(kind))

        out = {}

        # The "common" part
        out['common'] = self.meta

        # The "iterable" part
        iterables = []
        for pos, index in zip(chunk.positions, chunk.indices):
            frame = {'index': index,
                     'data': chunk.data.get(index),
                     'position': pos}
            if frame['data'] is None:
                frame['mask'] = None
            else:
                # ok we now know that we need a mask since data is not None
                # first look in chunk for a weight to this index, then
                # look for a 2d-weight in meta, then arbitrarily set
                # weight to ones.
                w = chunk.weights.get(index, self.meta.get('weight2d', np.ones_like(frame['data'])))
                frame['mask'] = (w > 0)
            iterables.append(frame)
        out['iterable'] = iterables
        return out

    def _mpi_pipeline_with_dictionaries(self, indices):
        """
        example processing pipeline using dictionaries.
        
        return :
            positions, data, weights
             -- Dictionaries. Keys are the respective scan point indices
                `positions` and `weights` may be empty. If so, the information
                is taken from the self.common dictionary
        
        """
        if self.load_in_parallel:
            # all nodes load raw_data and slice according to indices
            raw, pos, weights = self.load(indices=indices.node)

            # gather postion information as every node needs it later
            pos = parallel.gather_dict(pos)
        else:
            if parallel.master:
                raw, pos, weights = self.load(indices=indices.chunk)
            else:
                raw = {}
                pos = {}
                weights = {}
            # distribute raw data across nodes according to indices
            raw = parallel.bcast_dict(raw, indices.node)
            weights = parallel.bcast_dict(weights, indices.node)

        # (re)distribute position information - every node should now be 
        # aware of all positions
        parallel.bcast_dict(pos)

        # prepare data across nodes    
        data, weights = self.correct(raw, weights, self.common)

        return data, pos, weights

    def check(self, frames=None, start=None):
        """
        **Overwrite in child class**
        
        This method checks how many frames the preparation routine may
        process, starting from frame `start` at a request of `frames_requested`.
        
        This method is supposed to return the number of accessible frames
        for preparation and should determine if data acquistion for this
        scan is finished.
        
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
            is one of the following:
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
        **Overwrite in child class**
        
        Loads data according to node specific scanpoint indeces that have 
        been determined by the loadmanager from utils.parallel or otherwise
    
        Returns
        -------
        raw, positions, weight : dict
            Dictionaries whose keys are the given scan point `indices` 
            and whose values are the respective frame / position according 
            to the scan point index. `weight` and `positions` may be empty
        """
        # dummy fill
        raw = dict((i, i * np.ones(u.expect2(self.info.shape))) for i in indices)
        return raw, {}, {}

    def correct(self, raw, weights, common):
        """
        **Overwrite in child class**
        
        Place holder for dark and flatfield correction. If :any:`load` 
        already provides data in the form of photon counts, and no frame
        specific weight is needed, this method may be left as is
        
        May get *merged* with :any:`load` in future.
    
        Returns
        -------
        data,weights : dict
            Flat and dark-corrected data dictionaries. These dictionaries
            must have the same keys as the input `raw` and contain 
            corrected frames (`data`) and statistical weights (`weights`)
            which are zero for invalid or masked pixel other the number
            of detector counts that correspond to one photon count
        """
        # c=dict(indices=None,data=None,weight=None)
        data = raw
        return data, weights

    def _mpi_autocenter(self, data, weights):
        """
        Calculates the frame center across all nodes.
        Data and weights are dicts of the same length and different on each node
        """
        cen = dict([(k, u.mass_center(d * (weights[k] > 0))) for k, d in data.iteritems()])
        # for some nodes, cen may still be empty. Therefore we use gather_dict to be save
        cen = parallel.gather_dict(cen)
        parallel.barrier()
        # now master possesses all calculated centers
        if parallel.master:
            cen = np.array(cen.values()).mean(0)
        else:
            cen = np.array([0., 0.])
            # print cen
        parallel.allreduce(cen)
        return cen

    def report(self, what=None, shout=True):
        """
        Make a report on internal structure
        """
        what = what if what is not None else self.__dict__
        msg = u.verbose.report(what)
        if shout:
            logger.info(msg, extra={'allprocesses': True})
        else:
            return msg

    def _mpi_save_chunk(self, kind='link', chunk=None):
        """
        Saves data chunk to hdf5 file specified with `dfile`
        It works by gathering weights and data to the master node.
        Master node then writes to disk.
        
        In case you support parallel hdf5 writing, please modify this 
        function to suit your installation.
        
        2 out of 3 modes currently supported
        
        kind : 'merge','append','link'
            
            'append' : appends chunks of data in same file
            'link' : saves chunks in seperate files and adds ExternalLinks
            
        TODO: 
            * For the 'link case, saving still requires access to
              main file `dfile` even so for just adding the link.
              This may result in conflict if main file is polled often
              by seperate read process.
              Workaraound would be to write the links on startup in
              initialise() 
            
        """
        # gather all distributed dictionary data.
        c = chunk if chunk is not None else self.chunk
        
        # shallow copy
        todisk = dict(c)
        num = todisk.pop('num')
        ind = todisk.pop('indices_node')
        for k in ['data', 'weights']:
            if k in c.keys():
                if hasattr(c[k], 'iteritems'):
                    v = c[k]
                else:
                    v = dict(zip(ind, np.asarray(c[k])))
                parallel.barrier()
                # gather the content
                newv = parallel.gather_dict(v)
                todisk[k] = np.asarray([newv[i] for i in sorted(newv.keys())])

        parallel.barrier()
        # all information is at master node.
        if parallel.master:

            # form a dictionary
            if str(kind) == 'append':
                h5address = 'chunks/%d' % num
                io.h5append(self.dfile, {h5address: todisk})

            elif str(kind) == 'link':
                import h5py
                h5address = 'chunks/%d' % num
                hddaddress = self.dfile + '.part%03d' % num
                with h5py.File(self.dfile) as f:
                    f[h5address] = h5py.ExternalLink(hddaddress, '/')
                    f.close()
                io.h5write(hddaddress, todisk)
            elif str(kind) == 'merge':
                raise NotImplementedError('Merging all data into single chunk is not yet implemented')
        parallel.barrier()
        
class PtydScan(PtyScan):
    """
    PtyScan provided by native "ptyd" file format.
    """
    DEFAULT = GENERIC.copy()
    
    def __init__(self, pars=None, source=None,**kwargs):
        """
        PtyScan provided by native "ptyd" file format.
        
        :param source: Explicit source file. If not None or 'file', 
                       the data may get processed depending on user input
                       
        :param pars: Input like PtyScan
        """
        # create parameter set
        p = u.Param(self.DEFAULT.copy())

        # copy the label
        #if pars is not None:
        #    p.label = pars.get('label')
        
        if source is None or str(source)=='file':
            # this is the case of absolutely no additional work
            logger.info('No explicit source file was given. Will continue read only')
            source = pars['dfile']
            manipulate = False
        elif pars is None or len(pars)==0:
            logger.info('No parameters provided. Saving / modification disabled')
            manipulate = False
        else:
            logger.info('Explicit source file given. Modification is possible\n')
            dfile = pars['dfile']
            # check for conflict
            if str(u.unique_path(source))==str(u.unique_path(dfile)):
                logger.info('Source and Sink files are the same.')
                dfile = os.path.splitext(dfile)
                dfile = dfile[0] +'_n.'+ dfile[1]
                logger.info('Will instead save to %s if necessary.' % os.path.split(dfile)[1])
            
            pars['dfile']= dfile
            manipulate = True
            p.update(pars)
            
        
        # make sure the source exists.
        assert os.path.exists(u.unique_path(source)), 'Source File (%s) not found' % source
        
        self.source = source
        
        # get meta information
        meta = u.Param(io.h5read(self.source, 'meta')['meta'])
        
        # update given parameters when they are None
        if not manipulate:
            super(PtydScan, self).__init__(meta,  **kwargs)
        else:
            # overwrite only those set to None
            for k, v in meta.items():
                if p.get(k) is None:
                    p[k] = v
            # Initialize parent class and fill self
            super(PtydScan, self).__init__(p, **kwargs)
        
        # enforce that orientations are correct
        # Other instance attributes
        self._checked = {}
        self._ch_frame_ind = None


    def check(self, frames=None, start=None):
        """
        Implementation of the check routine for a .ptyd file format
        
        See also
        --------
        Ptyscan.check
        """

        if start is None:
            start = self.framestart
        if frames is None:
            frames = self.min_frames

        # Get info about size of currently available chunks.
        # Dead external links will produce None and are excluded
        with h5py.File(self.source, 'r') as f:
            d = {}
            chitems = sorted([(int(k), v) for k, v in f['chunks'].iteritems() if v is not None], key=lambda t: t[0])
            for chkey in chitems[0][1].keys():
                d[chkey] = np.array([((int(k),) + v[chkey].shape) for k, v in chitems if v is not None])
            f.close()

        self._checked = d
        allframes = int(sum([ch[1] for ch in d['data']]))
        self._ch_frame_ind = np.array([(dd[0], frame) for dd in d['data'] for frame in range(dd[1])])

        return min((frames, allframes - start)), allframes < start+frames

    def _coord_to_h5_calls(self, key, coord):
        return 'chunks/%d/%s' % (coord[0], key), slice(coord[1], coord[1] + 1)

    def load_common(self):
        """
        In ptyd, 'common' must exist
        """
        # this total buggy right now
        return {'weight2d' : self.info.weight2d, 'positions_scan' : None}

    def load(self, indices):
        """
        Load from ptyd. Due to possible chunked data, slicing frames is 
        non-trivial
        """
        # ok we need to communicate the some internal info
        parallel.barrier()
        self._ch_frame_ind=parallel.bcast(self._ch_frame_ind)
        parallel.barrier()
        parallel.bcast_dict(self._checked)
        
        # get the coordinates in the chunks
        coords = self._ch_frame_ind[indices]
        calls = {}
        for key in self._checked.keys():
            calls[key] = [self._coord_to_h5_calls(key, c) for c in coords]

        # get our data from the ptyd file
        out = {}
        with h5py.File(self.source, 'r') as f:
            for array, call in calls.iteritems():
                out[array] = [np.squeeze(f[path][slce]) for path, slce in call]
            f.close()

        # if the chunk provided indices, we use those instead of our own
        # Dangerous and not yet implemented
        # indices = out.get('indices',indices)

        # wrap in a dict 
        for k, v in out.iteritems():
            out[k] = dict(zip(indices, v))

        return (out.get(key, {}) for key in ['data', 'positions', 'weights'])


class MoonFlowerScan(PtyScan):
    """
    Test Ptyscan class producing a romantic ptychographic dataset of a moon
    illuminating flowers.
    """
    
    DEFAULT = GENERIC.copy().update(geometry.DEFAULT.copy())
    
    def __init__(self, pars = None, **kwargs):
        """
        Parent pars are for the 
        """
        p = geometry.DEFAULT.copy()
        if pars is not None:
            p.update(pars)
        # Initialize parent class
        super(MoonFlowerScan, self).__init__(p, **kwargs)
        
        from ptypy import resources
        from ptypy.core import xy
        
        # derive geometry from input
        G = geometry.Geo(pars = self.meta)
        #G._initialize(self.meta)
        
        # derive scan pattern
        pos = u.Param()
        pos.spacing = G.resolution * G.shape / 5.
        pos.layers = np.int(np.round(np.sqrt(self.num_frames)))+1
        pos.extent = pos.layers * pos.spacing
        pos.model = 'round'
        pos.count = self.num_frames
        self.pos = xy.from_pars(pos)

        # calculate pixel positions
        pixel = self.pos / G.resolution
        pixel -= pixel.min(0)
        self.pixel = np.round(pixel).astype(int) + 10
        frame = self.pixel.max(0) + 10 + G.shape
        self.G = G
        #from matplotlib import pyplot as plt
        #plt.figure(200);plt.imshow(u.imsave(G.propagator.pre_ifft))
        #plt.figure(101);plt.imshow(G.propagator.grids_det[0]**2+G.propagator.grids_det[1]**2)
        # get object
        self.obj = resources.flower_obj(frame) 
        
        # get probe
        moon = resources.moon_pr(self.G.shape)
        moon /= np.sqrt(u.abs2(moon).sum() / 1e8)
        self.pr = moon
        self.load_common_in_parallel = True
        
    def load_common(self):
        """
        Transmit positions
        """
        return {'weight2d': np.ones(self.pr.shape),
                'positions_scan': self.pos}

    def load(self, indices):
        """
        Forward propagation
        """
        # dummy fill
        p=self.pixel
        s=self.G.shape
        raw = {}
        for i in indices:
            raw[i]=np.random.poisson(u.abs2(self.G.propagator.fw(self.pr * self.obj[p[i][0]:p[i][0]+s[0],p[i][1]:p[i][1]+s[1]]))).astype(np.int32)
        return raw, {}, {}

        

class DataSource(object):
    """
    A source of data for ptychographic reconstructions. The important method is "feed_data", which returns
    packets of diffraction patterns with their meta-data.
    """
    def __init__(self, scans, frames_per_call=10000000, feed_format='dp'):
        """
        DataSource initialization.

        scans: a dictionnary of scan structures.
        frames_per_call: (optional) number of frames to load in one call. By default, load as many as possible.
        feed_format: the format in with the data is packaged. For now only 'dp' is implemented.
        """

        from ..experiment import PtyScanTypes

        self.frames_per_call = frames_per_call
        self.feed_format = feed_format
        self.scans = scans

        # sort after keys given
        self.labels = sorted(scans.keys())

        # empty list for the scans
        self.PS = []

        for label in self.labels:
            # we are making a copy of the root as we want to fill it
            scan = scans[label]
            s = scan['pars']

            # Copy other relevant information
            prep = s.data.copy()
            
            # Assign label
            prep.label = label
            
            source = prep.source
            recipe = prep.get('recipe',{})
            if prep.get('positions_theory') is None:
                prep.positions_theory = scan['pos_theory']
            
            prep.dfile = s.data_file
            #prep.geometry = s.geometry.copy()
            #prep.xy = s.xy.copy()
            
            if source is not None:
                source = source.lower()

            if source in PtyScanTypes:
                PS = PtyScanTypes[source]
                logger.info('Scan %s will be prepared with the recipe "%s"' % (label, source))
                self.PS.append(PS(prep, recipe= recipe))
            elif source.endswith('.ptyd') or source.endswith('.pty') or str(source)=='file':
                self.PS.append(PtydScan(prep, source=source))
            elif source=='test':
                self.PS.append(MoonFlowerScan(prep))
            elif source=='sim':
                from ..simulations import SimScan
                logger.info('Scan %s will simulated' % (label))
                self.PS.append(SimScan(prep,s.copy()))
            elif source=='empty' or source is None:
                prep.recipe = None
                logger.warning('Generating dummy PtyScan for scan `%s` - This label will source only zeros as data' % label)
                self.PS.append(PtyScan(prep))
            else:
                raise RuntimeError('Could not manage source "%s" for scan `%s`' % (str(source),label))

        # Initialize flags
        self.scan_current = -1
        self.data_available = True
        self.scan_total = len(self.PS)

    @property
    def scan_available(self):
        return self.scan_current < (self.scan_total - 1)

    def feed_data(self):
        """
        Yield data packages.
        """
        # get PtyScan instance to scan_number
        PS = self.PS[self.scan_current]
        label = self.labels[self.scan_current]
        
        # initialize if that has not been done yet
        if not PS.is_initialized:
            PS.initialize()

        msg = PS.auto(self.frames_per_call, self.feed_format)
        # if we catch a scan that has ended look for an unfinished scan
        while msg == EOS and self.scan_available:
            self.scan_current += 1
            PS = self.PS[self.scan_current]
            label = self.labels[self.scan_current]
            if not PS.is_initialized:
                PS.initialize()
            msg = PS.auto(self.frames_per_call, self.feed_format)

        self.data_available = (msg != EOS or self.scan_available)

        logger.debug(u.verbose.report(msg))
        if msg != WAIT and msg != EOS:
            # ok that would be a data package
            # attach inner label
            msg['common']['ptylabel'] = label
            #print u.verbose.report(msg)
            logger.info('Feeding data chunk')
            return msg
        else:
            return None

if __name__ == "__main__":
    u.verbose.set_level(3)
    MS = MoonFlowerScan(num_frames=100)
    MS.initialize()
    for i in range(50):
        msg = MS.auto(10)
        logger.info(u.verbose.report(msg), extra={'allprocesses': True})
        parallel.barrier()

