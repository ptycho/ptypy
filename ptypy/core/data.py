"""
data - diffraction data access

This module defines a PtyScan, a container to hold the experimental 
data of a ptychography scan. Instrument-specific reduction routines should
inherit PtyScan to prepare data for the Ptycho Instance in a uniform format.

The experiment specific child class only needs to overwrite 2 functions
of the base class:

check(self,frames,start):

    returns :
        (frames_available, end_of_scan)
        - the number of frames available from a starting point `start`
        - bool if the end of scan was reached (None if this routine doesn't know)

load(self,indices):
    loads data according to node specific scanpoint indeces that have 
    been determined by the loadmanager from utils.parallel
    returns :
        (raw, positions, weight)
        - dictionaries whose keys are `indices` and who hold the respective
          frame / position according to the scan point index
        - weight and positions may be empty

load_common(self):
    loads common arrays for all processes. excuted once on initialize()
    Especially scan point position and correction arrays like dark field, mask 
    and flat field are handy to be loaded here
    The returned dictionary is available throughout preparation at 
    self.common
    
    returns :
        (common)
        - a dictionary with only numpy arrays as values. These arrays are 
          the same among all processes. 
        - fill items `positions_scan` and `wieght2d` with the 
        
correct(self,raw,weights,common):
    place holder for dark and flat_field correction. 
    may get merged with load
    use common for correction 
    
    returns:
        (data,weights)
        - corrected data and weights
        
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
    import numpy as np
    import os
    import h5py
else:
    from .. import utils as u
    from .. import io
    import numpy as np
    import os
    import h5py

parallel = u.parallel
logger = u.verbose.logger

META = dict(
    label_original=None,
    label=None,
    version='0.1',
    shape=None,
    psize_det=None,
    lam=None,
    energy=None,
)

PTYD = dict(
    data={},  # frames, positions
    meta={},  # important to understand data. loaded by every process
    info={},  # this dictionary is not loaded from a ptyd. Mainly for documentation
    common={},
)

GENERIC = dict(
    filename=None,  # filename (e.g. 'foo.ptyd')
    label=None,  # a scan label (e.g. 'scan0001')
    chunk_format='.chunk%02d',  # Format for chunk file appendix.
    num_frames=None,  # frame size of saved ptyd or load. Can be None.
    roi=None,  # 2-tuple or int for the desired fina frame size
    save=None,  # None, 'merge', 'append', 'extlink'
    center=None,  # 2-tuple, 'auto', None. If 'auto', center is chosen automatically
    load_parallel='data',  # None, 'data', 'common', 'all'
    rebin=1,  # rebin diffraction data
    orientation=(False, False, False),  # 3-tuple switch, actions are (transpose, invert rows, invert cols)
    min_frames=parallel.size,  # minimum number of frames of one chunk if not at end of scan
    positions_theory=None # Theoretical position list
)

WAIT = 'msg1'
EOS = 'msgEOS'
CODES = {WAIT: 'Scan unfinished. More frames available after a pause',
         EOS: 'End of scan reached'}


class PtyScan(object):
    """\
    PtyScan: A single ptychography scan, created on the fly or read from file.
    BASECLASS
    
    Objectives:
     - Stand alone functionality
     - Can be produce .ptyd data formats
     - Child instances should be able to prepare from raw data
     - On-the-fly support in form of chunked data.
     - mpi capable, child classes should not worry about mpi
     
    """

    def __init__(self, pars=None, **kwargs):  # filename='./foo.ptyd',shape=None, save=True):
        """
        Class creation with minimum set of parameters, see DEFAULT dict in core/data.py
        Please note the the class creation does not necessarily load data.
        Call <cls_instance>.initialize() to begin
        """

        # Load default parameter structure
        info = u.Param(GENERIC.copy())
        info.update(pars)
        info.update(kwargs)

        # validate(pars, '.scan.preparation')

        # Prepare meta data
        self.meta = u.Param(META.copy())

        # Attempt to get number of frames.
        if info.num_frames is None:
            logger.info('Total number of frames for scan {0.label} not specified.'.format(info))
            logger.info('Looking for position information input parameter structure ....\n')

            # Check if we got position information in the parameters
            if info.get('positions_theory') is None:

                if info.get('pattern') is not None:
                    from ptypy.core import pattern

                    info.positions_theory = pattern.from_pars(info['pattern'])

                elif info.get('xy') is not None:
                    from ptypy.core import xy

                    info.positions_theory = xy.from_pars(info['xy'])

            if info.get('positions_theory') is not None:
                num = len(info.positions_theory)
                logger.info('%d positions found. Setting frame count to this number\n' % num)
                info.num_frames = len(info.positions_theory)

                # check if we got information on geometry from ptycho
        if info.get('geometry') is not None:
            for k, v in info.geometry.items():
                info[k] = v if info.get(k) is None else None
            info.roi = u.expect2(info.geometry.get('N')) if info.roi is None else info.roi

        # None for rebin should be allowed, as in "don't rebin".
        if info.rebin is None:
            info.rebin = 1

        self.info = info
        # update internal dict    

        # Print a report
        logger.info('Ptypy Scan instance got the following parameters:\n')
        logger.info(u.verbose.report(info))

        # Dump all input parameters as class attributes.
        # FIXME: This can lead to much confusion...
        self.__dict__.update(info)

        # check mpi settings
        lp = str(self.load_parallel)
        self.load_common_in_parallel = (lp == 'all' or lp == 'common')
        self.load_in_parallel = (lp == 'all' or lp == 'data')

        # set data chunk and frame counters to zero
        self.framestart = 0
        self.chunknum = 0
        self.start = 0

        # initialize flags
        self._flags = np.array([0, 0, 0], dtype=int)
        self.is_initialized = False

    def initialize(self):
        """
        Time for some read /write access
        """
        filename = self.filename
        if self.save is not None:
            # ok we want to create a .ptyd
            if parallel.master:
                if os.path.exists(filename):
                    backup = filename + '.old'
                    logger.warning('File %s already exist. Renamed to %s' % (filename, backup))
                    os.rename(filename, backup)
                # Prepare an empty file with the appropriate structure
                io.h5write(filename, PTYD.copy())
            # Wait for master
            parallel.barrier()

        self.common = self.load_common() if (parallel.master or self.load_common_in_parallel) else {}
        # broadcast
        if not self.load_common_in_parallel:
            parallel.bcast_dict(self.common)
        logger.info('\n ---------- Analysis of the "common" arrays  ---------- \n')

        # Check if weights (or mask) have been loaded by load_common.
        self.has_weight2d = self.common.has_key('weight2d')
        logger.info('Check for weight or mask,  "weight2d"  .... %s\n' % str(self.has_weight2d))

        # Check if positions have been loaded by load_common
        positions = self.common.get('positions_scan')
        self.has_positions = positions is not None
        logger.info('Check for positions, "positions_scan" .... %s' % str(self.has_positions))
        if self.has_positions:
            # Store positions in the info dictionary
            self.info['positions_scan'] = positions
            num_pos = len(positions)
            if self.num_frames is None:
                # Frame number was not known. We just set it now.
                logger.info('Setting number of frames for preparation from `None` to %d\n' % num_pos)
                self.num_frames = num_pos
            else:
                # Frame number was already specified. Maybe we didn't want to use everything?
                logger.warning('Going to prepare %d frames of %d\n' % (self.num_frames, num_pos))
                # FIXME: what is self.num_frames > num_pos?

        parallel.barrier()
        logger.info('#######  MPI Report: ########\n')
        self.report(what=self.common)
        parallel.barrier()
        logger.info(' ----------  Analyis done   ---------- \n\n')

        if self.save is not None and parallel.master:
            logger.info('Appending common dict to file %s\n' % self.filename)
            io.h5append(self.filename, common=self.common, info=self.info)
        # wait for master
        parallel.barrier()

        self.is_initialized = True

    def _finalize(self):
        """
        Last actions when Eon-of-Scan is reached
        """
        # maybe do this at end of everything
        if self.save is not None and parallel.master:
            io.h5append(self.filename, common=dict(self.common), info=dict(self.info))

    def load_common(self):
        """
        ! Overwrite in child class for custom behavior !
        
        Loads arrays that are common and needed for preparation of all 
        other data frames coming in. Any array loaded here and returned 
        in a dict will be distributed afterwards through a broadcoast. 
        It is not intended to load diffraction data in this step.
                       
        Main purpose ist to load dark field, flat field, etc
        Also positions may be handy to load here
        
        If `load_parallel` is set to `all` or common`, this function is 
        executed by all nodes, otherwise the master node executes this
        function and braodcasts the results to other nodes. 
        
        returns:
            common : dict of numpy arrays
                - fill items to keys `weight2d` or `positons_scan`
        """
        # dummy fill
        common = {}
        common['weight2d'] = np.ones(self.roi, dtype='bool')
        common['positions_scan'] = np.indices((self.num_frames, 2)).sum(0)
        return common

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
            self.end_of_scan = (s + self.frames_accessible >= self.num_frames) if eos is None else eos
        # wait for master
        parallel.barrier()
        # communicate result
        self._flags = parallel.bcast(self._flags)

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
            has_weights = (len(weights) > 0)

            if has_data:
                dsh = np.array(data.values()[0].shape[-2:])
            else:
                dsh = np.zeros([0, 0])

            # communicate result
            parallel.MPImax(dsh)

            if not has_weights:
                # peak at first item
                altweight = self.common.get('weight2d', np.ones(dsh))
                weights = dict.fromkeys(data.keys(), altweight)

            assert len(weights) == len(data), 'Data and Weight frames unbalanced %d vs %d' % (len(data), len(weights))

            cen = self.center

            # get center in diffraction image
            if str(cen) == 'auto':
                cen = self._mpi_autocenter(data, weights)
            elif cen is None:
                cen = dsh // 2
            else:
                # center is number or tuple
                cen = u.expect2(cen[-2:])

            # It is important to set center again in order to NOT have different centers for each chunk
            # the downside is that the center may be off if only a few diffraction patterns were
            # used for the analysis. It such case it is benficial to set the center in the parameters
            self.center = cen
            self.info.center = cen  # for documentation

            # make sure center is in the image frame
            assert (cen > 0).all() and (
                dsh - cen > 0).all(), 'Optical axes (center = (%.1f,%.1f) outside diffraction image frame' % tuple(cen)

            # adapt roi if not set
            self.roi = u.expect2(self.roi) if self.roi is not None else dsh
            self.shape = self.roi

            # determine if the arrays require further processing
            do_flip = np.array(self.orientation).any()
            do_crop = (np.abs(self.roi - dsh) > 0.5).any()
            do_rebin = (self.rebin != 1)

            if do_flip or do_crop or do_rebin:
                logger.info('Enter preprocessing (crop/pad %s, rebin %s, flip/rotate %s) ... \n' %
                            (str(do_crop), str(do_flip), str(do_rebin)))
                # we proceed with numpy arrays. That is probably now
                # more memory intensive but shorter in writing                       
                d = np.array([data[ind] for ind in indices.node])
                w = np.array([weights[ind] for ind in indices.node])

                # flip, rotate etc.
                if len(d) > 0:

                    # flip, rotate etc.
                    d, cen = switch_frame_orientation(d, self.orientation, cen)
                    w, tmp = switch_frame_orientation(w, self.orientation, cen)

                    # crop
                    d, cen = crop_pad_symmetric_2d(d, self.roi, cen)
                    w, tmp = crop_pad_symmetric_2d(w, self.roi, cen)

                    # rebin, check if rebinning is neither to strong nor impossible
                    rebin = self.rebin
                    if self.rebin in range(2, 6) and (((self.roi / float(rebin)) % 1) == 0.0).all():
                        mask = w > 0
                        d = rebin_2d(d, rebin)
                        w = rebin_2d(w, rebin)
                        mask = rebin_2d(mask, rebin)
                        w[mask < 1] = 0
                        cen /= float(rebin)
                        self.shape = u.expect2(self.roi) / rebin
                        if self.__dict__.get('psize_det') is not None:
                            self.psize_det = u.expect2(self.psize_det) * rebin

                # translate back to dictionaries
                data = dict(zip(indices.node, d))
                weights = dict(zip(indices.node, w))

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
                try:
                    chunk.positions = self.common.get('positions_scan', self.info.get('positions_theory'))[
                        indices.chunk]
                except:
                    logger.info('slicing position information failed')
                    chunk.positions = None
            else:
                # a dict : sort positions to indices.chunk
                # this may fail if there a less positions than scan points (unlikely)
                chunk.positions = np.asarray([positions[i] for i in indices.chunk])
                # positions complete

            # with first chunk we update meta
            if self.chunknum < 1:
                for k, v in self.meta.items():
                    self.meta[k] = self.__dict__.get(k, self.info.get(k)) if v is None else v
                self.meta['center'] = cen

                if self.save is not None and parallel.master:
                    io.h5append(self.filename, meta=dict(self.meta))
                parallel.barrier()

            self.chunk = chunk
            self.chunknum += 1

            return chunk

    def auto(self, frames, chunk_form='dp'):
        """
        Repeated calls to this function will process the data
        
        returns:
        ----------
        None  ,if scan's end is not reached, but no data could be prepared yet
        False ,if scan's end is reached
        a data package otherwise
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
            if self.save is not None:
                self._mpi_save_chunk()
            # delete chunk
            del self.chunk
            return out

    def return_chunk_as(self, chunk, kind='dp'):
        """
        Returns the loaded data chunk `chunk` in the format `kind`
        For now only kind=='dp' (data package) is valid.
        """
        # this is a bit ugly now
        if kind == 'dp':
            out = {}

            # The "common" part
            out['common'] = self.meta

            # The "iterable" part
            iterables = []
            for pos, index in zip(chunk.positions, chunk.indices):
                frame = {}
                frame['index'] = index
                frame['data'] = chunk.data.get(index)
                if frame['data'] is None:
                    frame['mask'] = None
                else:
                    # ok we now know that we need a mask since data is not None
                    # first look in chunk for a weight to this index, then
                    # look for a 2d-weight in meta, then arbitrarily set
                    # weight to ones. 
                    w = chunk.weights.get(index, self.meta.get('weight2d', np.ones_like(frame['data'])))
                    frame['mask'] = (w > 0)
                frame['position'] = pos
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
        ! Overwrite in child class !
        
        This method checks how many frames the preparation routine may
        process, starting from frame `start` at a request of `frames_requested`
        
        The method is supposed to return the number of accessible frames
        for preparation and should detemernie if data acquistion for this
        scan is finished
        
        Parameters
        -----------
        frames : (int) or None, Number of frames requested
        start  : (int) or None, scanpoint index to start checking from
        
        Returns 
        ---------------
        (frames_accessible, : Number of frames readable  
            end_of_scan)    : int or None
                               0 : end of the scan is not reached
                               1 : end of scan will be reached or is
                              None : can't say
                    
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

    def load(self, indices):
        """
        Overwrite in child class
        loads data according to indices given
        
        must return 3 dicts with indices as keys: one for the raw frames, one for the
        """
        # dummy fill
        raw = dict((i, i * np.ones(self.roi)) for i in indices)
        return raw, {}, {}

    def correct(self, raw, weights, common):
        """
        Overwrite in child class
        
        Corrects the raw data with arrays given in `common`
        Adjust the weights accordinlgy
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
        # now master possesses all calcuated centers
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

    def _mpi_save_chunk(self, kind='extlink', chunk=None):
        """
        Saves data chunk to hdf5 file specified with `filename`
        It works by gathering wieghts and data to the master node.
        Master node then writes to disk.
        
        In case you support parallel hdf5 writing, please modify this 
        function to suit your installation.
        
        2 out of 3 modes currently supported
        
        kind : 'merge','append','link'
            
            'append' : appends chunks of data in same file
            'extlink' : saves chunks in seperate files and adds ExternalLinks
            
        TODO: 
            * For the 'extlink case, saving still requires access to
              main file `filename` even so for just adding the link.
              This may result in conflict if main file is polled often
              by seperate read process.
              Workaraound would be to write the links on startup in
              initialise() 
            
        """
        # gather all distributed dictionary data.
        c = chunk if chunk is not None else self.chunk
        # shallow copy
        todisk = dict(self.chunk)
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
            if kind == 'append':
                h5address = 'chunks/%d' % num
                io.h5append(self.filename, {h5address: todisk})
            elif kind == 'extlink':
                import h5py

                h5address = 'chunks/%d' % num
                hddaddress = self.filename + '.part%03d' % num
                with h5py.File(self.filename) as f:
                    f[h5address] = h5py.ExternalLink(hddaddress, '/')
                    f.close()
                io.h5write(hddaddress, todisk)


class PtydScan(PtyScan):
    def __init__(self, pars=None, **kwargs):
        super(PtydScan, self).__init__(pars, **kwargs)

        self.meta = io.h5read(self.filename, 'meta')['meta']
        # update internal dict, making sure
        for k, v in self.meta.items():
            if self.__dict__.get(k) is None:
                self.__dict__[k] = v

    def check(self, frames=None, start=None):

        if start is None:
            start = self.framestart
        if frames is None:
            frames = self.minframes
        # Get info about size of currently available chunks.
        # Dead external links will produce None and are excluded
        with h5py.File(self.filename, 'r') as f:
            d = {}

            chitems = sorted([(int(k), v) for k, v in f['chunks'].iteritems() if v is not None], key=lambda t: t[0])
            for chkey in chitems[0][1].keys():
                d[chkey] = [(int(k), v[chkey].shape) for k, v in chitems if v is not None]
            f.close()

        self._checked = d
        self.allframes = int(sum([ch[1][0] for ch in d['data']]))
        self._ch_frame_ind = np.array([(dd[0], frame) for dd in d['data'] for frame in range(dd[1][0])])

        return self.allframes - start, None

    def _coord_to_h5_calls(self, key, coord):
        return ('chunks/%d/%s' % (coord[0], key), slice(coord[1], coord[1] + 1))

    def load_common(self):
        """
        In ptyd, 'common' does not necessarily exist. Only meta is essential
        """
        try:
            common = io.h5read(self.filename, 'common')['common']
        except:
            common = {}
        return common

    def load(self, indices):
        """
        Load from ptyd. Due to possible chunked data, slicing frames is 
        non-trivial
        """
        # get the coordinates in the chunks
        coords = self._ch_frame_ind[indices]
        calls = {}
        for key in self._checked.keys():
            calls[key] = [self._coord_to_h5_calls(key, c) for c in coords]

        # get our data from the ptyd file
        out = {}
        with h5py.File(self.filename, 'r') as f:
            for array, call in calls.iteritems():
                out[array] = [f[path][slce] for path, slce in call]
            f.close()

        # if the chunk provided indices, we use those instead of our own
        # Dangerous and not yet implemented
        # indices = out.get('indices',indices)

        # wrap in a dict 
        for k, v in out.iteritems():
            out[k] = dict(zip(indices, v))

        return (out.get(key, {}) for key in ['data', 'positions', 'weights'])


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
        labels = sorted(scans.keys())

        # empty list for the scans
        self.PS = []

        for label in labels:
            # we are making a copy of the root as we want to fill it
            s = scans[label]['pars']

            ptype = None  # preparation type
            logger.info(u.verbose.report(s))

            # Get parameters for preparation from raw data if required.
            if s.prepare_data:
                prep = u.Param(s.preparation.generic.copy())
                ptype = s.preparation.type
                prep.update(s.preparation[ptype])

            # copy other relevant information
            prep.filename = s.data_file
            prep.geometry = s.geometry.copy()
            prep.xy = s.xy.copy()

            if ptype is not None:
                PS = PtyScanTypes[ptype.lower()]
                logger.info('Scan %s will be prepared with the recipe "%s"' % (label, ptype))
                self.PS.append(PS(prep))
            elif prep.filename.endswith('.ptyd'):
                self.PS.append(PtydScan(prep))
            else:
                raise RuntimeError('Could not manage scan %s' % label)
                # logger.warning('Generating PtyScan for scan "%s" failed - This label will source no data')

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

        # initialize if that has not been done yet
        if not PS.is_initialized:
            PS.initialize()

        msg = PS.auto(self.frames_per_call, self.feed_format)
        # if we catch a scan that has ended look for an unfinished scan
        while msg == EOS and self.scan_available:
            self.scan_current += 1
            PS = self.PS[self.scan_current]
            if not PS.is_initialized:
                PS.initialize()
            msg = PS.auto(self.frames_per_call, self.feed_format)

        self.data_available = (msg != EOS or self.scan_available)
        logger.info(u.verbose.report(msg))
        if msg != WAIT and msg != EOS:
            # ok that would be a data package
            yield msg


def switch_frame_orientation(A, orientation, center=None):
    """
    Switches orientation of Array A along the last two axes (-2,-1)
        
    orientation : 3-tuple (transpose,flipud,fliplr)
    
    returns
    --------
        Flipped array, new center
    """
    # switch orientation
    if orientation[0]:
        axes = list(range(A.ndim - 2)) + [-1, -2]
        A = np.transpose(A, axes)
        center = (center[1], center[0]) if center is not None else None
    if orientation[1]:
        A = A[..., ::-1, :]
        center = (A.shape[-2] - 1 - center[0], center[1]) if center is not None else None
    if orientation[2]:
        A = A[..., ::-1]
        center = (center[0], A.shape[-1] - 1 - center[1]) if center is not None else None

    return A, np.array(center)


def rebin_2d(A, rebin=1):
    """
    Rebins array A symmetrically along last 2 axes with a factor `rebin`
    """
    newdim = np.asarray(A.shape[-2:]) / rebin
    return A.reshape(-1, newdim[0], rebin, newdim[1], rebin).sum(-1).sum(-2)


def crop_pad_symmetric_2d(A, newshape, center=None):
    """
    Crops or pads Array A symmetrically along the last two axes (-2,-1)
    around center `center` to a new shape `newshape`
    
    """
    # crop / pad symmetrically around center
    osh = np.array(A.shape[-2:])
    c = np.round(center) if center is not None else osh // 2
    sh = np.array(newshape[-2:])
    low = -c + sh // 2
    high = -osh + c + (sh + 1) // 2
    hplanes = np.array([[low[0], high[0]], [low[1], high[1]]])

    if (hplanes != 0).any():
        A = u.crop_pad(A, hplanes)

    return A, c + low


if __name__ == "__main__":
    u.verbose.set_level(3)
    shape = (40, 256, 256)
    PS = PtyScan(roi=256, save='extlink')
    PS.initialize()
    for i in range(50):
        msg = PS.auto()
        logger.info(u.verbose.report(msg), extra={'allprocesses': True})
        parallel.barrier()

