# -*- coding: utf-8 -*-
"""\
Scan loading recipe for the ID16A beamline at ESRF - near-field ptycho setup.

This file is part of the PTYPY package.

    :copyright: Copyright 2014 by the PTYPY team, see AUTHORS.
    :license: see LICENSE for details.
"""

import numpy as np
import os, re, glob

from .. import utils as u
from .. import io
from . import register
from ..core.data import PtyScan
from ..utils.verbose import log
from ..core.paths import Paths
from ..core import Ptycho

IO_par = Ptycho.DEFAULT['io']

logger = u.verbose.logger

# Paths to relevant information in h5 file.
H5_PATHS = u.Param()
H5_PATHS.energy = '{entry}/measurement/initial/energy'
H5_PATHS.current_start = '{entry}/instrument/source/current_start'
H5_PATHS.current_end = '{entry}/instrument/source/current_end'
H5_PATHS.exposure = 'entry1/instrument/%(detector_name)s/count_time'
H5_PATHS.sample_name = '{entry}/sample/name'
H5_PATHS.pixel_size = '{entry}/sample/pixel_size'
H5_PATHS.focus_position = '{entry}/focus_positions'
H5_PATHS.command = '{entry}/scan_type'
H5_PATHS.label = '{entry}/title'
H5_PATHS.experiment = '{entry}/experiment_identifier'

# These two entries are added to the structure post-measurement.
H5_PATHS.frames = '{entry}/ptycho/data'
H5_PATHS.motors = '{entry}/ptycho/motors'


@register()
class ID16AScan(PtyScan):
    """
    Subclass of PtyScan for ID16A beamline (specifically for near-field
    ptychography).

    Default data parameters. See :py:data:`.scan.data`

    Defaults:

    [name]
    default = 'ID16AScan'
    type = str
    help =

    [experimentID]
    type = str
    default = None
    help = Name of the experiment
    doc = If None, a default value will be provided by the recipe. **unused**
    userlevel = 2

    [dfile]
    type = file
    default = None
    help = File path where prepared data will be saved in the ``ptyd`` format.
    userlevel = 0

    [motors]
    default = ['spy', 'spz']
    type = list
    help = Motor names to determine the sample translation

    [motors_multiplier]
    default = 1e-6
    type = float
    help = Motor conversion factor to meters at ID16A beamline

    [base_path]
    default = None
    type = str
    help = Base path to read and write data

    [sample_name]
    default = None
    type = str
    help = Sample name - will be read from h5

    [label]
    type = str
    default = None
    help = The scan label
    doc = Unique string identifying the scan
    userlevel = 1

    [mask_file]
    default = None
    type = str
    help = Mask file name

    [flat_division]
    default = False
    type = bool
    help = Switch for flat division

    [dark_subtraction]
    default = False
    type = bool
    help = Switch for dark subtraction

    [distortion_h_file]
    default = '/data/id16a/inhouse1/instrument/img1/optique_peter_distortion/detector_distortion2d_v.edf'
    type = str
    help = The h and v are inverted here - that's on purpose!

    [distortion_v_file]
    default = '/data/id16a/inhouse1/instrument/img1/optique_peter_distortion/detector_distortion2d_h.edf'
    type = str
    help = The h and v are inverted here - that's on purpose!

    [whitefield_file]
    default = '/data/id16a/inhouse1/instrument/whitefield/white.edf'
    type = str
    help =

    [det_flat_field]
    default = None
    type = str
    help = path to detector flat field 

    [auto_center]
    type = bool
    default = False
    help = Determine if center in data is calculated automatically
    doc =
       - ``False``, no automatic centering
       - ``None``, only if :py:data:`center` is ``None``
       - ``True``, it will be enforced
    userlevel = 0

    [orientation]
    type = int, tuple, list
    default = (False, True, False)
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

    [recipe]
    default = u.Param()
    help = Specific additional parameters of ID16A

    """

    def __init__(self, pars=None, **kwargs):
        """
        Create a PtyScan object that will load ID16A data.
        """

        p = self.DEFAULT.copy(99)
        p.update(pars)

        # Initialise parent class
        super(ID16AScan, self).__init__(p, **kwargs)
        print(type(p.recipe))

        # Try to extract base_path to access data files
        if self.info.base_path is None:
            d = os.getcwd()
            base_path = None
            while True:
                if 'id16a' in os.listdir(d):
                    base_path = os.path.join(d, 'id16a')
                    break
                d, rest = os.path.split(d)
                if not rest:
                    break
            if base_path is None:
                raise RuntimeError('Could not guess base_path.')
            else:
                self.info.base_path = base_path

        # get the scan name
        self.scan_name = self.info.sample_name

        # filename analysis
        self.frame_wcard = os.path.join(self.info.base_path,self.info.sample_name+'*')
        self.filelist = sorted(glob.glob(self.frame_wcard))
        self.firstframe,self.ext = os.path.splitext(self.filelist[0])

        # count the number of available frames
        self.num_frames = len(self.filelist)

        # h5 path
        if self.ext == '.h5':
            self.h5_path = 'entry_0000/measurement/{}'.format(self.info.recipe.detector)

        # Create the ptyd file name if not specified
        if self.info.dfile is None:
            home = Paths(IO_par).home
            self.info.dfile = '%s/prepdata/data_%d.ptyd' % (
                home, self.info.label)
            log(3, 'Save file is %s' % self.info.dfile)

        log(4, u.verbose.report(self.info))

    def load_weight(self):
        """
        Function description see parent class. For now, this function
        will be used to load the mask.

        Returns
        -------
         weight2d : ndarray
            A two-dimensional array with a shape compatible to the raw
            diffraction data frames if provided from file
        """
        # FIXME: do something better here. (detector-dependent)
        # Load mask as weight
        if self.info.mask_file is not None:
            print('Loading detector mask')
            return io.h5read(self.info.mask_file, 'mask')['mask'].astype(np.float32)

    def load_positions(self):
        """
        Loads all positions for all diffraction patterns in this scan.

        Returns
        -------
        Positions : ndarray
            A (N,2)-array where *N* is the number of positions.
        """
        positions = []
        mmult = u.expect2(self.info.motors_multiplier)

        # Load positions
        if self.ext == '.h5':
            # From .h5 files
            for ii in self.filelist:
                projobj = io.h5read(ii,self.h5_path)[self.h5_path]
                metadata = projobj['parameters']
                motor_mne = str(metadata['motor_mne ']).split("'")[1].split() # motor names
                motor_pos = [eval(kk) for kk in str(metadata['motor_pos ']).split("'")[1].split()] # motor pos
                motor_idx = (motor_mne.index('spy'), motor_mne.index('spz')) # index motors
                positions.append((motor_pos[motor_idx[0]], \
                                  motor_pos[motor_idx[1]]))# translation motor positions
        else:
            # From .edf files
            for ii in self.filelist:
                data, meta = io.edfread(ii)
                positions.append((meta['motor'][self.info.motors[0]],
                                  meta['motor'][self.info.motors[1]]))

        return np.array(positions) * mmult[0]

    def load_common(self):
        """
        Loads anything that is common to all frames and stores it in dict.

        Returns
        -------
        common : dict
            contains information common to all frames such as dark,
            flat-field, detector flat-field, normalization couter,
            and distortion files
        """
        common = u.Param()

        # Load dark files
        if self.info.dark_subtraction:
            print('Loading the dark files...')
            dark = []
            if self.ext == '.h5':
                # From HDF5 files
                dark_files = sorted(glob.glob(os.path.join(self.info.base_path,'dark*.h5')))
                for ff in dark_files:
                    dobj = io.h5read(ff,self.h5_path)[self.h5_path]
                    d = dobj['data'].astype(np.float32)
                    dark.append(d)
            else:
                # From .edf files
                dark_files = sorted(glob.glob(os.path.join(self.info.base_path,'dark*.edf')))
                for ff in dark_files:
                    data, meta = io.edfread(ff)
                    dark.append(data.astype(np.float32))
            print('Averaging the dark files...')
            common.dark = np.array(dark).mean(axis=0)

            log(3, 'Dark loaded successfully.')

        # Load flat files
        if self.info.flat_division:
            print('Loading the flat files...')
            flat = []
            if self.ext == '.h5':
                # From HDF5 file
                flat_files = sorted(glob.glob(os.path.join(self.info.base_path,'ref*.h5')))
                for ff in flat_files:
                    flobj = io.h5read(ff,self.h5_path)[self.h5_path]
                    fl = flobj['data'].astype(np.float32)
                    flat.append(fl)
            else:
                # From .edf files
                flat_files = sorted(glob.glob(os.path.join(self.info.base_path,'ref*.edf')))
                for ff in flat_files:
                    data, meta = io.edfread(ff)
                    flat.append(data.astype(np.float32))
            print('Averaging the flat files...')
            common.flat = np.array(flat).mean(axis=0)

            log(3, 'Flat loaded successfully.')

        # Load detector flat field
        if self.info.det_flat_field is not None:
            # read flat-field file
            print('Reading flat-field file of the detector')
            flat_field,header = io.edfread(self.det_flat_field)
            flat_field = flat_field.astype(np.float32)/flat_field.mean()
            flat_field[np.where(flat_field==0)]=1 # put 1 where values are 0 for the division
            common.flat_field = flat_field

            log(3, 'Detector flat field loaded successfully.')

        # read the BPM5 ct values
        if self.info.recipe.use_bpm5_ct:
            print('Reading the values of the bpm5 ct')
            bpm5_ct_val = np.zeros(self.num_frames)
            for ii in range(self.num_frames):
                projobj = io.h5read(self.frame_format.format(ii),self.h5_path)[self.h5_path]
                #projobj = io.h5read(self._index_to_frame(ii),self.h5_path)[self.h5_path]
                # metadata
                metadata = projobj['parameters']
                counter_mne = str(metadata['counter_mne ']).split() # counter names
                counter_pos = [eval(kk) for kk in str(metadata['counter_pos ']).split()] # counter pos
                bpm5_ct_idx = counter_mne.index('bpm5_ct')  # index bpm5_ct
                bpm5_ct_val[ii] = (counter_pos[bpm5_ct_idx])  # value bpm5_ct

            # Check for spikes in bpm5_ct values and correct them if needed
            below_zero_values = np.where(bpm5_ct_val<0)
            spikes_pos = [ii for ii in below_zero_values[0]]
            below_mean_values = np.where(bpm5_ct_val<bpm5_ct_val.mean())
            if len(below_mean_values[0])==1:
                spikes_pos.append(below_mean_values[0][0])
            if len(spikes_pos)!=0:#below_zero_values[0].shape[0]!=0:
                # check for spikes in bpm5
                for ii in spikes_pos:#below_zero_values[0]:
                    if ii!=0 or ii!=len(bpm5_ct_val):
                        bpm5_ct_val[ii] = (bpm5_ct_val[ii-1] + bpm5_ct_val[ii+1])/2.
                    elif ii == 0:
                        bpm5_ct_val[ii] = bpm5_ct_val[ii+1]
                    elif ii == len(bpm5_ct_val):
                        bpm5_ct_val[ii] = bpm5_ct_val[ii-1]
            # normalization
            print('Normalizing the values of the bpm5 ct by the average')
            common.bpm5_ct_val = bpm5_ct_val/bpm5_ct_val.mean() # normalize by the mean value

            log(3, 'Values of bpm5_ct loaded successfully.')

        # Load white field
        #common.white = io.edfread(self.rinfo.whitefield_file)[0]

        # Load distortion files
        #dh = io.edfread(self.rinfo.distortion_h_file)[0]
        #dv = io.edfread(self.rinfo.distortion_v_file)[0]

        #common.distortion = (dh, dv)

        #return common._to_dict()

        return common

    def check(self, frames, start=0):
        """
        This method checks how many frames the preparation routine may
        process, starting from frame `start` at a request of `frames`.
        Returns the number of frames available from starting index `start`, and
        whether the end of the scan was reached.

        Parameters
        ----------
        frames : int or None
            Number of frames requested
        start : int or None
            Scanpoint index to start checking from

        Returns
        -------
        frame_accessible : int 
            Number of frames readable from a starting point `start`
        end_of_scan : bool or None
            Check if the end of scan was reached, otherwise None if this
            routine doesn't know
        """
        npos = self.num_frames
        frames_accessible = min((frames, npos - start))
        stop = self.frames_accessible + start
        return frames_accessible, (stop >= npos)

    def load(self, indices):
        """
        Loads data according to node specific scanpoint indices that have
        been determined by :py:class:`LoadManager` or otherwise.

        Returns
        -------
        raw, pos, weight : dict
            Dictionaries whose keys are the given scan point `indices`
            and whose values are the respective frame / position according
            to the scan point index. `weight` and `positions` may be empty
        
        Note
        ----
        If one weight (mask) is to be used for the whole scan, it should
        be loaded with load_weights(). The same goes for the positions, 
        which sould be loade with load_positions().
        """
        raw = {}
        pos = {}
        weights = {}

        # Load data
        if self.ext == '.h5':
            # From HDF5 file
            for idx in indices:
                #print('Loading {}'.format(self.filelist[idx]))
                projobj = io.h5read(self.filelist[idx],self.h5_path)[self.h5_path]
                raw[idx] = projobj['data'][0].astype(np.float32) # needs the index [0] to squeeze (faster thant np.squeeze)
        else:
            # From .edf files
            for idx in indices:
                #print('Loading {}'.format(self.filelist[idx]))
                data, meta = io.edfread(self.filelist[idx])
                raw[idx] = data.astype(np.float32)

        return raw, pos, weights

    def correct(self, raw, weights, common):
        """
        Place holder for dark and flatfield correction. Apply (eventual)
        corrections to the raw frames. Convert from "raw" frames to 
        usable data.
        
        Parameters
        ----------
        raw : dict
            Dict containing index matched data frames (np.array).
        weights : dict
            Dict containing possible weights.
        common : dict
            Dict containing possible dark and flat frames.
        
        Returns
        -------
        data, weights : dict
            Flat and dark-corrected data dictionaries. These dictionaries
            must have the same keys as the input `raw` and contain
            corrected frames (`data`) and statistical weights (`weights`)

        Note
        ----
        If the negative values results from the calculation, they will
        be forced to be equal to 0.
        """

        # Apply flat and dark, only dark, or no correction
        if self.info.flat_division and self.info.dark_subtraction:
            for j in raw:
                raw[j] = (raw[j] - common.dark) / (common.flat - common.dark)
                raw[j][raw[j] < 0] = 0 # put negative values to 0
        elif self.info.dark_subtraction:
            for j in raw:
                raw[j] = raw[j] - common.dark
                raw[j][raw[j] < 0] = 0 # put negative values to 0

        if self.info.det_flat_field is not None:
            for j in raw:
                print("Correcting detector flat-field: {}".format(self._index_to_frame(j)))
                raw[j] = raw[j] / common.flat_field
                raw[j][raw[j] < 0] = 0 # put negative values to 0

        if self.info.recipe.pad_crop is not None:
            newdim = (self.info.recipe.pad_crop,self.info.recipe.pad_crop)
            for j in raw:
                print('Reshaping projection {} to {}'.format(self._index_to_frame(j),newdim))
                raw[j],_ = u.crop_pad_symmetric_2d(raw[j],newdim)

        if self.info.recipe.use_bpm5_ct:
            for j in raw:
                print("Correcting for the bpm5_ct values for {}".format(self.frame_format.format(j)))
                raw[j] = raw[j] / common.bpm5_ct_val[j]
        
        data = raw

        # Sanity check
        #assert (raw.shape == (2048,2048)), (
        #    'Wrong frame dimension! Is this a Frelon camera?')

        # Whitefield correction
        #raw_wf = raw / common.white

        # Missing line correction
        #raw_wf_ml = raw_wf.copy()
        #raw_wf_ml[1024:,:] = raw_wf[1023:-1,1]
        #raw_wf_ml[1023,:] += raw_wf[1024,:]
        #raw_wf_ml[1023,:] *= .5

        # Undistort
        #raw_wl_ml_ud = undistort(raw_wf_ml, common.distortion)

        #data = raw_wl_ml_ud

        return data, weights

def undistort(frame, delta):
    """
    Frame distortion correction (linear interpolation)
    Any value outside the frame is replaced with a constant value (mean of
    the complete frame)

    Parameters
    ----------
    frame : ndarray
        The input frame data
    delta : 2-tuple
        Containing the horizontal and vertical displacements respectively.

    Returns
    -------
    outf : ndarray
        The corrected frame of same dimension and type as frame.

    """
    # FIXME: this function should attempt to use scipy.interpolate.interpn if available.

    deltah, deltav = delta

    sh = frame.shape
    x, y = np.meshgrid(np.arange(sh[0]), np.arange(sh[1]))

    sh = frame.shape

    ny = (y-deltav).flatten()
    nx = (x-deltah).flatten()

    nyf = np.floor(ny).astype(int)
    nyc = nyf+1
    nxf = np.floor(nx).astype(int)
    nxc = nxf+1

    pts_in = (nyc < sh[0]) & (nyc > 0) & (nxc < sh[1]) & (nxc > 0)
    pts_out = ~pts_in
    outf = np.zeros_like(frame)
    outf[pts_out] = frame.mean()
    nxf = nxf[pts_in]
    nxc = nxc[pts_in]
    nyf = nyf[pts_in]
    nyc = nyc[pts_in]
    nx = nx[pts_in]
    ny = ny[pts_in]

    #nxf = np.clip(nxf, 0, sh[1]-1)
    #nxc = np.clip(nxc, 0, sh[1]-1)
    #nyf = np.clip(nyf, 0, sh[0]-1)
    #nyc = np.clip(nyc, 0, sh[0]-1)

    fa = frame[ nyf, nxf ]
    fb = frame[ nyc, nxf ]
    fc = frame[ nyf, nxc ]
    fd = frame[ nyc, nxc ]

    wa = (nxc-nx) * (nyc-ny)
    wb = (nxc-nx) * (ny-nyf)
    wc = (nx-nxf) * (nyc-ny)
    wd = (nx-nxf) * (ny-nyf)

    outf[pts_in] = (wa*fa + wb*fb + wc*fc + wd*fd).astype(outf.dtype)
    #outf = (wa*fa + wb*fb + wc*fc + wd*fd).astype(outf.dtype)

    return outf.reshape(sh)
