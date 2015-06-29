# -*- coding: utf-8 -*-
"""\
Scan loading recipe for the ID16A beamline at ESRF - near-field ptycho setup.

This file is part of the PTYPY package.

    :copyright: Copyright 2014 by the PTYPY team, see AUTHORS.
    :license: GPLv2, see LICENSE for details.
"""

import numpy as np
import os
from .. import io
from .. import utils as u
from .. import core
from ..core.data import PtyScan

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

# Generic defaults
PREP_DEFAULT = u.Param()
PREP_DEFAULT.orientation = (False, True, False) # Orientation of Frelon frame - only LR flip

# Recipe defaults
DEFAULT = u.Param()
DEFAULT.experimentID = None         # Experiment identifier - will be read from h5
DEFAULT.energy = None               # Energy in keV - will be read from h5
DEFAULT.lam = None                  # 1.2398e-9 / DEFAULT.energy
DEFAULT.z = None                    # Distance from object to screen
DEFAULT.motors = ['spy', 'spz']     # 'Motor names to determine the sample translation'
DEFAULT.motors_multiplier = 1e-6    # 'Motor conversion factor to meters'
DEFAULT.base_path = None            # Base path to read and write data - can be guessed.
DEFAULT.sample_name = None          # Sample name - will be read from h5
DEFAULT.scan_label = None           # Scan label - will be read from h5
DEFAULT.flat_label = None           # Flat label - equal to scan_label by default
DEFAULT.dark_label = None           # Dark label - equal to scan_label by default

# These are home-made wrapped data 
DEFAULT.data_file_pattern = '{[base_path]}/{[sample_name]}/{[scan_label]}_data.h5'
DEFAULT.flat_file_pattern = '{[base_path]}/{[sample_name]}/{[flat_label]}_flat.h5'
DEFAULT.dark_file_pattern = '{[base_path]}/{[sample_name]}/{[dark_label]}_dark.h5'

# The h and v are inverted here - that's on purpose!
DEFAULT.distortion_h_file = '/data/id16a/inhouse1/instrument/img1/optique_peter_distortion/detector_distortion2d_v.edf'
DEFAULT.distortion_v_file = '/data/id16a/inhouse1/instrument/img1/optique_peter_distortion/detector_distortion2d_h.edf'
DEFAULT.whitefield_file = '/data/id16a/inhouse1/instrument/whitefield/white.edf'

class ID16AScan(PtyScan):
    """
    Subclass of PtyScan for ID16A beamline (specifically for near-field
    ptychography).
    """

    def __init__(self, pars=None, **kwargs):
        """
        Create a PtyScan object that will load ID16A data.

        :param pars: preparation parameters
        :param kwargs: Additive parameters
        """
        # Apply beamline-specific generic defaults
        pars = PREP_DEFAULT.copy().update(pars)
        pars.update(**kwargs)
        
        # Apply beamline parameters ("recipe")
        rinfo = DEFAULT.copy()
        rinfo.update(pars.recipe)
        
        # Initialise parent class with input parameters
        super(self.__class__, self).__init__(pars)

        # Store recipe parameters in self.info
        self.info.recipe = rinfo

        # Default scan label
        if self.info.label is not None:
            assert (self.info.label == rinfo.scan_label), 'Incompatible scan labels'
        self.info.label = rinfo.scan_label
        logger.info('Scan label: %s' % rinfo.scan_label)
        
        # Default flat and dark labels.
        if rinfo.flat_label is None:
            rinfo.flat_label = rinfo.scan_label
        if rinfo.dark_label is None:
            ringo.dark_label = ringo.scan_label

        # Attempt to extract base_path if missing
        if rinfo.base_path is None:
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
                rinfo.base_path = base_path
                logger.info('Base path: %s' % base_path)

        # Data file names
        rinfo.data_file = rinfo.data_file_pattern.format(rinfo)
        rinfo.dark_file = rinfo.dark_file_pattern.format(rinfo)
        rinfo.flat_file = rinfo.flat_file_pattern.format(rinfo)

        # Read metadata
        h = io.h5read(rinfo.data_file)
        entry = h.keys()[0]
        rinfo.entry = entry
        
        # Energy
        k = H5_PATHS.energy.format(rinfo)
        energy = float(io.h5read(rinfo.data_file, k)[k])
        if self.info.energy is not None:
            assert self.info.energy == energy, "Energy (%f keV) is read from file - please don't attempt to overwrite it" % energy

        # Attempt to extract experiment ID
        if rinfo.experimentID is None:
            # We use the path structure for this
            experimentID = os.path.split(os.path.split(rinfo.base_path[:-1])[0])[1]
            logger.info('experiment ID: %s' % experimentID)
            rinfo.experimentID = experimentID
        
        # Effective pixel size


        # Data file names
        rinfo.data_file = rinfo.data_file_pattern.format(rinfo)
        rinfo.dark_file = rinfo.dark_file_pattern.format(rinfo)
        rinfo.flat_file = rinfo.flat_file_pattern.format(rinfo)

        self.rinfo = rinfo
        self.info.recipe = rinfo

        logger.info(u.verbose.report(self.info))

    def load_common(self):
        """
        Load scanning positions, dark, white field and distortion files.
        """
        common = u.Param()

        h = io.h5read(self.rinfo.data_file)
        entry = h.keys()[0]
        
        # Get positions
        motor_positions = io.h5read(self.rinfo.data_file, H5_PATHS.motors)[H5_PATHS.motors]
        mmult = u.expect2(self.rinfo.motors_multiplier)
        pos_list = [mmult[i] * np.array(motor_positions[motor_name]) for i, motor_name in enumerate(self.rinfo.motors)]
        common.positions_scan = np.array(pos_list).T

        # Load dark
        h = io.h5read(self.rinfo.dark_file)
        entry_name = h.keys()[0]
        darks = h[entry_name]['ptycho']['data']
        if darks.ndim == 2:
            common.dark = darks
        else:
            common.dark = darks.median(axis=0)
        
        # Load white field
        common.white = io.edfread(self.rinfo.whitefield_file)[0]
        
        # Load distortion files
        dh = io.edfread(self.rinfo.distortion_h_file)[0]
        dv = io.edfread(self.rinfo.distortion_v_file)[0]

        common.distortion = (dh, dv)

        return common._to_dict()
        
    def check(self, frames, start=0):
        """
        Returns the number of frames available from starting index `start`, and whether the end of the scan
        was reached.

        :param frames: Number of frames to load
        :param start: starting point
        :return: (frames_available, end_of_scan)
        - the number of frames available from a starting point `start`
        - bool if the end of scan was reached (None if this routine doesn't know)
        """
        npos = self.num_frames
        frames_accessible = min((frames, npos-start))
        stop = self.frames_accessible + start
        return frames_accessible, (stop >= npos)
        
    def load(self, indices):
        """
        Load frames given by the indices.

        :param indices:
        :return:
        """
        raw = {}
        pos = {}
        weights = {}
        for j in indices:
            key = H5_PATHS.frame_pattern % self.rinfo
            raw[j] = io.h5read(self.rinfo.data_file, H5_PATHS.frame_pattern % self.rinfo, slice=j)[key].astype(np.float32)
            
        return raw, pos, weights
        
    def correct(self, raw, weights, common):
        """
        Apply (eventual) corrections to the raw frames. Convert from "raw" frames to usable data.
        :param raw:
        :param weights:
        :param common:
        :return:
        """
        # Sanity check
        assert (raw.shape == (2048,2048)), 'Wrong frame dimension! Is this a frelon camera?'
        
        # Whitefield correction
        raw_wf = raw / common.white

        # Missing line correction
        raw_wf_ml = raw_wf.copy()
        raw_wf_ml[1024:,:] = raw_wf[1023:-1,1]
        raw_wf_ml[1023,:] += raw_wf[1024,:]
        raw_wf_ml[1023,:] *= .5 
        
        # Undistort
        raw_wl_ml_ud = undistort(raw_wf_ml, common.distortion)
        
        data = raw_wl_ml_ud
        
        return data, weights

def undistort(frame, delta):
    """
    Frame distortion correction (linear interpolation)
    Any value outside the frame is replaced with a constang value (mean of
    the complete frame)
    
    Parameters
    ----------
    frame: ndarray
        the input frame data
    delta: 2-tuple 
        containing the horizontal and vertical displacements respectively.
    
    Returns
    -------
    ndarray
        The corrected frame of same dimension and type as frame.

    """
    # FIXME: this function should attempt to use scipy.interpolate.interpn if available.

    deltah, deltav = delta

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
    outf = np.zeros_like(ff)
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
