# -*- coding: utf-8 -*-
"""\
Scan loading recipe for the ID16A beamline at ESRF - near-field ptycho setup.

This file is part of the PTYPY package.

    :copyright: Copyright 2014 by the PTYPY team, see AUTHORS.
    :license: GPLv2, see LICENSE for details.
"""

import numpy as np
import os

from .. import utils as u
from .. import io
from ..utils.descriptor import defaults_tree
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


@defaults_tree.parse_doc('scandata.ID16AScan')
class ID16AScan(PtyScan):
    """
    Subclass of PtyScan for ID16A beamline (specifically for near-field
    ptychography).

    Defaults:

    [name]
    default = 'ID16AScan'
    type = str
    help = 

    [experimentID]
    default = None

    [motors]
    default = ['spy', 'spz']
    type = list
    help = Motor names to determine the sample translation

    [motors_multiplier]
    default = 1e-6
    type = float
    help = Motor conversion factor to meters

    [base_path]
    default = None
    type = str
    help = Base path to read and write data - can be guessed

    [sample_name]
    default = None
    type = str
    help = Sample name - will be read from h5

    [scan_label]
    default = None
    type = int
    help = Scan label - will be read from h5

    [flat_label]
    default = None
    type = int
    help = Flat label - equal to scan_label by default

    [dark_label]
    default = None
    type = int
    help = Dark label - equal to scan_label by default

    [mask_file]
    default = None
    type = str
    help = Mask file name

    [use_h5]
    default = False
    type = bool
    help = Load data from prepared h5 file

    [flat_division]
    default = False
    type = bool
    help = Switch for flat division

    [dark_subtraction]
    default = False
    type = bool
    help = Switch for dark subtraction

    [data_file_pattern]
    default = '{[base_path]}/{[sample_name]}/{[scan_label]}_data.h5'
    type = str
    help = 

    [flat_file_pattern]
    default = '{[base_path]}/{[sample_name]}/{[flat_label]}_flat.h5'
    type = str
    help = 

    [dark_file_pattern]
    default = '{[base_path]}/{[sample_name]}/{[dark_label]}_dark.h5'
    type = str
    help = 

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

    [auto_center]
    default = False

    [orientation]
    default = (False, True, False)
    """

    def __init__(self, pars=None, **kwargs):
        """
        Create a PtyScan object that will load ID16A data.

        :param pars: preparation parameters
        :param kwargs: Additive parameters
        """

        p = self.DEFAULT.copy(99)
        p.update(pars)

        # Initialise parent class
        super(ID16AScan, self).__init__(p, **kwargs)

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

        # Create the ptyd file name if not specified
        if self.info.dfile is None:
            home = Paths(IO_par).home
            self.info.dfile = '%s/prepdata/data_%d.ptyd' % (
                home, self.info.scan_label)
            log(3, 'Save file is %s' % self.info.dfile)

        log(4, u.verbose.report(self.info))

    def load_weight(self):
        """
        Function description see parent class. For now, this function will be
        used to load the mask.
        """
        # FIXME: do something better here. (detector-dependent)
        # Load mask as weight
        if self.info.mask_file is not None:
            return io.h5read(self.info.mask_file, 'mask')['mask'].astype(
                np.float32)

    def load_positions(self):
        """
        Load the positions and return as an (N, 2) array.
        """
        positions = []
        mmult = u.expect2(self.info.motors_multiplier)

        # Load positions
        if self.info.use_h5:
            # From prepared .h5 file
            data = io.h5read(self.info.base_path + '/raw/data.h5')
            for i in np.arange(1, len(data) + 1, 1):
                positions.append((data['data_%04d' % i]['positions'][0, 0],
                                  data['data_%04d' % i]['positions'][0, 1]))
        else:
            # From .edf files
            pos_files = []
            # Count available images given through scan_label
            for i in os.listdir(self.info.base_path +
                                self.info.scan_label):
                if i.startswith(self.info.scan_label):
                    pos_files.append(i)

            for i in np.arange(1, len(pos_files) + 1, 1):
                data, meta = io.edfread(self.info.base_path +
                                        self.info.scan_label + '/' +
                                        self.info.scan_label +
                                        '_%04d.edf' % i)

                positions.append((meta['motor'][self.info.motors[0]],
                                  meta['motor'][self.info.motors[1]]))

        return np.array(positions) * mmult[0]

    def load_common(self):
        """
        Load scanning positions, dark, white field and distortion files.
        """
        common = u.Param()

        #h = io.h5read(self.rinfo.data_file)
        #entry = h.keys()[0]

        # Get positions
        #motor_positions = io.h5read(self.rinfo.data_file,
        #                            H5_PATHS.motors)[H5_PATHS.motors]
        #mmult = u.expect2(self.rinfo.motors_multiplier)
        #pos_list = [mmult[i] * np.array(motor_positions[motor_name])
        #            for i, motor_name in enumerate(self.rinfo.motors)]
        #common.positions_scan = np.array(pos_list).T

        # Load dark
        #h = io.h5read(self.rinfo.dark_file)
        #entry_name = h.keys()[0]
        #darks = h[entry_name]['ptycho']['data']
        #if darks.ndim == 2:
        #    common.dark = darks
        #else:
        #    common.dark = darks.median(axis=0)

        # Load white field
        #common.white = io.edfread(self.rinfo.whitefield_file)[0]

        # Load distortion files
        #dh = io.edfread(self.rinfo.distortion_h_file)[0]
        #dv = io.edfread(self.rinfo.distortion_v_file)[0]

        #common.distortion = (dh, dv)

        #return common._to_dict()

        # Load dark
        if self.info.use_h5:
            # From prepared .h5 file
            dark = io.h5read(self.info.base_path + '/raw/dark.h5')
            common.dark = dark['dark_avg']['avgdata'].astype(np.float32)
        else:
            # From .edf files
            dark_files = []
            # Count available dark given through scan_label
            for i in os.listdir(self.info.base_path +
                                self.info.scan_label):
                if i.startswith('dark'):
                    dark_files.append(i)

            dark = []
            for i in np.arange(1, len(dark_files) + 1, 1):
                data, meta = io.edfread(self.info.base_path +
                                        self.info.scan_label + '/' +
                                        'dark_%04d.edf' % i)
                dark.append(data.astype(np.float32))

            common.dark = np.array(dark).mean(0)

        log(3, 'Dark loaded successfully.')

        # Load flat
        if self.info.use_h5:
            # From prepared .h5 file
            flat = io.h5read(self.info.base_path + '/raw/ref.h5')
            common.flat = flat['ref_avg']['avgdata'].astype(np.float32)
        else:
            # From .edf files
            flat_files = []
            # Count available dark given through scan_label
            for i in os.listdir(self.info.base_path +
                                self.info.scan_label):
                if i.startswith('flat'):
                    flat_files.append(i)

            flat = []
            for i in np.arange(1, len(flat_files) + 1, 1):
                data, meta = io.edfread(self.info.base_path +
                                        self.info.scan_label + '/' +
                                        'ref_%04d.edf' % i)
                flat.append(data.astype(np.float32))

            common.flat = np.array(flat).mean(0)

        log(3, 'Flat loaded successfully.')

        return common

    def check(self, frames, start=0):
        """
        Returns the number of frames available from starting index `start`, and
        whether the end of the scan was reached.

        :param frames: Number of frames to load
        :param start: starting point
        :return: (frames_available, end_of_scan)
        - the number of frames available from a starting point `start`
        - bool if the end of scan was reached
          (None if this routine doesn't know)
        """
        npos = self.num_frames
        frames_accessible = min((frames, npos - start))
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
        #for j in indices:
        #    key = H5_PATHS.frame_pattern % self.rinfo
        #    raw[j] = io.h5read(self.rinfo.data_file, H5_PATHS.frame_pattern %
        #                       self.rinfo, slice=j)[key].astype(np.float32)

        # Load data
        if self.info.use_h5:
            # From prepared .h5 file
            data = io.h5read(self.info.base_path + '/raw/data.h5')
            for j in indices:
                i = j + 1
                raw[j] = data['data_%04d' % i]['data'].astype(np.float32)
        else:
            # From .edf files
            for j in indices:
                i = j + 1
                data, meta = io.edfread(self.info.base_path +
                                        self.info.scan_label + '/' +
                                        self.info.scan_label +
                                        '_%04d.edf' % i)
                raw[j] = data.astype(np.float32)

        return raw, pos, weights

    def correct(self, raw, weights, common):
        """
        Apply (eventual) corrections to the raw frames. Convert from "raw"
        frames to usable data.
        :param raw:
        :param weights:
        :param common:
        :return:
        """
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

        # Apply flat and dark, only dark, or no correction
        if self.info.flat_division and self.info.dark_subtraction:
            for j in raw:
                raw[j] = (raw[j] - common.dark) / (common.flat - common.dark)
                raw[j][raw[j] < 0] = 0
            data = raw
        elif self.info.dark_subtraction:
            for j in raw:
                raw[j] = raw[j] - common.dark
                raw[j][raw[j] < 0] = 0
            data = raw
        else:
            data = raw

        # FIXME: this will depend on the detector type used.

        weights = weights

        return data, weights

def undistort(frame, delta):
    """
    Frame distortion correction (linear interpolation)
    Any value outside the frame is replaced with a constant value (mean of
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

# ID16A original subclassing of PtyScan
# *Deprecated*
#
# import numpy as np
# import re, glob, os
# from ptypy.core import data
# from ptypy import utils as u
# from ptypy import io
#
# try:
#     from Tkinter import Tk
#     from tkFileDialog import askopenfilename, askopenfilenames
#
#     print("Please, load the first frame of the ptychography experiment...")
#     Tk().withdraw()
#     pathfilename = askopenfilename(initialdir='.',
#                                    title='Please, load the first frame of the ptychography experiment...')
# except ImportError:
#     print(
#     'Please, give the full path for the first frame of the ptychography experiment')
#     pathfilename = raw_input('Path:')
#
# filename = pathfilename.rsplit('/')[-1]
# path = pathfilename[:pathfilename.find(pathfilename.rsplit('/')[-1])]
#
# default_recipe = u.Param(
#     first_frame=filename,
#     base_path=path,
# )
#
#
# class ID16Scan(data.PtyScan):
#     """
#     Class ID16Scan
#     Data preparation for far-field ptychography experiments at ID16A beamline - ESRF using FReLoN camera
#     First version by B. Enders (12/05/2015)
#     Modifications by J. C. da Silva (30/05/2015)
#     """
#
#     def __init__(self, pars=None):
#         super(ID16Scan, self).__init__(pars)
#         r = self.info
#         # filename analysis
#         body, ext = os.path.splitext(
#             os.path.expanduser(r.base_path + r.first_frame))
#         sbody = re.sub('\d+$', '', body)
#         num = re.sub(sbody, '', body)
#         # search string for glob
#         self.frame_wcard = re.sub('\d+$', '*', body) + ext
#         # format string for load
#         self.frame_format = sbody + '%0' + str(len(num)) + 'd' + ext
#         # count the number of available frames
#         self.num_frames = len(glob.glob(self.frame_wcard))
#
#     def _frame_to_index(self, fname):
#         body, ext = os.path.splitext(os.path.split(fname)[-1])
#         return int(re.sub(re.sub('\d+$', '', body), '', body)) - 1
#
#     def _index_to_frame(self, index):
#         return self.frame_format % (index + 1)
#
#     def _load_dark(self):
#         r = self.info
#         print('Loading the dark files...')
#         darklist = []
#         for ff in sorted(glob.glob(r.base_path + 'dark*.edf')):
#             d, dheader = io.image_read(ff)
#             darklist.append(d)
#         print('Averaging the dark files...')
#         darkavg = np.array(np.squeeze(darklist)).mean(axis=0)
#         return darkavg
#
#     def load(self, indices):
#         raw = {}
#         pos = {}
#         weights = {}
#         darkavg = self._load_dark()
#         for idx in indices:
#             r, header = io.image_read(self._index_to_frame(idx))
#             img1 = r - darkavg
#             raw[idx] = img1
#             pos[idx] = (
#             header['motor']['spy'] * 1e-6, header['motor']['spz'] * 1e-6)
#         return raw, pos, {}
#
#
# pars = dict(
#     label=None,  # label will be set internally
#     version='0.2',
#     shape=(700, 700),
#     psize=9.552e-6,
#     energy=17.05,
#     center=None,
#     distance=1.2,
#     dfile=filename[:filename.find('.')][:-4].lower() + '.ptyd',
#     # 'siemensstar30s.ptyd',  # filename (e.g. 'foo.ptyd')
#     chunk_format='.chunk%02d',  # Format for chunk file appendix.
#     save='append',  # None, 'merge', 'append', 'extlink'
#     auto_center=None,
#     # False: no automatic center,None only  if center is None, True it will be enforced
#     load_parallel='data',  # None, 'data', 'common', 'all'
#     rebin=2,  # None,  # rebin diffraction data
#     orientation=(True, True, False),
#     # None,int or 3-tuple switch, actions are (transpose, invert rows, invert cols)
#     min_frames=1,  # minimum number of frames of one chunk if not at end of scan
#     positions_theory=None,
#     # Theoretical position list (This input parameter may get deprecated)
#     num_frames=None,  # Total number of frames to be prepared
#     recipe=default_recipe,
# )
# u.verbose.set_level(3)
# IS = ID16Scan(pars)
# IS.initialize()
# IS.auto(400)
