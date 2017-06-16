# -*- coding: utf-8 -*-
"""\
Tools specific to the I13 beamline, Diamond.

TODO:
    * smarter way to guess parameters.

This file is part of the PTYPY package.

    :copyright: Copyright 2014 by the PTYPY team, see AUTHORS.
    :license: GPLv2, see LICENSE for details.
"""

__all__ = ['prepare_data', 'DataReader']

from matplotlib import pyplot as plt
import ptypy.utils as u
import numpy as np
import glob
import os
import fnmatch
import time
import re

from ptypy.utils.scripts import expect2, mass_center
from ptypy.utils.verbose import logger
from ptypy import io

lastDR = None

# I13DLS defaults
SAVE_FILENAME_PATTERN = '{write_path}/{scan_label}_data_{dpsize[0]:03d}x{dpsize[1]:03d}.h5'
WRITE_PATH_PATTERN = '{base_path}/processing/analysis/{scan_label}'
#READ_PATH_PATTERN = '{base_path}/raw/{scan_number:05d}/mpx'
NEXUS_PATTERN = '{base_path}/raw/{scan_number:05d}.nxs'
FRAME_IN_NEXUS_FILE = 'entry1/instrument/{detector_name}/data'
EXPOSURE_IN_NEXUS_FILE = 'entry1/instrument/{detector_name}/count_time'
MOTORS_IN_NEXUS_FILE = 'entry1/instrument/t1_sxy'
COMMAND_IN_NEXUS_FILE = 'entry1/scan_command'
LABEL_IN_NEXUS_FILE = 'entry1/entry_identifier'
EXPERIMENT_IN_NEXUS_FILE = 'entry1/experiment_identifier'
#FILENAME_PATTERN = '{read_path}/{index}.{file_extension}'
#INDEX_REPLACEMENT_STRING = '????'

# Standard parameter structure for prepare_data
DEFAULT = u.Param(
    verbose_level = 2, # Verbosity level
    base_path = None, #'The root directory where all experiment information can be found'
    detector_name = None, # That is the name of the detector in the nexus file
    experimentID = None, #'Experiment identifier'
    scan = None, #'Scan number' or file
    dpsize = None, #'Cropped array size for one frame'
    ctr = None, #'Center of the diffraction pattern.'
    energy = None, #'Nominal Radiation energy in keV'
    detector_pixel_size = None, #'Detector pixel size in meters'
    detector_distance = None, #'Detector distance to sample in meters'
    motors = ['t1_sy','t1_sx'], #'Motor names to determine the sample translation'
    motors_multiplier = 1e-6, #'Motor conversion factor to meters'
    mask = None, #'Mask or name or number of the file containing the mask'
    dark = None, #'Dark frame or name or number of the file containing the dark'
    flat = None, #'Flat frame or name of number the file containing the flat'
    flip = None, #None, 'lr','ud'
    rebin = 1,
    rotate = False,
)

def flip(A,what):
    if what is None:
        return A
    elif what == 'lr':
        return A[...,::-1,:]
    elif what == 'ud':
        return A[...,::-1]
    elif what in ['all', 'udlr']:
        return A[...,::-1,::-1]
    else:
        raise RuntimeError('Unknown flipping option %s' % str(what))

def load(filename,name=None):
    """\
    Helper function to read mask, flat and dark.
    """

    #try:
    #    a = io.h5read(filename)
    #except:
    #    a = io.loadmat(filename)

    if name is not None:
        return io.h5read(filename,name)[name]
    else:
        return io.h5read(filename).values()[0]    # Take first array we find - hint: better have a single array in the file.

class DataReader(object):
    """\
    I13 DLS data reading class.
    Set flat, dark and mask correction for all data read
    """

    def __init__(self, pars=None):#base_path, experimentID=None, mask=None, flat=None, dark=None):
        """\
        I13 DLS data reading class. The only mandatory argument for this constructor are 'base_path' and 'user'.

        Mask, flat or dark should be either None, scan numbers (integer) or file paths
        """
        # Semi-smart base path detection: we are looking for a "raw" directory
        p = u.Param(DEFAULT)
        if pars is not None:
            p.update(pars)

        try:
            verbose_level = p.verbose_level
            verbose.set_level(verbose_level)
        except:
            pass
        self.p=p
        base_path = p.base_path if p.base_path.endswith('/') else p.base_path + '/'
        experimentID = p.experimentID

        if base_path is None:
            d = os.getcwd()
            while True:
                if 'raw' in os.listdir(d):
                    base_path = d
                    break
                d,rest = os.path.split(d)
                if not rest:
                    break
            if base_path is None:
                raise RuntimeError('Could not figure out base_path')
            logger.debug( 'base_path: "%s" (automatically set).' % base_path)
        else:
            logger.debug( 'base_path: "%s".' % base_path)
        self.base_path = base_path

        self.nxs = u.Param()
        self.nxs.frame = FRAME_IN_NEXUS_FILE.format(**p)
        self.nxs.exp = EXPOSURE_IN_NEXUS_FILE.format(**p)
        self.nxs.motors = MOTORS_IN_NEXUS_FILE.format(**p)
        self.nxs.command = COMMAND_IN_NEXUS_FILE.format(**p)
        self.nxs.experiment = EXPERIMENT_IN_NEXUS_FILE.format(**p)
        self.nxs.label = LABEL_IN_NEXUS_FILE.format(**p)
        # Load mask, flat and dark

        try:
            self.mask = load(self.get_nexus_file(p.mask),self.nxs.frame)
        except:
            self.mask = p.mask
            #assert self.mask.shape[-2:] == sh

        try:
            self.dark = load(self.get_nexus_file(p.dark),self.nxs.frame)
        except:
            self.dark = p.dark
            #assert self.dark.shape[-2:] == sh

        try:
            self.flat = load(self.get_nexus_file(p.flat),self.nxs.frame)
        except:
            self.flat = p.flat
            #assert self.flat.shape[-2:] == sh


    def read(self, scan=None, **kwargs):
        """\
        Read in the data
        TODO: (maybe?) MPI to avoid loading all data in a single process for large scans.
        """

        scan=scan if scan is not None else self.p.scan
        logger.info( 'Processing scan number %s' % str(scan))

        self.scan = self.get_nexus_file(scan)
        logger.debug( 'Data will be read from path: %s' % self.scan)

        self.exp = load(self.scan,self.nxs.frame)
        try:
            self.motors = load(self.scan,self.nxs.motors)
        except:
            self.motors=None
        self.command = load(self.scan,self.nxs.command)
        self.data = load(self.scan, self.nxs.frame).astype(float)
        self.label = load(self.scan, self.nxs.label)[0]


        if self.p.experimentID is None:
            try:
                experimentID = load(self.scan, self.nxs.experiment)[0]
            except:
                logger.debug('Could not find experiment ID from nexus file %s.' % self.scan)
                experimentID = os.path.split(base_path[:-1])[1]
            logger.debug( 'experimentID: "%s" (automatically set).' % experimentID)
        else:
            logger.debug( 'experimentID: "%s".' % self.p.experimentID)
            self.experimentID = self.p.experimentID



        dpsize = self.p.dpsize
        ctr =  self.p.ctr

        sh = self.data.shape[-2:]
        fullframe = False
        if dpsize is None:
            dpsize = sh
            logger.debug( 'Full frames (%d x %d) will be saved (so no recentering).' % (sh))
            fullframe = True
        self.p.dpsize = expect2(dpsize)

        #data_filename = self.get_save_filename(scan_number, dpsize)
        #logger.info( 'Data will be saved to %s' % data_filename)
        f = self.data
        if not fullframe:
            # Compute center of mass
            if self.mask is None:
                ctr_auto = mass_center(f.sum(0))
            else:
                ctr_auto = mass_center(f.sum(0)*self.mask)
            print ctr_auto
            # Check for center position
            if ctr is None:
                ctr = ctr_auto
                logger.debug( 'Using center: (%d, %d)' % (ctr[0],ctr[1]))

            #elif ctr == 'inter':
                #import matplotlib as mpl
                #fig = mpl.pyplot.figure()
                #ax = fig.add_subplot(1,1,1)
                #ax.imshow(np.log(f))
                #ax.set_title('Select center point (hit return to finish)')
                #s = u.Multiclicks(ax, True, mode='replace')
                #mpl.pyplot.show()
                #s.wait_until_closed()
                #ctr = np.round(np.array(s.pts[0][::-1]));
                #logger.debug( 'Using center: (%d, %d) - I would have guessed it is (%d, %d)' % (ctr[0], ctr[1], ctr_auto[0], ctr_auto[1]))

            else:
                logger.debug( 'Using center: (%d, %d) - I would have guessed it is (%d, %d)' % (ctr[0], ctr[1], ctr_auto[0], ctr_auto[1]))

            self.dpsize = dpsize
            self.ctr = np.array(ctr)
            lim_inf = -np.ceil(ctr - dpsize/2.).astype(int)
            lim_sup = np.ceil(ctr + dpsize/2.).astype(int) - np.array(sh)
            hplane_list = [(lim_inf[0], lim_sup[0]), (lim_inf[1], lim_sup[1])]
            logger.debug( 'Going from %s to %s (hplane_list = %s)' % (str(sh), str(dpsize), str(hplane_list)))
            if self.mask is not None: self.mask = u.crop_pad(self.mask, hplane_list).astype(bool)
            if self.flat is not None: self.flat = u.crop_pad(self.flat, hplane_list,fillpar=1.)
            if self.dark is not None: self.dark = u.crop_pad(self.dark, hplane_list)
            if self.data is not None: self.data = u.crop_pad(self.data, hplane_list)
        # obsolete

        #return data, self.mask, self.flat, self.dark

    def prepare(self,scan=None,filename=None,dtype=np.uint32,**kwargs):

        self.p.update(kwargs) #rebin = rebin if rebin is not None else self.p.rebin

        scan = scan if scan is not None else self.p.scan

        self.read( scan, **kwargs)
        DS = u.Param()

        dark = self.dark
        data = self.data
        if dark is not None:
            if dark.ndim == 3:
                dark = dark.mean(0)
            dark = np.resize(dark,data.shape)

        flat = self.flat
        if flat is not None:
            if flat.ndim == 3:
                flat = flat.mean(0)
            flat = np.resize(flat,self.data.shape)
        #plt.ion();
        #plt.figure();plt.imshow(dark[0]);plt.colorbar()
        #plt.figure();plt.imshow((flat-dark)[0]);plt.colorbar()
        #plt.figure();plt.imshow(data[0]);plt.colorbar()
        if flat is not None and dark is not None:
            data = (data-dark )/(flat-dark)
        elif dark is not None:
            data = data - dark
        else:
            data = data
        # remove negative values
        data[data<0]=0
        #
        #plt.figure();plt.imshow(DS.data[0],vmin=0);plt.colorbar()
        #
        if self.mask is None:
            mask = np.ones_like(data,dtype=np.bool)
        #
        #DS.flat = self.flat
        #DS.dark = self.dark
        DS.scan_info = u.Param()
        s = DS.scan_info
        p = self.p
        #s.scan_number = p.scan_number
        s.scan_label = 'S'+self.label #'S%05d' % p.scan_number
        s.data_filename = self.scan #scandict['data_filename']
        s.wavelength = u.keV2m( p.energy )
        s.energy = p.energy
        rebin = self.p.rebin
        s.detector_pixel_size = p.detector_pixel_size * rebin if p.detector_pixel_size is not None else None
        p.dpsize = p.dpsize / rebin
        s.detector_distance = p.detector_distance
        s.initial_ctr = self.ctr / rebin
        if rebin!=1:
            sh = data.shape
            data = u.rebin(data,sh[0],sh[1]/rebin,sh[2]/rebin)
            mask = u.rebin(mask.astype(int),sh[0],sh[1]/rebin,sh[2]/rebin).astype(bool)

        data = flip(data,self.p.flip)
        mask = flip(mask,self.p.flip)

        DS.data = data.astype(dtype)
        DS.mask = mask
        DS.data[np.invert(DS.mask)]=0
        #s.date_collected = scandict['date']
        s.date_processed = time.asctime()
        s.exposure_time = self.exp

        #if meta is not None: s.raw_filenames = meta['filename']
        s.preparation_basepath = self.base_path
        s.preparation_other = {}
        s.shape = DS.data.shape

        s.positions_theory = None

        s.scan_command = self.command

        motors = p.motors
        if self.motors is not None:
            Nmotors = len(motors)
            logger.debug( 'Motors are : %s' % str(p.motors))
            mmult = u.expect2(p.motors_multiplier)

            pos_list = [mmult[i]*np.array(self.motors[motors[i]]) for i in range(Nmotors)]
            s.positions = np.array(pos_list).T
        else:
            s.positions = None

        self.p.scan_label = s.scan_label
        if filename is None:
            p.write_path = WRITE_PATH_PATTERN.format(**p)
            filename = SAVE_FILENAME_PATTERN.format(**p)

        s.data_filename = u.clean_path(filename)
        io.h5write(filename,DS)
        return DS

    def get_nexus_file(self, number):
        if str(number)==number or number is None:
            # ok this is not a number but a filename
            return number
        else:
            return NEXUS_PATTERN.format(base_path=self.base_path,
                                     scan_number=number)
