# -*- coding: utf-8 -*-
"""\
Tools specific to the cSAXS beamline, Swiss Light Source. 

TODO: 
    * smarter way to guess parameters.

This file is part of the PTYPY package.

    :copyright: Copyright 2014 by the PTYPY team, see AUTHORS.
    :license: GPLv2, see LICENSE for details.
"""

__all__ = ['prepare_data', 'DataReader']

import os
import numpy as np
import glob
import fnmatch
import time
import re   

from ..utils import verbose
from .. import parameters
from data_structure import DataScan
from .. import io
import spec

lastDR = None

# cSAXS defaults
SAVE_FILENAME_PATTERN = '{write_path}/S{scan_number:05d}_data_{dpsize[0]:03d}x{dpsize[1]:03d}.h5'
WRITE_PATH_PATTERN = '{base_path}/analysis/S{scan_number:05d}'
READ_PATH_PATTERN_CONVERT = '{base_path}/analysis/S{scan_number:05d}'
READ_PATH_PATTERN = '{base_path}/{pilatus_dir}/S{smin:05d}-{smax:05d}/S{scan_number:05d}'
FILENAME_PATTERN = '{read_path}/{prefix}{scan_number:05d}_{index}.{file_extension}'
FILENAME_MULTEXP_PATTERN = '{read_path}/{prefix}{scan_number:05d}_{index}_{exp}.{file_extension}'
PREFIX_PATTERN = '{user}_1_'
INDEX_REPLACEMENT_STRING = '?????'
_PILATUS_SLICES = {(407,487):(slice(636,1043), slice(494,981)),
              (407,1475):(slice(636,1043), slice(0,1475)),
              (831, 1475):(slice(424,1255), slice(0,1475)),
              (1679, 1475):(slice(0,1679), slice(0,1475))}

              
def prepare_data(params):
    """\
    Prepare data according to cSAXS conventions.
    
    Returns a complete DataScan object.

    TODO: Fragment this code in smaller pieces to make it easier to combine 
    with a GUI.
    """
    p = parameters.asParam(params, autoviv=False, none_if_absent=True)

    try:
        verbose_level = p.verbose_level
        verbose.set_level(verbose_level)
    except:
        pass
    
    base_path = p.base_path
    user = p.user
    pilatus_dir = p.pilatus_dir
    pilatus_mask = p.pilatus_mask
    if not p.spec_filename: 
        spec_filename = p.spec_file
    else:
        spec_filename = p.spec_filename
    scan_number = p.scan_number

    DR = DataReader(base_path=base_path, user=user, pilatus_dir=pilatus_dir, pilatus_mask=pilatus_mask, spec_filename=spec_filename)
    DC = DataConverter(base_path=base_path, spec_filename=DR.specinfo)

    # Prepare data 

    file_list, fpattern = DR.get_pilatus_files(scan_number)
    if file_list is None:
        verbose(1, 'Could not find the pilatus files (no match for %s or %s)' % fpattern)
        verbose(1, 'Trying to convert old data instead...')
        data, mask, scandict = DC.read(scan_number, dpsize=p.dpsize)
        meta = None
        #write_path = DR.get_write_path(scan_number)
        #verbose(2, 'Will write to path "%s"' % write_path)
    else:
        data, mask, meta, scandict = DR.read(scan_number, dpsize=p.dpsize, ctr=p.ctr, multexp=False)
            
    s = parameters.Param()
    s.scan_number = p.scan_number
    s.scan_label = 'S%05d' % p.scan_number
    s.data_filename = scandict['data_filename']
    s.wavelength = 1.2398e-9 / p.energy       
    s.energy = p.energy
    s.detector_pixel_size = p.detector_pixel_size
    s.detector_distance = p.detector_distance
    s.initial_ctr = scandict['ctr']
    s.date_collected = scandict['date'].ctime()
    s.date_processed = time.asctime()
    s.exposure_time = scandict['exposure_time']
    if scandict['exposure_time'] is None:
        s.exposure_time = p.exposure_time

    if meta is not None: s.raw_filenames = meta['filename']
    s.preparation_basepath = DR.base_path
    s.preparation_user = DR.user
    s.preparation_spec_file = DR.spec_filename
    s.shape = data.shape

    if p.scan_type is not None: 
        sp = p.scan_params
        if p.scan_type == 'raster':
            #raise RuntimeError('Raster needs to be implemented')
            scanpos = np.array(scan.raster_scan_positions(sp.nx, sp.ny, sp.step_size_x, sp.step_size_y))
        elif p.scan_type == 'round':
            scanpos = np.array(scan.round_scan_positions(sp.radius_in, sp.radius_out, sp.nr, sp.nth))
        elif p.scan_type == 'round_roi':
            scanpos = np.array(scan.round_scan_ROI_positions(sp.dr, sp.lx, sp.ly, sp.nth))
        elif p.scan_type == 'custom':
            scanpos = np.asarray(sp.positions)
        s.positions_theory = scanpos            
    else:
        s.positions_theory = None

    if scandict['spec_command'] is not None:
        # Extract scanning information from spec
        s.spec_command = scandict['spec_command']

        # Position counters
        motors = [] if p.motors is None else p.motors 
        Nmotors = len(motors)
        if Nmotors > 0:
            verbose(3, 'Motors are : %s' % str(motors))
            if np.isscalar(p.motors_multiplier):
                mmult = [p.motors_multiplier]*Nmotors
            else:
                mmult = p.motors_multiplier
            pos_list = [mmult[i]*np.array(scandict['counters'][motors[i]]) for i in range(Nmotors)]
            s.positions = np.array(pos_list).T
        else:
            s.positions = 0.
        # Scan type and parameters
        #TODO: positions = parse_spec_command(scandict.spec_command)

    else:
        s.positions = 0. + s.positions_theory

     # Put everything together in a DataScan object 
    DS = DataScan()

    DS.data = data
    DS.mask = mask
    s.autoviv=False
    DS.scan_info = s
    DS.save()
    return DS

class DataReaderBase(object):
    """\
    cSAXS data reading base class (contains what is in common between DataReader and DataConverter).
    """

    def __init__(self, base_path, spec_filename=None):
        """\
        cSAXS data reading class. The only mendatory argument for this constructor are 'base_path' and 'user'. 
        
        Optional arguments:
            * file_extension [default: '*']
            * pilatus_dir [default: 'pilatus']
            * pilatus_mask (file name or actual mask)
            * spec_file
        """
        
        # Semi-smart base path detection
        if base_path is None:
            d = os.getcwd()
            while True:
                if 'analysis' in os.listdir(d):
                    base_path = d
                    break
                d,rest = os.path.split(d)
                if not rest:
                    break
            if base_path is None:
                raise RuntimeError('Could not figure out base_path')
            verbose(1, 'base_path automatically set to "%s".' % base_path)          
        self.base_path = base_path

               
        self.specinfo = None
        if spec_filename is None:
            spec_base_path = None
            for d in os.listdir(base_path):
                if d.startswith('spec'):
                    spec_base_path = d
                    break
            if spec_base_path is None:
                raise RuntimeError('Could not find spec directory')
            matches = []
            spec_base_path = os.path.join(base_path, spec_base_path,'dat-files')
            for root, dirnames, filenames in os.walk(spec_base_path):
                for filename in fnmatch.filter(filenames, '*.dat'):
                    matches.append(os.path.join(root, filename))      
            if not matches:
                raise RuntimeError('Could not find a spec dat-file (looked into %s)' % spec_base_path)
            if len(matches) > 1:
                print len(matches)
                spec_filename = sorted([(os.stat(x).st_mtime, x) for x in matches])[-1][1]
                verbose(1, 'Found multiple spec files fitting the default location. Using the most recent.')
            else:
                spec_filename = matches[0]
        
        if isinstance(spec_filename, spec.SpecInfo):
            self.specinfo = spec_filename
            self.spec_filename = self.specinfo.spec_filename
        else:
            self.spec_filename = spec_filename
            if spec_filename is not None:
                verbose(1, 'Using spec file: %s' % spec_filename)
                if spec.lastSpecInfo is not None and spec.lastSpecInfo.spec_filename == spec_filename:
                    self.specinfo = spec.lastSpecInfo
                    print 'Using already parsed spec info.'
                else:
                    self.specinfo = spec.SpecInfo(spec_filename)
            else:
                verbose(1, 'Warning: could not open spec_file')
                self.specinfo = None

    def read(self, scan_number, dpsize=None, ctr=None, multexp=False, **kwargs):
        """\
        Read in the data
        TODO: (maybe?) MPI to avoid loading all data in a single process for large scans. 
        """        
        raise NotImplementedError()

    def test(self):
        """\
        TODO: Add simple tests here to check the existance of files and paths. 
        """
        pass

    def summary(self):
        """\
        TODO: Pretty-print content information.
        """
        return 'Not yet implemented'
       
    def get_write_path(self, scan_number):
        return WRITE_PATH_PATTERN.format(base_path=self.base_path, scan_number=scan_number)

    def get_save_filename(self, scan_number, dpsize):
        write_path = self.get_write_path(scan_number)
        return SAVE_FILENAME_PATTERN.format(write_path=write_path,
                                                 scan_number=scan_number,
                                                 dpsize=dpsize)

       
class DataConverter(DataReaderBase):
    """\
    cSAXS data conversion class.
    """
    def __init__(self, base_path, spec_filename=None):
       DataReaderBase.__init__(self, base_path, spec_filename)
       
    def read(self, scan_number, dpsize):
        """\
        Convert old prepared data into new format.
        """
        
        scaninfo = None
        if self.specinfo is not None:
            if not self.specinfo.scans.has_key(scan_number): self.specinfo.parse()
            scaninfo = self.specinfo.scans.get(scan_number, None)
            if scaninfo is None:
                raise RuntimeError('Scan #S %d could not be found in spec file!' % scan_number)

        # Load data
        read_path = self.get_read_path(scan_number)
        file_to_read = (read_path + '/S%05d_data_%dx%d.h5') % (scan_number, dpsize[0], dpsize[1])
        try:
            a = io.h5read(file_to_read)
            data = a['data']
            fmask = a['fmask']
        except IOError:
            file_to_read = (read_path + '/S%05d_data_%dx%d.mat') % (scan_number, dpsize[0], dpsize[1])
            a = io.loadmat(file_to_read)
            data = a['data'].transpose((2,0,1))
            fmask = a['fmask']
            if fmask.ndim==3:
                fmask=fmask.transpose((2,0,1))
        
       
        # Store additional info from spec file
        write_path = self.get_write_path(scan_number)
        data_filename = self.get_save_filename(scan_number, dpsize)
        ctr = tuple(np.asarray(dpsize)/2)
        
        scandict = {}
        scandict['write_path'] = write_path
        scandict['data_filename'] = data_filename
        scandict['read_path'] = read_path
        scandict['dpsize'] = dpsize
        scandict['ctr'] = ctr
        
        scandict['exposure_time'] = None

        if scaninfo is not None:
            scandict['date'] = scaninfo.date
            scandict['counters'] = scaninfo.data
            scandict['motors'] = scaninfo.motors
            scandict['spec_command'] = scaninfo.command
        else:
            scandict['date'] = None
            scandict['counters'] = None
            scandict['motors'] = None
            scandict['spec_command'] = None

        return data, fmask, scandict

    def get_read_path(self, scan_number):
        return READ_PATH_PATTERN_CONVERT.format(base_path=self.base_path, scan_number=scan_number)

class DataReader(DataReaderBase):
    """\
    cSAXS data reading class.
    """

    def __init__(self, base_path, user=None, file_extension='*', pilatus_dir=None, pilatus_mask=None, spec_filename=None):
        """\
        cSAXS data reading class. The only mendatory argument for this constructor are 'base_path' and 'user'. 
        
        Optional arguments:
            * file_extension [default: '*']
            * pilatus_dir [default: 'pilatus']
            * pilatus_mask (file name or actual mask)
            * spec_file
        """
        DataReaderBase.__init__(self, base_path, spec_filename)
        base_path = self.base_path

        # Semi-smart user determination
        if user is None:
            # use system login name
            import getpass
            user = getpass.getuser()
            verbose(1, 'User name automatically set to "%s".' % user)
        self.user = user
            
        self.file_extension = file_extension

        # Semi-smart pilatus directory determination
        if pilatus_dir is None:
            l = os.listdir(base_path)
            for x in l:
                if x.lower().startswith('pilatus'):
                    pilatus_dir = x
                    break
            if pilatus_dir is None:
                raise RuntimeError('Could not figure out pilatus_dir')
            verbose(1, 'pilatus_dir automatically set to "%s".' % pilatus_dir)                          
        self.pilatus_dir = pilatus_dir

        # Look for standard file location
        if pilatus_mask is None:
            candidates = glob.glob(base_path + '/matlab/ptycho/binary_mask*.mat')
            if not candidates:
                verbose(1, 'Could not find a pilatus mask in the default location')
            elif len(candidates) > 1:
                pilatus_mask = sorted([(os.stat(x).st_mtime, x) for x in candidates])[-1][1]
                verbose(1, 'Found multiple pilatus masks fitting the default location')
                verbose(1, 'Using the most recent (%s)' % pilatus_mask)
            else:
                pilatus_mask = candidates[0]
    
        self.pilatus_mask = pilatus_mask       
        if pilatus_mask is not None:
            if str(pilatus_mask) == pilatus_mask:
                verbose(1, 'Using pilatus mask: %s' % str(pilatus_mask))
            else:
                verbose(1, 'Using pilatus mask: <%d x %d array>' % pilatus_mask.shape)
            self.set_pilatus_mask(pilatus_mask)

        # [TODO] Find a more robust way of dealing with the prefix
        self.prefix = PREFIX_PATTERN.format(user=user)
        print PREFIX_PATTERN.format(user=user)
     
    def get_pilatus_files(self, scan_number):
        """\
        Return the list of pilatus files for a given scan_number and the glob pattern.
        Returns None if no file is found.
        """
        fpattern_wildcard = self.get_read_filename(scan_number, index=None)
        file_list = glob.glob(fpattern_wildcard)
        if file_list:
            return file_list, fpattern_wildcard
        else:
            nomatch = fpattern_wildcard
            # Try again with the exposure suffix
            fpattern_wildcard = self.get_read_filename(scan_number, index=None, exposure='00000')
            file_list = glob.glob(fpattern_wildcard)
            if file_list:
                return file_list, fpattern_wildcard
            else:
                return None, (nomatch, fpattern_wildcard)
          
    def read(self, scan_number, dpsize=None, ctr=None, multexp=False, **kwargs):
        """\
        Read in the data
        TODO: (maybe?) MPI to avoid loading all data in a single process for large scans. 
        """        
        verbose(2, 'Processing scan number %d' % scan_number)

        scaninfo = None
        if self.specinfo is not None:
            if not self.specinfo.scans.has_key(scan_number): self.specinfo.parse()
            scaninfo = self.specinfo.scans.get(scan_number, None)
            if scaninfo is None:
                raise RuntimeError('Scan #S %d could not be found in spec file!' % scan_number)

        write_path = self.get_write_path(scan_number)
        
        read_path = self.get_read_path(scan_number)
        verbose(2, 'Will read from path: %s' % read_path)

        if multexp:
            raise RuntimeError('Multiple exposure scans are not yet supported')

        verbose(3, 'Looking for a first file...')      
        file_list, fpattern_wildcard = self.get_pilatus_files(scan_number)
        if file_list is None:
            raise IOError('No file matching "%s" or "%s"' % fpattern_wildcard)
        verbose(3, 'Found %d files matching "%s"' % (len(file_list), fpattern_wildcard))
    
        multexp_stuff = """\
            # multiple exposures
            fpattern = p.filename_multexp_pattern.format(p, exp='*', index='{index}')
            this_fp = fpattern.format(index='00000')
            file_list = glob.glob(this_fp)
            p.multexp = True 
            verbose(3, 'Found a file matching "%s"' % this_fp)
            if not file_list:
                raise IOError('No file match!')
            else:
                verbose(2, 'This is a multiple-exposure dataset')
        """

        # Read in a first dataset to evaluate the size and (eventually) the center.
        first_filename = sorted(file_list)[0]
        f,meta = io.image_read(first_filename, doglob=True)

        amultexp_stuff = """\
        if num_exp > 1:
        
            exp_times = np.array([float(m['time_of_frame']) for m in meta])
    
            min_exp = exp_times.min()
            low_exp_index = exp_times.argmin()
    
            all_same = False
            if min_exp == exp_times.mean():
                # all exposure times are the same
                all_same = True
                verbose(2, 'Exposure time: %f' % min_exp)
            else:
                max_exp = exp_times.max()
                high_exp_index = exp_times.argmax()
                verbose(2, 'Lowest exposure time: %f (number %d)' % (min_exp, low_exp_index))
                verbose(2, 'Highest exposure time: %f (number %d)' % (max_exp, high_exp_index))
        """
        f = f[0]
        sh = f.shape

        if self.pilatus_mask is None:
            fmask = np.ones_like(f)
            verbose(3, 'pilatus_mask is not defined.')
        else:
            verbose(3, 'Using provided pilatus_mask')
            slicetuple = _PILATUS_SLICES.get(sh, None)
            if slicetuple is None:
                raise RuntimeError('Array shape %s is incompatible with known shapes.' % str(sh))
            fmask = self.pilatus_mask[slicetuple]

        fullframe = False
        if dpsize is None:
            dpsize = sh
            verbose(2, 'Full frames (%d x %d) will be saved (so no recentering).' % (sh))
            fullframe = True
        elif np.isscalar(dpsize):
                dpsize = (dpsize,dpsize)
        dpsize = np.array(dpsize)
 
        data_filename = self.get_save_filename(scan_number, dpsize)
        verbose(2, 'Data will be saved to %s' % data_filename)

        if not fullframe:
            # Compute center of mass
            f0 = (f*fmask).sum(axis=0)
            f1 = (f*fmask).sum(axis=1)
            c0 = (np.arange(len(f0))*f0).sum()/f0.sum()
            c1 = (np.arange(len(f1))*f1).sum()/f1.sum()

            ctr_auto = (c1, c0)
    
            # Check for center position
            if ctr is None:
                ctr = ctr_auto
                verbose(2, 'Using center: (%d, %d)' % ctr)
            elif ctr == 'inter':
                import matplotlib as mpl
                fig = mpl.pyplot.figure()
                ax = fig.add_subplot(1,1,1)
                ax.imshow(np.log(f))
                ax.set_title('Select center point (hit return to finish)')
                s = U.Multiclicks(ax, True, mode='replace')
                mpl.pyplot.show()
                s.wait_until_closed()
                ctr = np.round(np.array(s.pts[0][::-1]));
                verbose(2, 'Using center: (%d, %d) - I would have guessed it is (%d, %d)' % (ctr[0], ctr[1], ctr_auto[0], ctr_auto[1]))
            else:
                verbose(2, 'Using center: (%d, %d) - I would have guessed it is (%d, %d)' % (ctr[0], ctr[1], ctr_auto[0], ctr_auto[1]))
    
            ctr = np.array(ctr)
            lim_inf = ctr - dpsize/2.
            lim_sup = ctr + dpsize/2.
            if (lim_inf < 0).any() or (lim_sup >= np.array(sh)).any():
                verbose(1, 'Warning: chosen center is too close to the edge! Changing the coordinates to make it fit.')
                out_string = 'From ' + str(ctr)
                ctr -= lim_inf * (lim_inf < 0)
                ctr -= (lim_sup - np.array(sh)) * (lim_sup >= np.array(sh))
                lim_inf = ctr - dpsize/2.
                lim_sup = ctr + dpsize/2.
                out_string += ' to ' + str(ctr)
                verbose(1, out_string)
    
            fmask = fmask[lim_inf[0]:lim_sup[0], lim_inf[1]:lim_sup[1]]
    
        # Prepare the general meta-information dictionnary
        meta_list = dict([(k, []) for k in meta[0].keys()])
       
        if fullframe:
            f,meta = io.image_read(fpattern_wildcard, doglob=True)
        else:
            f,meta = io.image_read(fpattern_wildcard, doglob=True, roi=(lim_inf[0],lim_sup[0],lim_inf[1],lim_sup[1]))
    
        for mm in meta:
            for k,v in mm.items():
                meta_list[k].append(v)
    
        npts = len(f)
        verbose(2, 'Read %d files' % npts)
 
        # Store in a 3D array
        data = np.zeros((npts, dpsize[0], dpsize[1]), dtype=np.single);
        for nn in range(npts):
            data[nn,:,:] = f[nn]

        del f
        
        # Store additional info from spec file
        scandict = {}
        scandict['write_path'] = write_path
        scandict['data_filename'] = data_filename
        scandict['read_path'] = read_path
        scandict['dpsize'] = dpsize
        scandict['ctr'] = ctr
        
        try:
            rawheader = meta_list['rawheader'][0].lower()
            scandict['exposure_time'] = float(rawheader[rawheader.find('exposure_time'):].split()[1])
        except KeyError:
            pass

        if scaninfo is not None:
            scandict['date'] = scaninfo.date
            scandict['counters'] = scaninfo.data
            scandict['motors'] = scaninfo.motors
            scandict['spec_command'] = scaninfo.command
            if not scandict.has_key('exposure_time'):
                scandict['exposure_time'] = float(scaninfo.command.split()[-1])
        else:
            # Try to extract date from first frame
            tm = re.findall("([0-9]{4}-[0-9]{2}-[0-9]{2}[^\n]*)", rawheader)
            if tm: 
                t = tm[0].strip().split('.')[0]  # remove the fraction of second
                scandict['date'] = time.strptime(t, "%Y-%m-%dt%H:%M:%S")
            scandict['counters'] = None
            scandict['motors'] = None
            scandict['spec_command'] = None

        return data, fmask, meta_list, scandict


    def get_read_path(self, scan_number):
        return READ_PATH_PATTERN.format(base_path=self.base_path,
                                                  pilatus_dir=self.pilatus_dir,
                                                  smin=int(1000*np.floor(scan_number/1000.)),
                                                  smax=int(1000*np.floor(scan_number/1000.)+999),
                                                  scan_number=scan_number)

    def get_read_filename(self, scan_number, index=None, exposure=None):
        read_path = self.get_read_path(scan_number)
        if index is None:
            index = INDEX_REPLACEMENT_STRING
        else:
            try:
                index = '%05d' % index
            except TypeError:
                pass
        if exposure is None:
            filename = FILENAME_PATTERN.format(read_path=read_path,
                                                    prefix=self.prefix,
                                                    scan_number=scan_number,
                                                    index=index,
                                                    file_extension=self.file_extension)
        else:
            try:
                exposure = '%05d' % exposure
            except TypeError:
                pass
            filename = FILENAME_MULTEXP_PATTERN.format(read_path=read_path,
                                                            prefix=self.prefix,
                                                            scan_number=scan_number,
                                                            index=index,
                                                            exp=exposure,
                                                            file_extension=self.file_extension)
        return filename
        
    def set_pilatus_mask(self,mask):
        """\
        Stores a pilatus mask (or loads it if the provided input is a filename).
        """
        if type(mask)==str:
            try:
                mask,meta = io.image_read(mask)
            except IOError:
                mask = io.loadmat(mask)['mask']
                mask = np.rot90(mask,2)
        self.pilatus_mask = (mask != 0)

