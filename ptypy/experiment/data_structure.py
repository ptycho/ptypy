# -*- coding: utf-8 -*-
"""\
Utility function related to experiments.

This file is part of the PTYPY package.

    :copyright: Copyright 2014 by the PTYPY team, see AUTHORS.
    :license: GPLv2, see LICENSE for details.
"""
import numpy as np
import copy
import os

from ..utils import verbose, expect2, clean_path, mass_center
from .. import parameters
from .. import io

__all__ = ['DataScan']

DEFAULT = u.Param(
    scan_number =  None, #Scan number
    scan_label =  None, #A string associated with the scan number
    data_filename =  None, #The file name the data file is going to be saved to
    wavelength =  None, #The radiation wavelength in meters
    energy =  None, #The photon energy 
    detector_pixel_size =  None, #Detector pixel dimensions in meters
    detector_distance =  None, #Distance between the detector and the sample in meters
    initial_ctr =  None, #The coordinate that locate the center of the frame in the full detector frame
    date_collected =  None, #Date data was collected
    date_processed =  None, #Date data was processed
    exposure_time =  None, #Exposure time in seconds
    preparation_basepath =  None, #Base path used for the data preparation
    preparation_other =  None, #Other information about preparation (may be beamline-specific)
    shape =  None, #The shape of one frame
    positions = None, #Positions (if possible measured) of the sample for each frame in meters
    positions_theory = None, #Expected positions of the sample in meters
    scan_command =  None, #The command entered to run the scan
)


class DataScan(object):
    """\
    This class represents a single ptychography scan.
    """
    
    DEFAULT=DEFAULT

    def __init__(self, filename=None):
        """\
        Empty object unless a filename is provided
        """
        self.data = None
        self.mask = None
        self.flat = None
        self.dark = None
        self.scan_info = u.Param(self.DEFAULT)
        if filename is None: return
        
        self.load_info(filename)
        self.load_data(filename)

    def load_info(self, filename):
        """\
        Load only information (scan_info) from prepared file.
        """
        self.scan_info.update(u.asParam(io.h5read(filename, 'scan_info')['scan_info']))

    def load_data(self, filename=None, roidim=None, roictr=None):
        """\
        Load data.
        """
        if filename is None:
            filename = self.scan_info.data_filename
        verbose(3, 'Loading data file %s' % filename)

        dpsize = expect2(self.scan_info.shape[1:])
        verbose(3, 'Frame size is %dx%d' % tuple(dpsize))

        if roidim is not None:
            roidim = expect2(roidim)
            verbose(3, 'Loading a region of interest %dx%d' % tuple(roidim)) 
            if roictr is None:
                roictr = np.asarray(dpsize) // 2
            else:
                roictr = expect2(roictr)
            verbose(3, 'Region of interest centered at (%d,%d)' % tuple(roictr))

            self.asize = roidim
            slice_string = '%d:%d,%d:%d' % (int(np.ceil(roictr[0] - roidim[0]/2.)),\
                                              int(np.ceil(roictr[0] + roidim[0]/2.)),\
                                              int(np.ceil(roictr[1] - roidim[1]/2.)),\
                                              int(np.ceil(roictr[1] + roidim[1]/2.)))
            self.data = io.h5read(filename, ('data[:,%s]' % slice_string))['data']

            # Backward compatibility
            try:
                self.mask = io.h5read(filename, ('mask[...,%s]' % slice_string))['mask']
            except:
                self.mask = io.h5read(filename, ('fmask[...,%s]' % slice_string))['fmask']

            try:
                self.flat = io.h5read(filename, ('flat[...,%s]' % slice_string))['flat']
            except:
                self.flat = None
                verbose(3, 'No flat frame was found.')

            try:
                self.dark = io.h5read(filename, ('dark[...,%s]' % slice_string))['dark']
            except:
                self.dark = None
                verbose(3, 'No dark frame was found.')

        else:
            self.asize = self.scan_info.shape[1:]
            verbose(3, 'Loading full frames (%dx%d)' % tuple(self.asize))
            tmp = io.h5read(filename)
            self.data = tmp['data']
            self.mask = tmp['fmask'] if tmp.has_key('fmask') else tmp['mask']
            self.flat = tmp['flat'] if tmp.has_key('flat') else None
            self.dark = tmp['dark'] if tmp.has_key('dark') else None  

    def unload_data(self):
        """\
        Deletes the "data" and "fmask" arrays.
        """
        del self.data
        del self.mask

    def save(self, filename=None, force_overwrite=True):
        """\
        Store this dataset in a standard format.
        If filename is None, use default (scan_info.data_filename)
        """
        # Check if data is available and the right size
        if not hasattr(self, 'data'):
            raise RuntimeError("Attempting to save DataScan instance that does not contain data.")

        if self.data.shape != self.scan_info.shape:
            error_string = "Attempting to save DataScan instance with non-native data dimension "
            error_string += "[data.shape = %s, while scan_info.shape = %s]" % (str(self.data.shape), str(self.scan_info.shape))
            raise RuntimeError(error_string)

        if filename is None:
            filename = self.scan_info.data_filename           

        filename = clean_path(filename)
        if os.path.exists(filename):
            if force_overwrite:
                verbose(1, 'Save file exists but will be overwritten (force_overwrite is True)')
            elif not force_overwrite:
                raise RuntimeError('File %s exists! Operation cancelled.' % filename)
            elif force_overwrite is None:
                ans = raw_input('File %s exists! Overwrite? [Y]/N' % filename)
                if ans and ans.upper() != 'Y':
                    raise RuntimeError('Operation cancelled by user.') 

        h5opt = io.h5options['UNSUPPORTED']
        io.h5options['UNSUPPORTED'] = 'ignore'
        io.h5write(filename, data=self.data, mask=self.mask, flat=self.flat, dark=self.dark, scan_info=self.scan_info._to_dict())
        io.h5options['UNSUPPORTED'] = h5opt
        verbose(3, 'Scan %s data saved to %s.' % (self.scan_info.scan_label, filename))
        return
       
       
class Data(object):
    """\
    Global data container.
    
    Holds all data scans and provides a flat list of all diffraction patterns.
    """
    def __init__(self, scandict):
        pass
        
    def from_scandict(self, scandict):
        """\       
        scandict is a dictionary containing DataScan objects, named by the keys.
        """
        self.initial_scandict = scandict
        
        # Quantities that will have to be uniform for all scans
        wavelengths = []
        detector_distances = []
        detector_pixel_sizes = []
        dpsizes = []
        
        scan_names = []
        scans = {}
        for scan_name, scan in scandict.iteritems():
            scan_names.append(scan_name)
            scans[scan_name] = scan           
            wavelengths.append(scan['wavelength'])
            detector_distances.append(scan['detector_distance'])
            detector_pixel_sizes.append(scan['detector_pixel_size'])
            dpsizes.append(scan['dpsize'])

        # Uniform quantities
        if not np.allclose(wavelengths, wavelengths[0]):
            raise RuntimeError('Reconstruction from scans at different energies is not supported.')
        else:
            self.wavelength = 0. + wavelengths[0]

        if not np.allclose(detector_distances, detector_distances[0]):
            raise RuntimeError('Reconstruction from scans at different detector distances is not supported.')
        else:
            self.detector_distance = 0. + detector_distances[0]

        if not np.allclose(detector_pixel_sizes, detector_pixel_sizes[0]):
            raise RuntimeError('Reconstruction from scans with different detector pixel sizes is not supported.')
        else:
            self.detector_pixel_size = 0. + detector_pixel_sizes[0]

        # Reconstruction pixel size for near-field ptychography
        self.dxnf = self.detector_pixel_size

        if not np.allclose(dpsizes, dpsizes[0]):
            raise RuntimeError('Reconstruction from scans with different array sizes is not supported.')
        else:
            self.dpsize = 0. + dpsizes[0]
                
    def __getitem__(self, key):
        return self.scans[key]
    
    def keys(self):
        """ Same as self.scan_names """
        return self.scans.keys()

    def _prepare_files(self):
        pass
    
    def _load_intensities(self):
        pass        
    def _load_magnitudes(self):
        pass
    
    @property
    def energy(self):
        return 1.2398e-9 / self.wavelength
      
    @property
    def dx(self):
        """\
        Reconstruction pixel size for far-field ptychography.
        """
        return self.wavelength * self.detector_distance / (self.dpsize * self.detector_pixel_size)

