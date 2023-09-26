"""  
    Implementation of PtyScan subclasses to hold MAXIV-SoftiMAX scan data.
    The beamline is developing, as is the data format. 
"""

from ..core.data import PtyScan
from .. import utils as u
from . import register
logger = u.verbose.logger

import numpy as np
try:
	import hdf5plugin
except ImportError:
	logger.warning('Couldnt find hdf5plugin - better hope your h5py has bitshuffle!')
import h5py
import os.path
import time



@register()
class SoftiMAX_Sep2021(PtyScan):
    """
    Starting a fresh class here.

    Defaults:

    [name]
    default = SoftiMAX_Sep2021
    type = str
    help =

    [path]
    default = None
    type = str
    help = base directory of that beamtime including the visit folder
    doc =

    [scanNumber_data]
    default = None
    type = int, list, tuple
    help = Scan number or list of scan numbers
    doc =

    [scanNumber_background]
    default = None
    type = int, list, tuple
    help = Scan number or list of scan numbers
    doc =

    [xMotor]
    default = samx
    type = str
    help = Which x motor to use
    doc =

    [yMotor]
    default = samy
    type = str
    help = Which y motor to use
    doc =

    [xMotorFlipped]
    default = False
    type = bool
    help = Flip detector x positions
    doc =

    [yMotorFlipped]
    default = False
    type = bool
    help = Flip detector y positions
    doc =

    [maskfile]
    default = None
    type = str
    help = Arbitrary mask file
    doc = Hdf5 file containing an array called 'mask' at the root level.

    """

    def load_positions(self):
        """
        Provides the relative sample positions inside the scan. 
        """

        fullfilename = os.path.join(self.info.path, 'Max_'+str(self.info.scanNumber_data)+'.h5')
        self.frames_per_scan = {}

        xFlipper, yFlipper = 1, 1
        if self.info.xMotorFlipped:
            xFlipper = -1
            logger.warning("note: x motor is specified as flipped")
        if self.info.yMotorFlipped:
            yFlipper = -1
            logger.warning("note: y motor is specified as flipped")

        x, y = [], []
        with h5py.File(fullfilename, 'r') as hf:
            x.append(xFlipper * np.array(hf['entry/data/%s' % (self.info.xMotor)]))
            y.append(yFlipper * np.array(hf['entry/data/%s' % (self.info.yMotor)]))

        # make lists to two arrays
        x = np.array(x)*1e-6
        y = np.array(y)*1e-6 
        x = x-np.min(x)  
        y = y-np.min(y)  
        positions = np.vstack((y, x)).T
        return positions[0]

    def load(self, indices):
        """
        Provides the raw diffraction pattern from the detector (eiger .h5) file.
        Normalizes them, if there is normalization data. 
        """

        raw, weights, positions = {}, {}, {}

        fullfilename_background = os.path.join(self.info.path, 'Max_'+str(self.info.scanNumber_background)+'.h5')
        with h5py.File(fullfilename_background, 'r') as fp:
            data_background = fp['entry/data/data/']
            data_background = np.mean(data_background, axis=0)

        fullfilename_data       = os.path.join(self.info.path, 'Max_'+str(self.info.scanNumber_data)+'.h5')
        with h5py.File(fullfilename_data, 'r') as fp:
            for ind in indices:
                raw[ind] = fp['entry/data/data/'][ind]-data_background
                raw[ind][raw[ind]<=0] = 0

        return raw, positions, weights

    def load_weight(self):
        """
        Provides the mask used for every diffraction pattern in the whole scan
        This mask will have the shape of the first frame.
        """

        r, w, p = self.load(indices=(0,))
        data = r[0]
        mask = np.ones_like(data)
        mask[np.where(data < 0)] = 0

        if self.info.maskfile:
            with h5py.File(self.info.maskfile, 'r') as hf:
                mask2 = np.array(hf.get('mask'))
#            logger.info("loaded additional mask, %u x %u, sum %u" %
#                        (mask2.shape + (np.sum(mask2),)))
            mask = mask * mask2
#            logger.info("total mask, %u x %u, sum %u" %
#                        (mask.shape + (np.sum(mask),)))

        return mask

@register()
class SoftiMAX_Nov2021(PtyScan):


    """
    This class reads the positions from the meta data of the camera frames. Needs to be adjusted to any changes in the format provided by STXM control.

    Defaults:

    [name]
    default = SoftiMAX_Nov2021
    type = str
    help =

    [path]
    default = None
    type = str
    help = base directory of that beamtime including the visit folder
    doc =

    [scanNumber_data]
    default = None
    type = int, list, tuple
    help = Scan number or list of scan numbers
    doc =

    [scanNumber_background]
    default = None
    type = int, list, tuple
    help = Scan number or list of scan numbers
    doc =

    [xMotor]
    default = samx
    type = str
    help = Which x motor to use
    doc =

    [yMotor]
    default = samy
    type = str
    help = Which y motor to use
    doc =

    [xMotorFlipped]
    default = False
    type = bool
    help = Flip detector x positions
    doc =

    [yMotorFlipped]
    default = False
    type = bool
    help = Flip detector y positions
    doc =

    [maskfile]
    default = None
    type = str
    help = Arbitrary mask file
    doc = Hdf5 file containing an array called 'mask' at the root level.

    """

    def load_positions(self):

        """
        Provides the relative sample positions inside the scan. 
        """

        fullfilename = os.path.join(self.info.path, 'Max_'+str(self.info.scanNumber_data)+'.h5')
        self.frames_per_scan = {}

        xFlipper, yFlipper = 1, 1
        if self.info.xMotorFlipped:
            xFlipper = -1
            logger.warning("note: x motor is specified as flipped")
        if self.info.yMotorFlipped:
            yFlipper = -1
            logger.warning("note: y motor is specified as flipped")

        x, y = np.array([]), np.array([])

        with h5py.File(fullfilename, 'r') as f:
            for k in f['entry/data/data'].attrs.keys():
                n = f['/entry/data/data'].attrs[k]
                n2 = n.decode()
                n3 = n2.split()
                x = np.append(x,n3[0])
                y = np.append(y,n3[1])

        x = x.astype(float)
        y = y.astype(float)
        
        # make lists to two arrays
        x = xFlipper * x[np.newaxis,:,np.newaxis].T * 1e-6
        y = yFlipper * y[np.newaxis,:,np.newaxis].T * 1e-6
        x = x-np.min(x)  
        y = y-np.min(y)
        positions = np.vstack((y, x)).T 
        return positions[0]

    def load(self, indices):
        """
        Provides the raw diffraction pattern from the detector (eiger .h5) file.
        Normalizes them, if there is normalization data. 
        """

        raw, weights, positions = {}, {}, {}

        fullfilename_background = os.path.join(self.info.path, 'Max_'+str(self.info.scanNumber_background)+'.h5')
        with h5py.File(fullfilename_background, 'r') as fp:
            data_background = fp['entry/data/data/']
            data_background = np.mean(data_background, axis=0)

        fullfilename_data       = os.path.join(self.info.path, 'Max_'+str(self.info.scanNumber_data)+'.h5')
        with h5py.File(fullfilename_data, 'r') as fp:
            for ind in indices:
                raw[ind] = fp['entry/data/data/'][ind]-data_background
                raw[ind][raw[ind]<=0] = 0

        return raw, positions, weights

    def load_weight(self):
        """
        Provides the mask used for every diffraction pattern in the whole scan
        This mask will have the shape of the first frame.
        """

        r, w, p = self.load(indices=(0,))
        data = r[0]
        mask = np.ones_like(data)
        mask[np.where(data < 0)] = 0

        if self.info.maskfile:
            with h5py.File(self.info.maskfile, 'r') as hf:
                mask2 = np.array(hf.get('mask'))
#            logger.info("loaded additional mask, %u x %u, sum %u" %
#                        (mask2.shape + (np.sum(mask2),)))
            mask = mask * mask2
#            logger.info("total mask, %u x %u, sum %u" %
#                        (mask.shape + (np.sum(mask),)))

        return mask





@register()
class SoftiMAX_Dec2021(PtyScan):


    """
    This class reads the positions from the meta data of the camera frames. Needs to be adjusted to any changes in the format provided by STXM control.

    Defaults:

    [name]
    default = SoftiMAX_Nov2021
    type = str
    help =

    [path]
    default = None
    type = str
    help = base directory of that beamtime including the visit folder
    doc =

    [scanNumber_data]
    default = None
    type = int, list, tuple
    help = Scan number or list of scan numbers
    doc =

    [scanNumber_background]
    default = None
    type = int, list, tuple
    help = Scan number or list of scan numbers
    doc =

    [xMotor]
    default = samx
    type = str
    help = Which x motor to use
    doc =

    [yMotor]
    default = samy
    type = str
    help = Which y motor to use
    doc =

    [xMotorFlipped]
    default = False
    type = bool
    help = Flip detector x positions
    doc =

    [yMotorFlipped]
    default = False
    type = bool
    help = Flip detector y positions
    doc =

    [maskfile]
    default = None
    type = str
    help = Arbitrary mask file
    doc = Hdf5 file containing an array called 'mask' at the root level.

    """

    def load_positions(self):

        """
        Provides the relative sample positions inside the scan. 
        """

        fullfilename = os.path.join(self.info.path, str(self.info.scanNumber_data).zfill(6)+'.h5')
        self.frames_per_scan = {}

        xFlipper, yFlipper = 1, 1
        if self.info.xMotorFlipped:
            xFlipper = -1
            logger.warning("note: x motor is specified as flipped")
        if self.info.yMotorFlipped:
            yFlipper = -1
            logger.warning("note: y motor is specified as flipped")

        x, y = [], []
        with h5py.File(fullfilename, 'r') as hf:
            x = xFlipper * np.array(hf['entry/measurement/%s' % (self.info.xMotor)])
            y = yFlipper * np.array(hf['entry/measurement/%s' % (self.info.yMotor)])

        # make lists to two arrays
        x = np.array(x)*1e-6
        y = np.array(y)*1e-6 
        
        # hack: because the scanner is bumped
        #x += -0.0483*y
        x += -0.09*y

        x = x-np.min(x)  
        y = y-np.min(y)  
        positions = np.vstack((y, x))     
        return positions.T

    def load(self, indices):
        """
        Provides the raw diffraction pattern from the detector (eiger .h5) file.
        Normalizes them, if there is normalization data. 
        """

        raw, weights, positions = {}, {}, {}

        fullfilename_background = os.path.join(self.info.path, 'scan_'+str(self.info.scanNumber_background).zfill(6)+'_andor.h5')
        with h5py.File(fullfilename_background, 'r') as fp:
            data_background = fp['entry/data/data/']
            data_background = np.mean(data_background, axis=0)

        fullfilename_data       = os.path.join(self.info.path, 'scan_'+str(self.info.scanNumber_data).zfill(6)+'_andor.h5')
        with h5py.File(fullfilename_data, 'r') as fp:
            for ind in indices:
                raw[ind] = fp['entry/data/data/'][ind]-data_background
                raw[ind][raw[ind]<=0] = 0

        return raw, positions, weights

    def load_weight(self):
        """
        Provides the mask used for every diffraction pattern in the whole scan
        This mask will have the shape of the first frame.
        """

        r, w, p = self.load(indices=(0,))
        data = r[0]
        mask = np.ones_like(data)
        mask[np.where(data < 0)] = 0

        if self.info.maskfile:
            with h5py.File(self.info.maskfile, 'r') as hf:
                mask2 = np.array(hf.get('mask'))
#            logger.info("loaded additional mask, %u x %u, sum %u" %
#                        (mask2.shape + (np.sum(mask2),)))
            mask = mask * mask2
#            logger.info("total mask, %u x %u, sum %u" %
#                        (mask.shape + (np.sum(mask),)))

        return mask

@register()
class SoftiMAX_Dec2021_Abs(PtyScan):


    """
    This class reads the positions from the interferometers.

    Defaults:

    [name]
    default = SoftiMAX_Dec2021_Abs
    type = str
    help =

    [path]
    default = None
    type = str
    help = base directory of that beamtime including the visit folder
    doc =

    [scanNumber_data]
    default = None
    type = int, list, tuple
    help = Scan number or list of scan numbers
    doc =

    [scanNumber_background]
    default = None
    type = int, list, tuple
    help = Scan number or list of scan numbers
    doc =

    [xMotor]
    default = samx
    type = str
    help = Which x motor to use
    doc =

    [yMotor]
    default = samy
    type = str
    help = Which y motor to use
    doc =

    [xMotorFlipped]
    default = False
    type = bool
    help = Flip detector x positions
    doc =

    [yMotorFlipped]
    default = False
    type = bool
    help = Flip detector y positions
    doc =

    [maskfile]
    default = None
    type = str
    help = Arbitrary mask file
    doc = Hdf5 file containing an array called 'mask' at the root level.

    [energy]
    default = None
    type = float
    help = photon energy in keV, if None it will be read from the scan file
    doc =

    """

    def load_positions(self):

        """
        Provides the relative sample positions inside the scan. 
        """

        fullfilename = os.path.join(self.info.path, str(self.info.scanNumber_data).zfill(6)+'.h5')
        self.frames_per_scan = {}

        xFlipper, yFlipper = 1, 1
        if self.info.xMotorFlipped:
            xFlipper = -1
            logger.warning("note: x motor is specified as flipped")
        if self.info.yMotorFlipped:
            yFlipper = -1
            logger.warning("note: y motor is specified as flipped")

        x, y = [], []
        with h5py.File(fullfilename, 'r') as hf:
            x = xFlipper * np.array(hf['entry/measurement/%s' % (self.info.xMotor)])
            y = yFlipper * np.array(hf['entry/measurement/%s' % (self.info.yMotor)])

        # make lists to two arrays
        x = np.array(x)*1e-6 # 1e6 is the correct value for abs_x (April 2022)
        y = np.array(y)*1e-6 # 1e6 is the correct value for abs_y (April 2022)
        
        # hack: because the scanner is bumped
        #x += -0.0483*y
        #x += -0.09*y

        x = x-np.min(x)  
        y = y-np.min(y)  
        positions = np.vstack((y, x))     

        # set the photon energy
        if self.info.energy == None:	
            with h5py.File(fullfilename, 'r') as fp:
                self.meta.energy = fp['entry/snapshot/beamline_energy'][:] * 1e-3
        else:
            self.meta.energy = self.info.energy

        return positions.T

    def load(self, indices):
        """
        Provides the raw diffraction pattern from the detector (eiger .h5) file.
        Normalizes them, if there is normalization data. 
        """

        raw, weights, positions = {}, {}, {}

        fullfilename_background = os.path.join(self.info.path, 'scan_'+str(self.info.scanNumber_background).zfill(6)+'_andor.h5')
        with h5py.File(fullfilename_background, 'r') as fp:
            data_background = fp['entry/data/data/']
            data_background = np.mean(data_background, axis=0)

        fullfilename_data       = os.path.join(self.info.path, 'scan_'+str(self.info.scanNumber_data).zfill(6)+'_andor.h5')
        with h5py.File(fullfilename_data, 'r') as fp:
            for ind in indices:
                raw[ind] = fp['entry/data/data/'][ind]-data_background
                raw[ind][raw[ind]<=0] = 0

        return raw, positions, weights

    def load_weight(self):
        """
        Provides the mask used for every diffraction pattern in the whole scan
        This mask will have the shape of the first frame.
        """

        r, w, p = self.load(indices=(0,))
        data = r[0]
        mask = np.ones_like(data)
        mask[np.where(data < 0)] = 0
        print([mask.shape,mask.dtype])

        if self.info.maskfile:
            with h5py.File(self.info.maskfile, 'r') as hf:
                mask2 = np.array(hf.get('mask'))
                print([mask2.shape,mask2[0]])
            mask = mask * mask2
#            logger.info("total mask, %u x %u, sum %u" %
#                        (mask.shape + (np.sum(mask),)))
#            logger.info("loaded additional mask, %u x %u, sum %u" %
#                        (mask2.shape + (np.sum(mask2),)))

        return mask
