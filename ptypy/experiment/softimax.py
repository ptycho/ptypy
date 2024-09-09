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
class SoftimaxContrast(PtyScan):


    """
    This class reads the positions from the meta data of the camera frames. Needs to be adjusted to any changes in the format provided by STXM control.

    Defaults:

    [name]
    default = SoftimaxContrast
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

    [data_key]
    default = entry/data/data/
    type = str
    help = Key to the intensities and background entry in the hdf5 file
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

    [xMotorAngle]
    default = 0.0
    type = float
    help = Angle of the motor x axis relative to the lab x axis in degree
    doc =

    [yMotorAngle]
    default = 0.0
    type = float
    help = Angle of the motor y axis relative to the lab y axis in degree
    doc =

    [zDetectorAngle]
    default = 0.0
    type = float
    help = Relative rotation angle between the motor x and y axes and the detector pixel rows and columns in degree
    doc = If the Detector is mounted rotated around the beam axis relative to the scanning motors, use this angle to rotate the motor position into the detector frame of reference. The rotation angle is in mathematical positive sense from the motors to the detector pixel grid.

    [xyAxisSkewOffset]
    default = 0.0
    type = float
    help = Relative rotation angle beyond the expected 90 degrees between the motor x and y axes in degree
    doc = If for example the scanner is damaged and x and y end up not beeing perfectly under 90 degrees, this value can can be used to correct for that.

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
        xentry = {
                'abs_x':   'entry/measurement/abs_x',
                'INENC1':  'entry/measurement/panda0/INENC1.VAL_Value',
                'burst_x': 'entry/measurement/panda0/burst_x',
                }[self.info.xMotor]
        yentry = {
                'abs_y':   'entry/measurement/abs_y',
                'INENC2':  'entry/measurement/panda0/INENC2.VAL_Value',
                'burst_y': 'entry/measurement/panda0/burst_y',
                }[self.info.yMotor]

        with h5py.File(fullfilename, 'r') as hf:
            #           x = xFlipper * np.array(hf['entry/measurement/%s' % (self.info.xMotor)])
            #           y = yFlipper * np.array(hf['entry/measurement/%s' % (self.info.yMotor)])
            x = xFlipper * np.array(hf[xentry])
            y = yFlipper * np.array(hf[yentry])

        # make lists to two arrays
        #        x = np.array(x)*1e-6
        #        y = np.array(y)*1e-6
        x = {
                'abs_x':   np.array(x) * 1e-6,
                'INENC1':  np.array(x) * -1e-9,
                'burst_x': np.array(x) * -1e-9,
                }[self.info.xMotor]
        y = {
                'abs_y':   np.array(y) * 1e-6,
                'INENC2':  np.array(y) * 1e-9,
                'burst_y': np.array(y) * -1e-9,
                }[self.info.yMotor]
        
        chi_rad_x = 0
        chi_rad_y = 0
        # if the detector and motor frame of reference are roated around the beam axis
        if self.info.zDetectorAngle != 0:
            chi_rad_x = self.info.zDetectorAngle / 180.0 * np.pi
            chi_rad_y = 1.*chi_rad_x
            logger.info("x and y motor positions were roated by %.4f degree to align with the detector pixel grid" % (self.info.zDetectorAngle))
        # if x and y are not under 90 degrees to each other
        if self.info.xyAxisSkewOffset != 0:
            chi_rad_x += -0.5 * self.info.xyAxisSkewOffset / 180.0 * np.pi
            chi_rad_y += +0.5 * self.info.xyAxisSkewOffset / 180.0 * np.pi
            logger.info("x and y motor positions were skewed by %.4f degree to each other" % (self.info.xyAxisSkewOffset))
        x, y = np.cos(chi_rad_x)*x-np.sin(chi_rad_y)*y, np.sin(chi_rad_x)*x+np.cos(chi_rad_y)*y

        ## x = x-np.min(x)
        ## y = y-np.min(y)
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
            data_background = fp[self.info.data_key]
            data_background = np.mean(data_background, axis=0)

        fullfilename_data       = os.path.join(self.info.path, 'scan_'+str(self.info.scanNumber_data).zfill(6)+'_andor.h5')
        with h5py.File(fullfilename_data, 'r') as fp:
            for ind in indices:
                raw[ind] = fp[self.info.data_key][ind]-data_background
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
