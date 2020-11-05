"""  
    Implementation of PtyScan subclasses to hold CoSAXS scan data.
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

@register()
class CoSAXSStepscanNov2020(PtyScan):
    """
    Starting a fresh class here.

    Defaults:

    [name]
    default = CoSAXSStepscanNov2020
    type = str
    help =

    [path]
    default = None
    type = str
    help = base directory of that beamtime including the visit folder
    doc =

    [path_master_file]
    default = None
    type = str
    help = Path to the master .h5 file storing all the scan information
    doc =

    [scanNumber]
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

    [xMotorAngle]
    default = 0.0
    type = float
    help = Angle of the motor x axis relative to the lab x axis
    doc =

    [yMotorAngle]
    default = 0.0
    type = float
    help = Angle of the motor y axis relative to the lab y axis
    doc =

    [zDetectorAngle]
    default = 0.0
    type = float
    help = Relative rotation angle between the motor x and y axes and the detector pixel rows and columns in degree
    doc = If the Detector is mounted rotated around the beam axis relative to the scanning motors, use this angle to rotate the motor position into the detector frame of reference. The rotation angle is in mathematical positive sense from the motors to the detector pixel grid.

    [maskfile]
    default = None
    type = str
    help = Arbitrary mask file
    doc = Hdf5 file containing an array called 'mask' at the root level.

    [I0]
    default = None
    type = str
    help = Normalization channel
    doc =

    """

    def load_positions(self):
        """
        Provides the relative sample positions inside the scan. 
        """

        fullfilename = self.info.path_master_file
        self.frames_per_scan = {}

        xFlipper, yFlipper = 1, 1
        if self.info.xMotorFlipped:
            xFlipper = -1
            logger.warning("note: x motor is specified as flipped")
        if self.info.yMotorFlipped:
            yFlipper = -1
            logger.warning("note: y motor is specified as flipped")

        # if the x axis is tilted, take that into account.
        xCosFactor = np.cos(self.info.xMotorAngle / 180.0 * np.pi)
        yCosFactor = np.cos(self.info.yMotorAngle / 180.0 * np.pi)
        logger.info(
            "x and y motor angles result in multiplication by %.2f, %.2f" % (xCosFactor, yCosFactor))

        try:
            self.info.scanNumber = tuple(self.info.scanNumber)
        except TypeError:
            self.info.scanNumber = (self.info.scanNumber,)

        normdata, x, y = [], [], []
        for scan in self.info.scanNumber:
            entry = '/entry%d' % scan

            # may as well get normalization data here too
            if self.info.I0 is not None:
                with h5py.File(fullfilename, 'r') as hf:
                    normdata.append(np.array(hf['%s/measurement/%s' % (entry, self.info.I0)], dtype=float))
                logger.info('*** going to normalize by channel %s' % self.info.I0)

            with h5py.File(fullfilename, 'r') as hf:
                x.append(xFlipper * xCosFactor
                     * np.array(hf['%s/measurement/%s' % (entry, self.info.xMotor)]))
                y.append(yFlipper * yCosFactor
                     * np.array(hf['%s/measurement/%s' % (entry, self.info.yMotor)]))
                self.frames_per_scan[scan] = x[-1].shape[0]
        
        first_frames = [sum(list(self.frames_per_scan.values())[:i]) for i in range(len(self.frames_per_scan))]
        self.first_frame_of_scan = {scan:first_frames[i] for i, scan in enumerate(self.info.scanNumber)}
        if normdata:
            normdata = np.concatenate(normdata)
            self.normdata = normdata / np.mean(normdata)
        
        # make lists to two arrays
        x = np.concatenate(x)
        y = np.concatenate(y)   

        # if the detector and motor frame of reference are roated around the beam axis
        if self.info.zDetectorAngle != 0:
            chi_rad = self.info.zDetectorAngle / 180.0 * np.pi
            x, y    = np.cos(chi_rad)*x-np.sin(chi_rad)*y, np.sin(chi_rad)*x+np.cos(chi_rad)*y
            logger.info("x and y motor positions were roated by %.4f degree to align with the detector pixel grid" % (self.info.zDetectorAngle))

        positions = -np.vstack((y, x)).T * 1e-6
        return positions

    def load(self, indices):
        """
        Provides the raw diffraction pattern from the detector (eiger .h5) file.
        Normalizes them, if there is normalization data. 
        """

        raw, weights, positions = {}, {}, {}

        scan = self.info.scanNumber[0]
        filename = 'eiger_scan_%d_data.hdf5' % scan
        fullfilename = os.path.join(self.info.path, 'raw', filename)

        with h5py.File(fullfilename, 'r') as fp:
            for ind in indices:
                raw[ind] = fp['entry/data/data/'][ind]
                if self.info.I0:
                    raw[ind] = raw[ind] / self.normdata[ind]

        return raw, positions, weights

    def load_weight(self):
        """
        Provides the mask used for every diffraction pattern in the whole scan
        This mask will have the shape of the first frame.
        """

        r, w, p = self.load(indices=(0,))
        data = r[0]
        mask = np.ones_like(data)
        mask[np.where(data == 2**32-1)] = 0
        mask[np.where(data == 2**16-1)] = 0
        mask[np.where(data < 0)] = 0
        logger.info("took account of the built-in mask, %u x %u, sum %u, so %u masked pixels" %
                    (mask.shape + (np.sum(mask), np.prod(mask.shape)-np.sum(mask))))

        if self.info.maskfile:
            with h5py.File(self.info.maskfile, 'r') as hf:
                mask2 = np.array(hf.get('mask'))
            logger.info("loaded additional mask, %u x %u, sum %u, so %u masked pixels" %
                        (mask2.shape + (np.sum(mask2), np.prod(mask2.shape)-np.sum(mask2))))
            mask = mask * mask2
            logger.info("total mask, %u x %u, sum %u, so %u masked pixels" %
                    (mask.shape + (np.sum(mask), np.prod(mask.shape)-np.sum(mask))))

        return mask
