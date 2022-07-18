"""  Implementation of PtyScan subclasses to hold nanomax scan data. The
     beamline is developing, as is the data format. """

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
class NanomaxStepscanNov2018(PtyScan):
    """
    Starting a fresh class here.

    Defaults:

    [name]
    default = NanomaxStepscanNov2018
    type = str
    help =

    [path]
    default = None
    type = str
    help = Path to where the data is at
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

    [detector]
    default = 'pil100k'
    type = str
    help = Which detector to use, can be pil100k or merlin

    [maskfile]
    default = None
    type = str
    help = Arbitrary mask file
    doc = Hdf5 file containing an array called 'mask' at the root level.

    [I0]
    default = None
    type = str
    help = Normalization channel, like alba2/1 for example
    doc =

    """

    def load_positions(self):

        filename = self.info.path.strip('/').split('/')[-1] + '.h5'
        fullfilename = os.path.join(self.info.path, filename)
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
            entry = 'entry%d' % scan

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

        x = np.concatenate(x)
        y = np.concatenate(y)      
        positions = -np.vstack((y, x)).T * 1e-6
        return positions


    def load(self, indices):
        raw, weights, positions = {}, {}, {}

        hdfpath = 'entry_%%04u/measurement/%s/data' % {'pil100k': 'Pilatus', 'merlin': 'Merlin', 'pil1m': 'Pilatus'}[self.info.detector]

        for ind in indices:
            # work out in which scan to find this index
            for i in range(len(self.info.scanNumber)-1, -1, -1):
                scan = self.info.scanNumber[i]
                if ind >= self.first_frame_of_scan[scan]:
                    break
            frame = ind - self.first_frame_of_scan[scan]
            filename = 'scan_%04u_%s_0000.hdf5' % (
                    scan, {'pil100k': 'pil100k', 'merlin': 'merlin', 'pil1m':'pil1m'}[self.info.detector])
            fullfilename = os.path.join(self.info.path, filename)
            with h5py.File(fullfilename, 'r') as fp:
                raw[ind] = fp[hdfpath % frame][0]
                if self.info.I0:
                    raw[ind] = raw[ind] / self.normdata[ind]

        return raw, positions, weights

    def load_weight(self):
        """
        Provides the mask for the whole scan, the shape of the first 
        frame.
        """

        r, w, p = self.load(indices=(0,))
        data = r[0]
        mask = np.ones_like(data)
        mask[np.where(data == -2)] = 0
        logger.info("took account of the pilatus mask, %u x %u, sum %u" %
                    (mask.shape + (np.sum(mask),)))

        if self.info.maskfile:
            with h5py.File(self.info.maskfile, 'r') as hf:
                mask2 = np.array(hf.get('mask'))
            logger.info("loaded additional mask, %u x %u, sum %u" %
                        (mask2.shape + (np.sum(mask2),)))
            mask = mask * mask2
            logger.info("total mask, %u x %u, sum %u" %
                        (mask.shape + (np.sum(mask),)))

        return mask

@register()
class NanomaxFlyscanMay2019(PtyScan):
    """
    Starting a fresh subclass here, this class is mainly a cleanup
    plus I0 normalization.

    Defaults:

    [name]
    default = NanomaxFlyscanMay2019
    type = str
    help =

    [path]
    default = None
    type = str
    help = Path to where the data is at
    doc =

    [scanNumber]
    default = None
    type = int
    help = Scan number
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

    [detector]
    default = 'pil100k'
    type = str
    help = Which detector to use, can be pil100k or merlin

    [maskfile]
    default = None
    type = str
    help = Arbitrary mask file
    doc = Hdf5 file containing an array called 'mask' at the root level.

    [I0]
    default = None
    type = str
    help = Normalization channel, like counter1 for example
    doc =

    [firstLine]
    default = 0
    type = int
    help = The first line to be read

    [nMaxLines]
    type = int
    default = 0
    help = If positive, limit the number of lines to this value

    """

    def load_positions(self):
        filename = self.info.path.strip('/').split('/')[-1] + '.h5'
        fullfilename = os.path.join(self.info.path, filename)
        entry = 'entry%d' % self.info.scanNumber

        x, y = None, None
        with h5py.File(fullfilename, 'r') as hf:
            # get x positions
            xdataset = hf.get(entry + '/measurement/%s' % self.info.xMotor)
            xall = np.array(xdataset)
            # manually find shape by looking for zeros
            self.firstLine = self.info.firstLine
            self.lastLine = self.firstLine+self.info.nMaxLines-1 if self.info.nMaxLines > 0 else xall.shape[0]-1
            for i in range(xall.shape[1]):
                if xall[0, i] == 0:
                    Nsteps = i
                    break
            x = xall[self.firstLine:self.lastLine+1, :Nsteps].flatten()

            # get y positions
            ydataset = hf.get(entry + '/measurement/%s' % self.info.yMotor)
            yall = np.array(ydataset)
            # manually find shape by looking for zeros
            for i in range(yall.shape[1]):
                if yall[0, i] == 0:
                    Nsteps = i
                    break
            y = yall[self.firstLine:self.lastLine+1, :Nsteps].flatten()

            self.images_per_line = Nsteps

        if self.info.xMotorFlipped:
            x *= -1
            logger.warning("note: x motor is specified as flipped")
        if self.info.yMotorFlipped:
            y *= -1
            logger.warning("note: y motor is specified as flipped")

        # if the x axis is tilted, take that into account.
        xCosFactor = np.cos(self.info.xMotorAngle / 180.0 * np.pi)
        x *= xCosFactor
        logger.info(
            "x motor angle results in multiplication by %.2f" % xCosFactor)

        # if the y axis is tilted, take that into account.
        yCosFactor = np.cos(self.info.yMotorAngle / 180.0 * np.pi)
        y *= yCosFactor
        logger.info(
            "y motor angle results in multiplication by %.2f" % yCosFactor)

        # load normalization for the whole scan and index later
        if self.info.I0 is not None:
            with h5py.File(fullfilename, 'r') as hf:
                normdata = np.array(hf['%s/measurement/%s' % (entry, self.info.I0)], dtype=float)
            normdata = normdata[self.firstLine:self.lastLine+1, :Nsteps].flatten()
            self.normdata = normdata / np.mean(normdata)
            logger.info('*** going to normalize by channel %s - loaded %d values' % (self.info.I0, len(self.normdata)))

        positions = - np.vstack((y, x)).T * 1e-6
        return positions

    def load(self, indices):

        raw, weights, positions = {}, {}, {}
        hdfpath = 'entry_%%04u/measurement/%s/data' % {'pil100k': 'Pilatus', 'merlin': 'Merlin', 'pil1m': 'Pilatus'}[self.info.detector]
        filename = 'scan_%04u_%s_0000.hdf5' % (
                self.info.scanNumber, {'pil100k': 'pil100k', 'merlin': 'merlin', 'pil1m':'pil1m'}[self.info.detector])
        fullfilename = os.path.join(self.info.path, filename)

        # read the dataset
        for ind in indices:
            line = self.firstLine + ind // self.images_per_line
            image = ind % self.images_per_line
            with h5py.File(fullfilename, 'r') as hf:
                data = hf[hdfpath % line][image]
            raw[ind] = data
            if self.info.I0:
                raw[ind] = np.round(raw[ind] / self.normdata[ind]).astype(int)

        logger.info('loaded %d images' % len(raw))
        return raw, positions, weights

    def load_weight(self):
        """
        Provides the mask for the whole scan, the shape of the first 
        frame.
        """

        scannr = self.info.scanNumber
        hdfpath = 'entry_%%04u/measurement/%s/data' % {'pil100k': 'Pilatus', 'merlin': 'Merlin', 'pil1m': 'Pilatus'}[self.info.detector]
        filename = 'scan_%04u_%s_0000.hdf5' % (
                self.info.scanNumber, {'pil100k': 'pil100k', 'merlin': 'merlin', 'pil1m':'pil1m'}[self.info.detector])
        fullfilename = os.path.join(self.info.path, filename)

        with h5py.File(fullfilename, 'r') as hf:
            data = hf[hdfpath % 0]
            shape = data[0].shape
            mask = np.ones(shape)
            mask[np.where(data[0] == -2)] = 0
        logger.info("took account of the pilatus mask, %u x %u, sum %u" %
                    (mask.shape + (np.sum(mask),)))

        if self.info.maskfile:
            with h5py.File(self.info.maskfile, 'r') as hf:
                mask2 = np.array(hf.get('mask'))
            logger.info("loaded additional mask, %u x %u, sum %u" %
                        (mask2.shape + (np.sum(mask2),)))
            mask = mask * mask2
            logger.info("total mask, %u x %u, sum %u" %
                        (mask.shape + (np.sum(mask),)))

        return mask

@register()
class NanomaxStepscanSep2019(PtyScan):
    """
    This class loads data written with the nanomax pirate system

    Defaults:

    [name]
    default = NanomaxStepscanSep2019
    type = str
    help =

    [path]
    default = None
    type = str
    help = Path to where the data is at
    doc =

    [scanNumber]
    default = None
    type = int, list, tuple
    help = Scan number or list of scan numbers
    doc =

    [xMotor]
    default = sx
    type = str
    help = Which x motor to use
    doc =

    [yMotor]
    default = sy
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

    [detector]
    default = 'pilatus'
    type = str
    help = Which detector to use, can be pilatus or merlin

    [maskfile]
    default = None
    type = str
    help = Arbitrary mask file
    doc = Hdf5 file containing an array called 'mask' at the root level.

    [I0]
    default = None
    type = str
    help = Normalization channel, like ni/counter1 for example
    doc =

    """

    def load_positions(self):

        filename = '%06u.h5' % self.info.scanNumber
        fullfilename = os.path.join(self.info.path, filename)
        self.frames_per_scan = {}

        xFlipper, yFlipper = 1, 1
        if self.info.xMotorFlipped:
            xFlipper = -1
            logger.warning("note: x motor is specified as flipped")
        if self.info.yMotorFlipped:
            yFlipper = -1
            logger.warning("note: y motor is specified as flipped")

        # if the x/y axis is tilted with respect to the beam axis, take that into account.
        xCosFactor = np.cos(self.info.xMotorAngle / 180.0 * np.pi)
        yCosFactor = np.cos(self.info.yMotorAngle / 180.0 * np.pi)
        logger.info("x and y motor angles result in multiplication by %.2f, %.2f" % (xCosFactor, yCosFactor))

        try:
            self.info.scanNumber = tuple(self.info.scanNumber)
        except TypeError:
            self.info.scanNumber = (self.info.scanNumber,)

        normdata, x, y = [], [], []
        for scan in self.info.scanNumber:

            # may as well get normalization data here too
            if self.info.I0 is not None:
                with h5py.File(fullfilename, 'r') as hf:
                    normdata.append(np.array(hf['entry/measurement/%s' % (self.info.I0)], dtype=float))
                logger.info('*** going to normalize by channel %s' % self.info.I0)

            with h5py.File(fullfilename, 'r') as hf:
                x.append(xFlipper * xCosFactor
                     * np.array(hf['entry/measurement/%s' % (self.info.xMotor)]))
                y.append(yFlipper * yCosFactor
                     * np.array(hf['entry/measurement/%s' % (self.info.yMotor)]))
                self.frames_per_scan[scan] = x[-1].shape[0]
        
        first_frames = [sum(list(self.frames_per_scan.values())[:i]) for i in range(len(self.frames_per_scan))]
        self.first_frame_of_scan = {scan:first_frames[i] for i, scan in enumerate(self.info.scanNumber)}
        if normdata:
            normdata = np.concatenate(normdata)
            self.normdata = normdata / np.mean(normdata)

        # make list to arrays
        x = np.concatenate(x)
        y = np.concatenate(y)   

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
            
        # set minimum to zero so ptypy can work out the proper object size
        x -= np.min(x)
        y -= np.min(y)

        # put the two arrays together and express in [m]
        positions = -np.vstack((y, x)).T * 1e-6
        return positions


    def load(self, indices):
        raw, weights, positions = {}, {}, {}

        hdfpath = 'entry/measurement/%s/%%06u' % self.info.detector
        filename = '%06u.h5' % self.info.scanNumber
        fullfilename = os.path.join(self.info.path, filename)

        with h5py.File(fullfilename, 'r') as fp:
            for ind in indices:
                raw[ind] = fp[hdfpath % ind][0]
                if self.info.I0:
                    raw[ind] = raw[ind] / self.normdata[ind]

        return raw, positions, weights

    def load_weight(self):
        """
        Provides the mask for the whole scan, the shape of the first 
        frame.
        """

        r, w, p = self.load(indices=(0,))
        data = r[0]
        mask = np.ones_like(data)
        mask[np.where(data == -2)] = 0
        logger.info("took account of the pilatus mask, %u x %u, sum %u" %
                    (mask.shape + (np.sum(mask),)))

        if self.info.maskfile:
            with h5py.File(self.info.maskfile, 'r') as hf:
                mask2 = np.array(hf.get('mask'))
            logger.info("loaded additional mask, %u x %u, sum %u" %
                        (mask2.shape + (np.sum(mask2),)))
            mask = mask * mask2
            logger.info("total mask, %u x %u, sum %u" %
                        (mask.shape + (np.sum(mask),)))

        return mask


@register()
class NanomaxFlyscanDec2019(NanomaxFlyscanMay2019):
    """
    Flyscan class for nanomax contrast acquisitions.

    Defaults:

    [name]
    default = NanomaxFlyscanDec2019
    type = str
    help =

    """

    def load_positions(self):
        filename = '%06u.h5' % self.info.scanNumber
        entry = 'entry'

        fullfilename = os.path.join(self.info.path, filename)

        with h5py.File(fullfilename, 'r') as hf:
            x = hf['entry/measurement/npoint_buff/%s'%self.info.xMotor][()]
            y = hf['entry/measurement/npoint_buff/%s'%self.info.yMotor][()]
            self.images_per_line = x.shape[-1]
            x = x.flatten()
            y = y.flatten()

        if self.info.xMotorFlipped:
            x *= -1
            logger.warning("note: x motor is specified as flipped")
        if self.info.yMotorFlipped:
            y *= -1
            logger.warning("note: y motor is specified as flipped")

        # if the x axis is tilted, take that into account.
        xCosFactor = np.cos(self.info.xMotorAngle / 180.0 * np.pi)
        x *= xCosFactor
        logger.info(
            "x motor angle results in multiplication by %.2f" % xCosFactor)

        # if the y axis is tilted, take that into account.
        yCosFactor = np.cos(self.info.yMotorAngle / 180.0 * np.pi)
        y *= yCosFactor
        logger.info(
            "y motor angle results in multiplication by %.2f" % yCosFactor)

        # load normalization for the whole scan and index later
        if self.info.I0 is not None:
            with h5py.File(fullfilename, 'r') as hf:
                normdata = np.array(hf['%s/measurement/%s' % (entry, self.info.I0)], dtype=float)
            normdata = normdata[self.firstLine:self.lastLine+1, :Nsteps].flatten()
            self.normdata = normdata / np.mean(normdata)
            logger.info('*** going to normalize by channel %s - loaded %d values' % (self.info.I0, len(self.normdata)))

        positions = - np.vstack((y, x)).T * 1e-6
        return positions

    def load(self, indices):

        raw, weights, positions = {}, {}, {}

        filename = '%06u.h5' % self.info.scanNumber
        fullfilename = os.path.join(self.info.path, filename)

        # read the dataset
        for ind in indices:
            line = self.info.firstLine + ind // self.images_per_line
            image = ind % self.images_per_line
            with h5py.File(fullfilename, 'r') as hf:
                data = hf['entry/measurement/%s/%06u'%(self.info.detector, line)][image]
            raw[ind] = data
            if self.info.I0:
                raw[ind] = np.round(raw[ind] / self.normdata[ind]).astype(int)

        logger.info('loaded %d images' % len(raw))
        return raw, positions, weights

    def load_weight(self):
        """
        Provides the mask for the whole scan, the shape of the first 
        frame.
        """

        filename = '%06u.h5' % self.info.scanNumber
        fullfilename = os.path.join(self.info.path, filename)

        with h5py.File(fullfilename, 'r') as hf:
            data = hf['entry/measurement/%s/%06u'%(self.info.detector, 0)][0]
            shape = data.shape
            mask = np.ones(shape)
            mask[np.where(data[0] == -2)] = 0
        logger.info("took account of the pilatus mask, %u x %u, sum %u" %
                    (mask.shape + (np.sum(mask),)))

        if self.info.maskfile:
            with h5py.File(self.info.maskfile, 'r') as hf:
                mask2 = np.array(hf.get('mask'))
            logger.info("loaded additional mask, %u x %u, sum %u" %
                        (mask2.shape + (np.sum(mask2),)))
            mask = mask * mask2
            logger.info("total mask, %u x %u, sum %u" %
                        (mask.shape + (np.sum(mask),)))

        return mask

@register()
class NanomaxContrast(NanomaxStepscanSep2019):
    """
    This class loads data written with the nanomax pirate system,
    in a slightly matured state. Step and fly scan have the same
    format.

    Defaults:

    [name]
    default = NanomaxContrast
    type = str
    help =

    [energy]
    default = None
    type = float
    help = photon energy in keV, if None it will be read from the scan file
    doc =

    [cropOnLoad]
    default = True
    type = bool
    help = Only load the used bits of each detector frame
    doc =

    [cropOnLoad_y_lower]
    default = None
    type = int, list, tuple
    help = y-axis lower limit
    doc =

    [cropOnLoad_y_upper]
    default = None
    type = int, list, tuple
    help = y-axis upper limit
    doc =

    [cropOnLoad_x_lower]
    default = None
    type = int, list, tuple
    help = x-axis lower limit
    doc =

    [cropOnLoad_x_upper]
    default = None
    type = int, list, tuple
    help = x-axis upper limit
    doc =

    """

    def load(self, indices):
        raw, weights, positions = {}, {}, {}

        filename = '%06u.h5' % self.info.scanNumber
        fullfilename = os.path.join(self.info.path, filename)

        # crop on load is requested, but the actual indices to crop are not yet defined
        if self.info.cropOnLoad and self.info.cropOnLoad_y_lower == None:
            
            # center of the diffraction patterns is not explicitly given
            if self.info.center==None:
                # requires to load the first frame and to find the center of mass there
                with h5py.File(fullfilename, 'r') as fp:
                    frame = fp['entry/measurement/%s/frames'%self.info.detector][0]
                # and to mask the hot pixels ... sadly this will have double with self.load_weight
                mask = np.ones_like(frame)
                if self.info.detector == 'pilatus':
                    mask[np.where(frame < 0)] = 0
                if self.info.detector == 'eiger':
                    mask[np.where(frame == 2**32-1)] = 0
                    mask[np.where(frame == 2**16-1)] = 0
                if self.info.maskfile:
                    with h5py.File(self.info.maskfile, 'r') as hf:
                        mask2 = np.array(hf.get('mask'))
                    mask *= mask2
                # now find the center of mass can be estimated using the ptypy internal function and make it integers
                self.info.center = u.scripts.mass_center(frame*mask)
                self.info.center = [int(x) for x in self.info.center]
                logger.info('Estimated the center of the (first) diffraction pattern to be %s' % (self.info.center,))

            # the center of the full frames is (now) known, and thus the indices for the cropping can be defined
            cy, cx   = self.info.center
            d        = self.info.shape
            logger.info('Found the center of the full frames at %s' % (self.info.center,))
            logger.info('Will crop all diffraction patterns on load to a size of %s' % self.info.shape)
            self.info.cropOnLoad_y_lower, self.info.cropOnLoad_x_lower = int(cy)-d//2, int(cx)-d//2
            self.info.cropOnLoad_y_upper, self.info.cropOnLoad_x_upper = self.info.cropOnLoad_y_lower+d, self.info.cropOnLoad_x_lower+d

            # the (temporary) center needs to be redefined for the cropped frames
            tmp_center_y, tmp_center_x = d//2, d//2

            # if the lower crop indices are negative, set them zero
            if self.info.cropOnLoad_y_lower<0:
                tmp_center_y += self.info.cropOnLoad_y_lower
                self.info.cropOnLoad_y_lower = 0
            if self.info.cropOnLoad_x_lower<0:
                tmp_center_x += self.info.cropOnLoad_x_lower
                self.info.cropOnLoad_x_lower = 0    
            # no need to have something similar for too large upper indices due to the way python slices arrays

            # now fix the new center
            self.info.center = (tmp_center_y, tmp_center_x)
        
        # set the photon energy
        if self.info.energy == None:	
            with h5py.File(fullfilename, 'r') as fp:
                self.meta.energy = fp['entry/snapshot/energy'][:] * 1e-3
        else:
            self.meta.energy = self.info.energy

        # actually loading the detector frames
        with h5py.File(fullfilename, 'r') as fp:
            for ind in indices:
                # load only a cropped bit of the full frame
                if self.info.cropOnLoad:
                    raw[ind] = fp['entry/measurement/%s/frames'%self.info.detector][ind,self.info.cropOnLoad_y_lower:self.info.cropOnLoad_y_upper, self.info.cropOnLoad_x_lower:self.info.cropOnLoad_x_upper]
                # load the full raw frame                
                else:	
                    raw[ind] = fp['entry/measurement/%s/frames'%self.info.detector][ind]
                # if there is I0 information, use it to normalize the just loaded frame                
                if self.info.I0:
                    logger.info('normalizing frame %u by %f' % (ind, self.normdata[ind]))
                    logger.info('hack! assuming mask = 2**32-1 when I0-normalizing')
                    msk = np.where(raw[ind] == 2**32-1)
                    raw[ind] = np.round(raw[ind] / self.normdata[ind]).astype(raw[ind].dtype)
                    raw[ind][msk] = 2**32-1

        return raw, positions, weights


    def load_weight(self):
        """
        Provides the mask for the whole scan, the shape of the first 
        frame.
        """

        r, w, p = self.load(indices=(0,))
        data = r[0]
        mask = np.ones_like(data)
        if self.info.detector == 'pilatus':
            mask[np.where(data < 0)] = 0
        if self.info.detector == 'eiger':
            mask[np.where(data == 2**32-1)] = 0
            mask[np.where(data == 2**16-1)] = 0
        logger.info("took account of the built-in mask, %u x %u, sum %u, so %u masked pixels" %
                    (mask.shape + (np.sum(mask), np.prod(mask.shape)-np.sum(mask))))

        if self.info.maskfile:
            with h5py.File(self.info.maskfile, 'r') as hf:
                mask2 = np.array(hf.get('mask'))
                if self.info.cropOnLoad:
                    mask2 = mask2[self.info.cropOnLoad_y_lower:self.info.cropOnLoad_y_upper, 
                                  self.info.cropOnLoad_x_lower:self.info.cropOnLoad_x_upper]
            logger.info("loaded additional mask, %u x %u, sum %u, so %u masked pixels" %
                        (mask2.shape + (np.sum(mask2), np.prod(mask2.shape)-np.sum(mask2))))
            mask = mask * mask2
            logger.info("total mask, %u x %u, sum %u, so %u masked pixels" %
                    (mask.shape + (np.sum(mask), np.prod(mask.shape)-np.sum(mask))))

        return mask


