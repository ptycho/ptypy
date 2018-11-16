"""  Implementation of PtyScan subclasses to hold nanomax scan data. The
     beamline is developing, as is the data format. Thus, commissioning
     and user experiments rely on different classes depending on
     measurement campaign. """

import numpy as np
import h5py

from ..core.data import PtyScan
from .. import utils as u
from . import register

logger = u.verbose.logger


class NanomaxBase(PtyScan):
    """
    Defaults:

    [dataPath]
    default = None
    type = str
    help = Path to folder containing the Sardana master file
    doc =

    [datafile]
    default = None
    type = str
    help = Sardana master file
    doc =

    [maskfile]
    default = None
    type = str
    help = Arbitrary mask file
    doc = Hdf5 file containing an array called 'mask' at the root level.

    [pilatusPath]
    default = None
    type = str
    help = Path to folder containing detector image files
    doc =

    [pilatusPattern]
    default = None
    type = str
    help = Format string for detector image files
    doc = A format string with two integer fields, the first holds the scan number while the second holds the image number.

    [scannr]
    default = None
    type = int
    help = Scan number
    doc =
    """
    pass


@register()
class NanomaxStepscanNov2016(NanomaxBase):
    """
    Loads Nanomax step scan data in the format of week Nov/Dec 2016

    Defaults:

    [name]
    default = NanomaxStepscanNov2016
    type = str
    help =
    doc =

    """

    def __init__(self, pars=None, **kwargs):
        self.p = self.DEFAULT.copy(99)
        self.p.update(pars)
        super(NanomaxStepscanNov2016, self).__init__(self.p)

    def load_positions(self):
        fileName = self.p.dataPath + self.p.datafile
        entry = 'entry%d' % self.p.scannr

        with h5py.File(fileName, 'r') as hf:
            x = np.array(hf.get(entry + '/measurement/samx'))
            y = np.array(hf.get(entry + '/measurement/samy'))

        positions = -np.vstack((y, x)).T * 1e-6
        return positions

    def load(self, indices):

        raw, weights, positions = {}, {}, {}
        scannr = self.p.scannr
        path = self.p.pilatusPath
        filepattern = self.p.pilatusPattern
        if not (path[-1] == '/'):
            path += '/'

        data = []
        for im in range(self.info.positions_scan.shape[0]):
            with h5py.File(path + filepattern % (scannr, im), 'r') as hf:
                dataset = hf.get('entry_0000/measurement/Pilatus/data')
                data.append(np.array(dataset)[0])

        # pick out the requested indices
        for i in indices:
            raw[i] = data[i]

        return raw, positions, weights

    def load_weight(self):
        """
        Provides the mask for the whole scan, the shape of the first 
        frame.
        """

        scannr = self.p.scannr
        path = self.p.pilatusPath
        pattern = self.p.pilatusPattern
        if not (path[-1] == '/'):
            path += '/'

        if self.p.maskfile:
            with h5py.File(self.p.maskfile, 'r') as hf:
                mask = np.array(hf.get('mask'))
            logger.info("loaded mask, %u x %u, sum %u" %
                        (mask.shape + (np.sum(mask),)))
        else:
            filename = self.p.dataPath + self.p.datafile
            with h5py.File(path + pattern % (scannr, 0), 'r') as hf:
                data = hf.get('entry_0000/measurement/Pilatus/data')
                shape = np.asarray(data[0]).shape
                mask = np.ones(shape)
            logger.info("created dummy mask, %u x %u, sum %u" %
                        (mask.shape + (np.sum(mask),)))
        return mask


@register()
class NanomaxStepscanMay2017(NanomaxBase):
    """
    Loads Nanomax step scan data in the format of May 2017.

    Defaults:

    [name]
    default = NanomaxStepscanMay2017
    type = str
    help =
    doc =

    [hdfPath]
    default = 'entry_0000/measurement/Pilatus/data'
    type = str
    help = Path to image array within detector hdf5 file
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
    doc = Use this if the stage is mounted at an angle around the y axis, the sign doesn't matter as a cos factor is added.

    """

    def __init__(self, pars=None, **kwargs):
        self.p = self.DEFAULT.copy(99)
        self.p.update(pars)
        super(NanomaxStepscanMay2017, self).__init__(self.p)

    def load_positions(self):
        fileName = self.p.dataPath + self.p.datafile
        entry = 'entry%d' % self.p.scannr

        xFlipper, yFlipper = 1, 1
        if self.p.xMotorFlipped:
            xFlipper = -1
            logger.warning("note: x motor is specified as flipped")
        if self.p.yMotorFlipped:
            yFlipper = -1
            logger.warning("note: y motor is specified as flipped")

        # if the x axis is tilted, take that into account.
        xCosFactor = np.cos(self.p.xMotorAngle / 180.0 * np.pi)
        logger.info(
            "x motor angle results in multiplication by %.2f" % xCosFactor)

        with h5py.File(fileName, 'r') as hf:
            x = xFlipper * \
                np.array(hf.get(entry + '/measurement/samx')) * xCosFactor
            y = yFlipper * np.array(hf.get(entry + '/measurement/samy'))

        positions = -np.vstack((y, x)).T * 1e-6
        return positions

    def load(self, indices):

        raw, weights, positions = {}, {}, {}
        scannr = self.p.scannr
        path = self.p.pilatusPath
        filepattern = self.p.pilatusPattern
        if not (path[-1] == '/'):
            path += '/'

        data = []
        for im in range(self.info.positions_scan.shape[0]):
            with h5py.File(path + filepattern % (scannr, im), 'r') as hf:
                dataset = hf.get(self.p.hdfPath)
                data.append(np.array(dataset)[0])

        # pick out the requested indices
        for i in indices:
            raw[i] = data[i]

        return raw, positions, weights

    def load_weight(self):
        """
        Provides the mask for the whole scan, the shape of the first 
        frame.
        """

        scannr = self.p.scannr
        path = self.p.pilatusPath
        pattern = self.p.pilatusPattern
        if not (path[-1] == '/'):
            path += '/'

        filename = self.p.dataPath + self.p.datafile
        with h5py.File(path + pattern % (scannr, 0), 'r') as hf:
            data = hf.get(self.p.hdfPath)
            shape = np.asarray(data[0]).shape
            mask = np.ones(shape)
            mask[np.where(data[0] == -2)] = 0
        logger.info("took account of the pilatus mask, %u x %u, sum %u" %
                    (mask.shape + (np.sum(mask),)))

        if self.p.maskfile:
            with h5py.File(self.p.maskfile, 'r') as hf:
                mask2 = np.array(hf.get('mask'))
            logger.info("loaded additional mask, %u x %u, sum %u" %
                        (mask2.shape + (np.sum(mask2),)))
            mask = mask * mask2
            logger.info("total mask, %u x %u, sum %u" %
                        (mask.shape + (np.sum(mask),)))

        return mask


@register()
class NanomaxFlyscanJune2017(NanomaxStepscanMay2017):
    """
    Loads Nanomax fly scan data in the format of June 2017.

    Defaults:

    [name]
    default = NanomaxFlyscanJune2017
    type = str
    help =

    [detNormalizationFilePattern]
    default = None
    type = str
    help = Format string for detector file containing data over which to normalize
    
    [detNormalizationIndices]
    default = None
    type = str
    help = Indices over which to normalize

    [nMaxLines]
    type = int
    default = 0
    help = If positive, limit the number of lines to this value

    """

    def __init__(self, pars=None, **kwargs):
        self.p = self.DEFAULT.copy(99)
        self.p.update(pars)
        super(NanomaxFlyscanJune2017, self).__init__(self.p)

    def load_positions(self):
        fileName = self.p.dataPath + self.p.datafile
        entry = 'entry%d' % self.p.scannr

        x, y = None, None
        with h5py.File(fileName, 'r') as hf:
            # get fast x positions
            xdataset = hf.get(entry + '/measurement/AdLinkAI_buff')
            xall = np.array(xdataset)
            # manually find shape by looking for zeros
            Ny = xall.shape[0]
            if self.p.nMaxLines > 0:
                Ny = self.p.nMaxLines
            for i in range(xall.shape[1]):
                if xall[0, i] == 0:
                    Nx = i
                    break
            x = xall[:Ny, :Nx].flatten()

            # get slow y positions
            ydataset = hf.get(entry + '/measurement/samy')
            yall = np.array(ydataset)[:Ny+1]
            if not (len(yall) == Ny):
                raise Exception('Something''s wrong with the positions')
            y = np.repeat(yall, Nx)

        if self.p.xMotorFlipped:
            x *= -1
            logger.warning("note: x motor is specified as flipped")
        if self.p.yMotorFlipped:
            y *= -1
            logger.warning("note: y motor is specified as flipped")

        # if the x axis is tilted, take that into account.
        xCosFactor = np.cos(self.p.xMotorAngle / 180.0 * np.pi)
        x *= xCosFactor
        logger.info(
            "x motor angle results in multiplication by %.2f" % xCosFactor)

        positions = - np.vstack((y, x)).T * 1e-6
        return positions

    def load(self, indices):

        raw, weights, positions = {}, {}, {}
        scannr = self.p.scannr
        path = self.p.pilatusPath
        pattern = self.p.pilatusPattern
        normfile = self.p.detNormalizationFilePattern
        normind = self.p.detNormalizationIndices

        # read the entire dataset
        done = False
        line = 0
        data = []
        while not done:
            try:
                with h5py.File(path + pattern % (scannr, line), 'r') as hf:
                    logger.info('loading data: ' + pattern % (scannr, line))
                    dataset = hf.get(self.p.hdfPath)
                    linedata = np.array(dataset)
                if normfile:
                    dtype = linedata.dtype
                    linedata = np.array(linedata, dtype=float)
                    with h5py.File(path + normfile % (scannr, line), 'r') as hf:
                        logger.info('loading normalization data: ' +
                                    normfile % (scannr, line))
                        dataset = hf.get(
                            self.p.detNormalizationHdfPath)
                        normdata = np.array(dataset)
                        if not normind:
                            shape = linedata[0].shape
                            normind = [0, shape[0], 0, shape[1]]
                        norm = np.mean(normdata[:, normind[0]:normind[
                                       1], normind[2]:normind[3]], axis=(1, 2))
                        if line == 0:
                            norm0 = norm[0]
                        norm /= norm0  # to avoid dividing integers by huge numbers
                        logger.debug("normalizing line by: %s" % str(norm))
                        for i in range(len(norm)):
                            linedata[i, :, :] = linedata[i] / norm[i]
                    linedata = np.array(np.round(linedata), dtype=dtype)

                data.append(linedata)
                line += 1
                if line == self.p.nMaxLines:
                    done = True
            except IOError:
                done = True
        logger.info("loaded %d lines of Pilatus data" % len(data))
        data = np.concatenate(data, axis=0)

        # pick out the requested indices
        for i in indices:
            raw[i] = data[i]

        return raw, positions, weights

    def load_weight(self):
        """
        Provides the mask for the whole scan, the shape of the first 
        frame.
        """

        scannr = self.p.scannr
        path = self.p.pilatusPath
        pattern = self.p.pilatusPattern
        if not (path[-1] == '/'):
            path += '/'

        filename = self.p.dataPath + self.p.datafile
        with h5py.File(path + pattern % (scannr, 0), 'r') as hf:
            data = hf.get(self.p.hdfPath)
            shape = np.asarray(data[0]).shape
            mask = np.ones(shape)
            mask[np.where(data[0] == -2)] = 0
        logger.info("took account of the pilatus mask, %u x %u, sum %u" %
                    (mask.shape + (np.sum(mask),)))

        if self.p.maskfile:
            with h5py.File(self.p.maskfile, 'r') as hf:
                mask2 = np.array(hf.get('mask'))
            logger.info("loaded additional mask, %u x %u, sum %u" %
                        (mask2.shape + (np.sum(mask2),)))
            mask = mask * mask2
            logger.info("total mask, %u x %u, sum %u" %
                        (mask.shape + (np.sum(mask),)))

        return mask
