"""  Implementation of PtyScan subclasses to hold nanomax scan data. The
     beamline is developing, as is the data format. Thus, commissioning
     and user experiments rely on different classes depending on
     measurement campaign. """


import ptypy
from ptypy.core.data import PtyScan
import ptypy.utils as u

import h5py
import numpy as np
import time

logger = u.verbose.logger

# new recipe for this one
RECIPE = u.Param()
RECIPE.dataPath = None
RECIPE.datafile = None
RECIPE.maskfile = None
RECIPE.pilatusPath = None
RECIPE.pilatusPattern = None
RECIPE.scannr = None


class NanomaxStepscanNov2016(PtyScan):
    """
    Loads Nanomax step scan data in the format of week Nov/Dec 2016
    """

    def __init__(self, pars=None, **kwargs):

        p = PtyScan.DEFAULT.copy(depth=10)
        p.recipe = RECIPE.copy()
        p.update(pars, in_place_depth=10)
        super(NanomaxStepscanNov2016, self).__init__(p)

    def load_positions(self):
        fileName = self.info.recipe.dataPath + self.info.recipe.datafile
        entry = 'entry%d' % self.info.recipe.scannr

        with h5py.File(fileName, 'r') as hf:
            x = np.array(hf.get(entry + '/measurement/samx'))
            y = np.array(hf.get(entry + '/measurement/samy'))

        positions = -np.vstack((y, x)).T * 1e-6
        return positions

    def load(self, indices):

        raw, weights, positions = {}, {}, {}
        scannr = self.info.recipe.scannr
        path = self.info.recipe.pilatusPath
        filepattern = self.info.recipe.pilatusPattern
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

        scannr = self.info.recipe.scannr
        path = self.info.recipe.pilatusPath
        pattern = self.info.recipe.pilatusPattern
        if not (path[-1] == '/'):
            path += '/'

        if self.info.recipe.maskfile:
            with h5py.File(self.info.recipe.maskfile, 'r') as hf:
                mask = np.array(hf.get('mask'))
            logger.info("loaded mask, %u x %u, sum %u" %
                        (mask.shape + (np.sum(mask),)))
        else:
            filename = self.info.recipe.dataPath + self.info.recipe.datafile
            with h5py.File(path + pattern % (scannr, 0), 'r') as hf:
                data = hf.get('entry_0000/measurement/Pilatus/data')
                shape = np.asarray(data[0]).shape
                mask = np.ones(shape)
            logger.info("created dummy mask, %u x %u, sum %u" %
                        (mask.shape + (np.sum(mask),)))
        return mask

# new recipe for this one too
RECIPE = u.Param()
RECIPE.dataPath = None
RECIPE.datafile = None
RECIPE.maskfile = None
RECIPE.pilatusPath = None
RECIPE.pilatusPattern = None
RECIPE.hdfPath = 'entry_0000/measurement/Pilatus/data'
RECIPE.scannr = None
RECIPE.xMotorFlipped = None
RECIPE.yMotorFlipped = None
RECIPE.xMotorAngle = 0.0


class NanomaxStepscanMay2017(PtyScan):
    """
    Loads Nanomax step scan data in the format of May 2017.
    """

    def __init__(self, pars=None, **kwargs):

        p = PtyScan.DEFAULT.copy(depth=10)
        p.recipe = RECIPE.copy()
        p.update(pars, in_place_depth=10)
        super(NanomaxStepscanMay2017, self).__init__(p)

    def load_positions(self):
        fileName = self.info.recipe.dataPath + self.info.recipe.datafile
        entry = 'entry%d' % self.info.recipe.scannr

        xFlipper, yFlipper = 1, 1
        if self.info.recipe.xMotorFlipped:
            xFlipper = -1
            logger.warning("note: x motor is specified as flipped")
        if self.info.recipe.yMotorFlipped:
            yFlipper = -1
            logger.warning("note: y motor is specified as flipped")

        # if the x axis is tilted, take that into account.
        xCosFactor = np.cos(self.info.recipe.xMotorAngle / 180.0 * np.pi)
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
        scannr = self.info.recipe.scannr
        path = self.info.recipe.pilatusPath
        filepattern = self.info.recipe.pilatusPattern
        if not (path[-1] == '/'):
            path += '/'

        data = []
        for im in range(self.info.positions_scan.shape[0]):
            with h5py.File(path + filepattern % (scannr, im), 'r') as hf:
                dataset = hf.get(self.info.recipe.hdfPath)
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

        scannr = self.info.recipe.scannr
        path = self.info.recipe.pilatusPath
        pattern = self.info.recipe.pilatusPattern
        if not (path[-1] == '/'):
            path += '/'

        filename = self.info.recipe.dataPath + self.info.recipe.datafile
        with h5py.File(path + pattern % (scannr, 0), 'r') as hf:
            data = hf.get(self.info.recipe.hdfPath)
            shape = np.asarray(data[0]).shape
            mask = np.ones(shape)
            mask[np.where(data[0] == -2)] = 0
        logger.info("took account of the pilatus mask, %u x %u, sum %u" %
                    (mask.shape + (np.sum(mask),)))

        if self.info.recipe.maskfile:
            with h5py.File(self.info.recipe.maskfile, 'r') as hf:
                mask2 = np.array(hf.get('mask'))
            logger.info("loaded additional mask, %u x %u, sum %u" %
                        (mask2.shape + (np.sum(mask2),)))
            mask = mask * mask2
            logger.info("total mask, %u x %u, sum %u" %
                        (mask.shape + (np.sum(mask),)))

        return mask

# new recipe for this one too
RECIPE = u.Param()
RECIPE.dataPath = None
RECIPE.datafile = None
RECIPE.maskfile = None
RECIPE.detFilePath = None
RECIPE.detFilePattern = None
RECIPE.detNormalizationFilePattern = None
RECIPE.detNormalizationIndices = None
RECIPE.hdfPath = 'entry_0000/measurement/Pilatus/data'
RECIPE.scannr = None
RECIPE.xMotorFlipped = None
RECIPE.yMotorFlipped = None
RECIPE.xMotorAngle = 0.0


class NanomaxFlyscanJune2017(PtyScan):
    """
    Loads Nanomax fly scan data in the format of June 2017.
    """

    def __init__(self, pars=None, **kwargs):
        p = PtyScan.DEFAULT.copy(depth=10)
        p.recipe = RECIPE.copy()
        p.update(pars, in_place_depth=10)
        super(NanomaxFlyscanJune2017, self).__init__(p)

    def load_positions(self):
        fileName = self.info.recipe.dataPath + self.info.recipe.datafile
        entry = 'entry%d' % self.info.recipe.scannr

        x, y = None, None
        with h5py.File(fileName, 'r') as hf:
            # get fast x positions
            xdataset = hf.get(entry + '/measurement/AdLinkAI_buff')
            xall = np.array(xdataset)
            # manually find shape by looking for zeros
            Ny = xall.shape[0]
            for i in range(xall.shape[1]):
                if xall[0, i] == 0:
                    Nx = i
                    break
            x = xall[:, :Nx].flatten()

            # get slow y positions
            ydataset = hf.get(entry + '/measurement/samy')
            yall = np.array(ydataset)
            if not (len(yall) == Ny):
                raise Exception('Something''s wrong with the positions')
            y = np.repeat(yall, Nx)

        if self.info.recipe.xMotorFlipped:
            x *= -1
            logger.warning("note: x motor is specified as flipped")
        if self.info.recipe.yMotorFlipped:
            y *= -1
            logger.warning("note: y motor is specified as flipped")

        # if the x axis is tilted, take that into account.
        xCosFactor = np.cos(self.info.recipe.xMotorAngle / 180.0 * np.pi)
        x *= xCosFactor
        logger.info(
            "x motor angle results in multiplication by %.2f" % xCosFactor)

        positions = - np.vstack((y, x)).T * 1e-6
        return positions

    def load(self, indices):

        raw, weights, positions = {}, {}, {}
        scannr = self.info.recipe.scannr
        path = self.info.recipe.detFilePath
        pattern = self.info.recipe.detFilePattern
        normfile = self.info.recipe.detNormalizationFilePattern
        normind = self.info.recipe.detNormalizationIndices

        # read the entire dataset
        done = False
        line = 0
        data = []
        while not done:
            try:
                with h5py.File(path + pattern % (scannr, line), 'r') as hf:
                    logger.info('loading data: ' + pattern % (scannr, line))
                    dataset = hf.get(self.info.recipe.hdfPath)
                    linedata = np.array(dataset)
                if normfile:
                    dtype = linedata.dtype
                    linedata = np.array(linedata, dtype=float)
                    with h5py.File(path + normfile % (scannr, line), 'r') as hf:
                        logger.info('loading normalization data: ' +
                                    normfile % (scannr, line))
                        dataset = hf.get(
                            self.info.recipe.detNormalizationHdfPath)
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

        scannr = self.info.recipe.scannr
        path = self.info.recipe.detFilePath
        pattern = self.info.recipe.detFilePattern
        if not (path[-1] == '/'):
            path += '/'

        filename = self.info.recipe.dataPath + self.info.recipe.datafile
        with h5py.File(path + pattern % (scannr, 0), 'r') as hf:
            data = hf.get(self.info.recipe.hdfPath)
            shape = np.asarray(data[0]).shape
            mask = np.ones(shape)
            mask[np.where(data[0] == -2)] = 0
        logger.info("took account of the pilatus mask, %u x %u, sum %u" %
                    (mask.shape + (np.sum(mask),)))

        if self.info.recipe.maskfile:
            with h5py.File(self.info.recipe.maskfile, 'r') as hf:
                mask2 = np.array(hf.get('mask'))
            logger.info("loaded additional mask, %u x %u, sum %u" %
                        (mask2.shape + (np.sum(mask2),)))
            mask = mask * mask2
            logger.info("total mask, %u x %u, sum %u" %
                        (mask.shape + (np.sum(mask),)))

        return mask
