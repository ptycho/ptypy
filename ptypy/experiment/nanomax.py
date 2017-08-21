"""  Implementation of PtyScan subclasses to hold nanomax scan data. The
format is very preliminary, and therefore these classes are subject
to frequent change.  """

import ptypy
from ptypy.core.data import PtyScan
import ptypy.utils as u

import h5py
import numpy as np
import time

logger = u.verbose.logger

# All "additional" parameters must come from the recipe tree, and the recipe
# is filled in from the script where data preparation is initiated.
RECIPE = u.Param()
RECIPE.dataPath = None
RECIPE.datafile = None
RECIPE.maskfile = None
RECIPE.scan_shape = None		# read only the first N frames
RECIPE.stepsize = None  	# step size in m

# These are the paths within hdf5 files
NEXUS_DATA_PATH = 'entry/detector/data'
NEXUS_MASK_PATH = 'mask'


class NanomaxTmpScan(PtyScan):
    """
    Basic class to load Nanomax data after the completion of a scan.
    """

    def __init__(self, pars=None, **kwargs):

        # Get the default parameter tree and add a recipe branch. Here, the
        # parameter tree is that corresponding to a scan.data tree, I guess it
        # will eventually become a scans.xxx.data tree.
        p = PtyScan.DEFAULT.copy(depth=10)
        p.recipe = RECIPE.copy()
        p.update(pars, in_place_depth=10)
        # for some reason this removes defaults:
        # p.update(kwargs)

        # For this primitive scan type we need the shape of the scan
        if p.recipe.scan_shape is None:
            raise RuntimeError('We need the shape of a nanomax scan!')

        # This is a good place to extract geometrical information from
        # the data files, if it is not supplied from the parameters
        # through the reconstruction script. We could update the
        # parameter tree here before it gets passed to the PtyScan
        # constructor.

        # Call the base class constructor with the new updated parameters.
        # This constructor only reads params, it doesn't modify them. We've
        # already put the kwargs in p, so we don't need to pass them here.
        super(NanomaxTmpScan, self).__init__(p)

        # From now on, all the input parameters are available through self.info.
        # Some stuff is available also through self.meta.
        # should probably instantiate Geo here somewhere and get distance,
        # psize, shape from somewhere.

    def load_positions(self):
        # returning positions from here causes self.num_frames to get
        # set in self.initialize(), so here we can limit how many frames
        # to read. Here we do this in the recipe.shape parameter.

        # The docs say that positions become available as
        # self.positions, but instead it's in self.info.positions_scan.

        filename = self.info.recipe.dataPath + self.info.recipe.datafile

        # first get the total number of positions
        with h5py.File(filename, 'r') as hf:
            data = hf.get(NEXUS_DATA_PATH)
            data = np.array(data)
            nPositions = data.shape[0]
        # Then generate positions to match. This is based on a very early
        # scan format and assumes that the scan is sqrt(Nframes)-by-
        # sqrt(Nframes).
        positions = []
        for motor_y in range(self.info.recipe.scan_shape[0]):
            # tried to adapt this to Bjoern's info here
            # https://github.com/ptycho/ptypy-dev/issues/39
            # and also to my own notes 2016-09-30.
            for motor_x in range(self.info.recipe.scan_shape[1]):
                # in nanomax coordinates:
                x = motor_x
                y = -motor_y
                # in ptypy coordinates:
                positions.append(np.array([-y, -x])
                                 * self.info.recipe.stepsize)

        return np.array(positions)

    def load(self, indices):
        # returns three dicts: raw, positions, weights, whose keys are the
        # scan pont indices. If one weight (mask) is to be used for the whole
        # scan, it should be loaded with load_weights(). The same goes for the
        # positions.

        # Probably this should slice up the big array on reading, but won't
        # bother now.

        raw, weights, positions = {}, {}, {}

        filename = self.info.recipe.dataPath + self.info.recipe.datafile
        with h5py.File(filename) as hf:
            data = hf.get(NEXUS_DATA_PATH)
            for i in indices:
                raw[i] = np.asarray(data[i])
                #weights[i] = np.ones(data[i].shape)

        return raw, positions, weights

    def load_weight(self):
        """
        Provides the mask for the whole scan, the shape of the first 
        frame.
        """
        if self.info.recipe.maskfile:
            filename = self.info.recipe.dataPath + self.info.recipe.maskfile
            with h5py.File(filename) as hf:
                data = hf.get(NEXUS_MASK_PATH)
                mask = np.asarray(data)
        else:
            filename = self.info.recipe.dataPath + self.info.recipe.datafile
            with h5py.File(filename) as hf:
                data = hf.get(NEXUS_DATA_PATH)
                shape = np.asarray(data[0]).shape
                mask = np.ones(shape)
        logger.info("loaded mask, %u x %u, sum %u" %
                    (mask.shape + (np.sum(mask),)))
        return mask


class NanomaxTmpScanOnline(NanomaxTmpScan):
    """
    Class which loads mimicks loading Nanomax temporary data on the fly,
    at the moment by pretending that data is rolling in slowly. This 
    forms the basis for a proper on the fly loading class.
    """

    def __init__(self, *args, **kwargs):
        # keep track of a starting time to fake slow data acquisition
        self.t0 = time.time()
        super(NanomaxTmpScanOnline, self).__init__(*args, **kwargs)

    def check(self, frames=None, start=None):
        """
        Let the loader know how many frames are available.
        """

        # there are internal PtyScan counters which keep track of the
        # next frame and the number of frames to get at a time. Fall
        # back on these as done in the PtyScan class.
        if start is None:
            start = self.framestart
        if frames is None:
            frames = self.min_frames

        # dispense a given number of scan positions per second:
        positions_per_second = 1
        frames_total = np.prod(self.info.recipe.scan_shape)
        t = int(time.time() - self.t0)
        frames_done = min(t * positions_per_second + 10, frames_total)

        # work out how many frames are currently available and whether
        # the scan is done.
        frames_accessible = min(frames_done - start, frames)
        end_of_scan = int(start + frames_accessible >= frames_total)
        self.debug("%u/%u done, %u available, requesting %u frames starting at %u, eos=%s" %
                   (frames_done, frames_total, frames_accessible, frames,
                    start, str(end_of_scan)))

        return frames_accessible, end_of_scan

    def load(self, indices):
        t0 = time.time()
        # Here we supply both images and positions, but not the mask.
        raw, weights, positions = {}, {}, {}
        filename = self.info.recipe.dataPath + self.info.recipe.datafile

        # data and positions
        with h5py.File(filename) as hf:
            data = hf.get(NEXUS_DATA_PATH)
            for i in indices:
                raw[i] = np.asarray(data[i])
                vertical = i // self.info.recipe.scan_shape[1]
                horizontal = i % self.info.recipe.scan_shape[1]
                positions[i] = (np.array([vertical, -horizontal])
                                * self.info.recipe.stepsize)
                #weights[i] = np.ones(raw[i].shape)
        return raw, positions, weights

    def load_positions(self):
        """ 
        Now we don't want to load any positions beforehand. Returning None
        here will make self.initialize() understand that the positions are
        not available, as shown in the terminal output.
        """
        return None

    def load_weight(self):
        """
        Provides the mask for the whole scan, the shape of the first 
        frame.
        """
        filename = self.info.recipe.dataPath + self.info.recipe.datafile
        with h5py.File(filename) as hf:
            data = hf.get(NEXUS_DATA_PATH)
            shape = np.asarray(data[0]).shape
        return np.ones(shape)


# new recipe for this one
RECIPE = u.Param()
RECIPE.dataPath = None
RECIPE.datafile = None
RECIPE.maskfile = None
RECIPE.pilatusPath = None
RECIPE.pilatusPattern = None
RECIPE.scannr = None


class NanomaxFlyscanWeek48(PtyScan):
    """
    Loads Nanomax fly scan data in the format of week 48
    """

    def __init__(self, pars=None, **kwargs):

        raise NotImplementedError("this subclass needs updating")
        p = PtyScan.DEFAULT.copy(depth=10)
        p.recipe = RECIPE.copy()
        p.update(pars, in_place_depth=10)
        super(NanomaxFlyscanWeek48, self).__init__(p)

    def load_positions(self):
        fileName = self.info.recipe.dataPath + self.info.recipe.datafile
        entry = 'entry%d' % self.info.recipe.scannr

        x, y = None, None
        with h5py.File(fileName, 'r') as hf:
            # get fast x positions
            xdataset = hf.get(entry + '/measurement/AdLinkAI')
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

        positions = - np.vstack((x, y)).T * 1e-6
        return positions

    def load(self, indices):

        raw, weights, positions = {}, {}, {}
        scannr = self.info.recipe.scannr
        path = self.info.recipe.pilatusPath
        pattern = self.info.recipe.pilatusPattern

        # read the entire dataset
        done = False
        line = 0
        data = []
        while not done:
            try:
                with h5py.File(path + pattern % (scannr, line), 'r') as hf:
                    logger.info('loading data: ' + pattern % (scannr, line))
                    dataset = hf.get('entry_0000/measurement/Pilatus/data')
                    data.append(np.array(dataset))
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
        path = self.info.recipe.pilatusPath
        pattern = self.info.recipe.pilatusPattern

        if self.info.recipe.maskfile:
            raise NotImplementedError('No masks for this scan type yet!')
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


class NanomaxStepscanWeek48(PtyScan):
    """
    Loads Nanomax step scan data in the format of week 48
    """

    def __init__(self, pars=None, **kwargs):

        p = PtyScan.DEFAULT.copy(depth=10)
        p.recipe = RECIPE.copy()
        p.update(pars, in_place_depth=10)
        super(NanomaxStepscanWeek48, self).__init__(p)

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
    Loads Nanomax step scan data in the format of week 48
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
    Loads Nanomax fly scan data in the format of june 2017.
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
