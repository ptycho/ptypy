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
# is filled in from the script where data preparation is initiated. Here's the
# default.
RECIPE = u.Param()
RECIPE.dataPath = None
RECIPE.datafile = None
RECIPE.maskfile = None
RECIPE.scan_shape = None		# read only the first N frames
RECIPE.stepsize = None  	# step size in m

# These are the paths within hdf5 files
NEXUS_DATA_PATH = 'entry/detector/data'


def positions_from_hdf5(filename, stepsize, shape):
    # first get the total number of positions
    with h5py.File(filename, 'r') as hf:
        data = hf.get(NEXUS_DATA_PATH)
        data = np.array(data)
        nPositions = data.shape[0]
    # Then generate positions to match. This is based on a very early
    # scan format and assumes that the scan is sqrt(Nframes)-by-
    # sqrt(Nframes).
    positions = []
    for motor_y in range(shape[0]):
        # tried to adapt this to Bjoern's info here
        # https://github.com/ptycho/ptypy-dev/issues/39
        for motor_x in range(shape[1]):
            positions.append(np.array([-motor_y, -motor_x]) * stepsize)
    return np.array(positions)


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
        return positions_from_hdf5(filename, self.info.recipe.stepsize, self.info.recipe.scan_shape)

    def load(self, indices):
        # returns three dicts: raw, positions, weights, whose keys are the
        # scan pont indices. If one weight (mask) is to be used for the whole
        # scan, it should be loaded with load_weights(). The same goes for the
        # positions. We don't really need a mask here, but it must be
        # provided, otherwise it's given the shape of self.info.shape, and
        # then there's a shape mismatch in some multiplication.

        # Probably this should slice up the big array on reading, but won't
        # bother now.

        raw, weights, positions = {}, {}, {}

        filename = self.info.recipe.dataPath + self.info.recipe.datafile
        with h5py.File(filename) as hf:
            data = hf.get(NEXUS_DATA_PATH)
            for i in indices:
                raw[i] = np.asarray(data[i])
                weights[i] = np.ones(data[i].shape)

        return raw, positions, weights


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
        positions_per_second = 5
        frames_total = np.prod(self.info.recipe.scan_shape)
        t = int(time.time() - self.t0)
        frames_done = min(t * positions_per_second + 10, frames_total)

        # work out how many frames are currently available and whether
        # the scan is done.
        frames_accessible = min(frames_done - start, frames)
        end_of_scan = int(start + frames_accessible >= frames_total)
        print "****** %u/%u done, %u available, requesting %u frames starting at %u, eos=%s" % (frames_done, frames_total, frames_accessible, frames, start, str(end_of_scan))

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
                positions[i] = -(np.array([vertical, horizontal])
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
