"""
PtyScan subclass to load data from the 5.3.2.1 beamline of the ALS.
Based on one particular data set, and possibly not generally valid for
future experiments at that beamline.
"""

import ptypy
from ptypy.core.data import PtyScan
import ptypy.utils as u
import h5py
import numpy as np
import time

logger = u.verbose.logger

# Default recipe tree.
RECIPE = u.Param()
RECIPE.dataPath = None
RECIPE.datafile = None
RECIPE.maskfile = None
RECIPE.energy = 0.820
RECIPE.CXI_PATHS = u.Param()
RECIPE.CXI_PATHS.CXI_DATA_PATH = 'entry_1/data_1/data'
RECIPE.CXI_PATHS.CXI_MASK_PATH = 'mask'
RECIPE.CXI_PATHS.CXI_POS_PATH = 'entry_1/data_1/translation'
RECIPE.CXI_PATHS.CXI_DISTANCE = 'entry_1/instrument_1/detector_1/distance'
RECIPE.CXI_PATHS.CXI_PSIZES = [
    'entry_1/instrument_1/detector_1/x_pixel_size',
    'entry_1/instrument_1/detector_1/y_pixel_size'
]


class ALS5321Scan(PtyScan):
    """
    Basic class to load 5.3.2.1 data offline.
    """

    def __init__(self, pars=None, **kwargs):
        # Get the default parameter tree and add a recipe branch.
        p = PtyScan.DEFAULT.copy(depth=10)
        p.recipe = RECIPE.copy()
        p.update(pars, in_place_depth=10)

        # Extract geometrical information
        filename = p.recipe.dataPath + p.recipe.datafile
        with h5py.File(filename, 'r') as hf:
            p.energy = RECIPE.energy
            p.distance = hf.get(RECIPE.CXI_PATHS.CXI_DISTANCE).value
            p.psize = [hf.get(RECIPE.CXI_PATHS.CXI_PSIZES[0]).value,
                       hf.get(RECIPE.CXI_PATHS.CXI_PSIZES[1]).value]

        # Call the base class constructor with the new updated parameters.
        # This constructor only reads params, it doesn't modify them. We've
        # already put the kwargs in p, so we don't need to pass them here.
        super(ALS5321Scan, self).__init__(p)

    def load_positions(self):
        filename = self.info.recipe.dataPath + self.info.recipe.datafile

        # first get the total number of positions
        with h5py.File(filename, 'r') as hf:
            data = hf.get(self.info.recipe.CXI_PATHS.CXI_POS_PATH)
            positions = np.asarray(data)[:,:2]
        positions = np.fliplr(positions)
        return positions

    def load(self, indices):
        raw, weights, positions = {}, {}, {}
        filename = self.info.recipe.dataPath + self.info.recipe.datafile
        with h5py.File(filename) as hf:
            data = hf.get(self.info.recipe.CXI_PATHS.CXI_DATA_PATH)
            for i in indices:
                raw[i] = np.asarray(data[i])
        return raw, positions, weights

    def load_weight(self):
        filename = self.info.recipe.dataPath + self.info.recipe.maskfile
        with h5py.File(filename) as hf:
            mask = hf.get(self.info.recipe.CXI_PATHS.CXI_MASK_PATH)
            mask = np.asarray(mask)
        return mask
