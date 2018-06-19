"""
PtyScan subclass to load data from the 5.3.2.1 beamline of the ALS.
Based on one particular data set, and possibly not generally valid for
future experiments at that beamline.
"""

import h5py
import numpy as np

from ..core.data import PtyScan
from .. import utils as u
from . import register

logger = u.verbose.logger


@register()
class ALS5321Scan(PtyScan):
    """
    Basic class to load 5.3.2.1 data offline.

    Defaults:

    [name]
    default = ALS5321Scan
    type = str
    help =

    [dataPath]
    default = None
    type = str
    help = Path to folder containing the data

    [dataile]
    default = None
    type = str
    help = CXI data file

    [maskfile]
    default = None
    type = str
    help = Optional mask file
    doc = Should contain an array called 'mask' at the root level.

    [energy]
    default = 0.820

    [CXI_PATHS]
    default = None
    type = Param
    help = Container for CXI path options

    [CXI_PATHS.CXI_DATA_PATH]
    default = 'entry_1/data_1/data'
    type = str
    help = Data path within the CXI file

    [CXI_PATHS.CXI_MASK_PATH]
    default = 'mask'
    type = str
    help = Mask path within the CXI file

    [CXI_PATHS.CXI_POS_PATH]
    default = 'entry_1/data_1/translation'
    type = str
    help = Positions path within the CXI file

    [CXI_PATHS.CXI_DISTANCE]
    default = 'entry_1/instrument_1/detector_1/distance'
    type = str
    help = Distance path within the CXI file

    [CXI_PATHS.CXI_PSIZES]
    default = ['entry_1/instrument_1/detector_1/x_pixel_size', 'entry_1/instrument_1/detector_1/y_pixel_size']
    type = list
    help = Pixel size path within the CXI file

    """

    def __init__(self, pars=None, **kwargs):
        # Get the default parameter tree.
        p = self.DEFAULT.copy(99)
        p.update(pars)

        # Call the base class constructor with the new updated parameters.
        # This constructor only reads params, it doesn't modify them. 
        super(ALS5321Scan, self).__init__(p, **kwargs)

        # Extract geometrical information
        filename = p.dataPath + p.datafile
        with h5py.File(filename, 'r') as hf:
            self.info.energy = p.energy
            self.info.distance = hf.get(p.CXI_PATHS.CXI_DISTANCE).value
            self.info.psize = [hf.get(p.CXI_PATHS.CXI_PSIZES[0]).value,
                       hf.get(p.CXI_PATHS.CXI_PSIZES[1]).value]

    def load_positions(self):
        filename = self.info.dataPath + self.info.datafile

        # first get the total number of positions
        with h5py.File(filename, 'r') as hf:
            data = hf.get(self.info.CXI_PATHS.CXI_POS_PATH)
            positions = np.asarray(data)[:,:2]
        positions = np.fliplr(positions)
        return positions

    def load(self, indices):
        raw, weights, positions = {}, {}, {}
        filename = self.info.dataPath + self.info.datafile
        with h5py.File(filename) as hf:
            data = hf.get(self.info.CXI_PATHS.CXI_DATA_PATH)
            for i in indices:
                raw[i] = np.asarray(data[i])
        return raw, positions, weights

    def load_weight(self):
        filename = self.info.dataPath + self.info.maskfile
        with h5py.File(filename) as hf:
            mask = hf.get(self.info.CXI_PATHS.CXI_MASK_PATH)
            mask = np.asarray(mask)
        return mask
