"""
PtyScan subclass to load data from the 7.0.1.1 beamline of the ALS.
Based on one particular data set, and possibly not generally valid for
future experiments at that beamline.
Written by: Damian Guenzing (dguenzing@lbl.gov/git:gnzng) during the PtyPy
workshop 2024.
"""

import h5py
import numpy as np
from ptypy.core.data import PtyScan
from ptypy.experiment import register
from ptypy import utils as u

logger = u.verbose.logger


@register()
class ALS7011Scan(PtyScan):
    """
    Starting a fresh class for the BL7.0.1.1 beamline of the ALS. This class
    will use the use the h5 files created by the bluesky exporter. Also the
    h5 files created with the h5repair function from
    https://github.com/ALS-Scattering/BL7011 will work with this class.

    Defaults:

    [name]
    default = ALS7011Scan
    type = str
    help =

    [fpath_h5]
    default = None
    type = str
    help =

    [energy]
    default = 0.708
    type = float
    help =

    [distance]
    default = 0.205
    type = float
    help =

    [psize]
    default = 15e-6
    type = float
    help =
    """

    def load_positions(self):
        """
        loads the positions from the given path self.info.fpath_positions
        """

        with h5py.File(self.info.fpath_h5, "r") as f:
            x = f["entry1"]["instrument_1"]["labview_data"]["sample_translate"][:]
            y = f["entry1"]["instrument_1"]["labview_data"]["sample_lift"][:]

        # convert to meters:
        x = np.array(x) * 1e-6
        y = np.array(y) * 1e-6

        positions = np.array([x, y]).T

        return positions

    def load(self, indices):
        """
        loads the diffraction patterns for a given list of indices
        from the given file self.info.fpath_diffraction_patterns
        """

        raw, weights, positions = {}, {}, {}

        with h5py.File(self.info.fpath_h5, "r") as f:
            raw_all = f["entry1"]["instrument_1"]["detector_1"]["data"][:]

        for ind in indices:
            raw[ind] = raw_all[ind, 0, :, :]

            print(ind, raw[ind].shape)
        return raw, weights, positions

    def load_weight(self):
        """
        Provides the mask for the whole scan,
        the shape of the first frame.
        """

        r, w, p = self.load(indices=(0,))
        first_frame = r[0]

        mask = np.ones_like(first_frame)

        return mask
