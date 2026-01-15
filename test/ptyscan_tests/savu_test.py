"""\
Test for the I08 beamline ptyscan, Diamond Light Source.

This file is part of the PTYPY package.
    :copyright: Copyright 2014 by the PTYPY team, see AUTHORS.
    :license: see LICENSE for details.
"""

import unittest
from .. import utils as tu
from ptypy.experiment.savu import Savu
from ptypy import utils as u
import h5py as h5
import numpy as np

class SavuTest(unittest.TestCase):
    @unittest.skip("this won't work unless we figure out how to treat the data")
    def test_savu(self):
        r = u.Param()
        r.base_path = tu.get_test_data_path('dls')
        r.mask = h5.File(r.base_path+'processing/mask2.hdf')['mask'][...]
        r.data = h5.File(r.base_path+'raw/68862.nxs')['entry1/merlin_sw_hdf/data'][...]
        r.positions = np.array([h5.File(r.base_path+'raw/68862.nxs')['entry1/merlin_sw_hdf/lab_sy'][...],
                                h5.File(r.base_path+'raw/68862.nxs')['entry1/merlin_sw_hdf/lab_sx'][...]]).T
        tu.PtyscanTestRunner(Savu,r)

if __name__ == "__main__":
    unittest.main()
