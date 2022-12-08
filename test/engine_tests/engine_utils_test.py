"""
Test for the DM engine.

This file is part of the PTYPY package.
    :copyright: Copyright 2014 by the PTYPY team, see AUTHORS.
    :license: see LICENSE for details.
"""

import unittest
from test import utils as tu
import numpy as np
from ptypy import utils as u
from ptypy.core import Ptycho
from ptypy.engines import utils as eu

np.random.seed(1234)

def get_ptycho(model='Full', base_dir='./'):
    p = u.Param()
    p.verbose_level = 3
    p.io = u.Param()
    p.io.interaction = u.Param()
    p.io.interaction.active = False
    p.io.home = base_dir
    p.io.rfile = base_dir + model + "_test.ptyr"
    p.io.autosave = u.Param(active=False)
    p.io.autoplot = u.Param(active=False)
    p.ipython_kernel = False
    p.scans = u.Param()
    p.scans.MF = u.Param()
    p.scans.MF.name = model
    p.scans.MF.propagation = 'farfield'
    p.scans.MF.data = u.Param()
    p.scans.MF.data.name = 'MoonFlowerScan'
    p.scans.MF.data.positions_theory = None
    p.scans.MF.data.auto_center = None
    p.scans.MF.data.min_frames = 1
    p.scans.MF.data.orientation = None
    p.scans.MF.data.num_frames = 100
    p.scans.MF.data.energy = 6.2
    p.scans.MF.data.shape = 64
    p.scans.MF.data.chunk_format = '.chunk%02d'
    p.scans.MF.data.rebin = None
    p.scans.MF.data.experimentID = None
    p.scans.MF.data.label = None
    p.scans.MF.data.version = 0.1
    p.scans.MF.data.dfile = base_dir + model + "_test.ptyd"
    p.scans.MF.data.psize = 0.000172
    p.scans.MF.data.load_parallel = None
    p.scans.MF.data.distance = 7.0
    p.scans.MF.data.save = None
    p.scans.MF.data.center = 'fftshift'
    p.scans.MF.data.photons = 100000000.0
    p.scans.MF.data.psf = 0.0
    p.scans.MF.data.density = 0.2
    p.scans.MF.coherence = u.Param()
    p.scans.MF.coherence.num_probe_modes = 1 # currently breaks when this is =2

    # init data is level 2
    P = Ptycho(p, level=2)

    return P

class FourierUpdateTest(unittest.TestCase):
    def test_legacy_general_UNITY(self):
        P = get_ptycho(model='Full', base_dir='./')
        pod = list(P.pods.values())[0]
        diff = pod.di_view
        e = {}
        pods = list(diff.pods.values())
        for p in pods:
            print(pod.exit)
            e[p.ex_view.ID] = pod.exit.copy()
        error_LEGACY = eu.basic_fourier_update_LEGACY(diff, None, alpha=1.0, LL_error=False)
        res = {}
        for p in pods:
            print(pod.exit)
            res[p.ex_view.ID] = pod.exit.copy()
            pod.exit = e[p.ex_view.ID]

        error = eu.basic_fourier_update(diff, None, alpha=1.0, LL_error=False)
        for p in pods:
            print(pod.exit)
            #np.testing.assert_array_equal(res[p.ex_view.ID], pod.exit,
            #                              "Exit wave data diverges after fourier update")
        np.testing.assert_array_equal(error_LEGACY, error, "Error metrics diverge")

if __name__ == "__main__":
    unittest.main()
