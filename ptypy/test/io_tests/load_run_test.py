'''
A test for load_run - might get expanded and renamed in the future, but I depend on it, so for now this is just protection
'''


import unittest
import tempfile
import numpy as np
import os
from copy import deepcopy

from ptypy.core import Ptycho
import ptypy.utils as u

class LoadRunTest(unittest.TestCase):
    def test_load_run_with_data(self):
        outpath = tempfile.mkdtemp(prefix='something')

        file_path = outpath + os.sep + 'reconstruction.ptyr'
        p = u.Param()
        p.verbose_level = 3
        p.io = u.Param()
        p.io.home = outpath
        p.io.rfile = file_path
        p.io.autosave = u.Param(active=False)
        p.io.autoplot = u.Param(active=False)
        p.ipython_kernel = False
        p.scans = u.Param()
        p.scans.MF = u.Param()
        p.scans.MF.name = 'Full'
        p.scans.MF.propagation = "farfield"
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
        p.scans.MF.data.dfile = file_path
        p.scans.MF.data.psize = 0.000172
        p.scans.MF.data.load_parallel = None
        p.scans.MF.data.distance = 7.0
        p.scans.MF.data.save = None
        p.scans.MF.data.center = 'fftshift'
        p.scans.MF.data.photons = 100000000.0
        p.scans.MF.data.psf = 0.0
        p.scans.MF.data.density = 0.2
        p.engines = u.Param()
        p.engines.engine00 = u.Param()
        p.engines.engine00.name = 'DM'
        p.engines.engine00.numiter = 5
        p.engines.engine00.alpha =1
        p.engines.engine00.probe_update_start = 2
        p.engines.engine00.overlap_converge_factor = 0.05
        p.engines.engine00.overlap_max_iterations = 10
        p.engines.engine00.probe_inertia = 1e-3
        p.engines.engine00.object_inertia = 0.1
        p.engines.engine00.fourier_relax_factor = 0.01
        p.engines.engine00.obj_smooth_std = 20

        P = Ptycho(p, level=2) # this is what we should get back
        Pcomp = deepcopy(P)
        P.init_communication()
        P.init_engine()
        P.run()
        P.finalize()


        b = Ptycho.load_run(file_path)
        np.testing.assert_equal(type(b), type(Pcomp))
        np.testing.assert_equal(b.__dict__, Pcomp.__dict__) # this is as far as I want to go for now



    @unittest.skip("To be filled in")
    def test_load_run_no_data(self):
        b = Ptycho.load_run(self.parameters['ptyr_file'], False)  # load in the run but without the data

        p = b.p
        pass