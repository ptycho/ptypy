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
    def test_load_run(self):

        # in the core tests, the moonflower defaults to random noise. We need to keep this consistent for this check,
        # so we set the random seed
        np.random.seed(1)

        outpath = tempfile.mkdtemp(prefix='something')

        file_path = outpath + os.sep + 'reconstruction.ptyr'
        p = u.Param()
        p.verbose_level = 0
        p.io = u.Param()
        p.io.home = outpath
        p.io.rfile = file_path
        p.io.autosave = u.Param(active=True)
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
        p.scans.MF.data.add_poisson_noise = False
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

        # external check!
        # from ptypy import io
        # content = io.h5read(file_path, 'content')['content']
        # self.maxDiff =None
        # print content.pars.io._to_dict(Recursive=True)
        # set_vals = P.p._to_dict(Recursive=True)
        # print set_vals
        # file_vals = content.pars._to_dict(Recursive=True)
        # for name, val in file_vals.items():
        #     self.assertEqual(file_vals[name], set_vals[name])
        # self.assertDictEqual(content.pars._to_dict(Recursive=True), set_vals)
        np.random.seed(1)

        b = Ptycho.load_run(file_path)
        np.testing.assert_equal(type(b), type(Pcomp))
        for name, st in b.mask.storages.items():
            np.testing.assert_equal(st.data, P.mask.storages[name].data)

        for name, st in b.diff.storages.items():
            np.testing.assert_equal(st.data, P.diff.storages[name].data)

        for name, st in b.probe.storages.items():
            np.testing.assert_equal(st.data, P.probe.storages[name].data)

        for name, st in b.obj.storages.items():
            np.testing.assert_equal(st.data, P.obj.storages[name].data)


