from ptypy.core import Ptycho
from ptypy import utils as u
from test import utils as tu
import tempfile
import unittest

class PrepAndRunMoonFlowerTest(unittest.TestCase):

    def test_dm_single_probe(self):
        p = u.Param()
        p.verbose_level = 3
        p.io = u.Param()
        p.io.interaction = u.Param()
        p.io.interaction.active = False
        p.io.home = "./"
        p.io.rfile = "None.ptyr"
        p.io.autosave = u.Param(active=False)
        p.io.autoplot = u.Param(active=False)
        p.ipython_kernel = False
        p.scans = u.Param()
        p.scans.MF = u.Param()
        p.scans.MF.name = 'Full'
        p.scans.MF.data= u.Param()
        p.scans.MF.data.name = 'MoonFlowerScan'
        p.scans.MF.data.shape = 64
        p.scans.MF.data.num_frames = 100
        p.scans.MF.data.save = None
        p.scans.MF.data.density = 0.2
        p.scans.MF.data.photons = 1e8
        p.scans.MF.data.psf = 0.0
        p.engines = u.Param()
        p.engines.engine00 = u.Param()
        p.engines.engine00.name = 'DM'
        p.engines.engine00.numiter = 5
        P = Ptycho(p,level=5)

    def test_dm_multiple_probes(self):
        p = u.Param()
        p.verbose_level = 3
        p.io = u.Param()
        p.io.interaction = u.Param()
        p.io.interaction.active = False
        p.io.home = "./"
        p.io.rfile = "None.ptyr"
        p.io.autosave = u.Param(active=False)
        p.io.autoplot = u.Param(active=False)
        p.ipython_kernel = False
        p.scans = u.Param()
        p.scans.MF = u.Param()
        p.scans.MF.name = 'Full'
        p.scans.MF.data= u.Param()
        p.scans.MF.data.name = 'MoonFlowerScan'
        p.scans.MF.data.shape = 64
        p.scans.MF.data.num_frames = 200
        p.scans.MF.data.save = None
        p.scans.MF.data.density = 0.15
        p.scans.MF.data.photons = 1e8
        p.scans.MF.data.psf = 0.7
        p.scans.MF.coherence = u.Param()
        p.scans.MF.coherence.num_probe_modes = 6
        p.scans.MF.illumination = u.Param()
        p.scans.MF.illumination.diversity = u.Param(noise=(1.0, 1.0))
        p.engines = u.Param()
        p.engines.engine00 = u.Param()
        p.engines.engine00.name = 'DM'
        p.engines.engine00.numiter = 5
        p.engines.engine00.fourier_relax_factor = 0.05
        P = Ptycho(p,level=5)

    def test_dm_resample(self):
        p = u.Param()
        p.verbose_level = 3
        p.io = u.Param()
        p.io.interaction = u.Param()
        p.io.interaction.active = False
        p.io.home = "./"
        p.io.rfile = "None.ptyr"
        p.io.autosave = u.Param(active=False)
        p.io.autoplot = u.Param(active=False)
        p.ipython_kernel = False
        p.scans = u.Param()
        p.scans.MF = u.Param()
        p.scans.MF.name = 'Full'
        p.scans.MF.data= u.Param()
        p.scans.MF.data.name = 'MoonFlowerScan'
        p.scans.MF.data.shape = 64
        p.scans.MF.data.num_frames = 100
        p.scans.MF.data.save = None
        p.scans.MF.data.density = 0.2
        p.scans.MF.data.photons = 1e8
        p.scans.MF.data.psf = 0.0
        p.scans.MF.resample = 2
        p.engines = u.Param()
        p.engines.engine00 = u.Param()
        p.engines.engine00.name = 'DM'
        p.engines.engine00.numiter = 5
        P = Ptycho(p,level=5)

    def test_ml_single_probe(self):
        p = u.Param()
        p.verbose_level = 3
        p.io = u.Param()
        p.io.interaction = u.Param()
        p.io.interaction.active = False
        p.io.home = "./"
        p.io.rfile = "None.ptyr"
        p.io.autosave = u.Param(active=False)
        p.io.autoplot = u.Param(active=False)
        p.ipython_kernel = False
        p.scans = u.Param()
        p.scans.MF = u.Param()
        p.scans.MF.name = 'Full'
        p.scans.MF.data= u.Param()
        p.scans.MF.data.name = 'MoonFlowerScan'
        p.scans.MF.data.shape = 64
        p.scans.MF.data.num_frames = 100
        p.scans.MF.data.save = None
        p.scans.MF.data.density = 0.2
        p.scans.MF.data.photons = 1e8
        p.scans.MF.data.psf = 0.
        p.engines = u.Param()
        p.engines.engine00 = u.Param()
        p.engines.engine00.name = 'ML'
        p.engines.engine00.ML_type = 'Gaussian'
        p.engines.engine00.reg_del2 = True
        p.engines.engine00.reg_del2_amplitude = 1.
        p.engines.engine00.scale_precond = True
        #p.engines.engine00.scale_probe_object = 1.
        p.engines.engine00.smooth_gradient = 20.
        p.engines.engine00.smooth_gradient_decay = 1/50.
        p.engines.engine00.floating_intensities = False
        p.engines.engine00.numiter = 5
        P = Ptycho(p,level=5)

    def test_ml_resample(self):
        p = u.Param()
        p.verbose_level = 3
        p.io = u.Param()
        p.io.interaction = u.Param()
        p.io.interaction.active = False
        p.io.home = "./"
        p.io.rfile = "None.ptyr"
        p.io.autosave = u.Param(active=False)
        p.io.autoplot = u.Param(active=False)
        p.ipython_kernel = False
        p.scans = u.Param()
        p.scans.MF = u.Param()
        p.scans.MF.name = 'Full'
        p.scans.MF.data= u.Param()
        p.scans.MF.data.name = 'MoonFlowerScan'
        p.scans.MF.data.shape = 64
        p.scans.MF.data.num_frames = 100
        p.scans.MF.data.save = None
        p.scans.MF.data.density = 0.2
        p.scans.MF.data.photons = 1e8
        p.scans.MF.data.psf = 0.
        p.scans.MF.resample = 2
        p.engines = u.Param()
        p.engines.engine00 = u.Param()
        p.engines.engine00.name = 'ML'
        p.engines.engine00.ML_type = 'Gaussian'
        p.engines.engine00.reg_del2 = True
        p.engines.engine00.reg_del2_amplitude = 1.
        p.engines.engine00.scale_precond = True
        #p.engines.engine00.scale_probe_object = 1.
        p.engines.engine00.smooth_gradient = 20.
        p.engines.engine00.smooth_gradient_decay = 1/50.
        p.engines.engine00.floating_intensities = False
        p.engines.engine00.numiter = 5
        P = Ptycho(p,level=5)

if __name__ == '__main__':
    unittest.main()
