"""
This script is a test for ptychographic reconstruction in the absence
of actual data. It uses the test Scan class
`ptypy.core.data.MoonFlowerScan` to provide "data".
"""
import ptypy
from ptypy.core import Ptycho
from ptypy import utils as u
p = u.Param()

# for verbose output
p.verbose_level = 3

# set home path
p.io = u.Param()
p.io.home = "/tmp/ptypy/"
p.io.autosave = None

# max 100 frames (128x128px) of diffraction data
p.scans = u.Param()
p.scans.MF = u.Param()
p.scans.MF.data= u.Param()
p.scans.MF.data.source = 'test'
p.scans.MF.data.shape = 128
p.scans.MF.data.num_frames = 100
p.scans.MF.data.save = None

## special recipe paramters for this scan ##
p.scans.MF.data.recipe = u.Param()
# position distance in fraction of illumination frame
p.scans.MF.data.recipe.density = 0.2
# total number of photon in empty beam
p.scans.MF.data.recipe.photons = 1e8
# Gaussian FWHM of possible detector blurring
p.scans.MF.data.recipe.psf = 0.

# attach a reconstrucion engine
p.engines = u.Param()
p.engines.engine00 = u.Param()
p.engines.engine00.name = 'DM'
p.engines.engine00.numiter = 30

# prepare and run
P = Ptycho(p,level=5)
