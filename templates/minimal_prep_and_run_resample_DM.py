"""
This script is a test for ptychographic reconstruction in the absence
of actual data. It uses the test Scan class
`ptypy.core.data.MoonFlowerScan` to provide "data".
"""

from ptypy.core import Ptycho
from ptypy import utils as u
p = u.Param()

# for verbose output
p.verbose_level = 3

# set home path
p.io = u.Param()
p.io.home = "/tmp/ptypy/"
p.io.autosave = None

# max 200 frames (128x128px) of diffraction data
p.scans = u.Param()
p.scans.MF = u.Param()
# now you have to specify which ScanModel to use with scans.XX.name,
# just as you have to give 'name' for engines and PtyScan subclasses.
p.scans.MF.name = 'Vanilla' # or 'Full'
p.scans.MF.data= u.Param()
p.scans.MF.data.name = 'MoonFlowerScan'
p.scans.MF.data.shape = 128
p.scans.MF.data.num_frames = 200
p.scans.MF.data.save = None

# position distance in fraction of illumination frame
p.scans.MF.data.density = 0.2
# total number of photon in empty beam
p.scans.MF.data.photons = 1e8
# Gaussian FWHM of possible detector blurring
p.scans.MF.data.psf = 0.

# Resample by a factor of 2
p.scans.MF.resample = 2

# attach a reconstrucion engine
p.engines = u.Param()
p.engines.engine00 = u.Param()
p.engines.engine00.name = 'DM'
p.engines.engine00.numiter = 100
p.engines.engine00.probe_support = 0.05

# prepare and run
P = Ptycho(p,level=4)
P.run()