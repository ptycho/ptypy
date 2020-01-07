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
p.frames_per_block = 100
# set home path
p.io = u.Param()
p.io.home = "/dls/tmp/clb02321/dumps/ptypy"
p.io.autosave = u.Param(active=True)
p.io.autoplot = u.Param(active=False)
# max 200 frames (128x128px) of diffraction data
p.scans = u.Param()
p.scans.i13 = u.Param()
# now you have to specify which ScanModel to use with scans.XX.name,
# just as you have to give 'name' for engines and PtyScan subclasses.
p.scans.i13.name = 'BlockFull' # or 'Full'
p.scans.i13.data= u.Param()
p.scans.i13.data.name = 'MoonFlowerScan'
p.scans.i13.data.shape = 512
p.scans.i13.data.num_frames = 10000
p.scans.i13.data.save = None

p.scans.i13.illumination = u.Param()
p.scans.i13.coherence = u.Param(num_probe_modes=1)
p.scans.i13.illumination.diversity = u.Param()
p.scans.i13.illumination.diversity.noise = (0.5, 1.0)
p.scans.i13.illumination.diversity.power = 0.1

# position distance in fraction of illumination frame
p.scans.i13.data.density = 0.2
# total number of photon in empty beam
p.scans.i13.data.photons = 1e8
# Gaussian FWHM of possible detector blurring
p.scans.i13.data.psf = 0.2

# attach a reconstrucion engine
p.engines = u.Param()
p.engines.engine00 = u.Param()
p.engines.engine00.name = 'DM_pycuda_stream'
p.engines.engine00.numiter = 1000
p.engines.engine00.numiter_contiguous = 20
p.engines.engine00.probe_update_start = 1

# prepare and run
P = Ptycho(p,level=5)
#P.run()
P.print_stats()
#u.pause(10)
