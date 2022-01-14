"""
This script is a test for ptychographic reconstruction in the absence
of actual data. It uses the test Scan class
`ptypy.core.data.MoonFlowerScan` to provide "data".
"""

from ptypy.core import Ptycho
from ptypy import utils as u
from ptypy.accelerate.cuda_pycuda.engines import SDR_pycuda
p = u.Param()

# for verbose output
p.verbose_level = 3

# Frames per block
p.frames_per_block = 200

# set home path
p.io = u.Param()
p.io.home = "/tmp/ptypy/"
p.io.autosave = u.Param(active=False)
p.io.interaction = u.Param(active=False)
p.io.interaction.client = u.Param()
p.io.interaction.client.poll_timeout = 1

# max 200 frames (128x128px) of diffraction data
p.scans = u.Param()
p.scans.MF = u.Param()
# now you have to specify which ScanModel to use with scans.XX.name,
# just as you have to give 'name' for engines and PtyScan subclasses.
p.scans.MF.name = 'BlockFull'
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
p.scans.MF.data.psf = 0.0
p.scans.MF.coherence = u.Param()
p.scans.MF.coherence.num_probe_modes = 3

# attach a reconstrucion engine
p.engines = u.Param()
p.engines.engine00 = u.Param()
p.engines.engine00.name = 'SDR_pycuda'
p.engines.engine00.numiter = 100
p.engines.engine00.alpha = 0 # alpha=0, tau=1 behaves like ePIE
p.engines.engine00.tau = 1

# prepare and run
P = Ptycho(p,level=5)