"""
This script is a test for ptychographic reconstruction in the absence
of actual data. It uses the test Scan class
`ptypy.core.data.MoonFlowerScan` to provide "data".
"""

from ptypy.core import Ptycho
from ptypy import utils as u
import ptypy
ptypy.load_gpu_engines(arch="cuda")

import tempfile
tmpdir = tempfile.gettempdir()

p = u.Param()

# for verbose output
p.verbose_level = "info"
p.frames_per_block = 400
# set home path
p.io = u.Param()
p.io.home =  "/".join([tmpdir, "ptypy"])
p.io.autosave = u.Param(active=False)
p.io.autoplot = u.Param(active=False)
p.io.interaction = u.Param(active=False)

# max 200 frames (128x128px) of diffraction data
p.scans = u.Param()
p.scans.MF = u.Param()
# now you have to specify which ScanModel to use with scans.XX.name,
# just as you have to give 'name' for engines and PtyScan subclasses.
p.scans.MF.name = 'BlockFull' 
p.scans.MF.data= u.Param()
p.scans.MF.data.name = 'MoonFlowerScan'
p.scans.MF.data.shape = 128
p.scans.MF.data.num_frames = 100
p.scans.MF.data.save = None

p.scans.MF.illumination = u.Param(diversity=None)
p.scans.MF.coherence = u.Param(num_probe_modes=1)
# position distance in fraction of illumination frame
p.scans.MF.data.density = 0.2
# total number of photon in empty beam
p.scans.MF.data.photons = 1e8
# Gaussian FWHM of possible detector blurring
p.scans.MF.data.psf = 0.

# attach a reconstrucion engine
p.engines = u.Param()
p.engines.engine00 = u.Param()
p.engines.engine00.name = 'ML_pycuda'
p.engines.engine00.numiter = 300
p.engines.engine00.numiter_contiguous = 5
p.engines.engine00.reg_del2 = True                      # Whether to use a Gaussian prior (smoothing) regularizer
p.engines.engine00.reg_del2_amplitude = 1.             # Amplitude of the Gaussian prior if used
p.engines.engine00.scale_precond = True
p.engines.engine00.smooth_gradient = 20.
p.engines.engine00.smooth_gradient_decay = 1/50.
p.engines.engine00.floating_intensities = False

p.engines.engine01 = u.Param()
p.engines.engine01.name = 'ML_pycuda'
p.engines.engine01.numiter = 20
p.engines.engine01.numiter_contiguous = 5
p.engines.engine01.reg_del2 = False
p.engines.engine01.reg_del2_amplitude = 1.
p.engines.engine01.floating_intensities = False
p.engines.engine01.probe_support = 0.5

# prepare and run
if __name__ == "__main__":
    P = Ptycho(p,level=5)
