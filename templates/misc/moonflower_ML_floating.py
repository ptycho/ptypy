"""
This script is a test for ptychographic reconstruction in the absence
of actual data. It uses the test Scan class
`ptypy.core.data.MoonFlowerScan` to provide "data".
"""
#import ptypy
from ptypy.core import Ptycho
from ptypy import utils as u
import ptypy
ptypy.load_gpu_engines(arch="cuda")
ptypy.load_gpu_engines(arch="serial")

import tempfile
tmpdir = tempfile.gettempdir()

p = u.Param()

# for verbose output
p.verbose_level = "info"

# set home path
p.io = u.Param()
p.io.home = "/".join([tmpdir, "ptypy"])

# saving intermediate results
p.io.autosave = u.Param(active=False)

# opens plotting GUI if interaction set to active)
p.io.autoplot = u.Param(active=False)
p.io.interaction = u.Param(active=False)

# max 100 frames (128x128px) of diffraction data
p.scans = u.Param()
p.scans.MF = u.Param()
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
p.scans.MF.data.psf = 0.

# attach a reconstrucion engine
p.engines = u.Param()
p.engines.engine00 = u.Param()
p.engines.engine00.name = 'ML_pycuda'
p.engines.engine00.ML_type = 'Gaussian'
p.engines.engine00.reg_del2 = True                      # Whether to use a Gaussian prior (smoothing) regularizer
p.engines.engine00.reg_del2_amplitude = 1.             # Amplitude of the Gaussian prior if used
p.engines.engine00.scale_precond = True
#p.engines.engine00.scale_probe_object = 1.
p.engines.engine00.smooth_gradient = 20.
p.engines.engine00.smooth_gradient_decay = 1/50.
p.engines.engine00.floating_intensities = True
p.engines.engine00.floating_intensities_high_pass = True
p.engines.engine00.numiter = 300

# prepare and run
P = Ptycho(p,level=5)
