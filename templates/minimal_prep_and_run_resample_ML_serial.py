"""
This script is a test for ptychographic reconstruction in the absence
of actual data. It uses the test Scan class
`ptypy.core.data.MoonFlowerScan` to provide "data".
"""
#import ptypy
from ptypy.core import Ptycho
from ptypy import utils as u
from ptypy.accelerate.base.engines import ML_serial

p = u.Param()

# for verbose output
p.verbose_level = 4
p.frames_per_block = 100

# set home path
p.io = u.Param()
p.io.home = "/tmp/ptypy/"
p.io.autosave = u.Param(active=False)
#p.io.autoplot = u.Param()
#p.io.autoplot.dump = True
#p.io.autoplot = False

# max 100 frames (128x128px) of diffraction data
p.scans = u.Param()
p.scans.MF = u.Param()
p.scans.MF.name = 'Full'
p.scans.MF.data= u.Param()
p.scans.MF.data.name = 'MoonFlowerScan'
p.scans.MF.data.shape = 128
p.scans.MF.data.num_frames = 100
p.scans.MF.data.save = None

# position distance in fraction of illumination frame
p.scans.MF.data.density = 0.2
# total number of photon in empty beam
p.scans.MF.data.photons = 1e8
# Gaussian FWHM of possible detector blurring
p.scans.MF.data.psf = 0.

# Resample by a factor of 2
p.scans.MF.resample = 2

p.engines = u.Param()
p.engines.engine01 = u.Param()
p.engines.engine01.name = 'ML_serial'
p.engines.engine01.reg_del2 = True                  # Whether to use a Gaussian prior (smoothing) regularizer
p.engines.engine01.reg_del2_amplitude = 1.             # Amplitude of the Gaussian prior if used
p.engines.engine01.scale_precond = True
p.engines.engine01.scale_probe_object = 1.
p.engines.engine01.smooth_gradient = 20.
p.engines.engine01.smooth_gradient_decay = 1/50.
p.engines.engine01.floating_intensities = True
p.engines.engine01.numiter = 300

# prepare and run
P = Ptycho(p,level=4)
P.run()
P.finalize()

