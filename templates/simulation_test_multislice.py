# A simulation template to test and demonstrate the "OPR" version of Difference map.
# Note that it is better to use "ptypy.plotclient --layout minimal" otherwise 92 probes will be plotted.

from ptypy import utils as u
from ptypy.core import Ptycho
import numpy as np


p = u.Param()
p.verbose_level = 3
p.data_type = "single"
p.run = 'test_multislice'
p.io = u.Param()
p.io.home = "/tmp/ptypy/"
p.io.autosave = u.Param()
p.io.autosave.interval = 20
p.io.autoplot =u.Param(active=True)
p.io.interaction = u.Param()

p.scans = u.Param()
p.scans.MF = u.Param()
p.scans.MF.name = 'Full'
p.scans.MF.coherence = u.Param(num_object_modes=2)
p.scans.MF.propagation = 'farfield'

# this next bit is generating simulated data
p.scans.MF.data = u.Param()
p.scans.MF.data.name = 'MoonFlowerMultisliceScan'
p.scans.MF.data.number_object_slices = 2
p.scans.MF.data.slice_separation = 0.1e-6
p.scans.MF.data.positions_theory = None
p.scans.MF.data.auto_center = None
p.scans.MF.data.min_frames = 1
p.scans.MF.data.orientation = None
p.scans.MF.data.num_frames = 100
p.scans.MF.data.energy = 6.2
p.scans.MF.data.shape = 256
p.scans.MF.data.chunk_format = '.chunk%02d'
p.scans.MF.data.rebin = None
p.scans.MF.data.experimentID = None
p.scans.MF.data.label = None
p.scans.MF.data.version = 0.1
p.scans.MF.data.dfile = None
p.scans.MF.data.psize = 0.000172
p.scans.MF.data.load_parallel = None
p.scans.MF.data.distance = 7.0
p.scans.MF.data.save = None
p.scans.MF.data.center = 'fftshift'
p.scans.MF.data.photons = 1e9
p.scans.MF.data.psf = 0.0
p.scans.MF.data.density = 0.2


p.scans.MF.illumination = u.Param()
p.scans.MF.illumination.model = None
p.scans.MF.illumination.aperture = u.Param()
p.scans.MF.illumination.aperture.diffuser = None
p.scans.MF.illumination.aperture.form = "circ"
p.scans.MF.illumination.aperture.size = 3e-6
p.scans.MF.illumination.aperture.edge = 10


p.engines = u.Param()
p.engines.engine00 = u.Param()
p.engines.engine00.name = 'DM'
p.engines.engine00.numiter = 150
p.engines.engine00.numiter_contiguous = 5
p.engines.engine00.overlap_max_iterations = 2
p.engines.engine00.fourier_relax_factor = 0.01
p.engines.engine00.slice_separation = 0.1e-6

P = Ptycho(p, level=5)

