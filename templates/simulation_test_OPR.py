# A simulation template to test and demonstrate the "OPR" version of Difference map.
# Note that it is better to use "ptypy.plotclient --layout minimal" otherwise 92 probes will be plotted.

from ptypy import utils as u
from ptypy.core import Ptycho
import numpy as np


p = u.Param()
p.verbose_level = 5
p.data_type = "single"
p.run = 'test_indep_probes'
p.io = u.Param()
p.io.home = "/tmp/ptypy/"
p.io.autosave = u.Param()
p.io.autosave.interval = 20
p.io.autosave.store_OPR_iter = None
p.io.autoplot =u.Param(active=True)
p.io.interaction = u.Param()

p.scans = u.Param()
p.scans.MF = u.Param()
p.scans.MF.name = 'Full'
p.scans.MF.propagation = 'farfield'
p.scans.MF.data = u.Param()
p.scans.MF.data.name = 'MoonFlowerScan'
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
p.scans.MF.data.photons = 100000000.0
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
p.engines.engine00.numiter = 30
p.engines.engine00.numiter_contiguous = 5
p.engines.engine00.overlap_max_iterations = 2
p.engines.engine00.fourier_relax_factor = 0.01

p.engines.engine01 = u.Param()
p.engines.engine01.name = 'DMOPR'
p.engines.engine01.numiter = 100
p.engines.engine01.numiter_contiguous = 5
p.engines.engine01.overlap_max_iterations = 2
p.engines.engine01.fourier_relax_factor = 0.01
p.engines.engine01.IP_metric = 1.
p.engines.engine01.subspace_dim = 10


P = Ptycho(p, level=4)

# Mess up the positions in a predictible way (for MPI)
a = 0.
for pname, pod in P.pods.iteritems():
    pod.ob_view.coord += 3e-7 * np.array([np.sin(a), np.cos(a)])
    a += 4.
P.obj.reformat()

# Run
P.run()
