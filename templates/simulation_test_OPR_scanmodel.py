# A simulation template to test and demonstrate the "OPR" version of Difference map.
# Note that it is better to use "ptypy.plotclient --layout minimal" otherwise 92 probes will be plotted.

from ptypy import utils as u
from ptypy.core import Ptycho
import numpy as np


p = u.Param()
p.verbose_level = 4
p.data_type = "single"
p.run = 'test_indep_probes'
p.io = u.Param()
p.io.home = "~/tmp/ptypy/"
p.io.autosave = u.Param()
p.io.autosave.interval = 20
p.io.autoplot = u.Param()
p.io.autoplot.active = False
p.io.interaction = u.Param()
p.io.interaction.active = False

p.scans = u.Param()
p.scans.MF = u.Param()
p.scans.MF.name = 'OPRModel'
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

p.engines = u.Param()
p.engines.engine01 = u.Param()
p.engines.engine01.name = 'DMOPR'
p.engines.engine01.numiter = 500
p.engines.engine01.numiter_contiguous = 5
p.engines.engine01.overlap_max_iterations = 2
p.engines.engine01.fourier_relax_factor = 0.01
p.engines.engine01.probe_support = None
p.engines.engine01.subspace_dim = 10
p.engines.engine01.subspace_start = 30
p.engines.engine01.IP_metric = 1

# p.engines.engine02 = u.Param()
# p.engines.engine02.name = 'MLOPR'
# p.engines.engine02.numiter = 100
# p.engines.engine02.numiter_contiguous = 5
# p.engines.engine02.reg_del2 = True
# p.engines.engine02.reg_del2_amplitude = 1.0
# p.engines.engine02.scale_precond = True
# p.engines.engine02.subspace_dim = 10

P = Ptycho(p, level=4)

# Mess up the positions in a predictible way (for MPI)
a = 0.
for pname, pod in P.pods.iteritems():
    pod.ob_view.coord += 3e-7 * np.array([np.sin(a), np.cos(a)])
    #pod.diff *= np.random.uniform(0.1,1)
    a += 4.
P.obj.reformat()

# Run
P.run()
