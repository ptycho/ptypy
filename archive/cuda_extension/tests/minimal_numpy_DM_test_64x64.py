from ptypy.core import Ptycho
from ptypy import utils as u
import cProfile
p = u.Param()
p.verbose_level = 3
p.io = u.Param()
p.io.autosave = u.Param(active=False)
p.io.autoplot = u.Param(active=True)
p.ipython_kernel = False
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
p.scans.MF.data.add_poisson_noise = False
p.scans.MF.coherence = u.Param()
p.scans.MF.coherence.num_probe_modes = 1  # currently breaks when this is =2

p.engines = u.Param()

# attach a reconstrucion engine
p.engines = u.Param()
p.engines.engine00 = u.Param()
p.engines.engine00.name = 'DMNpy'
p.engines.engine00.numiter = 50
# p.engines.engine00.overlap_max_iterations = 1
# prepare and run


P = Ptycho(p, level=4)
P.run()
# cProfile.run('P.run()',
#              '/home/clb02321/Desktop/profiling_thing_with_realspace_error.prof',
#              'tottime')