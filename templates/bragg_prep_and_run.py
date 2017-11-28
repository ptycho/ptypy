from ptypy.core import Ptycho
from ptypy import utils as u

p = u.Param()
p.run = 'no_mpi'

# for verbose output
p.verbose_level = 4

# no 2d plotting of 3d data
p.io = u.Param()
#p.io.home = './'
p.io.autoplot = u.Param()
p.io.autoplot.layout = 'bragg3d'
p.io.autoplot.dump = True

# illumination for simulation and reconstruction
illumination = u.Param()
illumination.aperture = u.Param()
illumination.aperture.size = 3e-6
illumination.aperture.form = 'circ'

p.scans = u.Param()
p.scans.scan01 = u.Param()
p.scans.scan01.name = 'Bragg3dModel'
p.scans.scan01.illumination = illumination
p.scans.scan01.data= u.Param()
p.scans.scan01.data.name = 'Bragg3dSimScan'
p.scans.scan01.data.illumination = illumination
p.scans.scan01.sample = u.Param()
p.scans.scan01.sample.fill = 1e-3

p.engines = u.Param()
p.engines.engine00 = u.Param()
p.engines.engine00.name = 'DM'
p.engines.engine00.numiter = 50
p.engines.engine00.probe_update_start = 100000
p.engines.engine00.probe_support = None

# prepare and run
P = Ptycho(p,level=5)
