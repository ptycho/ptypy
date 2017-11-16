from ptypy.core import Ptycho
from ptypy import utils as u

p = u.Param()

# for verbose output
p.verbose_level = 3

# no 2d plotting of 3d data
p.io = u.Param()
p.io.autoplot = u.Param()
p.io.autoplot.layout = 'bragg3d'

p.scans = u.Param()
p.scans.scan01 = u.Param()
p.scans.scan01.name = 'Bragg3dModel'
p.scans.scan01.data= u.Param()
p.scans.scan01.data.name = 'Bragg3dSimScan'
p.scans.scan01.data.theta_bragg = 10.0
p.scans.scan01.sample = u.Param()
p.scans.scan01.sample.fill = 1e-3

p.engines = u.Param()
p.engines.engine00 = u.Param()
p.engines.engine00.name = 'DM'
p.engines.engine00.numiter = 100000
p.engines.engine00.probe_update_start = 100000
p.engines.engine00.probe_support = None

# prepare and run
P = Ptycho(p,level=5)
