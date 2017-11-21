"""
This template uses a PtyScan class which simulates the numerical data
from Berenguer et al. PRB 88 (2013) 144101, and then reconstructs the
data with PtyPy.
"""

from ptypy.core import Ptycho
from ptypy import utils as u

p = u.Param()

# for verbose output
p.verbose_level = 3

# specify the Bragg 3d plot client
p.io = u.Param()
p.io.autoplot = u.Param()
p.io.autoplot.layout = 'bragg3d'

# illumination for simulation and reconstruction
illumination = u.Param()
illumination.aperture = u.Param()
illumination.aperture.size = 3e-6
illumination.aperture.form = 'circ'

p.scans = u.Param()
p.scans.scan01 = u.Param()
p.scans.scan01.name = 'Bragg3dModel'
p.scans.scan01.illumination = illumination # for reconstruction
p.scans.scan01.data= u.Param()
p.scans.scan01.data.name = 'Bragg3dSimScan'
p.scans.scan01.data.illumination = illumination # for data simulation
p.scans.scan01.data.shape = 256
p.scans.scan01.data.n_rocking_positions = 100
p.scans.scan01.data.psize = 55e-6
p.scans.scan01.data.rocking_step = .01
p.scans.scan01.sample = u.Param()
p.scans.scan01.sample.fill = 1e-3

p.engines = u.Param()
p.engines.engine00 = u.Param()
p.engines.engine00.name = 'DM'
p.engines.engine00.numiter = 100
p.engines.engine00.probe_update_start = 100000
p.engines.engine00.probe_support = None

# prepare and run
P = Ptycho(p,level=5)
