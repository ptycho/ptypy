"""
Simulates and then inverts 3d Bragg ptycho data.
"""
from ptypy.core import Ptycho
from ptypy import utils as u
import ptypy
ptypy.load_ptyscan_module("Bragg3dSim")
import tempfile

p = u.Param()
p.run = 'Si110_stripes'

# for verbose output
p.verbose_level = "info"

# use special plot layout for 3d data
p.io = u.Param()
p.io.home = tempfile.mkdtemp('braggtest')
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
p.engines.engine00.name = 'DM_3dBragg'
p.engines.engine00.numiter = 100
p.engines.engine00.probe_update_start = 100000
p.engines.engine00.probe_support = None
p.engines.engine00.sample_support = u.Param()
p.engines.engine00.sample_support.coefficient = 0.0
p.engines.engine00.sample_support.type = 'thinlayer'
p.engines.engine00.sample_support.shrinkwrap = u.Param()
p.engines.engine00.sample_support.shrinkwrap.cutoff = .3
p.engines.engine00.sample_support.shrinkwrap.smooth = None
p.engines.engine00.sample_support.shrinkwrap.start = 15
p.engines.engine00.sample_support.shrinkwrap.plot = True

# prepare and run
if __name__ == "__main__":
    P = Ptycho(p,level=5)
