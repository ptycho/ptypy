"""
Simulates and then inverts 3d Bragg ptycho data.
"""

from ptypy.core import Ptycho
from ptypy import utils as u

p = u.Param()
p.run = '3dbpp_example'

# for verbose output
p.verbose_level = 3

# use special plot layout for 3d data
p.io = u.Param()
p.io.home = '/tmp/ptypy/'
p.io.autoplot = u.Param()
p.io.autoplot.layout = 'bragg3d'
p.io.autoplot.dump = True

# illumination for simulation and reconstruction
illumination = u.Param()
illumination.aperture = u.Param()
illumination.aperture.size = 40e-9
illumination.aperture.form = 'circ'
illumination.propagation = u.Param()
illumination.propagation.parallel = -5e-6

# reconstruction
p.scans = u.Param()
p.scans.scan01 = u.Param()
p.scans.scan01.name = 'Bragg3dProjectionModel'
p.scans.scan01.illumination = illumination
p.scans.scan01.sample = u.Param()
p.scans.scan01.sample.fill = 1e-6
p.scans.scan01.r3_shape = 128//4
p.scans.scan01.r3_spacing = 4*10e-9

# simulation
p.scans.scan01.data= u.Param()
p.scans.scan01.data.shape = 128
p.scans.scan01.data.distance = .5
p.scans.scan01.data.r3_shape = 128
p.scans.scan01.data.r3_spacing = 10e-9
p.scans.scan01.data.name = 'Bragg3dProjectionSimScan'
p.scans.scan01.data.dry_run = False
p.scans.scan01.data.illumination = illumination
p.scans.scan01.data.debug_index = 227

p.engines = u.Param()
p.engines.engine00 = u.Param()
p.engines.engine00.name = 'DM_3dBragg'
p.engines.engine00.alpha = .2
p.engines.engine00.numiter = 1000
p.engines.engine00.probe_update_start = 100000
p.engines.engine00.probe_support = None
p.engines.engine00.sample_support = u.Param()
p.engines.engine00.sample_support.coefficient = 0.0
p.engines.engine00.sample_support.type = 'thinlayer'
p.engines.engine00.sample_support.size = 400e-9
p.engines.engine00.sample_support.shrinkwrap = None
##p.engines.engine00.sample_support.shrinkwrap = u.Param()
#p.engines.engine00.sample_support.shrinkwrap.cutoff = .3
#p.engines.engine00.sample_support.shrinkwrap.smooth = None
#p.engines.engine00.sample_support.shrinkwrap.start = 15
#p.engines.engine00.sample_support.shrinkwrap.plot = True

# prepare and run
P = Ptycho(p,level=5)
