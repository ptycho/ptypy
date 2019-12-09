import ptypy
from ptypy.core import Ptycho
from ptypy import utils as u
import ptypy.simulations as sim
import numpy as np


### PTYCHO PARAMETERS
p = u.Param()
p.verbose_level = 3
p.data_type = "single"

p.run = None
p.io = u.Param()
p.io.home = "/tmp/ptypy/"
p.io.autosave = None
p.io.autoplot = u.Param()
p.io.autoplot.layout='minimal'

# Simulation parameters
sim = u.Param()
sim.energy = u.keV2m(1.0)/6.32e-7
sim.distance = 15e-2
sim.psize = 24e-6
sim.shape = 256
sim.xy = u.Param()
sim.xy.model = "round"
sim.xy.spacing = 0.3e-3
sim.xy.steps = 60
sim.xy.extent = (5e-3,10e-3)

sim.illumination = u.Param()
sim.illumination.model = None
sim.illumination.photons = int(1e9)
sim.illumination.aperture = u.Param()
sim.illumination.aperture.diffuser = None#(0.7,3)
sim.illumination.aperture.form = "circ"
sim.illumination.aperture.size = 1.0e-3
sim.illumination.aperture.edge = 2
sim.illumination.aperture.central_stop = None
sim.illumination.propagation = u.Param()
sim.illumination.propagation.focussed = None
sim.illumination.propagation.parallel = 0.03
sim.illumination.propagation.spot_size = None

ptypy_path = ptypy.__file__.strip('ptypy.__init__.py')
sim.sample = u.Param()
sim.sample.model = -u.rgb2complex(u.imload('%s/resources/ptypy_logo_1M.png' % ptypy_path)[::-1,:,:-1])
sim.sample.process = u.Param()
sim.sample.process.offset = (0,0)
sim.sample.process.zoom = 0.5
sim.sample.process.formula = None
sim.sample.process.density = None
sim.sample.process.thickness = None
sim.sample.process.ref_index = None
sim.sample.process.smoothing = None
sim.sample.fill = 1.0+0.j
sim.plot=False
sim.detector = u.Param(dtype=np.uint32,full_well=2**32-1,psf=None)

# Scan model and initial value parameters
p.scans = u.Param()
p.scans.ptypy = u.Param()
p.scans.ptypy.name = 'Full'

p.scans.ptypy.coherence = u.Param()
p.scans.ptypy.coherence.num_probe_modes=1

p.scans.ptypy.illumination = u.Param()
p.scans.ptypy.illumination.model=None
p.scans.ptypy.illumination.aperture = u.Param()
p.scans.ptypy.illumination.aperture.diffuser = None
p.scans.ptypy.illumination.aperture.form = "circ"
p.scans.ptypy.illumination.aperture.size = 1.0e-3
p.scans.ptypy.illumination.aperture.edge = 10

# Scan data (simulation) parameters
p.scans.ptypy.data = u.Param()
p.scans.ptypy.data.name = 'SimScan'
p.scans.ptypy.data.update(sim)

# Reconstruction parameters
p.engines = u.Param()
p.engines.engine00 = u.Param()
p.engines.engine00.name = 'DM'
p.engines.engine00.numiter = 40
p.engines.engine00.fourier_relax_factor = 0.05

u.verbose.set_level(3)
P = Ptycho(p,level=5)
