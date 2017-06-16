import ptypy
from ptypy.core import Ptycho
from ptypy import utils as u
import ptypy.simulations as sim
import numpy as np
p = u.Param()


### PTYCHO PARAMETERS
p.verbose_level = 3
p.data_type = "single"

p.run = None
p.io = u.Param()
p.io.home = "/tmp/ptypy/"
p.io.run = None
p.io.autosave = None
p.io.autoplot = u.Param()
p.io.autoplot.layout='minimal'

p.scan = u.Param()
p.scan.geometry = u.Param()
p.scan.geometry.energy = u.keV2m(1.0)/6.32e-7
p.scan.geometry.lam = None
p.scan.geometry.distance = 15e-2
p.scan.geometry.psize = 24e-6
p.scan.geometry.shape = 256
p.scan.geometry.propagation = "farfield"


sim = u.Param()
sim.xy = u.Param()
sim.xy.model = "round"
sim.xy.spacing = 0.3e-3
sim.xy.steps = 60
sim.xy.extent = (5e-3,10e-3)

sim.illumination = u.Param()
sim.illumination.model = None
sim.illumination.photons = 1e9
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

sim.sample = u.Param()
sim.sample.model = -u.rgb2complex(u.imload('../resources/ptypy_logo_1M.png')[::-1,:,:-1])
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
sim.detector = dict(dtype=np.uint32,full_well=2**32-1,psf=None)

p.scans = u.Param()
p.scans.ptypy = u.Param()
p.scans.ptypy.data = u.Param()
p.scans.ptypy.data.source = 'sim'
p.scans.ptypy.data.recipe = sim.copy(depth=4)

p.scans.ptypy.coherence = u.Param()
p.scans.ptypy.coherence.num_probe_modes=1

p.scans.ptypy.illumination = u.Param()
p.scans.ptypy.illumination.model=None
p.scans.ptypy.illumination.aperture = u.Param()
p.scans.ptypy.illumination.aperture.diffuser = None
p.scans.ptypy.illumination.aperture.form = "circ"
p.scans.ptypy.illumination.aperture.size = 1.0e-3
p.scans.ptypy.illumination.aperture.edge = 10

p.engine = u.Param()
p.engine.common = u.Param()
p.engine.common.numiter = 100
p.engine.common.numiter_contiguous = 1
p.engine.common.probe_support = 0.9
p.engine.common.probe_inertia = 0.01
p.engine.common.object_inertia = 0.1
p.engine.common.clip_object = None

p.engine.DM = u.Param()
p.engine.DM.name = "DM"
p.engine.DM.alpha = 1
p.engine.DM.probe_update_start = 2
p.engine.DM.update_object_first = True
p.engine.DM.overlap_converge_factor = 0.05
p.engine.DM.overlap_max_iterations = 100
p.engine.DM.fourier_relax_factor = 0.05
p.engine.DM.obj_smooth_std = 5

p.engine.ML = u.Param()

p.engines = u.Param()
p.engines.engine00 = u.Param()
p.engines.engine00.name = 'DM'
p.engines.engine00.numiter = 40
p.engines.engine00.fourier_relax_factor = 0.05


u.verbose.set_level(3)
P = Ptycho(p,level=5)



