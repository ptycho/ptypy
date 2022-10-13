"""
This script is a test for ptychographic reconstruction in the absence
of actual data. It uses a simulated Au Siemens star pattern under  
experimental farfield conditions and with a focused beam in the hard X-ray regime.
"""
from ptypy.core import Ptycho
from ptypy import utils as u
import ptypy
ptypy.load_gpu_engines(arch="cuda")

import numpy as np
import tempfile
tmpdir = tempfile.gettempdir()

p = u.Param()

### PTYCHO PARAMETERS
p.verbose_level = "info"

p.data_type = "single"
p.run = None

p.io = u.Param()
p.io.home = "/".join([tmpdir, "ptypy"])
p.io.autosave = u.Param(active=False)
p.io.autoplot = u.Param(active=False, layout="weak")
p.io.interaction = u.Param(active=False)

# Simulation parameters
sim = u.Param()
sim.energy = 17.0
sim.distance = 2.886
sim.psize = 51e-6
sim.shape = 256
sim.xy = u.Param()
sim.xy.model = "round"
sim.xy.spacing = 250e-9
sim.xy.steps = 30
sim.xy.extent = 4e-6

sim.illumination = u.Param()
sim.illumination.model = None
sim.illumination.photons = 3e8
sim.illumination.aperture = u.Param()
sim.illumination.aperture.diffuser = None
sim.illumination.aperture.form = "rect"
sim.illumination.aperture.size = 35e-6
sim.illumination.aperture.central_stop = None
sim.illumination.propagation = u.Param()
sim.illumination.propagation.focussed = 0.08
sim.illumination.propagation.parallel = 0.0014
sim.illumination.propagation.spot_size = None

sim.sample = u.Param()
sim.sample.model = u.xradia_star((1000,1000),minfeature=3,contrast=0.0)
sim.sample.process = u.Param()
sim.sample.process.offset = (100,100)
sim.sample.process.zoom = 1.0
sim.sample.process.formula = "Au"
sim.sample.process.density = 19.3
sim.sample.process.thickness = 2000e-9
sim.sample.process.ref_index = None
sim.sample.process.smoothing = None
sim.sample.fill = 1.0+0.j

#sim.detector = 'FRELON_TAPER'
sim.detector = 'GenericCCD32bit'
sim.verbose_level = 1
sim.psf = 1. # emulates partial coherence
sim.plot = False

# Scan model and initial value parameters
p.scans = u.Param()
p.scans.scan00 = u.Param()
p.scans.scan00.name = 'BlockFull'

p.scans.scan00.coherence = u.Param()
p.scans.scan00.coherence.num_probe_modes = 4
p.scans.scan00.coherence.num_object_modes = 1
p.scans.scan00.coherence.energies = [1.0]

p.scans.scan00.sample = u.Param()
p.scans.scan00.sample.model = 'stxm'
p.scans.scan00.sample.process =  None

# (copy the simulation illumination and change specific things)
p.scans.scan00.illumination = sim.illumination.copy(99)
p.scans.scan00.illumination.aperture.form = 'circ'
p.scans.scan00.illumination.propagation.focussed = 0.06
p.scans.scan00.illumination.diversity = u.Param()
p.scans.scan00.illumination.diversity.power = 0.1
p.scans.scan00.illumination.diversity.noise = (np.pi,3.0)

# Scan data (simulation) parameters
p.scans.scan00.data = u.Param()
p.scans.scan00.data.name = 'SimScan'
p.scans.scan00.data.update(sim)
p.scans.scan00.data.save = None

# Reconstruction parameters
p.engines = u.Param()
p.engines.engine00 = u.Param()
p.engines.engine00.name = 'DM_pycuda'
p.engines.engine00.numiter = 150
p.engines.engine00.fourier_relax_factor = 0.05
p.engines.engine00.numiter_contiguous = 1
p.engines.engine00.probe_support = 0.7
p.engines.engine00.probe_inertia = 0.01
p.engines.engine00.object_inertia = 0.1
p.engines.engine00.clip_object = (0, 1.)
p.engines.engine00.alpha = 1
p.engines.engine00.probe_update_start = 2
p.engines.engine00.update_object_first = True
p.engines.engine00.overlap_converge_factor = 0.05
p.engines.engine00.overlap_max_iterations = 100
p.engines.engine00.obj_smooth_std = 5

u.verbose.set_level("info")
if __name__ == "__main__":
    P = Ptycho(p,level=5)
