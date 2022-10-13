"""
This script is a test for ptychographic reconstruction in the absence
of actual data. It uses a simulated Au Siemens star pattern under  
experimental farfield conditions in the hard X-ray regime.
"""
from ptypy.core import Ptycho
from ptypy import utils as u
import ptypy
ptypy.load_gpu_engines(arch="cuda")

import tempfile
tmpdir = tempfile.gettempdir()

### PTYCHO PARAMETERS
p = u.Param()
p.verbose_level = "info"
p.run = None

p.data_type = "single"
p.run = None
p.io = u.Param()
p.io.home = "/".join([tmpdir, "ptypy"])
p.io.autosave = u.Param(active=False)
p.io.autoplot = u.Param(active=False, layout="nearfield")
p.io.interaction = u.Param(active=False)

# Simulation parameters
sim = u.Param()
sim.energy = 9.7
sim.distance = 8.46e-2
sim.psize = 100e-9
sim.shape = 1024
sim.xy = u.Param()
sim.xy.override = u.parallel.MPIrand_uniform(0.0,10e-6,(20,2))
#sim.xy.positions = np.random.normal(0.0,3e-6,(20,2))
sim.verbose_level = 1

sim.illumination = u.Param()
sim.illumination.model = None
sim.illumination.photons = 1e11
sim.illumination.aperture = u.Param()
sim.illumination.aperture.diffuser = (8.0, 10.0)
sim.illumination.aperture.form = "circ"
sim.illumination.aperture.size = 90e-6
sim.illumination.aperture.central_stop = 0.15
sim.illumination.propagation = u.Param()
sim.illumination.propagation.focussed = None#0.08
sim.illumination.propagation.parallel = 0.005
sim.illumination.propagation.spot_size = None

sim.sample = u.Param()
sim.sample.model = u.xradia_star((1200,1200),minfeature=3,contrast=0.8)
sim.sample.process = u.Param()
sim.sample.process.offset = (0,0)
sim.sample.process.zoom = 1.0
sim.sample.process.formula = "Au"
sim.sample.process.density = 19.3
sim.sample.process.thickness = 700e-9
sim.sample.process.ref_index = None
sim.sample.process.smoothing = None
sim.sample.fill = 1.0+0.j

sim.detector = 'GenericCCD32bit'
sim.plot = False

# Scan model and initial value parameters
p.scans = u.Param()
p.scans.scan00 = u.Param()
p.scans.scan00.name = 'BlockFull'

p.scans.scan00.coherence = u.Param()
p.scans.scan00.coherence.num_probe_modes = 1
p.scans.scan00.coherence.num_object_modes = 1
p.scans.scan00.coherence.energies = [1.0]

p.scans.scan00.sample = u.Param()

# (copy simulation illumination and modify some things)
p.scans.scan00.illumination = sim.illumination.copy(99)
p.scans.scan00.illumination.aperture.size = 105e-6
p.scans.scan00.illumination.aperture.central_stop = None

# Scan data (simulation) parameters
p.scans.scan00.data=u.Param()
p.scans.scan00.data.name = 'SimScan'
p.scans.scan00.data.propagation = 'nearfield'
p.scans.scan00.data.save = None #'append'
p.scans.scan00.data.shape = None
p.scans.scan00.data.num_frames = None
p.scans.scan00.data.update(sim)

# Reconstruction parameters
p.engines = u.Param()
p.engines.engine00 = u.Param()
p.engines.engine00.name = 'DM_pycuda'
p.engines.engine00.numiter = 100
p.engines.engine00.object_inertia = 1.
p.engines.engine00.numiter_contiguous = 1
p.engines.engine00.probe_support = None
p.engines.engine00.probe_inertia = 0.001
p.engines.engine00.obj_smooth_std = 10
p.engines.engine00.clip_object = None
p.engines.engine00.alpha = 1
p.engines.engine00.probe_update_start = 2
p.engines.engine00.update_object_first = True
p.engines.engine00.overlap_converge_factor = 0.5
p.engines.engine00.overlap_max_iterations = 100
p.engines.engine00.fourier_relax_factor = 0.05

p.engines.engine01 = u.Param()
p.engines.engine01.name = 'ML_pycuda'
p.engines.engine01.numiter = 50

if __name__ == "__main__":
    P = Ptycho(p,level=5)

