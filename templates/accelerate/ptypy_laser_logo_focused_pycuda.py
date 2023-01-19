"""
This script is a test for ptychographic reconstruction in the absence
of actual data. It uses a simulated Au Siemens star pattern under  
experimental farfield conditions and with a focused optical laser beam.
"""
from ptypy.core import Ptycho
from ptypy import utils as u
import ptypy.simulations as sim
import ptypy
ptypy.load_gpu_engines(arch="cuda")

import pathlib
import numpy as np
import tempfile
tmpdir = tempfile.gettempdir()

### PTYCHO PARAMETERS
p = u.Param()
p.verbose_level = "info"
p.data_type = "single"

p.run = None
p.io = u.Param()
p.io.home = "/".join([tmpdir, "ptypy"])
p.io.autosave = u.Param(active=False)
p.io.autoplot = u.Param(active=False, layout="minimal")
p.io.interaction = u.Param(active=False)

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

imgfile = "/".join([str(pathlib.Path(__file__).parent.resolve()), '../../resources/ptypy_logo_1M.png'])
sim.sample = u.Param()
sim.sample.model = -u.rgb2complex(u.imload(imgfile)[::-1,:,:-1])
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
p.scans.ptypy.name = 'BlockFull'

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
p.engines.engine00.name = 'DM_pycuda'
p.engines.engine00.numiter = 40
p.engines.engine00.fourier_relax_factor = 0.05

u.verbose.set_level("info")
if __name__ == "__main__":
    P = Ptycho(p,level=5)
