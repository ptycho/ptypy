"""
This script is a test for ptycho-tomographic reconstructions.
"""
from ptypy.core import Ptycho
from ptypy import utils as u
import ptypy.simulations as sim
import ptypy.utils.tomo as tu
from ptypy.custom import ML_separate_grads_ptychotomo
import random

import astra
import numpy as np
import tempfile
tmpdir = tempfile.gettempdir()


### PTYCHO PARAMETERS
p = u.Param()
p.verbose_level = "info"
p.data_type = "single"

p.run = None
p.io = u.Param()
p.io.home = "/".join([tmpdir, "ptypy_letizia"])
p.io.autosave = u.Param(active=True)
p.io.autoplot = u.Param(active=False)
p.io.autoplot.layout='minimal'

# Simulation parameters
sim = u.Param()
sim.energy = u.keV2m(1.0)/6.32e-7
sim.distance = 15e-2
sim.psize = 24e-6
sim.shape = 32
sim.xy = u.Param()
sim.xy.model = "round"
sim.xy.spacing = 0.3e-3
sim.xy.steps = 9
sim.xy.extent = (5e-3,5e-3)

pshape = 56
n_angles = 19
n_frames = 76

angles = np.linspace(0, np.pi, n_angles, endpoint=True)
pgeom = astra.create_proj_geom("parallel3d", 1.0, 1.0, pshape, pshape, angles)
vgeom = astra.create_vol_geom(pshape, pshape, pshape)
rmap = tu.refractive_index_map(pshape)
proj_real_id, proj_real = astra.create_sino3d_gpu(rmap.real, pgeom, vgeom)
proj_imag_id, proj_imag = astra.create_sino3d_gpu(rmap.imag, pgeom, vgeom)
proj = np.moveaxis(proj_real + 1j * proj_imag, 1,0)

sim.extra = u.Param()
repeated_angles = []
for angle in angles:
    repeated_angles += n_frames*[angle]
sim.extra.vals = np.array(repeated_angles)
sim.extra.ind = np.arange(len(repeated_angles)).astype(int)

sim.illumination = u.Param()
sim.illumination.model = None
sim.illumination.photons = int(1e9)
sim.illumination.aperture = u.Param()
sim.illumination.aperture.diffuser = None
sim.illumination.aperture.form = "circ"
sim.illumination.aperture.size = 1.0e-3
sim.illumination.aperture.edge = 10
sim.illumination.aperture.central_stop = None
sim.illumination.propagation = u.Param()
sim.illumination.propagation.focussed = None
sim.illumination.propagation.parallel = 0.13
sim.illumination.propagation.spot_size = None


sim.projections = [np.exp(1j * slice).reshape((1, 56, 56)) for slice in proj]  
sim.tomo_angles = n_angles
sim.n_frames_per_angle = n_frames


sim.sample = u.Param()
sim.sample.process = u.Param()
sim.sample.process.offset = (0,0)
sim.sample.process.formula = None
sim.sample.process.density = None
sim.sample.process.thickness = None
sim.sample.process.ref_index = None
sim.sample.process.smoothing = None
sim.sample.fill = 1.0+0.j
sim.plot=False
sim.detector = u.Param(dtype=np.uint32,full_well=2**32-1,psf=None)

# Scan model
p.scans = u.Param()
p.scans.sim = u.Param() 
p.scans.sim.name = 'BlockFull3D' 

p.scans.sim.coherence = u.Param()
p.scans.sim.coherence.num_probe_modes=1
p.scans.sim.n_frames_per_angle = n_frames

p.scans.sim.illumination = u.Param()
p.scans.sim.illumination.model=None
p.scans.sim.illumination.aperture = u.Param()
p.scans.sim.illumination.aperture.diffuser = None
p.scans.sim.illumination.aperture.form = "circ"
p.scans.sim.illumination.aperture.size = 1.0e-3
p.scans.sim.illumination.aperture.edge = 15
p.scans.sim.illumination.propagation = u.Param()
p.scans.sim.illumination.propagation.focussed = None
p.scans.sim.illumination.propagation.parallel = 0.03
p.scans.sim.illumination.propagation.spot_size = None

p.scans.sim.sample = u.Param()
p.scans.sim.sample.diversity = u.Param()
p.scans.sim.sample.diversity.power = 1

# Scan data (simulation) parameters
p.scans.sim.data = u.Param()
p.scans.sim.data.name = 'SimScan3D'
p.scans.sim.data.update(sim)

# Reconstruction parameters
p.engines = u.Param()
p.engines.engine = u.Param()
p.engines.engine.name = 'MLPtychoTomo'
p.engines.engine.n_angles = n_angles
p.engines.engine.init_vol_zero = True
p.engines.engine.numiter = 200
p.engines.engine.numiter_contiguous = 10
p.engines.engine.probe_support = None
p.engines.engine.probe_fourier_support = None

u.verbose.set_level("info")

if __name__ == "__main__":
    P = Ptycho(p,level=5)
