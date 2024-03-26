"""
This script is a test for ptycho-tomographic reconstructions.
"""
from ptypy.core import Ptycho
from ptypy import utils as u
import ptypy.simulations as sim
import ptypy.utils.tomo as tu

from ptypy.custom import DM_ptycho_tomo

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

sim.illumination = u.Param()
sim.illumination.model = None
sim.illumination.photons = int(1e9)
sim.illumination.aperture = u.Param()
sim.illumination.aperture.diffuser = None
sim.illumination.aperture.form = "circ"
sim.illumination.aperture.size = 1.0e-3
sim.illumination.aperture.edge = 2
sim.illumination.aperture.central_stop = None
sim.illumination.propagation = u.Param()
sim.illumination.propagation.focussed = None
sim.illumination.propagation.parallel = 0.03
sim.illumination.propagation.spot_size = None

nangles = 5
pshape = 64
Afwd = tu.forward_projector_matrix_tomo(pshape, nangles)
rmap = tu.refractive_index_map(pshape).ravel()
proj = (Afwd @ rmap).reshape(nangles,pshape,pshape)

sim.sample = u.Param()
#sim.sample.model = proj[0]
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
scan = u.Param()
scan.name = 'BlockFull'

scan.coherence = u.Param()
scan.coherence.num_probe_modes=1

scan.illumination = u.Param()
scan.illumination.model=None
scan.illumination.aperture = u.Param()
scan.illumination.aperture.diffuser = None
scan.illumination.aperture.form = "circ"
scan.illumination.aperture.size = 1.0e-3
scan.illumination.aperture.edge = 10

# Scan data (simulation) parameters
scan.data = u.Param()
scan.data.name = 'SimScan'
#scan.data.update(sim)

# Iterate over nr. of tomographic angles
print('##########################')
p.scans = u.Param()
for i in range(nangles):
    simi = sim.copy(depth=99)
    simi.sample.model = proj[i]
    scani = scan.copy(depth=99)
    scani.data.update(simi)
    setattr(p.scans, f"scan{i}", scani)

# Reconstruction parameters
p.engines = u.Param()
p.engines.engine00 = u.Param()
p.engines.engine00.name = 'DMPtychoTomo'
p.engines.engine00.numiter = 40
p.engines.engine00.fourier_relax_factor = 0.05
p.engines.engine00.probe_center_tol = 1

u.verbose.set_level("info")

if __name__ == "__main__":
    P = Ptycho(p,level=5)
