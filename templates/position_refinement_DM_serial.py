"""
This script is a test for ptychographic reconstruction in the absence
of actual data. It uses the test Scan class
`ptypy.core.data.MoonFlowerScan` to provide "data".
"""

import numpy as np
from ptypy.core import Ptycho
from ptypy import utils as u

from ptypy.accelerate.base.engines import DM_serial


p = u.Param()

# for verbose output
p.verbose_level = 3
p.frames_per_block = 100
# set home path
p.io = u.Param()
p.io.home = "~/dumps/ptypy/"
p.io.autosave = u.Param(active=True, interval=500)
p.io.autoplot = u.Param(active=False)#True, interval=100)
p.io.interaction = u.Param(active=False)

# max 200 frames (128x128px) of diffraction data
p.scans = u.Param()
p.scans.MF = u.Param()
# now you have to specify which ScanModel to use with scans.XX.name,
# just as you have to give 'name' for engines and PtyScan subclasses.
p.scans.MF.name = 'BlockFull' # or 'Full'
p.scans.MF.data= u.Param()
p.scans.MF.data.name = 'MoonFlowerScan'
p.scans.MF.data.shape = 128
p.scans.MF.data.num_frames = 200
p.scans.MF.data.save = None

p.scans.MF.illumination = u.Param(diversity=None)
p.scans.MF.coherence = u.Param(num_probe_modes=1)
# p.scans.MF.illumination.diversity=u.Param()
# p.scans.MF.illumination.diversity.power = 0.1
# p.scans.MF.illumination.diversity.noise = (np.pi, 3.0)
# position distance in fraction of illumination frame
p.scans.MF.data.density = 0.2
# total number of photon in empty beam
p.scans.MF.data.photons = 1e8
# Gaussian FWHM of possible detector blurring
p.scans.MF.data.psf = 0.
#p.scans.MF.data.add_poisson_noise = False

# attach a reconstrucion engine
p.engines = u.Param()
p.engines.engine00 = u.Param()
p.engines.engine00.name = 'DM_serial'
p.engines.engine00.probe_support = 1
p.engines.engine00.numiter = 100
p.engines.engine00.numiter_contiguous = 10
p.engines.engine00.position_refinement = u.Param()
p.engines.engine00.position_refinement.start = 50
p.engines.engine00.position_refinement.stop = 950
p.engines.engine00.position_refinement.interval = 10
p.engines.engine00.position_refinement.nshifts = 32
p.engines.engine00.position_refinement.amplitude = 5e-7
p.engines.engine00.position_refinement.max_shift = 1e-6
p.engines.engine00.position_refinement.method = "GridSearch"

# prepare and run
P = Ptycho(p, level=4)

# Mess up the positions
a = 0.

coords = []
coords_start = []
for pname, pod in P.pods.items():

    # Save real position
    coords.append(np.copy(pod.ob_view.coord))
    before = pod.ob_view.coord
    psize = pod.pr_view.psize
    perturbation = psize * ((3e-7 * np.array([np.sin(a), np.cos(a)])) // psize)
    new_coord = before + perturbation # make sure integer number of pixels shift
    pod.ob_view.coord = new_coord
    coords_start.append(np.copy(pod.ob_view.coord))
    #pod.diff *= np.random.uniform(0.1,1)y
    a += 4.

np.savetxt("positions_theory.txt", coords)
np.savetxt("positions_start.txt", coords_start)
P.obj.reformat()# update the object storage

# Run
P.run()
P.finalize()

