"""
This script is a test for ptychographic reconstruction in the absence
of actual data. It uses the test Scan class
`ptypy.core.data.MoonFlowerScan` to provide "data".
"""
from ptypy.core import Ptycho
from ptypy import utils as u
import ptypy
ptypy.load_gpu_engines("cuda")
import time

import os
import getpass
from pathlib import Path
username = getpass.getuser()
tmpdir = os.path.join('/dls/tmp', username, 'dumps', 'ptypy')
Path(tmpdir).mkdir(parents=True, exist_ok=True)

p = u.Param()

# for verbose output
p.verbose_level = "info"
p.frames_per_block = 500
# set home path
p.io = u.Param()
p.io.home = tmpdir
p.io.autosave = u.Param(active=False)
p.io.autoplot = u.Param(active=False)
p.io.interaction = u.Param()
p.io.interaction.server = u.Param(active=False)
p.io.benchmark = "all"

# max 200 frames (128x128px) of diffraction data
p.scans = u.Param()
p.scans.I08 = u.Param()
# now you have to specify which ScanModel to use with scans.XX.name,
# just as you have to give 'name' for engines and PtyScan subclasses.
p.scans.I08.name = 'BlockFull'
p.scans.I08.data= u.Param()
p.scans.I08.data.name = 'MoonFlowerScan'
p.scans.I08.data.shape = 128
p.scans.I08.data.num_frames = 10000 # real is 50000
p.scans.I08.data.save = None

p.scans.I08.illumination = u.Param()
p.scans.I08.coherence = u.Param(num_probe_modes=20)
p.scans.I08.illumination.diversity = u.Param()
p.scans.I08.illumination.diversity.noise = (0.5, 1.0)
p.scans.I08.illumination.diversity.power = 0.1

# position distance in fraction of illumination frame
p.scans.I08.data.density = 0.05
# total number of photon in empty beam
p.scans.I08.data.photons = 1e8
# Gaussian FWHM of possible detector blurring
p.scans.I08.data.psf = 1.5

# attach a reconstrucion engine
p.engines = u.Param()
p.engines.engine00 = u.Param()
p.engines.engine00.name = 'DM_pycuda'
p.engines.engine00.numiter = 1000
p.engines.engine00.numiter_contiguous = 20
p.engines.engine00.probe_update_start = 1
p.engines.engine00.probe_update_cuda_atomics = False
p.engines.engine00.object_update_cuda_atomics = True

# prepare and run
P = Ptycho(p,level=4)
t1 = time.perf_counter()
P.run()
t2 = time.perf_counter()
P.print_stats()
print('Elapsed Compute Time: {} seconds'.format(t2-t1))

