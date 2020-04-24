"""
This script is a test for ptychographic reconstruction in the absence
of actual data. It uses the test Scan class
`ptypy.core.data.MoonFlowerScan` to provide "data".
"""

from ptypy.core import Ptycho
from ptypy import utils as u
import time

import os
import getpass
from pathlib import Path
username = getpass.getuser()
tmpdir = os.path.join('/dls/tmp', username, 'dumps', 'ptypy')
Path(tmpdir).mkdir(parents=True, exist_ok=True)

p = u.Param()

# for verbose output
p.verbose_level = 3
p.frames_per_block = 73
# set home path
p.io = u.Param()
p.io.home = tmpdir
p.io.autosave = u.Param(active=True, interval=50000)
p.io.autoplot = u.Param(active=False)
p.io.interaction = u.Param()
p.io.interaction.server = u.Param(active=False)

# max 200 frames (128x128px) of diffraction data
p.scans = u.Param()
p.scans.insane = u.Param()
# now you have to specify which ScanModel to use with scans.XX.name,
# just as you have to give 'name' for engines and PtyScan subclasses.
p.scans.insane.name = 'BlockFull' # or 'Full'
p.scans.insane.data= u.Param()
p.scans.insane.data.name = 'MoonFlowerScan'
p.scans.insane.data.shape = 210
p.scans.insane.data.num_frames = 531 # real is 50000
p.scans.insane.data.save = None
p.scans.insane.data.block_wait_count = 1

p.scans.insane.illumination = u.Param()
p.scans.insane.coherence = u.Param(num_probe_modes=3, num_object_modes=2)
p.scans.insane.illumination.diversity = u.Param()
p.scans.insane.illumination.diversity.noise = (0.5, 1.0)
p.scans.insane.illumination.diversity.power = 0.1

# position distance in fraction of illumination frame
p.scans.insane.data.density = 0.05
# total number of photon in empty beam
p.scans.insane.data.photons = 1e8
# Gaussian FWHM of possible detector blurring
p.scans.insane.data.psf = 0.0

# attach a reconstrucion engine
p.engines = u.Param()
p.engines.engine00 = u.Param()
p.engines.engine00.name = 'DM_pycuda_stream'
p.engines.engine00.numiter = 200
p.engines.engine00.numiter_contiguous = 10
p.engines.engine00.probe_update_start = 1
#p.engines.engine00.probe_update_cuda_atomics = False
#p.engines.engine00.object_update_cuda_atomics = True

# prepare and run
P = Ptycho(p,level=4)
t1 = time.perf_counter()
P.run()
t2 = time.perf_counter()
P.print_stats()
P.finalize()
print('Elapsed Compute Time: {} seconds'.format(t2-t1))

