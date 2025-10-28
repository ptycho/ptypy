
from ptypy.core import Ptycho
from ptypy import utils as u

import tempfile
tmpdir = tempfile.gettempdir()

p = u.Param()

# for verbose output
p.verbose_level = "info"

# set home path
p.io = u.Param()
p.io.home = "/".join([tmpdir, "ptypy"])
p.io.autosave = None
p.io.interaction = u.Param(active=False)
p.scans = u.Param()
p.scans.MF = u.Param()
p.scans.MF.name = 'BlockFull'
p.scans.MF.data = u.Param()
p.scans.MF.data.name = 'QuickScan'
p.scans.MF.data.shape = 32
p.scans.MF.data.num_frames = 200000
p.scans.MF.data.save = None

# position distance in fraction of illumination frame
p.scans.MF.data.density = 0.02

# attach a reconstrucion engine
p.engines = u.Param()
p.engines.engine00 = u.Param()
p.engines.engine00.name = 'DM'
p.engines.engine00.numiter = 80

# prepare and run
P = Ptycho(p,level=1)
for scan in P.model.scans.values():
    scan.max_frames_per_block=1000
P.model.new_data()
P.model.new_data()
P.model.new_data()
u.verbose.set_level("info")
if u.parallel.master:
    P.print_stats()

