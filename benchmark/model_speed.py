
from ptypy import utils as u
from ptypy.core import Ptycho, Vanilla

p = u.Param()
p.verbose_level = 3
p.io = u.Param()
p.io.home = "/tmp/ptypy/"
p.scans = u.Param()
p.scans.MF = u.Param()
p.scans.MF.data= u.Param()
p.scans.MF.name = 'Vanilla'
p.scans.MF.data.name = 'QuickScan'
p.scans.MF.data.num_frames = 50000
p.scans.MF.data.shape = 32
Ptycho(p,level=2)
