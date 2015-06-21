
import ptypy
from ptypy import utils as u
p = u.Param()

### PTYCHO PARAMETERS
p.verbose_level = 3                               # (00) Verbosity level

p.data_type = "single"                            # (01) Reconstruction floatine number precision
p.interaction = None

p.scan = u.Param()
p.scans = u.Param()
p.scans.MF = u.Param()
p.scans.MF.data= u.Param()
p.scans.MF.data.source = 'test'
p.scans.MF.data.shape = 128
p.scans.MF.data.num_frames = 100
p.scans.MF.data.save = None
p.scans.MF.data.dfile = 'sample.ptyd'

P = ptypy.core.Ptycho(p,level=2)

