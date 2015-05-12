
from ptypy.core import Ptycho
from ptypy import utils as u

p = u.Param()
### PTYCHO PARAMETERS
p.verbose_level = 3                               # (00) Verbosity level

p.data_type = "single"                            # (01) Reconstruction floatine number precision

p.paths = u.Param()
p.interaction = u.Param()
p.model = u.Param()

p.scans = u.Param()
p.scans.MF = u.Param()
p.scans.MF.data = u.Param()
p.scans.MF.data.shape=512
p.scans.MF.data.dfile = '/tmp/ptypy/test.ptyd'
p.scans.MF.data.save = 'append'
p.scans.MF.data.source = 'test'
p.scans.MF.data.num_frames = 200
p.scans.MF.if_conflict_use_meta = True
p.engines = u.Param()
p.engines.engine00= u.Param()
p.engines.engine00.name = 'DM'
p.engines.engine00.numiter = 200
p.engines.engine00.fourier_relax_factor = 0.05
p.engines.engine01 = u.Param()
p.engines.engine01.name = 'ML'
p.engines.engine01.numiter = 100

P = Ptycho(p,level=3)
P.save_run(kind='minimal')

