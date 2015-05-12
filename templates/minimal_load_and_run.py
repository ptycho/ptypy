
import ptypy
from ptypy.core import Ptycho
from ptypy import utils as u
p = u.Param()

### PTYCHO PARAMETERS
p.verbose_level = 3                               # (00) Verbosity level

p.data_type = "single"                            # (01) Reconstruction floatine number precision

p.paths = u.Param()

p.model = u.Param()
p.scans = u.Param()
p.scans.MF = u.Param()
p.scans.MF.data= u.Param()
p.scans.MF.data.source = 'sample.ptyd'#'file'
p.scans.MF.data.dfile = None#'sample.ptyd'

p.engines = u.Param()                                  # (100) empty structure to be filled with engines
p.engines.engine00 = u.Param()
p.engines.engine00.name = 'DM'
p.engines.engine00.numiter = 100
p.engines.engine00.fourier_relax_factor = 0.05
#p.engines.engine01 = u.Param()
#p.engines.engine01.name = 'ML'
#p.engines.engine01.numiter = 100

P = Ptycho(p,level=5)
#P.save_run(kind='minimal')
print 'done'
