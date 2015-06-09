
import ptypy
from ptypy.core import Ptycho
from ptypy import utils as u

p = u.Param()
p.verbose_level = 3
p.data_type = "single"

p.scans = u.Param()
p.scans.MF = u.Param()
p.scans.MF.data= u.Param()
p.scans.MF.data.source = 'sample.ptyd'#'file'
p.scans.MF.data.dfile = None#'sample.ptyd'

p.engines = u.Param()              
p.engines.engine00 = u.Param()
p.engines.engine00.name = 'DM'
p.engines.engine00.numiter = 30
p.engines.engine01 = u.Param()
p.engines.engine01.name = 'ML'
p.engines.engine01.numiter = 20

P = Ptycho(p,level=5)
