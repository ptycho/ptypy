
import ptypy
from ptypy.core import Ptycho
from ptypy import utils as u
p = u.Param()
p.verbose_level = 3                              

p.data_type = "single"n

p.paths = u.Param()
p.paths.home = "/tmp/ptypy/"  
p.autosave = None

p.model = u.Param()
p.scans = u.Param()
p.scans.MF = u.Param()
p.scans.MF.data= u.Param()
p.scans.MF.data.source = 'test'
p.scans.MF.data.shape = 128
p.scans.MF.data.num_frames = 100
p.scans.MF.data.save = None
p.scans.MF.data.dfile = 'sample.ptyd'

p.engines = u.Param()                              
p.engines.engine00 = u.Param()
p.engines.engine00.name = 'DM'
p.engines.engine00.numiter = 30
p.engines.engine00.alpha = 1
p.engines.engine00.fourier_relax_factor = 0.05
#p.engines.engine01 = u.Param()
#p.engines.engine01.name = 'ML'
#p.engines.engine01.numiter = 40

P = Ptycho(p,level=5)
