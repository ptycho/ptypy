"""
This script is a test for ptychographic reconstruction after an
experiment has been carried out and the data is available in ptypy's
data file format in "/tmp/ptypy/sample.ptyd"
"""
import ptypy
from ptypy.core import Ptycho
from ptypy import utils as u

p = u.Param()
p.verbose_level = 3
p.io = u.Param()
p.io.home = "/tmp/ptypy/"

p.scans = u.Param()
p.scans.MF = u.Param()
p.scans.MF.data= u.Param()
p.scans.MF.name = 'Vanilla'
p.scans.MF.data.name = 'PtydScan'
p.scans.MF.data.source = '/tmp/ptypy/sample.ptyd'#'file'
p.scans.MF.data.dfile = 'out.ptyd'

p.engines = u.Param()
p.engines.engine00 = u.Param()
p.engines.engine00.name = 'DM'
p.engines.engine00.numiter = 30
p.engines.engine01 = u.Param()
p.engines.engine01.name = 'ML'
p.engines.engine01.numiter = 20

P = Ptycho(p,level=5)
