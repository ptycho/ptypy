"""
This script is a test for ptychographic reconstruction after an
experiment has been carried out and the data is available in ptypy's
data file format in the current directory as "sample.ptyd". Use together
with `ptypy_make_sample_ptyd.py`.
"""
from ptypy.core import Ptycho
from ptypy import utils as u

import tempfile
tmpdir = tempfile.gettempdir()

p = u.Param()
p.verbose_level = "info"
p.io = u.Param()
p.io.home = "/".join([tmpdir, "ptypy"])

p.scans = u.Param()
p.scans.MF = u.Param()
p.scans.MF.data= u.Param()
p.scans.MF.name = 'BlockVanilla'
p.scans.MF.data.name = 'PtydScan'
p.scans.MF.data.source = 'file'
p.scans.MF.data.dfile = 'sample.ptyd'

p.engines = u.Param()
p.engines.engine00 = u.Param()
p.engines.engine00.name = 'DM'
p.engines.engine00.numiter = 80
p.engines.engine01 = u.Param()
p.engines.engine01.name = 'ML'
p.engines.engine01.numiter = 20

if __name__ == "__main__":
    P = Ptycho(p,level=5)
