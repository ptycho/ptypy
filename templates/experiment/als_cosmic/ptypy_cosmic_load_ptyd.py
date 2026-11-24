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
p.scans.MF.name = 'BlockFull'
p.scans.MF.data.name = 'PtydScan'
p.scans.MF.data.source = 'file'
p.scans.MF.data.dfile = 'test.ptyd'

p.scans.MF.illumination = u.Param()
p.scans.MF.illumination.aperture = u.Param()
p.scans.MF.illumination.aperture.diffuser = None
p.scans.MF.illumination.aperture.form = "circ"
p.scans.MF.illumination.aperture.size = 0.0001
p.scans.MF.illumination.aperture.central_stop = .2
p.scans.MF.illumination.propagation = u.Param()
p.scans.MF.illumination.propagation.focussed = 0.005
p.scans.MF.illumination.propagation.parallel = 0.00005
p.scans.MF.illumination.propagation.spot_size = None

p.engines = u.Param()
p.engines.engine00 = u.Param()
p.engines.engine00.name = 'DM'
p.engines.engine00.numiter = 200
p.engines.engine01 = u.Param()
p.engines.engine01.name = 'ML'
p.engines.engine01.numiter = 50

if __name__ == "__main__":
    P = Ptycho(p,level=5)
