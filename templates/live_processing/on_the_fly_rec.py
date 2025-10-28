"""
This script is a test for ptychographic reconstruction after an
experiment has been carried out and the data is available in ptypy's
data file format in "/tmp/ptypy/sample.ptyd"

This file is intended to be used in conjuction with
"on_the_fly_ptyd.py" although the script is absolutely generic.

Tested to work with mpi
"""
import ptypy
from ptypy.core import Ptycho
from ptypy import utils as u

p = u.Param()
p.verbose_level = 3
p.io = u.Param()
p.io.home = "/tmp/ptypy/"
p.io.autosave = None
p.io.autoplot = u.Param()
p.io.autoplot.layout = 'minimal'

p.scans = u.Param()
p.scans.MF = u.Param()
p.scans.MF.name = 'Full'
p.scans.MF.data= u.Param()
p.scans.MF.data.name = 'PtydScan'
p.scans.MF.data.dfile ='/tmp/ptypy/sample.ptyd'
p.scans.MF.data.min_frames = 10

p.engines = u.Param()
p.engines.engine00 = u.Param()
p.engines.engine00.name = 'DM'
p.engines.engine00.numiter = 100
p.engines.engine00.numiter_contiguous = 2
p.engines.engine00.probe_support = None
p.engines.engine00.probe_update_start = 2
p.engines.engine00.probe_inertia = 0.01
p.engines.engine00.object_inertia = 0.1
p.engines.engine00.update_object_first = True
p.engines.engine00.overlap_converge_factor = 0.5
p.engines.engine00.overlap_max_iterations = 100
p.engines.engine00.fourier_relax_factor = 0.05

if __name__ == "__main__":
    P = Ptycho(p,level=5)
