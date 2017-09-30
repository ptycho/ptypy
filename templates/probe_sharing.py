'''
An example showing how to share a probe across two scans
'''


import sys
import time
import ptypy
from ptypy import utils as u
from ptypy.core import Ptycho
import tempfile
u.verbose.set_level(3)
tmp = tempfile.mkdtemp()+'/%s'

def make_sample(outpath):
    data = u.Param()
    u.verbose.logger.info('Output going in:%s', outpath)
    data.dfile = outpath
    data.shape = 128
    data.num_frames = 100
    data.save = 'append'
    data.label = None
    data.psize = 172e-6
    data.energy = 6.2
    data.center = 64, 64
    data.distance = 7.0
    data.auto_center = None
    data.rebin = None
    data.orientation = None
# create PtyScan instance
    MF = ptypy.core.data.MoonFlowerScan(data)
    MF.initialize()
    for i in range(2): # autoprocess data
        msg = MF.auto(100) # logs the out put of .auto() to terminal prompt
        u.verbose.logger.info(u.verbose.report(msg), extra={'allprocesses':True})

## make some sample data
make_sample(tmp % 'scan1.ptyd')
make_sample(tmp % 'scan2.ptyd')


p = u.Param()
p.verbose_level = 3
p.io = u.Param()
p.io.home = "/tmp/ptypy/"
p.autosave = None

p.scans = u.Param()
p.scans.MF_01 = u.Param()
p.scans.MF_01.data= u.Param()
p.scans.MF_01.data.label = 'MF_01'
p.scans.MF_01.data.source ='file'
p.scans.MF_01.data.dfile = tmp % 'scan1.ptyd'

p.scans.MF_02 = u.Param()
p.scans.MF_02.data= u.Param()
p.scans.MF_02.data.label = 'MF_02'
p.scans.MF_02.data.source = 'file'
p.scans.MF_02.data.dfile = tmp % 'scan2.ptyd'
p.scans.MF_02.sharing = u.Param()
p.scans.MF_02.sharing.probe_share_with = 'MF_01'

p.engine = u.Param()
p.engine.common = u.Param()
p.engine.common.numiter = 100
p.engine.common.numiter_contiguous = 1
p.engine.common.probe_support = None
p.engine.common.probe_update_start = 2
p.engine.common.clip_object = None   # [0,1]

p.engine.DM = u.Param()
p.engine.DM.name = "DM"
p.engine.DM.alpha = 1
p.engine.DM.probe_inertia = 0.01
p.engine.DM.object_inertia = 0.1
p.engine.DM.update_object_first = True
p.engine.DM.obj_smooth_std = 10
p.engine.DM.overlap_converge_factor = 0.5
p.engine.DM.overlap_max_iterations = 100
p.engine.DM.fourier_relax_factor = 0.05

p.engines = u.Param()
p.engines.engine00 = u.Param()
p.engines.engine00.name = 'DM'
p.engines.engine00.numiter = 30


P = Ptycho(p,level=5)














