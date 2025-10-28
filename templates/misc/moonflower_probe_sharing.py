'''
An example showing how to share a probe across two scans
'''
import ptypy
from ptypy import utils as u
from ptypy.core import Ptycho
import tempfile
u.verbose.set_level(3)
tmp = tempfile.mkdtemp()+'/%s'
tmpdir = tempfile.gettempdir()

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
p.verbose_level = "info"
p.io = u.Param()
p.io.home = "/".join([tmpdir, "ptypy"])
p.io.autosave = u.Param(active=False)

p.scans = u.Param()

p.scans.MF_01 = u.Param()
p.scans.MF_01.name= "Vanilla"
p.scans.MF_01.data= u.Param()
p.scans.MF_01.data.name = 'PtydScan'
p.scans.MF_01.data.label = 'MF_01'
p.scans.MF_01.data.source ='file'
p.scans.MF_01.data.dfile = tmp % 'scan1.ptyd'

p.scans.MF_02 = u.Param()
p.scans.MF_02.name= "Vanilla"
p.scans.MF_02.data= u.Param()
p.scans.MF_02.data.name = 'PtydScan'
p.scans.MF_02.data.label = 'MF_02'
p.scans.MF_02.data.source = 'file'
p.scans.MF_02.data.dfile = tmp % 'scan2.ptyd'


p.engines = u.Param()
p.engines.DM = u.Param()
p.engines.DM.name = "DM"
p.engines.DM.alpha = 1
p.engines.DM.probe_inertia = 0.01
p.engines.DM.object_inertia = 0.1
p.engines.DM.update_object_first = True
p.engines.DM.obj_smooth_std = 10
p.engines.DM.overlap_converge_factor = 0.5
p.engines.DM.overlap_max_iterations = 5
p.engines.DM.fourier_relax_factor = 0.05
p.engines.DM.numiter = 60



if __name__ == "__main__":
    P = Ptycho(p,level=3)

    s1, s2 = list(P.probe.storages.values())
    # Transfer views
    for v in s2.views:
        v.storage = s1
        v.storageID = s1.ID
    P.probe.reformat()

    # Unforunately we need to delete the storage here due to DM being unable 
    # to ignore unused storages. This is due to the /=nrm division in the
    # probe update 
    P.probe.storages.pop(s2.ID)

    P.print_stats()
    P.init_engine()
    P.run()
    P.finalize()













