# A simulation template to test and demonstrate the "independent probe" variation of Difference map.
# Note that it is better to use "ptypy.plotclient --layout minimal" otherwise 92 probes will be plotted.

from ptypy import utils as u
p = u.Param()

p.verbose_level = 3
p.data_type = "single"
p.run = 'test_indep_probes'
p.io = u.Param()
p.io.home = "/tmp/ptypy/"
p.io.autosave = u.Param()
p.io.autosave.interval = 20
p.io.autoplot = False
p.io.interaction = u.Param()
p.scan = u.Param()
p.scans = u.Param()
p.scans.MF = u.Param()
p.scans.MF.if_conflict_use_meta = True
p.scans.MF.data = u.Param()
p.scans.MF.data.source = 'test'
p.scans.MF.data.recipe = u.Param()
p.scans.MF.data.recipe.density = 0.2
p.scans.MF.data.shape = 128
p.scans.MF.data.num_frames = 100
p.scans.MF.data.save = None
p.scans.MF.illumination = u.Param()
p.scans.MF.illumination.model=None
p.scans.MF.illumination.aperture = u.Param()
p.scans.MF.illumination.aperture.diffuser = None
p.scans.MF.illumination.aperture.form = "circ"
p.scans.MF.illumination.aperture.size = 3e-6
p.scans.MF.illumination.aperture.edge = 10


p.engines = u.Param()
p.engines.engine00 = u.Param()
p.engines.engine00.name = 'DMIP'
p.engines.engine00.numiter = 300
p.engines.engine00.numiter_contiguous = 5
p.engines.engine00.overlap_max_iterations = 2
p.engines.engine00.fourier_relax_factor = 0.01
p.engines.engine00.IP_metric = 5.

from ptypy.core import Ptycho
P = Ptycho(p, level=4)

import ptypy
import numpy as np
MF = ptypy.core.data.MoonFlowerScan({'source':'test', 'num_frames':100, 'shape':128, 'recipe':{'density':.1}})

pixel = MF.pos / MF.geo.resolution
#pixel -= pixel.min(0)
#pixel = np.round(pixel).astype(int) + 10
px = np.round(pixel)
px -= px.min()
px += 10

i0, j0 = px.min(axis=0)
i1, j1 = px.max(axis=0) + 128

init_obj = MF.obj[i0:i1, j0:j1]
#P.obj.S.values()[0].data[:] = init_obj * np.random.normal(1., .8, size=init_obj.shape)

#P.probe.S.values()[0].data[:] = MF.pr * np.random.normal(1., .8, size=MF.pr.shape)

for name, pod in P.pods.iteritems():
    if pod.active:
        pod.exit = pod.probe * pod.object

P.run()
