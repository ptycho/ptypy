import ptypy
from ptypy.core import Ptycho
from ptypy import utils as u
import numpy as np
import sys

"""
Reconstruction script for already prepared far-field ptychography data taken at ID16A beamline - ESRF
First version by B. Enders (12/05/2015)
Modifications by J. C. da Silva (30/05/2015)
"""

try:
    filename = sys.argv[1]
except IndexError:
    print('Missing arguments with .ptyd file')
    print('Usage: {} <ptyd filename>'.format(sys.argv[0]))

p = u.Param()

### PTYCHO PARAMETERS
p.verbose_level = 3

p.data_type = "single"

p.io = u.Param()
p.io.autoplot = u.Param()
p.io.autoplot.interval = 1
p.io.autoplot.layout = 'weak'
p.io.autoplot.threaded = False
p.io.autoplot.make_movie = False
p.io.autoplot.dump = True

p.autosave = None
p.paths = u.Param()

p.model = u.Param()

p.scan = u.Param()
p.scans = u.Param()

# Scan entry for the Siemensstar
p.scans.sstar = u.Param()
p.scans.sstar.if_conflict_use_meta = True

p.scans.sstar.data= u.Param()
p.scans.sstar.data.source = filename
p.scans.sstar.data.dfile = filename

p.scans.sstar.illumination = u.Param()
p.scans.sstar.illumination.aperture = u.Param()
p.scans.sstar.illumination.aperture.form = 'rect'
p.scans.sstar.illumination.aperture.size = 60e-6
p.scans.sstar.illumination.propagation=u.Param()
p.scans.sstar.illumination.propagation.focussed = 0.1
p.scans.sstar.illumination.propagation.parallel = 2e-3
p.scans.sstar.illumination.diversity = u.Param()
p.scans.sstar.illumination.diversity.power = 0.1
p.scans.sstar.illumination.diversity.noise = (1,2)

p.scans.sstar.sample= u.Param()

p.scans.sstar.coherence = u.Param()
p.scans.sstar.coherence.num_probe_modes = 3

p.engines = u.Param()
p.engines.engine00 = u.Param()
p.engines.engine00.name = 'DM'
p.engines.engine00.numiter =200
p.engines.engine00.fourier_relax_factor = 0.02
p.engines.engine00.overlap_converge_factor = .05
p.engines.engine00.overlap_max_iterations = 10
p.engines.engine00.update_object_first = True
p.engines.engine00.probe_update_start = 2
p.engines.engine00.object_smooth_std = 5
p.engines.engine00.probe_inertia = 0.01
p.engines.engine00.object_inertia = 0.1
p.engines.engine01 = u.Param()
p.engines.engine01.name = 'ML'
p.engines.engine01.numiter = 100

p.engine = u.Param()
p.engine.ML = u.Param()
p.engine.ML.floating_intensities = True

P = Ptycho(p,level=5)
