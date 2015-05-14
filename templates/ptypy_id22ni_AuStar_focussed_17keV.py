import numpy as np
import ptypy
from ptypy.core import Ptycho
from ptypy import utils as u
p = u.Param()

### PTYCHO PARAMETERS
p.verbose_level = 3 

p.interaction = u.Param()

p.data_type = "single"
p.run = None
p.autosave = None
p.paths = u.Param()
p.paths.home = "/tmp/ptypy/" 
p.paths.run = None
p.autoplot = u.Param()
p.autoplot.layout='weak'


sim = u.Param()
sim.xy = u.Param()
sim.xy.model = "round" 
sim.xy.spacing = 250e-9   
sim.xy.steps = 30        
sim.xy.extent = 4e-6 

sim.illumination = u.Param()
sim.illumination.model = None
sim.illumination.photons = 3e8
sim.illumination.aperture = u.Param()
sim.illumination.aperture.diffuser = None
sim.illumination.aperture.form = "rect"
sim.illumination.aperture.size = 35e-6
sim.illumination.aperture.central_stop = None
sim.illumination.propagation = u.Param()
sim.illumination.propagation.focussed = 0.08 
sim.illumination.propagation.parallel = 0.0014
sim.illumination.propagation.spot_size = None

sim.sample = u.Param()
sim.sample.model = u.xradia_star((1000,1000),minfeature=3,contrast=0.0)
sim.sample.process = u.Param()
sim.sample.process.offset = (100,100) 
sim.sample.process.zoom = 1.0 
sim.sample.process.formula = "Au" 
sim.sample.process.density = 19.3 
sim.sample.process.thickness = 2000e-9
sim.sample.process.ref_index = None
sim.sample.process.smoothing = None
sim.sample.fill = 1.0+0.j

#sim.detector = 'FRELON_TAPER'
sim.detector = 'GenericCCD32bit'
sim.verbose_level = 1
sim.coherence = u.Param()
sim.coherence.num_probe_modes = 1
sim.psf = 1.
sim.plot = False

p.scan = sim.copy(depth=4)

p.scan.geometry = u.Param()
p.scan.geometry.energy = 17.0
p.scan.geometry.lam = None
p.scan.geometry.distance = 2.886
p.scan.geometry.psize = 51e-6
p.scan.geometry.shape = 256
p.scan.geometry.propagation = "farfield"


p.scan.coherence = u.Param()
p.scan.coherence.num_probe_modes = 4
p.scan.coherence.num_object_modes = 1
p.scan.coherence.energies = [1.0]


p.scan.sample.model = 'stxm'
p.scan.sample.process =  None
p.scan.illumination.aperture.form = 'circ'
p.scan.illumination.propagation.focussed = 0.06
p.scan.illumination.diversity = u.Param()
p.scan.illumination.diversity.power = 0.1
p.scan.illumination.diversity.noise = (np.pi,3.0)

p.scans = u.Param()
p.scans.sim = u.Param()
p.scans.sim.data=u.Param()
p.scans.sim.data.source = 'sim' 
p.scans.sim.data.recipe = sim
p.scans.sim.data.save = None


p.engine = u.Param()
p.engine.common = u.Param()
p.engine.common.numiter = 100                    # (85) Total number of iterations
p.engine.common.numiter_contiguous = 1            # (86) Number of iterations to be executed in one go
p.engine.common.probe_support = 0.7               # (87) Fraction of valid probe area (circular) in probe frame
p.engine.common.probe_inertia = 0.01             # (88) Probe fraction kept from iteration to iteration
p.engine.common.object_inertia = 0.1              # (89) Object fraction kept from iteration to iteration
p.engine.common.clip_object = [0,1.]                # (91) Clip object amplitude into this intrervall

p.engine.DM = u.Param()
p.engine.DM.name = "DM"                           # (93) abbreviation of algorithm
p.engine.DM.alpha = 1                             # (94) Parameters that makes the difference
p.engine.DM.probe_update_start = 2                # (95) Number of iterations before probe update starts
p.engine.DM.update_object_first = True            # (96) If False update object before probe
p.engine.DM.overlap_converge_factor = 0.05        # (97) Loop overlap constraint until probe change is smaller than this fraction
p.engine.DM.overlap_max_iterations = 100           # (98) Maximum iterations to be spent inoverlap constraint
p.engine.DM.fourier_relax_factor = 0.1           # (99) If rms of model vs diffraction data is smaller than this fraction, Fourier constraint is met
p.engine.DM.obj_smooth_std = 5 

p.engine.ML = u.Param()

p.engines = u.Param()                                  # (100) empty structure to be filled with engines
p.engines.engine00 = u.Param()
p.engines.engine00.name = 'DM'
p.engines.engine00.numiter = 150
p.engines.engine00.fourier_relax_factor = 0.05

u.verbose.set_level(3)
P = Ptycho(p,level=5)
#if u.parallel.master:
    #P.plot_overview()
#u.pause(100.)
#P.save_run(kind='minimal')



