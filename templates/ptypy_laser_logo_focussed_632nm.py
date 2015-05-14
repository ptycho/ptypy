import ptypy
from ptypy.core import Ptycho
from ptypy import utils as u
import ptypy.simulations as sim
import numpy as np
p = u.Param()


### PTYCHO PARAMETERS
p.verbose_level = 3                  
p.data_type = "single"                            
p.autosave = None
p.autoplot = u.Param()
p.autoplot.layout='minimal'
p.run = None
p.paths = u.Param()
p.paths.home = "/tmp/ptypy/"             
p.paths.run = None

p.scan = u.Param()
p.scan.geometry = u.Param()
p.scan.geometry.energy = u.keV2m(1.0)/6.32e-7
p.scan.geometry.lam = None
p.scan.geometry.distance = 15e-2
p.scan.geometry.psize = 24e-6
p.scan.geometry.shape = 256
p.scan.geometry.propagation = "farfield"


sim = u.Param()
sim.xy = u.Param()
sim.xy.model = "round"                
sim.xy.spacing = 0.3e-3                
sim.xy.steps = 60        
sim.xy.extent = (5e-3,10e-3)      

sim.illumination = u.Param()
sim.illumination.model = None
sim.illumination.photons = 1e9                # (49) number of photons in illumination
sim.illumination.aperture = u.Param()
sim.illumination.aperture.diffuser = None#(0.7,3)
sim.illumination.aperture.form = "circ"
sim.illumination.aperture.size = 1.0e-3
sim.illumination.aperture.edge = 2
sim.illumination.aperture.central_stop = None
sim.illumination.propagation = u.Param()
sim.illumination.propagation.focussed = None 
sim.illumination.propagation.parallel = 0.03
sim.illumination.propagation.spot_size = None

sim.sample = u.Param()
sim.sample.model = -u.rgb2complex(u.imload('../resources/ptypy_logo_1M.png')[::-1,:,:-1])
sim.sample.process = u.Param()
sim.sample.process.offset = (0,0)                   
sim.sample.process.zoom = 0.5                         
sim.sample.process.formula = None                    
sim.sample.process.density = None                     # (56) density in [g/ccm]
sim.sample.process.thickness = None                 # (57) max thickness of sample
sim.sample.process.ref_index = None                   # (58) assigned refractive index
sim.sample.process.smoothing = None                  # (59) smooth the projection with gaussian kernel of with x pixels
sim.sample.fill = 1.0+0.j 
sim.plot=False
sim.detector = dict(dtype=np.uint32,full_well=2**32-1,psf=None)

p.scans = u.Param()
p.scans.ptypy = u.Param()
p.scans.ptypy.data = u.Param()
p.scans.ptypy.data.source = 'sim' 
p.scans.ptypy.data.recipe = sim.copy(depth=4)

p.scans.ptypy.coherence = u.Param()
p.scans.ptypy.coherence.num_probe_modes=1

p.scans.ptypy.illumination = u.Param()
p.scans.ptypy.illumination.model=None
p.scans.ptypy.illumination.aperture = u.Param()
p.scans.ptypy.illumination.aperture.diffuser = None
p.scans.ptypy.illumination.aperture.form = "circ"
p.scans.ptypy.illumination.aperture.size = 1.0e-3
p.scans.ptypy.illumination.aperture.edge = 10

p.engine = u.Param()
p.engine.common = u.Param()
p.engine.common.numiter = 100                    # (85) Total number of iterations
p.engine.common.numiter_contiguous = 1            # (86) Number of iterations to be executed in one go
p.engine.common.probe_support = 0.9               # (87) Fraction of valid probe area (circular) in probe frame
p.engine.common.probe_inertia = 0.01             # (88) Probe fraction kept from iteration to iteration
p.engine.common.object_inertia = 0.1              # (89) Object fraction kept from iteration to iteration
p.engine.common.clip_object = None                # (91) Clip object amplitude into this intrervall

p.engine.DM = u.Param()
p.engine.DM.name = "DM"                           # (93) abbreviation of algorithm
p.engine.DM.alpha = 1                             # (94) Parameters that makes the difference
p.engine.DM.probe_update_start = 2                # (95) Number of iterations before probe update starts
p.engine.DM.update_object_first = True            # (96) If False update object before probe
p.engine.DM.overlap_converge_factor = 0.05        # (97) Loop overlap constraint until probe change is smaller than this fraction
p.engine.DM.overlap_max_iterations = 100           # (98) Maximum iterations to be spent inoverlap constraint
p.engine.DM.fourier_relax_factor = 0.05           # (99) If rms of model vs diffraction data is smaller than this fraction, Fourier constraint is met
p.engine.DM.obj_smooth_std = 5               # (90) Gaussian smoothing (pixel) of kept object fraction

p.engine.ML = u.Param()

p.engines = u.Param()                                  # (100) empty structure to be filled with engines
p.engines.engine00 = u.Param()
p.engines.engine00.name = 'DM'
p.engines.engine00.numiter = 20
p.engines.engine00.fourier_relax_factor = 0.05
#p.engines.engine01 = u.Param()
#p.engines.engine01.name = 'ML'
#p.engines.engine01.numiter = 30

u.verbose.set_level(3)
P = Ptycho(p,level=5)



