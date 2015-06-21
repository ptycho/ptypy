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

p.io = u.Param()
p.io.home = "/tmp/ptypy/" 
p.io.run = None
p.io.autoplot = u.Param()
p.io.autoplot.layout='weak'
p.io.autosave = None

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
p.engine.common.numiter = 100       
p.engine.common.numiter_contiguous = 1    
p.engine.common.probe_support = 0.7    
p.engine.common.probe_inertia = 0.01    
p.engine.common.object_inertia = 0.1  
p.engine.common.clip_object = [0,1.]        

p.engine.DM = u.Param()
p.engine.DM.name = "DM"             
p.engine.DM.alpha = 1                
p.engine.DM.probe_update_start = 2     
p.engine.DM.update_object_first = True   
p.engine.DM.overlap_converge_factor = 0.05 
p.engine.DM.overlap_max_iterations = 100    
p.engine.DM.fourier_relax_factor = 0.1     
p.engine.DM.obj_smooth_std = 5 

p.engine.ML = u.Param()

p.engines = u.Param()                          
p.engines.engine00 = u.Param()
p.engines.engine00.name = 'DM'
p.engines.engine00.numiter = 150
p.engines.engine00.fourier_relax_factor = 0.05

u.verbose.set_level(3)
P = Ptycho(p,level=5)




