import ptypy
from ptypy.core import Ptycho
from ptypy import utils as u
import ptypy.simulations as sim
import numpy as np
p = u.Param()

### PTYCHO PARAMETERS
p.verbose_level = 3                  
p.data_type = "single"                            
p.run = None
p.io = u.Param()
p.io.home = "/tmp/ptypy/"             
p.io.run = None
p.io.autosave = None

p.scan = u.Param()
p.scan.geometry = u.Param()
p.scan.geometry.energy = 6.2        
p.scan.geometry.lam = None    
p.scan.geometry.distance = 7    
p.scan.geometry.psize = 172e-6       
p.scan.geometry.shape = 256          
p.scan.geometry.propagation = "farfield"     

p.scan.illumination = u.Param()
p.scan.illumination.model = None
p.scan.illumination.aperture = u.Param() 
p.scan.illumination.propagation = None 
p.scan.illumination.diversity = None 

p.scan.sample = u.Param()
p.scan.sample.model = 'stxm'
p.scan.sample.process = None
p.scan.sample.diversity = None

p.scan.coherence = u.Param()
p.scan.coherence.Nprobe_modes = 1               
p.scan.coherence.Nobject_modes = 1               
p.scan.coherence.energies = [1.0]                

p.scan.sharing = u.Param()
p.scan.sharing.object_shared_with = None 
p.scan.sharing.object_share_power = 1  
p.scan.sharing.probe_shared_with = None 
p.scan.sharing.probe_share_power = 1      

sim = u.Param()
sim.xy = u.Param()
sim.xy.model = "round"                
sim.xy.spacing = 1e-6                
sim.xy.steps = 10        
sim.xy.extent = 10e-6      

sim.illumination = u.Param()
sim.illumination.model = None
sim.illumination.photons = 1e8       
sim.illumination.aperture = u.Param()
sim.illumination.aperture.diffuser = None
sim.illumination.aperture.form = "circ"
sim.illumination.aperture.size = 2.5e-6
sim.illumination.aperture.edge = 1
sim.illumination.aperture.central_stop = None
sim.illumination.propagation = u.Param()
sim.illumination.propagation.focussed = None 
sim.illumination.propagation.parallel = 0.004
sim.illumination.propagation.spot_size = None

sim.sample = u.Param()
sim.sample.model = 255-u.imload('../resources/tree.bmp').astype(float).mean(-1)
sim.sample.process = u.Param()
sim.sample.process.offset = (100,400)                    
sim.sample.process.zoom = 1.0                         
sim.sample.process.formula = "Ca"                     
sim.sample.process.density = 1.5                     
sim.sample.process.thickness = 20e-6  
sim.sample.process.ref_index = None 
sim.sample.process.smoothing = None  
sim.sample.fill = 1.0+0.j 

sim.verbose_level = 1
sim.plot = False

p.scans = u.Param()
p.scans.CircSim = u.Param()
p.scans.CircSim.data=u.Param()
p.scans.CircSim.data.source = 'sim' 
p.scans.CircSim.data.recipe = sim.copy(depth=4)
p.scans.CircSim.data.recipe.illumination.aperture.form = "circ"
p.scans.CircSim.data.save = None

p.scans.RectSim = u.Param()
p.scans.RectSim.sharing= u.Param(object_share_with = 'CircSim')
p.scans.RectSim.data=u.Param()
p.scans.RectSim.data.source = 'sim' 
p.scans.RectSim.data.recipe = sim.copy(depth=4)
p.scans.RectSim.data.recipe.illumination.aperture.form = "rect"
p.scans.RectSim.data.save = None


p.engine = u.Param()
p.engine.common = u.Param()
p.engine.common.numiter = 100     
p.engine.common.numiter_contiguous = 1  
p.engine.common.probe_support = 0.7        
p.engine.common.probe_inertia = 0.01   
p.engine.common.object_inertia = 0.1     
p.engine.common.obj_smooth_std = 20   
p.engine.common.clip_object = None    
p.engine.common.probe_update_start = 2        

p.engine.DM = u.Param()
p.engine.DM.name = "DM"                    
p.engine.DM.alpha = 1                       
p.engine.DM.update_object_first = True      
p.engine.DM.overlap_converge_factor = 0.05      
p.engine.DM.overlap_max_iterations = 100    
p.engine.DM.fourier_relax_factor = 0.1   

p.engine.ML = u.Param()

p.engines = u.Param()                        
p.engines.engine00 = u.Param()
p.engines.engine00.name = 'DM'
p.engines.engine00.numiter = 50
p.engines.engine00.fourier_relax_factor = 0.05
p.engines.engine01 = u.Param()
p.engines.engine01.name = 'ML'
p.engines.engine01.numiter = 100



u.verbose.set_level(3)
P = Ptycho(p,level=5)


