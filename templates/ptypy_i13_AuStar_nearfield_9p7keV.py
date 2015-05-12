
import ptypy
from ptypy.core import Ptycho
from ptypy import utils as u
import numpy as np
p = u.Param()

### PTYCHO PARAMETERS
p.verbose_level = 3                               # (00) Verbosity level

p.data_type = "single"                            # (01) Reconstruction floatine number precision
p.run = None
p.paths = u.Param()
p.paths.home = "/tmp/ptypy/"             # (03) Relative base path for all other paths
p.paths.run = None

p.autoplot =  u.Param()
p.autoplot.layout ='nearfield'

p.scan = u.Param()
p.scan.source = None                             # (14) Address or path to data ressource.
#p.scan.tags = "file"                             # (15) tags (comma seperated) describing what kind of data this is
p.scan.geometry = u.Param()
p.scan.geometry.energy = 9.7                    # (17) Energy (in keV)
p.scan.geometry.lam = None                       # (18) wavelength
p.scan.geometry.distance = 8.46e-2                        # (19) distance from object to screen
p.scan.geometry.psize = 100e-9                 # (20) Pixel size in Detector plane
p.scan.geometry.shape = 1024                         # (22) Number of detector pixels
p.scan.geometry.propagation = "nearfield"           # (23) propagation type

p.scan.if_conflict_use_meta = False

sim = u.Param()
sim.xy = u.Param()
sim.xy.override = u.parallel.MPIrand_uniform(0.0,10e-6,(20,2))                       # (35) override
#sim.xy.positions = np.random.normal(0.0,3e-6,(20,2))
sim.verbose_level = 1

sim.illumination = u.Param()
sim.illumination.model = None
sim.illumination.photons = 1e11                # (49) number of photons in illumination
sim.illumination.aperture = u.Param()
sim.illumination.aperture.diffuser = (8.0, 10.0)
sim.illumination.aperture.form = "circ"
sim.illumination.aperture.size = 90e-6
sim.illumination.aperture.central_stop = 0.15
sim.illumination.propagation = u.Param()
sim.illumination.propagation.focussed = None#0.08 
sim.illumination.propagation.parallel = 0.005
sim.illumination.propagation.spot_size = None

sim.sample = u.Param()
sim.sample.model = u.xradia_star((1200,1200),minfeature=3,contrast=0.8)# (52) 'diffraction', None, path to a file or nd-array 
sim.sample.process = u.Param()
sim.sample.process.offset = (0,0)                     # (53) offset between center of object array and scan pattern
sim.sample.process.zoom = 1.0                         # (54) None, scalar or 2-tupel
sim.sample.process.formula = "Au"                     # (55) chemical formula (string)
sim.sample.process.density = 19.3                     # (56) density in [g/ccm]
sim.sample.process.thickness = 700e-9                 # (57) max thickness of sample
sim.sample.process.ref_index = None                   # (58) assigned refractive index
sim.sample.process.smoothing = None                  # (59) smooth the projection with gaussian kernel of with x pixels
sim.sample.fill = 1.0+0.j                     # (62) if object is smaller than the objectframe, fill with fill

sim.detector = 'GenericCCD32bit' 
sim.plot = False

p.scan.update(sim.copy(depth=4))
p.scan.coherence = u.Param()
p.scan.coherence.Nprobe_modes = 1               # (65) 
p.scan.coherence.Nobject_modes = 1               # (66) 
p.scan.coherence.energies = [1.0]                # (67) 

p.scan.sharing = u.Param()
p.scan.sharing.object_shared_with = None         # (71) `scan_label` of scan for the shared obejct
p.scan.sharing.object_share_power = 1            # (72) contribution to the shared object
p.scan.sharing.probe_shared_with = None          # (73) `scan_label` of scan for the shared probe
p.scan.sharing.probe_share_power = 1             # (74) contribution to the shared probe

p.scan.sample = u.Param()
#p.scan.sample.model = 'stxm'
#p.scan.sample.process = None
#p.scan.sample.diversity = None
p.scan.xy = u.Param()
p.scan.xy.model=None
#p.scan.illumination = sim.illumination.copy()
p.scan.illumination.model = 'stxm'

p.scans = u.Param()
p.scans.sim = u.Param()
p.scans.sim.data=u.Param()
p.scans.sim.data.source = 'sim' 
p.scans.sim.data.recipe = sim.copy(depth=4) 
p.scans.sim.data.save = None #'append'
p.scans.sim.data.shape = None
p.scans.sim.data.num_frames = None

p.engine = u.Param()
p.engine.common = u.Param()
p.engine.common.numiter = 100                    # (85) Total number of iterations
p.engine.common.numiter_contiguous = 1            # (86) Number of iterations to be executed in one go
p.engine.common.probe_support = None               # (87) Fraction of valid probe area (circular) in probe frame
p.engine.common.probe_inertia = 0.001             # (88) Probe fraction kept from iteration to iteration
p.engine.common.object_inertia = 0.1              # (89) Object fraction kept from iteration to iteration
p.engine.common.obj_smooth_std = 10               # (90) Gaussian smoothing (pixel) of kept object fraction
p.engine.common.clip_object = None                # (91) Clip object amplitude into this intrervall

p.engine.DM = u.Param()
p.engine.DM.name = "DM"                           # (93) abbreviation of algorithm
p.engine.DM.alpha = 1                             # (94) Parameters that makes the difference
p.engine.DM.probe_update_start = 2                # (95) Number of iterations before probe update starts
p.engine.DM.update_object_first = True            # (96) If False update object before probe
p.engine.DM.overlap_converge_factor = 0.5        # (97) Loop overlap constraint until probe change is smaller than this fraction
p.engine.DM.overlap_max_iterations = 100           # (98) Maximum iterations to be spent inoverlap constraint
p.engine.DM.fourier_relax_factor = 0.05           # (99) If rms of model vs diffraction data is smaller than this fraction, Fourier constraint is met

p.engine.ML = u.Param()

p.engines = u.Param()                                  # (100) empty structure to be filled with engines
p.engines.engine00 = u.Param()
p.engines.engine00.name = 'DM'
p.engines.engine00.numiter = 100
p.engines.engine00.object_inertia = 1.
p.engines.engine00.fourier_relax_factor = 0.1
#p.engines.engine01 = u.Param()
#p.engines.engine01.name = 'ML'
#p.engines.engine01.numiter = 50


P = Ptycho(p,level=5)
##P.save_run(kind='minimal')
#P.plot_overview()
#while True:         
    #u.pause(10)
