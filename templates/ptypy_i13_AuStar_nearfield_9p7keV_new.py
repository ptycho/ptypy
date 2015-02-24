
import ptypy
from ptypy.core import Ptycho
from ptypy import utils as u
import numpy as np
p = u.Param()

### PTYCHO PARAMETERS
p.verbose_level = 3                               # (00) Verbosity level

p.data_type = "single"                            # (01) Reconstruction floatine number precision

p.paths = u.Param()
p.paths.base_dir = "./"             # (03) Relative base path for all other paths
p.paths.run = None                                # (04) Name of reconstruction run
p.paths.data_dir = "analysis/%(run)s/"                    # (05) directory where diffraction data is stored
p.paths.data_file = "%(label)s.ptyd"              
p.paths.plot_dir = "plots/%(run)s/"               # (06) directory to dump plot images
p.paths.plot_file = "%(run)s_%(engine)s_%(iterations)04d.png"# (07) filename for dumping plots
p.paths.plot_interval = 2                         # (08) iteration interval for dumping plots
p.paths.save_dir = "recons/%(run)s/"              # (09) directory to save final reconstruction
p.paths.save_file = "%(run)s_%(engine)s_%(iterations)04d.ptyr"# (10) filename for saving 
p.paths.dump_dir = "dumps/%(run)s/"               # (11) directory to save intermediate results
p.paths.dump_file = "%(run)s_%(engine)s_%(iterations)04d.ptyr"# (12) 


p.model = u.Param()
p.model.source = None                             # (14) Address or path to data ressource.
#p.model.tags = "file"                             # (15) tags (comma seperated) describing what kind of data this is
p.model.geometry = u.Param()
p.model.geometry.energy = 9.7                    # (17) Energy (in keV)
p.model.geometry.lam = None                       # (18) wavelength
p.model.geometry.z = 8.46e-2                        # (19) distance from object to screen
p.model.geometry.psize_det = 100e-9                 # (20) Pixel size in Detector plane
p.model.geometry.psize_sam = None                 # (21) Pixel size in Sample plane
p.model.geometry.shape = 1024                         # (22) Number of detector pixels
p.model.geometry.prop_type = "nearfield"           # (23) propagation type

p.model.if_conflict_use_meta = False

sim = u.Param()
sim.xy = u.Param()
sim.xy.scan_type = 'custom'                # (25) None,'round', 'raster', 'round_roi','custom'
sim.xy.dr = 2e-6                             # (26) round,round_roi :width of shell
sim.xy.nr = 10                                # (27) round : number of intervals (# of shells - 1) 
sim.xy.nth = 5                                # (28) round,round_roi: number of points in the first shell
sim.xy.lx = 10e-6                            # (29) round_roi: Width of ROI
sim.xy.ly = 10e-6                            # (30) round_roi: Height of ROI
sim.xy.nx = 10                                # (31) raster scan: number of steps in x
sim.xy.ny = 10                                # (32) raster scan: number of steps in y
sim.xy.dx = 1e-6                              # (33) raster scan: step size (grid spacing)
sim.xy.dy = 1e-6                              # (34) raster scan: step size (grid spacing)
sim.xy.positions = u.parallel.MPIrand_uniform(0.0,10e-6,(20,2))                       # (35) override
#sim.xy.positions = np.random.normal(0.0,3e-6,(20,2))

sim.illumination = u.Param()
sim.illumination.probe_type = "parallel"         # (37) 'focus' or 'parallel'
sim.illumination.incoming = None              # (38) `None`, path to a file or any python evaluable statement yielding a 2d numpy array. If `None` defaults to array of ones
sim.illumination.phase_noise_rms = 8.0        # (39) phase noise amplitude on incoming wave before aperture 
sim.illumination.phase_noise_mfs = 10.0        # (40) phase noise minimum feature size on incoming wave before aperture 
sim.illumination.aperture_type = "circ"       # (41) type of aperture: use 
sim.illumination.aperture_size = 90e-6        # (42) aperture diameter (meter)
sim.illumination.aperture_edge = 2            # (43) edge width of aperture (pixel)
sim.illumination.focal_dist = 0.08            # (44) distance from aperture to focus (meter)
sim.illumination.prop_dist = 5e-3           # (45) focus: propagation distance (meter) from focus. parallel: propagation distance (meter) from aperture 
sim.illumination.UseConjugate = False         # (46) use the conjugate of the probe instef of the probe
sim.illumination.antialiasing = 1.0           # (47) antialiasing factor used when generating the probe
sim.illumination.spot_size = None             # (48) focal spot diameter (meter)
sim.illumination.photons = 1e11                # (49) number of photons in illumination
sim.illumination.probe = None                 # (50) override if not None

sim.sample = u.Param()
sim.sample.source = u.xradia_star((1000,1000),minfeature=3,contrast=0.8)# (52) 'diffraction', None, path to a file or nd-array 
sim.sample.offset = (0,0)                     # (53) offset between center of object array and scan pattern
sim.sample.zoom = 1.0                         # (54) None, scalar or 2-tupel
sim.sample.formula = "Au"                     # (55) chemical formula (string)
sim.sample.density = 19.3                     # (56) density in [g/ccm]
sim.sample.thickness = 700e-9                 # (57) max thickness of sample
sim.sample.ref_index = None                   # (58) assigned refractive index
sim.sample.smoothing_mfs = None                  # (59) smooth the projection with gaussian kernel of with x pixels
sim.sample.noise_rms = None                   # (60) noise applied, relative to 2*pi in phase and relative to 1 in amplitude
sim.sample.noise_mfs = 10                     # (61) see noise rms.
sim.sample.fill = 1.0+0.j                     # (62) if object is smaller than the objectframe, fill with fill
sim.sample.obj = None                         # (63) override

sim.detector = 'GenericCCD32bit' 

p.model.coherence = u.Param()
p.model.coherence.Nprobe_modes = 1               # (65) 
p.model.coherence.Nobject_modes = 1               # (66) 
p.model.coherence.energies = [1.0]                # (67) 

p.model.sharing = u.Param()
p.model.sharing.scan_per_probe = 1                # (69) number of scans per object
p.model.sharing.scan_per_object = 1.              # (70) number of scans per probe
p.model.sharing.object_shared_with = None         # (71) `scan_label` of scan for the shared obejct
p.model.sharing.object_share_power = 1            # (72) contribution to the shared object
p.model.sharing.probe_shared_with = None          # (73) `scan_label` of scan for the shared probe
p.model.sharing.probe_share_power = 1             # (74) contribution to the shared probe

p.model.sample = u.Param()
p.model.sample.source=None
p.model.xy = u.Param()
p.model.xy.scan_type=None
p.model.illumination = sim.illumination.copy()
p.model.illumination.phase_noise_rms = None

p.scans = u.Param()
p.scans.sim = u.Param()
p.scans.sim.data=u.Param()
p.scans.sim.data.source = 'sim' 
p.scans.sim.data.recipe = sim 
p.scans.sim.data.save = 'append'
#p.scans.scan002 = u.Param()
#p.scans.scan002.sharing = u.Param()
#p.scans.scan002.sharing.probe_shared_with = 'scan001'
#p.scans.scan002.tags = 'empty'


p.interaction = u.Param()
#p.interaction.primary_address = "tcp://127.0.0.2" # (80) 
#p.interaction.primary_port = 5570                 # (81) 
#p.interaction.port_range = "5571:5580"     # (82) 

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


P = Ptycho(p,level=3)
P.save_run(kind='minimal')
         
