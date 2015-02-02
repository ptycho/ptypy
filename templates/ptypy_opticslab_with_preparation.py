
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
p.paths.save_file = "%(run)s_%(engine)s_%(iterations)04d.h5"# (10) filename for saving 
p.paths.dump_dir = "dumps/%(run)s/"               # (11) directory to save intermediate results
p.paths.dump_file = "%(run)s_%(engine)s_%(iterations)04d.h5"# (12) 


p.model = u.Param()
p.model.source = None                             # (14) Address or path to data ressource.
p.model.tags = "file"                             # (15) tags (comma seperated) describing what kind of data this is
p.model.geometry = u.Param()
p.model.geometry.energy = u.keV2m(1.0)/5.32e-7    #None                    # (17) Energy (in keV)
p.model.geometry.lam = 5.32e-7                       # (18) wavelength
p.model.geometry.z = 0.152                        # (19) distance from object to screen
p.model.geometry.psize_det = 3*24e-6                # (20) Pixel size in Detector plane
p.model.geometry.psize_sam = None                 # (21) Pixel size in Sample plane
p.model.geometry.N = 256                          # (22) Number of detector pixels
p.model.geometry.prop_type = "farfield"           # (23) propagation type

p.model.xy = u.Param()
p.model.xy.scan_type = None                # (25) None,'round', 'raster', 'round_roi','custom'

p.model.illumination = u.Param()
p.model.illumination.probe_type = "parallel"         # (37) 'focus' or 'parallel'
p.model.illumination.incoming = None              # (38) `None`, path to a file or any python evaluable statement yielding a 2d numpy array. If `None` defaults to array of ones
p.model.illumination.phase_noise_rms = None        # (39) phase noise amplitude on incoming wave before aperture 
p.model.illumination.phase_noise_mfs = 2.0        # (40) phase noise minimum feature size on incoming wave before aperture 
p.model.illumination.aperture_type = "circ"       # (41) type of aperture: use 
p.model.illumination.aperture_size = 1.0e-3       # (42) aperture diameter (meter)
p.model.illumination.aperture_edge = 2            # (43) edge width of aperture (pixel)
p.model.illumination.focal_dist = 0.08            # (44) distance from aperture to focus (meter)
p.model.illumination.prop_dist = 10e-3           # (45) focus: propagation distance (meter) from focus. parallel: propagation distance (meter) from aperture 
p.model.illumination.UseConjugate = False         # (46) use the conjugate of the probe instef of the probe
p.model.illumination.antialiasing = 2.0           # (47) antialiasing factor used when generating the probe
p.model.illumination.spot_size = None             # (48) focal spot diameter (meter)
p.model.illumination.photons = None                # (49) number of photons in illumination
p.model.illumination.probe = None                 # (50) override if not None

p.model.sample = u.Param()
p.model.sample.source = None

p.model.coherence = u.Param()
p.model.coherence.Nprobe_modes = 1                # (65) 
p.model.coherence.Nobject_modes = 1               # (66) 
p.model.coherence.energies = [1.0]                # (67) 

p.model.sharing = u.Param()
p.model.sharing.scan_per_probe = 1                # (69) number of scans per object
p.model.sharing.scan_per_object = 1.              # (70) number of scans per probe
p.model.sharing.object_shared_with = None         # (71) `scan_label` of scan for the shared obejct
p.model.sharing.object_share_power = 1            # (72) contribution to the shared object
p.model.sharing.probe_shared_with = None          # (73) `scan_label` of scan for the shared probe
p.model.sharing.probe_share_power = 1             # (74) contribution to the shared probe


# make a scan
s1 = u.Param()
#s1.load_from = "fli_spec_multexp"

s1.data_file = None #"Sim_id22ni_AuStar_defoc_1p40.ptyd"# (14) Address or path to data ressource.
s1.prepare_data = True
s1.preparation = u.Param()
s1.preparation.generic = u.Param()
s1.preparation.generic.rebin = 3
s1.preparation.generic.roi = 768
s1.preparation.generic.orientation = (True,True,True)

s1.preparation.type = "fli_spec_multexp"
s1.preparation.fli_spec_multexp = u.Param()
s1.preparation.fli_spec_multexp.data_number = 74
s1.preparation.fli_spec_multexp.dark_number = 72

# list the scan
p.scans = u.Param()
p.scans.scan001 = s1

p.interaction = None

p.engine = u.Param()
p.engine.common = u.Param()
p.engine.common.numiter = 100                    # (85) Total number of iterations
p.engine.common.numiter_contiguous = 1            # (86) Number of iterations to be executed in one go
p.engine.common.probe_support = 0.7               # (87) Fraction of valid probe area (circular) in probe frame
p.engine.common.probe_inertia = 0.001             # (88) Probe fraction kept from iteration to iteration
p.engine.common.object_inertia = 0.1              # (89) Object fraction kept from iteration to iteration
p.engine.common.obj_smooth_std = 20               # (90) Gaussian smoothing (pixel) of kept object fraction
p.engine.common.clip_object = None                # (91) Clip object amplitude into this intrervall

p.engine.DM = u.Param()
p.engine.DM.name = "DM"                           # (93) abbreviation of algorithm
p.engine.DM.alpha = 1                             # (94) Parameters that makes the difference
p.engine.DM.probe_update_start = 2                # (95) Number of iterations before probe update starts
p.engine.DM.update_object_first = True            # (96) If False update object before probe
p.engine.DM.overlap_converge_factor = 0.05        # (97) Loop overlap constraint until probe change is smaller than this fraction
p.engine.DM.overlap_max_iterations = 100           # (98) Maximum iterations to be spent inoverlap constraint
p.engine.DM.fourier_relax_factor = 0.1           # (99) If rms of model vs diffraction data is smaller than this fraction, Fourier constraint is met

p.engine.ML = u.Param()

p.engines = u.Param()                                  # (100) empty structure to be filled with engines
p.engines.engine00 = u.Param()
p.engines.engine00.name = 'DM'
p.engines.engine00.numiter = 100
p.engines.engine00.fourier_relax_factor = 0.05
#p.engines.engine01 = u.Param()
#p.engines.engine01.name = 'ML'
#p.engines.engine01.numiter = 10

### UNCOMMENT FOR RECONTRUCTION ####
P = Ptycho(p,level=3)
P.save_run(kind='minimal')


