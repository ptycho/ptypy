from ptypy.core import Ptycho
from ptypy import utils as u
from ptypy import load_gpu_engines, load_ptyscan_module

# Cupy engins and HDF5 data loader
load_gpu_engines("cupy")
load_ptyscan_module("hdf5_loader")

# Path to raw data
# Available to download from https://zenodo.org/records/11501765
input_path = "/dls/science/groups/imaging/ptypy_tutorials/dls_i13_butterfly/"
data_path = f"{input_path}/raw/excalibur_306517_vds.h5"
mask_path = f"{input_path}/processing/masks/excalibur_512x512.h5"
position_path = f"{input_path}/processing/pos/306517.h5"
initial_probe_path = f"{input_path}/processing/ptypy/testing/303079_ML_pycuda_1500.ptyr"

# Path to output
output_path = f"/dls/science/users/iat69393/ptypy-paper/output/"

# Read initial probe from a previous reconstruction and extract main mode
probe = u.load_from_ptyr(initial_probe_path, what="probe")
probe_main = u.ortho(probe)[1][0]

# Switch reconstruction engine
engine = "sdr" # select from epie, sdr or ml

# Parameter Tree
p = u.Param()
p.verbose_level = "info"
p.data_type = "single"
p.run = "i13_butterfly"
p.dry_run = False

# Input / Output
p.io = u.Param()
p.io.home = f"{output_path}"
p.io.rfile = f"{output_path}/%(run)s_%(engine)s_%(iterations)04d.ptyr" 
p.io.autosave = u.Param(active=False)
p.io.autoplot = u.Param(active=False)
p.io.interaction = u.Param(active=False)
p.io.interaction.server = u.Param(active=False)

# TODO: remove this before publication
# as this parameter will be deprecated soonif args.benchmark:
p.io.benchmark = "all"

# Definition of scan
p.scans = u.Param()
p.scans.scan_00 = u.Param()

# Scan model and frames per block
if engine in ["epie", "ml"]:
    p.frames_per_block = 1260 # full data size (single block)
    p.scans.scan_00.name = 'BlockGradFull'    
if engine == "sdr":
    p.scans.scan_00.name = 'BlockFull'
    p.frames_per_block = 264

# Initial model for illumination
# p.scans.scan_00.illumination = u.Param()
# p.scans.scan_00.illumination.model = None
# p.scans.scan_00.illumination.photons = None
# p.scans.scan_00.illumination.aperture = u.Param()
# p.scans.scan_00.illumination.aperture.form = "circ"
# p.scans.scan_00.illumination.aperture.size = 400e-6
# p.scans.scan_00.illumination.propagation = u.Param()
# p.scans.scan_00.illumination.propagation.focussed = 0.469484
# p.scans.scan_00.illumination.propagation.parallel = 18.5e-3
# p.scans.scan_00.illumination.propagation.antialiasing = 1
    

p.scans.scan_00.illumination = u.Param()
p.scans.scan_00.illumination.model = probe_main
p.scans.scan_00.illumination.photons = None
p.scans.scan_00.illumination.aperture = u.Param()
p.scans.scan_00.illumination.aperture.form = None
p.scans.scan_00.illumination.diversity = u.Param()
p.scans.scan_00.illumination.diversity.power = 0.1
p.scans.scan_00.illumination.diversity.noise = [0.5,1.0]

# Initial model for object
p.scans.scan_00.sample = u.Param()
p.scans.scan_00.sample.model = None
p.scans.scan_00.sample.diversity = None
p.scans.scan_00.sample.process = None

# Coherence modes
p.scans.scan_00.coherence = u.Param()
p.scans.scan_00.coherence.num_probe_modes = 1
p.scans.scan_00.coherence.num_object_modes = 1

# Data loading
p.scans.scan_00.data = u.Param()
p.scans.scan_00.data.name = 'Hdf5Loader'

# Diffraction data
p.scans.scan_00.data.intensities = u.Param()
p.scans.scan_00.data.intensities.file = data_path
p.scans.scan_00.data.intensities.key = "data"

# Positions rescaled to SI unites
p.scans.scan_00.data.positions = u.Param()
p.scans.scan_00.data.positions.file = position_path
p.scans.scan_00.data.positions.slow_key = "slow"
p.scans.scan_00.data.positions.slow_multiplier = 1e-6
p.scans.scan_00.data.positions.fast_key = "fast"
p.scans.scan_00.data.positions.fast_multiplier = 1e-6

# Bad-pixel mask (1 is valid, 0 is bad)
p.scans.scan_00.data.mask = u.Param()
p.scans.scan_00.data.mask.file = mask_path
p.scans.scan_00.data.mask.key = "data"

# Geometry and other data properties
p.scans.scan_00.data.orientation = 0
p.scans.scan_00.data.distance = 14.65
p.scans.scan_00.data.energy = 9.7
p.scans.scan_00.data.psize = 55e-6
p.scans.scan_00.data.auto_center = False
p.scans.scan_00.data.center = (903.5, 1018.)
p.scans.scan_00.data.shape = (512,512)
p.scans.scan_00.data.save = None

# Reconstruction engines
p.engines = u.Param()

# ML reconstruction engine
if engine == "ml":
    p.engines.engine_ml = u.Param()
    p.engines.engine_ml.name = 'ML_cupy'
    p.engines.engine_ml.numiter = 500
    p.engines.engine_ml.numiter_contiguous = 10
    p.engines.engine_ml.ML_type = 'Gaussian'
    p.engines.engine_ml.floating_intensities = False
    p.engines.engine_ml.probe_support = None
    p.engines.engine_ml.reg_del2 = True
    p.engines.engine_ml.reg_del2_amplitude = .01
    p.engines.engine_ml.scale_precond = False
    p.engines.engine_ml.scale_probe_object = 1.
    p.engines.engine_ml.wavefield_precond = False
    p.engines.engine_ml.probe_update_start = 0
    p.engines.engine_ml.fft_lib = "cuda"

# ePIE reconstruction engine
if engine == "epie":
    p.engines.engine_ep = u.Param()
    p.engines.engine_ep.name = "EPIE_cupy"
    p.engines.engine_ep.numiter = 500
    p.engines.engine_ep.numiter_contiguous = 10
    p.engines.engine_ep.alpha = 0.9
    p.engines.engine_ep.beta = 0.1
    p.engines.engine_ep.object_norm_is_global = True
    p.engines.engine_ep.compute_log_likelihood = True
    p.engines.engine_ep.compute_exit_error = False
    p.engines.engine_ep.fft_lib = "cuda"

# sDR reconstruction engine
if engine == "sdr":
    p.engines.engine_dr = u.Param()
    p.engines.engine_dr.name = "SDR_cupy"
    p.engines.engine_dr.numiter = 500
    p.engines.engine_dr.numiter_contiguous = 10
    p.engines.engine_dr.sigma = 0.5
    p.engines.engine_dr.tau = 0.1
    p.engines.engine_dr.compute_log_likelihood = True
    p.engines.engine_dr.compute_exit_error = False
    p.engines.engine_dr.fft_lib = "cuda"

# Execute reconstruction
P = Ptycho(p,level=5)
