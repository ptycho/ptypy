from ptypy.core import Ptycho
from ptypy import utils as u
from ptypy import load_gpu_engines, load_ptyscan_module

# CuPy engines and HDF5 data loader
load_gpu_engines("cupy")
load_ptyscan_module("hdf5_loader")

# Path to raw data
# Available to download from https://zenodo.org/records/11501765
input_path = "/dls/p99/data/2023/cm33854-5/processing/benedikt/ptypy-benchmark/"
data_path  = f"{input_path}/i14-logo-128.h5"
initial_probe_path = f"{input_path}/scan_103937.ptyr"

# Path to output
output_path = "/dls/science/users/iat69393/ptypy-paper/output/"

# Switch for position refinement
do_position_refinement = True

# Parameter Tree
p = u.Param()
p.frames_per_block = 10000
p.verbose_level = "info"
p.data_type = "single"
p.run = f"i14_logo_posref_{do_position_refinement}".lower()
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
# as this parameter will be deprecated soon
p.io.benchmark = "all"

# Definition of scan
p.scans = u.Param()
p.scans.scan_00 = u.Param()
p.scans.scan_00.name = 'BlockFull'

# Initial model for illumination
p.scans.scan_00.illumination = u.Param()
p.scans.scan_00.illumination.model = "recon"
p.scans.scan_00.illumination.recon = u.Param()
p.scans.scan_00.illumination.recon.rfile = initial_probe_path
p.scans.scan_00.illumination.photons = None
p.scans.scan_00.illumination.aperture = u.Param()
p.scans.scan_00.illumination.aperture.form = None
p.scans.scan_00.illumination.diversity = u.Param()
p.scans.scan_00.illumination.diversity.power = 0.1
p.scans.scan_00.illumination.diversity.noise = [0.5,0.2]

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
p.scans.scan_00.data.name = 'Hdf5LoaderFast'

# Diffraction data
p.scans.scan_00.data.intensities = u.Param()
p.scans.scan_00.data.intensities.file = data_path
p.scans.scan_00.data.intensities.key = "data"

# Positions rescaled to SI units
p.scans.scan_00.data.positions = u.Param()
p.scans.scan_00.data.positions.file = data_path
p.scans.scan_00.data.positions.slow_key = "posy"
p.scans.scan_00.data.positions.slow_multiplier = 1e-3
p.scans.scan_00.data.positions.fast_key = "posx"
p.scans.scan_00.data.positions.fast_multiplier = 1e-3

# Photon energy loaded from file in keV
p.scans.scan_00.data.recorded_energy = u.Param()
p.scans.scan_00.data.recorded_energy.key = "energy"
p.scans.scan_00.data.recorded_energy.file = data_path
p.scans.scan_00.data.recorded_energy.multiplier = 1

# Detector distance loaded from file in SI units
p.scans.scan_00.data.recorded_distance = u.Param()
p.scans.scan_00.data.recorded_distance.key = "distance"
p.scans.scan_00.data.recorded_distance.file = data_path
p.scans.scan_00.data.recorded_distance.multiplier = 0.001

# Bad-pixel mask (1 is valid, 0 is bad)
p.scans.scan_00.data.mask = u.Param()
p.scans.scan_00.data.mask.file = data_path
p.scans.scan_00.data.mask.key = "mask"

# Geometry and other data properties
p.scans.scan_00.data.orientation = 0
p.scans.scan_00.data.psize = 55e-6
p.scans.scan_00.data.shape = (128,128)
p.scans.scan_00.data.save = None
p.scans.scan_00.data.load_parallel = "data"

# Reconstruction engine (DM)
p.engines = u.Param()
p.engines.engine = u.Param()
p.engines.engine.name = "DM_cupy"
p.engines.engine.numiter = 300
p.engines.engine.numiter_contiguous = 100
p.engines.engine.alpha = 0.99
p.engines.engine.probe_support = None
p.engines.engine.probe_fourier_support = None
p.engines.engine.overlap_converge_factor = 0.001
p.engines.engine.probe_update_start = 0
p.engines.engine.update_object_first = True
p.engines.engine.obj_smooth_std = 20
p.engines.engine.probe_inertia = 0.001
p.engines.engine.object_inertia = 0.001
p.engines.engine.fourier_power_bound = 0.25
p.engines.engine.record_local_error = False
p.engines.engine.fft_lib = "cuda"

# Perform position refinement if required
if do_position_refinement:
    p.engines.engine.position_refinement = u.Param()
    p.engines.engine.position_refinement.start = 100
    p.engines.engine.position_refinement.stop = 200
    p.engines.engine.position_refinement.interval = 10
    p.engines.engine.position_refinement.nshifts = 8
    p.engines.engine.position_refinement.amplitude = 50.0e-9
    p.engines.engine.position_refinement.max_shift = 100.0e-9
    p.engines.engine.position_refinement.record = False

# Execute reconstruction
P = Ptycho(p,level=5)
