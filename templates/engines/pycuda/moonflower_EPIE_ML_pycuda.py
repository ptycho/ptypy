"""
This script is a test for ptychographic reconstruction in the absence
of actual data. It uses the test Scan class
`ptypy.core.data.MoonFlowerScan` to provide "data".
"""
from ptypy.core import Ptycho
from ptypy import utils as u
import ptypy
ptypy.load_gpu_engines(arch="cuda")

import tempfile
tmpdir = tempfile.gettempdir()

p = u.Param()

# for verbose output
p.verbose_level = "info"

# set home path
p.io = u.Param()
p.io.home =  "/".join([tmpdir, "ptypy"])
p.io.autosave = u.Param(active=False)
p.io.autoplot = u.Param(active=False)
p.io.interaction = u.Param(active=False)

# max 200 frames (128x128px) of diffraction data
p.scans = u.Param()
p.scans.MF = u.Param()
# now you have to specify which ScanModel to use with scans.XX.name,
# just as you have to give 'name' for engines and PtyScan subclasses.
p.scans.MF.name = 'GradFull'
p.scans.MF.data= u.Param()
p.scans.MF.data.name = 'MoonFlowerScan'
p.scans.MF.data.shape = 128
p.scans.MF.data.num_frames = 200
p.scans.MF.data.save = None

# position distance in fraction of illumination frame
p.scans.MF.data.density = 0.2
# total number of photon in empty beam
p.scans.MF.data.photons = 1e8
# Gaussian FWHM of possible detector blurring
p.scans.MF.data.psf = 0.

p.scans.MF.illumination=u.Param()
p.scans.MF.illumination.diversity = None

p.scans.MF.coherence=u.Param()
p.scans.MF.coherence.num_probe_modes = 1
p.scans.MF.coherence.num_object_modes = 1

# attach a reconstrucion engine
p.engines = u.Param()
p.engines.engine00 = u.Param()
p.engines.engine00.name = 'EPIE_pycuda'
p.engines.engine00.numiter = 200
p.engines.engine00.probe_center_tol = None
p.engines.engine00.compute_log_likelihood = True
p.engines.engine00.object_norm_is_global = True
p.engines.engine00.alpha = 1
p.engines.engine00.beta = 1
p.engines.engine00.probe_update_start = 2

p.engines.engine01 = u.Param()
p.engines.engine01.name = 'ML_pycuda'
p.engines.engine01.ML_type = 'Gaussian'
p.engines.engine01.reg_del2 = True 
p.engines.engine01.reg_del2_amplitude = 1.
p.engines.engine01.scale_precond = True
p.engines.engine01.scale_probe_object = 1.
p.engines.engine01.numiter = 100

# prepare and run
if __name__ == "__main__":
    P = Ptycho(p,level=5)