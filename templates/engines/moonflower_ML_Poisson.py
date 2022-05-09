"""
This script is a test for ptychographic reconstruction in the absence
of actual data. It uses the test Scan class
`ptypy.core.data.MoonFlowerScan` to provide "data".
"""
from ptypy.core import Ptycho
from ptypy import utils as u

import tempfile
tmpdir = tempfile.gettempdir()

if __name__ == "__main__":

    p = u.Param()

    # for verbose output
    p.verbose_level = "info"

    # set home path
    p.io = u.Param()
    p.io.home =  "/".join([tmpdir, "ptypy"])

    # saving intermediate results
    p.io.autosave = u.Param(active=False)

    # opens plotting GUI if interaction set to active)
    p.io.autoplot = u.Param(active=True)
    p.io.interaction = u.Param(active=True)

    # max 100 frames (128x128px) of diffraction data
    p.scans = u.Param()
    p.scans.MF = u.Param()
    p.scans.MF.name = 'BlockFull'
    p.scans.MF.data= u.Param()
    p.scans.MF.data.name = 'MoonFlowerScan'
    p.scans.MF.data.shape = 128
    p.scans.MF.data.num_frames = 200
    p.scans.MF.data.save = None

    # position distance in fraction of illumination frame
    p.scans.MF.data.density = 0.2
    # total number of photon in empty beam
    p.scans.MF.data.photons = 1e5
    # Gaussian FWHM of possible detector blurring
    p.scans.MF.data.psf = 0.

    # attach a reconstrucion engine
    p.engines = u.Param()
    p.engines.engine00 = u.Param()
    p.engines.engine00.name = 'DM'
    p.engines.engine00.numiter = 100
    p.engines.engine00.fourier_power_bound = 0.01

    p.engines.engine01 = u.Param()
    p.engines.engine01.name = 'ML'
    p.engines.engine01.ML_type = 'Poisson'
    p.engines.engine01.reg_del2 = False                     # Whether to use a Gaussian prior (smoothing) regularizer
    p.engines.engine01.reg_del2_amplitude = 0.1             # Amplitude of the Gaussian prior if used
    p.engines.engine01.probe_update_start = 0
    p.engines.engine01.scale_precond = True
    #p.engines.engine01.scale_probe_object = 1.
    p.engines.engine01.smooth_gradient = 20.
    p.engines.engine01.smooth_gradient_decay = 1/50.
    p.engines.engine01.floating_intensities = False
    p.engines.engine01.numiter = 100
    #p.engines.engine01.intensity_renormalization = None # turn off likelihood scaling
    p.engines.engine01.all_line_coeffs = True # use all polynomial coefficients in linesearch

    # prepare and run
    P = Ptycho(p, level=5)
