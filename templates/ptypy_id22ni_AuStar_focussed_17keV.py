"""
import ptypy
from matplotlib import pyplot as plt
from ptypy import utils as u
from ptypy import *
import numpy as np
from pyE17 import utils as U
import os
import subprocess
import sys
import time
"""
import ptypy
from ptypy.core import Ptycho
from ptypy import utils as u
import numpy as np
p = u.Param()

### PTYCHO PARAMETERS
p.verbose_level = 3                               # (00) Verbosity level

p.interaction = u.Param()

p.data_type = "single"                            # (01) Reconstruction floatine number precision
p.run = None
p.autosave = None
p.paths = u.Param()
p.paths.home = "/tmp/ptypy/"             # (03) Relative base path for all other paths
p.paths.run = None


sim = u.Param()
sim.xy = u.Param()
sim.xy.scan_type = "round_roi"                # (25) None,'round', 'raster', 'round_roi','custom'
sim.xy.dr = 250e-9                             # (26) round,round_roi :width of shell
sim.xy.nr = 10                                # (27) round : number of intervals (# of shells - 1) 
sim.xy.nth = 5                                # (28) round,round_roi: number of points in the first shell
sim.xy.lx = 4e-6                            # (29) round_roi: Width of ROI
sim.xy.ly = 4e-6                            # (30) round_roi: Height of ROI
sim.xy.nx = 10                                # (31) raster scan: number of steps in x
sim.xy.ny = 10                                # (32) raster scan: number of steps in y
sim.xy.dx = 1e-6                              # (33) raster scan: step size (grid spacing)
sim.xy.dy = 1e-6                              # (34) raster scan: step size (grid spacing)
sim.xy.positions = None                       # (35) override

sim.illumination = u.Param()
sim.illumination.model = None
sim.illumination.photons = 3e8                # (49) number of photons in illumination
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
sim.sample.model = u.xradia_star((1000,1000),minfeature=3,contrast=0.0)# (52) 'diffraction', None, path to a file or nd-array 
sim.sample.process = u.Param()
sim.sample.process.offset = (200,200)                     # (53) offset between center of object array and scan pattern
sim.sample.process.zoom = 1.0                         # (54) None, scalar or 2-tupel
sim.sample.process.formula = "Au"                     # (55) chemical formula (string)
sim.sample.process.density = 19.3                     # (56) density in [g/ccm]
sim.sample.process.thickness = 2000e-9                 # (57) max thickness of sample
sim.sample.process.ref_index = None                   # (58) assigned refractive index
sim.sample.process.smoothing = None                  # (59) smooth the projection with gaussian kernel of with x pixels
sim.sample.fill = 1.0+0.j                     # (62) if object is smaller than the objectframe, fill with fill

#sim.detector = 'FRELON_TAPER'
sim.detector = 'GenericCCD32bit'
sim.verbose_level = 1
sim.coherence = u.Param()
sim.coherence.Nprobe_modes = 1
sim.psf = 1.

p.scan = sim.copy(depth=4)

p.scan.geometry = u.Param()
p.scan.geometry.energy = 17.0                    # (17) Energy (in keV)
p.scan.geometry.lam = None                       # (18) wavelength
p.scan.geometry.distance = 2.886                        # (19) distance from object to screen
p.scan.geometry.psize = 51e-6                # (20) Pixel size in Detector plane
p.scan.geometry.shape = 256                          # (22) Number of detector pixels
p.scan.geometry.propagation = "farfield"           # (23) propagation type


p.scan.coherence = u.Param()
p.scan.coherence.Nprobe_modes = 4                # (65) 
p.scan.coherence.Nobject_modes = 1               # (66) 
p.scan.coherence.energies = [1.0]                # (67) 


p.scan.sample.model = 'stxm'
p.scan.sample.process =  None
p.scan.illumination.aperture.form = 'circ'
p.scan.illumination.propagation.focussed = 0.06
p.scan.illumination.diversity = u.Param()
p.scan.illumination.diversity.power = 0.1
p.scan.illumination.diversity.noise = (np.pi,2.0)

p.scans = u.Param()
p.scans.sim = u.Param()
p.scans.sim.data=u.Param()
p.scans.sim.data.source = 'sim' 
p.scans.sim.data.recipe = sim
p.scans.sim.data.save = None


p.engine = u.Param()
p.engine.common = u.Param()
p.engine.common.numiter = 100                    # (85) Total number of iterations
p.engine.common.numiter_contiguous = 1            # (86) Number of iterations to be executed in one go
p.engine.common.probe_support = 0.7               # (87) Fraction of valid probe area (circular) in probe frame
p.engine.common.probe_inertia = 0.01             # (88) Probe fraction kept from iteration to iteration
p.engine.common.object_inertia = 0.1              # (89) Object fraction kept from iteration to iteration
p.engine.common.obj_smooth_std = 5               # (90) Gaussian smoothing (pixel) of kept object fraction
p.engine.common.clip_object = [0,1.]                # (91) Clip object amplitude into this intrervall

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
p.engines.engine00.numiter = 150
p.engines.engine00.fourier_relax_factor = 0.05

u.verbose.set_level(3)
P = Ptycho(p,level=3)
if u.parallel.master:
    P.plot_overview()
u.pause(100.)
#P.save_run(kind='minimal')



