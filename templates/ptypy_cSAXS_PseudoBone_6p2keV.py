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
import ptypy.simulations as sim
import numpy as np
p = u.Param()

### PTYCHO PARAMETERS
p.verbose_level = 3                               # (00) Verbosity level

p.data_type = "single"                            # (01) Reconstruction floatine number precision
p.autosave = None
p.data_type = "single"                            # (01) Reconstruction floatine number precision
p.run = None
p.paths = u.Param()
p.paths.home = "/tmp/ptypy/"             # (03) Relative base path for all other paths
p.paths.run = None

p.scan = u.Param()
p.scan.geometry = u.Param()
p.scan.geometry.energy = 6.2                    # (17) Energy (in keV)
p.scan.geometry.lam = None                       # (18) wavelength
p.scan.geometry.distance = 7                        # (19) distance from object to screen
p.scan.geometry.psize = 172e-6                # (20) Pixel size in Detector plane
p.scan.geometry.shape = 256                          # (22) Number of detector pixels
p.scan.geometry.propagation = "farfield"           # (23) propagation type

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
p.scan.coherence.Nprobe_modes = 1                # (65) 
p.scan.coherence.Nobject_modes = 1               # (66) 
p.scan.coherence.energies = [1.0]                # (67) 

p.scan.sharing = u.Param()
p.scan.sharing.object_shared_with = None         # (71) `scan_label` of scan for the shared obejct
p.scan.sharing.object_share_power = 1            # (72) contribution to the shared object
p.scan.sharing.probe_shared_with = None          # (73) `scan_label` of scan for the shared probe
p.scan.sharing.probe_share_power = 1             # (74) contribution to the shared probe

sim = u.Param()
sim.xy = u.Param()
sim.xy.model = "round"                # (25) None,'round', 'raster', 'round_roi','custom'
sim.xy.spacing = 1e-6                             # (26) round,round_roi :width of shell
sim.xy.steps = 10                                # (27) round : number of intervals (# of shells - 1) 
sim.xy.extent = 10e-6                            # (29) round_roi: Width of ROI

sim.illumination = u.Param()
sim.illumination.model = None
sim.illumination.photons = 1e8                # (49) number of photons in illumination
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
sim.sample.model = 255-u.imload('../resources/tree.bmp').astype(float).mean(-1)# (52) 'diffraction', None, path to a file or nd-array 
sim.sample.process = u.Param()
sim.sample.process.offset = (100,400)                     # (53) offset between center of object array and scan pattern
sim.sample.process.zoom = 1.0                         # (54) None, scalar or 2-tupel
sim.sample.process.formula = "Ca"                     # (55) chemical formula (string)
sim.sample.process.density = 1.5                     # (56) density in [g/ccm]
sim.sample.process.thickness = 20e-6                 # (57) max thickness of sample
sim.sample.process.ref_index = None                   # (58) assigned refractive index
sim.sample.process.smoothing = None                  # (59) smooth the projection with gaussian kernel of with x pixels
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
p.scans.RectSim.sharing= u.Param(object_shared_with = 'CircSim')
p.scans.RectSim.data=u.Param()
p.scans.RectSim.data.source = 'sim' 
p.scans.RectSim.data.recipe = sim.copy(depth=4)
p.scans.RectSim.data.recipe.illumination.aperture.form = "rect"
p.scans.RectSim.data.save = None


p.engine = u.Param()
p.engine.common = u.Param()
p.engine.common.numiter = 100                    # (85) Total number of iterations
p.engine.common.numiter_contiguous = 1            # (86) Number of iterations to be executed in one go
p.engine.common.probe_support = 0.7               # (87) Fraction of valid probe area (circular) in probe frame
p.engine.common.probe_inertia = 0.01             # (88) Probe fraction kept from iteration to iteration
p.engine.common.object_inertia = 0.1              # (89) Object fraction kept from iteration to iteration
p.engine.common.obj_smooth_std = 20               # (90) Gaussian smoothing (pixel) of kept object fraction
p.engine.common.clip_object = None                # (91) Clip object amplitude into this intrervall
p.engine.common.probe_update_start = 2                # (95) Number of iterations before probe update starts

p.engine.DM = u.Param()
p.engine.DM.name = "DM"                           # (93) abbreviation of algorithm
p.engine.DM.alpha = 1                             # (94) Parameters that makes the difference
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
p.engines.engine01 = u.Param()
p.engines.engine01.name = 'ML'
p.engines.engine01.numiter = 100
p.simulation = u.Param()
p.simulation.position_noise = 1e-10               # (104) 
p.simulation.detector = 'PILATUS_300K' #None #


u.verbose.set_level(3)
P = Ptycho(p,level=5)
#if u.parallel.master:
#    P.plot_overview()
#u.pause(100.)

