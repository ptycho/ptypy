"""
Reconstruction script for ptypy version 0.2.0

-- Additional parameters can be manually added to the tree --
--    Please refer to documentation for further details    --
"""

import numpy as np
import ptypy
from ptypy.core import Ptycho
from ptypy import utils as u

base_path = '/data/2016_07_ID16A_NFP/'

#### ID16A Recipe #####

r = u.Param()
r.base_path = base_path
r.scan_label = 'fibbed_chip_5nm_94d_17p_6px_singlenfp1'
r.flat_division = False
r.dark_subtraction = True
r.date = '2016-08-02'
r.use_h5 = False
#r.mask_file = [1,2,3]
#r.motors = ['spy', 'spz']


#### Ptypy Parameter Tree ######################################################


## General parameter container
p = u.Param()

p.verbose_level = 3						## (00) Verbosity level
p.data_type = "single"					## (01) Reconstruction floating number precision
p.run = None							## (02) Reconstruction identifier
p.dry_run = False						## (03) Dry run switch
p.ipython_kernel = False				## New feature?


## (04) Global parameters for I/O
p.io = u.Param()

p.io.home = "./" 						## (05) Base directory for all I/O
p.io.rfile = "recons/%(run)s/%(run)s_%(engine)s_%(iterations)04d.ptyr" ## (06) Reconstruction file name (or format string)


## (07) Auto-save options
p.io.autosave = u.Param()

p.io.autosave.active = False 			## (08) Activation switch
p.io.autosave.interval = 100			## (09) Auto-save interval
p.io.autosave.rfile = "dumps/%(run)s/%(run)s_%(engine)s_%(iterations)04d.ptyr" ## (10) Auto-save file name (or format string)


## (11) Server / Client parameters
p.io.interaction = u.Param()

p.io.interaction.active = True			## (12) Activation switch
p.io.interaction.address = "tcp://127.0.0.1" 	## (13) The address the server is listening to.
p.io.interaction.port = 5560 			## (14) The port the server is listening to.
p.io.interaction.connections = 10		## (15) Number of concurrent connections on the server


## (16) Plotting client parameters
p.io.autoplot = u.Param()

p.io.autoplot.imfile = "plots/%(run)s/%(run)s_%(engine)s_%(iterations)04d.png" ## (17) Plot images file name (or format string)
p.io.autoplot.interval = 1				## (18) Number of iterations between plot updates
p.io.autoplot.threaded = False 			## (19) Live plotting switch
p.io.autoplot.layout = u.Param()		## (20) Options for default plotter or template name
p.io.autoplot.dump = False				## (21) Switch to dump plots as image files
p.io.autoplot.make_movie = False		## (22) Produce reconstruction movie after the reconstruction.


## (23) Scan parameters
p.scan = u.Param()

## (51) Physical parameters
p.scan.geometry = u.Param()

p.scan.geometry.precedence = None		## (52) Where geometry parameters take precence over others
p.scan.geometry.energy = 17.05			## (53) Energy (in keV)
p.scan.geometry.lam = None				## (54) Wavelength
p.scan.geometry.distance = 0.002		## (55) Distance from object to detector
p.scan.geometry.psize = 10e-9			## (56) Pixel size in Detector plane
p.scan.geometry.resolution = None		## (57) Pixel size in Sample plane
p.scan.geometry.propagation = "nearfield" 	## (58) Propagation type


## (67) Illumination model (probe)
p.scan.illumination = u.Param()

p.scan.illumination.model =  None 		## (68) Type of illumination model
p.scan.illumination.recon = None		## (70) Parameters to load from previous reconstruction
#p.scan.illumination.recon.rfile = "\*.ptyr" 	## (71) Path to a ``.ptyr`` compatible file
p.scan.illumination.stxm = None			## (74) Parameters to initialize illumination from diffraction data
#p.scan.illumination.stxm.label = None 	## (75) Scan label of diffraction that is to be used for probe estimate
p.scan.illumination.aperture = u.Param()		## (76) Beam aperture parameters
p.scan.illumination.aperture.form = 'rect'		## (77) One of None, 'rect' or 'circ'
p.scan.illumination.aperture.size = 560e-6		## (79) Aperture width or diameter
#p.scan.illumination.propagation = u.Param()		## (83) Parameters for propagation after aperture plane
#p.scan.illumination.propagation.parallel = 1.206 	## (84) Parallel propagation distance
#p.scan.illumination.propagation.focussed = 0.343	## (85) Propagation distance from aperture to focus
p.scan.illumination.diversity = None 			## (88) Probe mode(s) diversity parameters


## (92) Initial object modelization parameters
p.scan.sample = u.Param()

p.scan.sample.model = None 			## (93) Type of initial object model
p.scan.sample.fill = 1				## (94) Default fill value
p.scan.sample.recon = None			## (95) Parameters to load from previous reconstruction
#p.scan.sample.process = u.Param()			## (101) Model processing parameters
#p.scan.sample.process.offset = (0,0)		## (102) Offset between center of object array and scan pattern
#p.scan.sample.process.zoom = 1.0			## (103) Zoom value for object simulation
#p.scan.sample.process.formula = None		## (104) Chemical formula
#p.scan.sample.process.density = 19.3		## (105) Density in [g/ccm]
#p.scan.sample.process.thickness = 2000e-9 	## (106) Maximum thickness of sample
#p.scan.sample.process.ref_index = None		## (107) Assigned refractive index
#p.scan.sample.process.smoothing = None		## (108) Smoothing scale
#p.scan.sample.diversity = u.Param()		## (109) Probe mode(s) diversity parameters
#p.scan.sample.diversity.noise = (None, 10)	## (110) Noise in the generated modes of the illumination

## (113) Coherence parameters
p.scan.coherence = u.Param()

p.scan.coherence.num_probe_modes = 1 		## (114) Number of probe modes
p.scan.coherence.num_object_modes = 1 		## (115) Number of object modes


## (119) Param container for instances of `scan` parameters
p.scans = u.Param()
p.scans.ID16A = u.Param()

p.scans.ID16A.if_conflict_use_meta = True 	## (25) Give priority to metadata relative to input parameters


## (26) Data preparation parameters
p.scans.ID16A.data = u.Param()

p.scans.ID16A.data.recipe = r 				## (27) Data preparation recipe container
p.scans.ID16A.data.source = 'id16a_nfp'		## (28) Describes where to get the data from.
p.scans.ID16A.data.dfile = 'prepdata/fibbed_chip.ptyd' ## (29) Prepared data file path
p.scans.ID16A.data.shape = 2048				## (31) Shape of the region of interest cropped from the raw data
p.scans.ID16A.data.save = 'append'			## (32) Saving mode
#p.scans.ID16A.data.center = (1280,1080)	## (33) Center (pixel) of the optical axes in raw data
p.scans.ID16A.data.psize = 5e-9				## (34) Detector pixel size before rebinning
p.scans.ID16A.data.distance = 0.002			## (35) Sample-to-detector distance
p.scans.ID16A.data.rebin = 2				## (36) Rebinning factor
p.scans.ID16A.data.orientation = 6			## (37) Data frame orientation
p.scans.ID16A.data.energy = 17.05			## (38) Photon energy of the incident radiation
p.scans.ID16A.data.load_parallel = "data" 	## (43) Determines what will be loaded in parallel

## (46) Scan sharing parameters
p.scans.ID16A.sharing = u.Param()

p.scans.ID16A.sharing.object_share_with = None	## (47) Label or index of scan to share object with
p.scans.ID16A.sharing.object_share_power = 1    ## (48) Relative power for object sharing
p.scans.ID16A.sharing.probe_share_with = None	## (49) Label or index of scan to share probe with
p.scans.ID16A.sharing.probe_share_power = 1     ## (50) Relative power for probe sharing


## (121) Reconstruction engine parameters
p.engine = u.Param()


## (122) Parameters common to all engines
p.engine.common = u.Param()

p.engine.common.name = "DM" 				## (123) Name of engine.
p.engine.common.numiter = 500				## (124) Total number of iterations
p.engine.common.numiter_contiguous = 1 		## (125) Number of iterations without interruption
p.engine.common.probe_support = None 		## (126) Fraction of valid probe area (circular) in probe frame
p.engine.common.clip_object = None			## (127) Clip object amplitude into this intrervall


## (128) Parameters for Difference map engine
p.engine.DM = u.Param()

p.engine.DM.alpha = 1 						## (129) Difference map parameter
p.engine.DM.probe_update_start = 1 			## (130) Number of iterations before probe update starts
p.engine.DM.update_object_first = True 		## (131) If False update object before probe
p.engine.DM.overlap_converge_factor = 0.05	## (132) Threshold for interruption of the inner overlap loop
p.engine.DM.overlap_max_iterations = 100	## (133) Maximum of iterations for the overlap constraint inner loop
p.engine.DM.probe_inertia = 0.001 			## (134) Weight of the current probe estimate in the update
p.engine.DM.object_inertia = 0.1			## (135) Weight of the current object in the update
p.engine.DM.fourier_relax_factor = 0.05		## (136) If rms error of model vs diffraction data is smaller than this fraction, Fourier constraint is met
p.engine.DM.obj_smooth_std = 20				## (137) Gaussian smoothing (pixel) of the current object prior to update


## (138) Maximum Likelihood parameters
p.engine.ML = u.Param()

p.engine.ML.type = "gaussian" 				## (139) Likelihood model. One of 'gaussian', 'poisson' or 'euclid'
p.engine.ML.floating_intensities = False	## (140) If True, allow for adaptative rescaling of the diffraction pattern intensities (to correct for incident beam intensity fluctuations).
#p.engine.ML.intensity_renormalization = 1	## (141) A rescaling of the intensity so they can be interpreted as Poisson counts.
#p.engine.ML.reg_del2 = True				## (142) Whether to use a Gaussian prior (smoothing) regularizer.
#p.engine.ML.reg_del2_amplitude = 0.01 		## (143) Amplitude of the Gaussian prior if used.
#p.engine.ML.smooth_gradient = 0			## (144) Smoothing preconditioner. If 0, not used, if > 0 gaussian filter if < 0 Hann window.
#p.engine.ML.scale_precond = False			## (145) Whether to use the object/probe scaling preconditioner.
#p.engine.ML.scale_probe_object = 1. 		## (146) Relative scale of probe to object.
#p.engine.ML.probe_update_start = 0			## (147) Number of iterations before probe update starts


## (148) Container for instances of "engine" parameters
p.engines = u.Param()

## (149) First engines entry
p.engines.engine_00 = u.Param()
p.engines.engine_00.name = 'DM'
p.engines.engine_00.numiter = 500
p.engines.engine_01 = u.Param()
p.engines.engine_01.name = 'ML'
p.engines.engine_01.numiter = 1

P = Ptycho(p, level=4)

P.run()
P.finalize()
