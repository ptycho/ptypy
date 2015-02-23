
import ptypy
from ptypy.core import Ptycho, data
from ptypy import utils as u
import ptypy.simulations as sim
from detector import Detector, conv

import numpy as np

DEFAULT = u.Param(
    pos_noise = 1e-10,  # (float) unformly distributed noise in xy experimental positions
    pos_scale = 0,      # (float, list) amplifier for noise. Will be extended to match number of positions. Maybe used to only put nois on individual points  
    pos_drift = 0,      # (float, list) drift or offset paramter. Noise independent drift. Will be extended like pos_scale.
    detector = 'PILATUS_300K',
    frame_size = None ,   # (None, or float, 2-tuple) final frame size when saving if None, no cropping/padding happens
    psf = 2,          # (None or float, 2-tuple, array) Parameters for gaussian convolution or convolution kernel after propagation
                        # use it for simulating partial coherence
)


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
p.model.geometry.energy = 6.2                    # (17) Energy (in keV)
p.model.geometry.lam = None                       # (18) wavelength
p.model.geometry.distance = 7                        # (19) distance from object to screen
p.model.geometry.psize = 172e-6                # (20) Pixel size in Detector plane
p.model.geometry.shape = 256                          # (22) Number of detector pixels
p.model.geometry.propagation = "farfield"           # (23) propagation type

p.model.xy = u.Param()
p.model.xy.scan_type = "round_roi"                # (25) None,'round', 'raster', 'round_roi','custom'
p.model.xy.dr = 1e-6                             # (26) round,round_roi :width of shell
p.model.xy.nr = 10                                # (27) round : number of intervals (# of shells - 1) 
p.model.xy.nth = 5                                # (28) round,round_roi: number of points in the first shell
p.model.xy.lx = 10e-6                            # (29) round_roi: Width of ROI
p.model.xy.ly = 10e-6                            # (30) round_roi: Height of ROI
p.model.xy.nx = 10                                # (31) raster scan: number of steps in x
p.model.xy.ny = 10                                # (32) raster scan: number of steps in y
p.model.xy.dx = 1e-6                              # (33) raster scan: step size (grid spacing)
p.model.xy.dy = 1e-6                              # (34) raster scan: step size (grid spacing)
p.model.xy.positions = None                       # (35) override

p.model.illumination = u.Param()
p.model.illumination.probe_type = "parallel"         # (37) 'focus' or 'parallel'
p.model.illumination.incoming = None              # (38) `None`, path to a file or any python evaluable statement yielding a 2d numpy array. If `None` defaults to array of ones
p.model.illumination.phase_noise_rms = None        # (39) phase noise amplitude on incoming wave before aperture 
p.model.illumination.phase_noise_mfs = 2.0        # (40) phase noise minimum feature size on incoming wave before aperture 
p.model.illumination.aperture_type = "circ"       # (41) type of aperture: use 
p.model.illumination.aperture_size = 2.5e-6        # (42) aperture diameter (meter)
p.model.illumination.aperture_edge = 1            # (43) edge width of aperture (pixel)
p.model.illumination.focal_dist = 0.08            # (44) distance from aperture to focus (meter)
p.model.illumination.prop_dist = 4e-3           # (45) focus: propagation distance (meter) from focus. parallel: propagation distance (meter) from aperture 
p.model.illumination.UseConjugate = False         # (46) use the conjugate of the probe instef of the probe
p.model.illumination.antialiasing = 2.0           # (47) antialiasing factor used when generating the probe
p.model.illumination.spot_size = None             # (48) focal spot diameter (meter)
p.model.illumination.photons = 1e7                # (49) number of photons in illumination
p.model.illumination.probe = None                 # (50) override if not None

p.model.sample = u.Param()
p.model.sample.source = ptypy.resources.flower_obj((512,512))
p.model.sample.offset = (0,0)                     # (53) offset between center of object array and scan pattern
p.model.sample.zoom = None                         # (54) None, scalar or 2-tupel
p.model.sample.formula = None                     # (55) chemical formula (string)
p.model.sample.density = 1.5                      # (56) density in [g/ccm]
p.model.sample.thickness = 20e-6                  # (57) max thickness of sample
p.model.sample.ref_index = None                   # (58) assigned refractive index
p.model.sample.smoothing_mfs = None               # (59) smooth the projection with gaussian kernel of with x pixels
p.model.sample.noise_rms = None                   # (60) noise applied, relative to 2*pi in phase and relative to 1 in amplitude
p.model.sample.noise_mfs = 10                     # (61) see noise rms.
p.model.sample.fill = 1.0+0.j                     # (62) if object is smaller than the objectframe, fill with fill
p.model.sample.obj = None                         # (63) override

p.model.coherence = u.Param()
p.model.coherence.Nprobe_modes = 1                # (65) 
p.model.coherence.Nobject_modes = 1               # (66) 
p.model.coherence.energies = [1.0]                # (67) 

p.model.sharing = u.Param()
p.model.sharing.scan_per_probe = 1                # (69) number of scans per object
p.model.sharing.scan_per_object = 2.              # (70) number of scans per probe
p.model.sharing.object_shared_with = None         # (71) `scan_label` of scan for the shared obejct
p.model.sharing.object_share_power = 1            # (72) contribution to the shared object
p.model.sharing.probe_shared_with = None          # (73) `scan_label` of scan for the shared probe
p.model.sharing.probe_share_power = 1             # (74) contribution to the shared probe



p.scans = u.Param()
p.scans.scan001 = u.Param()
p.scans.scan001.illumination = u.Param(aperture_type = "circ")
p.scans.scan001.data=u.Param()
p.scans.scan001.data.source = 'empty'


class SimulatedScan(data.PtyScan):
    """
    Test Ptyscan class producing a romantic ptychographic dataset of a moon
    illuminating flowers.
    """
    
    def __init__(self, pars = None,scan_pars=None,**kwargs):

        # Initialize parent class
        super(SimulatedScan, self).__init__(pars, **kwargs)
        from ptypy.core.manager import scan_DEFAULT
        
        rinfo = DEFAULT.copy()
        #rinfo.update(pars.recipe)
        
        # we will use ptypy to figure out everything
        pp = u.Param()
        pp.interaction = None
        pp.verbose_level = 1
        pp.data_type = 'single'
        
        # get scan parameters
        if scan_pars is None:
            pp.model = scan_DEFAULT.copy()
        else:
            pp.model = scan_pars.copy()
            
        # update changes specified in recipe
        pp.model.update(rinfo)

        # Create a Scan that will deliver empty diffraction patterns
        pp.scans=u.Param()
        pp.scans.sim = u.Param()
        pp.scans.sim.data=u.Param()
        pp.scans.sim.data.source ='empty'
        pp.scans.sim.data.shape = p.model.geometry.shape
        pp.scans.sim.data.auto_center = False
        
        # Now we let Ptycho sort out things
        P=Ptycho(pp,level=2)
        P.modelm.new_data()
        
        u.parallel.barrier()
        P.print_stats()
        ############################################################
        #Additional manipulation on position and sample place here##
        ############################################################        
        
        # Simulate diffraction signal
        for name,pod in P.pods.iteritems():
            if not pod.active: continue
            pod.diff += conv(u.abs2(pod.fw(pod.exit)),rinfo.psf)
        
        # Simulate detector reponse
        if p.detector is not None:
            Det = Detector(p.detector)
            save_dtype = Det.dtype
            for ID,Sdiff in P.diff.S.items():
                # get the mask storage too although their content will be overriden
                Smask = P.mask.S[ID]
                dat, mask = Det.filter(Sdiff.data)
                Sdiff.fill(dat)
                Smask.fill(mask) 
        else:
            save_dtype = None 
        
        # Create 'raw' ressource buffers. We will let the master node keep them
        # as memary may be short (Not that this is the most efficient type)
        self.P=P
        
SC = SimulatedScan(None,p.model)
