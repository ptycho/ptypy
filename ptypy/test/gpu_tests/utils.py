'''    
Created on 4 Jan 2018

@author: clb02321
'''
import unittest
from ptypy.core import Ptycho
from ptypy import utils as u

def print_array_info(a, name):
    print("{}: {}, {}".format(name, a.shape, a.dtype))

def get_ptycho_instance(label=None, num_modes=1, frame_size=64, scan_length=8):
    '''
    new ptypy probably has a better way of doing this.
    '''
    p = u.Param()
    p.verbose_level = 0
    p.data_type = "single"
    p.run = label
    p.io = u.Param()
    p.io.home = "/tmp/ptypy/"
    p.io.interaction = u.Param(active=False)
    p.io.autoplot = u.Param(active=False)
    p.scans = u.Param()
    p.scans.MF = u.Param()
    p.scans.MF.name = 'Full'
    p.scans.MF.propagation = 'farfield'
    p.scans.MF.data = u.Param()
    p.scans.MF.data.name = 'MoonFlowerScan'
    p.scans.MF.data.positions_theory = None
    p.scans.MF.data.auto_center = None
    p.scans.MF.data.min_frames = 1
    p.scans.MF.data.orientation = None
    p.scans.MF.data.num_frames =scan_length
    p.scans.MF.data.energy = 6.2
    p.scans.MF.data.shape = frame_size
    p.scans.MF.data.chunk_format = '.chunk%02d'
    p.scans.MF.data.rebin = None
    p.scans.MF.data.experimentID = None
    p.scans.MF.data.label = None
    p.scans.MF.data.version = 0.1
    p.scans.MF.data.dfile = None
    p.scans.MF.data.psize = 0.000172
    p.scans.MF.data.load_parallel = None
    p.scans.MF.data.distance = 7.0
    p.scans.MF.data.save = None
    p.scans.MF.data.center = 'fftshift'
    p.scans.MF.data.photons = 100000000.0
    p.scans.MF.data.psf = 0.0
    p.scans.MF.data.add_poisson_noise = False
    p.scans.MF.data.density = 0.2
    p.scans.MF.illumination = u.Param()
    p.scans.MF.illumination.model = None
    p.scans.MF.illumination.aperture = u.Param()
    p.scans.MF.illumination.aperture.diffuser = None
    p.scans.MF.illumination.aperture.form = "circ"
    p.scans.MF.illumination.aperture.size = 3e-6
    p.scans.MF.illumination.aperture.edge = 10
    p.scans.MF.coherence = u.Param()
    p.scans.MF.coherence.num_probe_modes = num_modes
    p.engines = u.Param()
    p.engines.DM = u.Param()
    p.engines.DM.name = 'DM'
    p.engines.DM.numiter = 5
    p.engines.DM.alpha = 1
    p.engines.DM.probe_update_start = 2
    p.engines.DM.overlap_converge_factor = 0.05
    p.engines.DM.overlap_max_iterations = 10
    p.engines.DM.probe_inertia = 1e-3
    p.engines.DM.object_inertia = 0.1
    p.engines.DM.fourier_relax_factor = 0.01
    p.engines.DM.obj_smooth_std = 20

    P = Ptycho(p, level=4)
    P.di = P.diff
    P.ma = P.mask
    P.ex = P.exit
    P.pr = P.probe
    P.ob = P.obj
    return P

