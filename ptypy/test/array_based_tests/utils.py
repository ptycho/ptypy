'''    
Created on 4 Jan 2018

@author: clb02321
'''
import unittest
from ptypy.core import Ptycho
from ptypy import utils as u


def get_ptycho_instance(label=None, num_modes=1, size=64, length=8):
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
    p.scans.MF.data.num_frames =length
    p.scans.MF.data.energy = 6.2
    p.scans.MF.data.shape = size
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
    P = Ptycho(p, level=4)
    P.di = P.diff
    P.ma = P.mask
    P.ex = P.exit
    P.pr = P.probe
    P.ob = P.obj
    return P
