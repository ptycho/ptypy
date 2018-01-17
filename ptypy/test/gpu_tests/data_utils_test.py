'''    
Created on 4 Jan 2018

@author: clb02321
'''
import unittest
import numpy as np
from ptypy.gpu import data_utils as du
from ptypy.core import Ptycho
from ptypy import utils as u


class DataUtilsTest(unittest.TestCase):
    '''
    tests the conversion between pods and numpy arrays
    '''

    def setUp(self):
        '''
        new ptypy probably has a better way of doing this.
        '''
        p = u.Param()
        p.verbose_level = 0
        p.data_type = "single"
        p.run = 'test_indep_probes'
        p.io = u.Param()
        p.io.home = "/tmp/ptypy/"
        p.io.interaction = u.Param()
        p.io.autoplot = u.Param(active=False)
        p.scan = u.Param()
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
        p.scans.MF.data.num_frames = 100
        p.scans.MF.data.energy = 6.2
        p.scans.MF.data.shape = 256
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
        p.scans.MF.data.density = 0.2
        p.scans.MF.illumination = u.Param()
        p.scans.MF.illumination.model = None
        p.scans.MF.illumination.aperture = u.Param()
        p.scans.MF.illumination.aperture.diffuser = None
        p.scans.MF.illumination.aperture.form = "circ"
        p.scans.MF.illumination.aperture.size = 3e-6
        p.scans.MF.illumination.aperture.edge = 10
        self.PtychoInstance = Ptycho(p, level=4)

    def test_pod_to_numpy(self):
        '''
        does this even run?
        '''
        du.pod_to_arrays(self.PtychoInstance, 'S0000')


    def test_numpy_to_pod(self):
        pass

    def test_numpy_pod_consistency(self):
        pass

if __name__ == "__main__":
    unittest.main()
