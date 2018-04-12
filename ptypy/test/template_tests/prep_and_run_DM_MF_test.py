"""
This script is a test for ptychographic reconstruction in the absence
of actual data. It uses the test Scan class
`ptypy.core.data.MoonFlowerScan` to provide "data".
"""
#import ptypy
from ptypy.core import Ptycho
from ptypy import utils as u
import unittest

# class PrepAndRunDMMFTest(unittest.TestCase):
#     def test_prep_and_run(self):
# p = u.Param()
#
# # for verbose output
# p.verbose_level = 4
#
# # set home path
# p.io = u.Param()
# p.io.home = "/tmp/ptypy/"
# p.io.autosave = None
# #p.io.autoplot = u.Param()
# #p.io.autoplot.dump = True
# #p.io.autoplot = False
#
# # max 100 frames (128x128px) of diffraction data
# p.scans = u.Param()
# p.scans.MF = u.Param()
# p.scans.MF.name = 'Full'
# p.scans.MF.data= u.Param()
# p.scans.MF.data.name = 'MoonFlowerScan'
# p.scans.MF.data.shape = 128
# p.scans.MF.data.num_frames = 100
# p.scans.MF.data.save = None
#
# # position distance in fraction of illumination frame
# p.scans.MF.data.density = 0.2
# # total number of photon in empty beam
# p.scans.MF.data.photons = 1e8
# # Gaussian FWHM of possible detector blurring
# p.scans.MF.data.psf = 0.0
#
# p.engines = u.Param()
# p.engines.engine00 = u.Param()
# p.engines.engine00.name = 'DM'
# p.engines.engine00.numiter = 5
#
# # prepare and run
# P = Ptycho(p,level=5)
