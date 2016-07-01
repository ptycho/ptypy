"""
This script creates a sample *.ptyd data file using the built-in
test Scan `ptypy.core.data.MoonFlowerScan`
"""
import sys
import time
import ptypy
from ptypy import utils as u


import unittest


class MakeSamplePtydTest(unittest.TestCase):
    def test_make_sample(self):
        # for verbose output
        u.verbose.set_level(3)
        
        # create data parameter branch
        data = u.Param()
        data.dfile = 'sample.ptyd'
        data.shape = 128
        data.num_frames = 100
        data.save = 'append'
        data.label=None 
        data.psize=1.0
        data.energy=1.0
        data.center=(64,64)
        data.distance = 1.0
        data.auto_center = None 
        data.rebin = None  
        data.orientation = None  
        
        # create PtyScan instance
        MF = ptypy.core.data.MoonFlowerScan(data)
        MF.initialize()
        for i in range(2):
            # autoprocess data
            msg = MF.auto(100)
            time.sleep(2)
            # logs the out put of .auto() to terminal prompt
            u.verbose.logger.info(u.verbose.report(msg), extra={'allprocesses': True})


if __name__ == '__main__':
    unittest.main()
