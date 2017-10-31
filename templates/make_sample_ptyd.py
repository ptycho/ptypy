"""
This script creates a sample *.ptyd data file using the built-in
test Scan `ptypy.core.data.MoonFlowerScan`
"""
import sys
import time
import ptypy
from ptypy import utils as u
from ptypy import experiment
# for verbose output
u.verbose.set_level(3)

# create data parameter branch
data = u.Param()
data.source = 'test'
data.dfile = 'sample.ptyd'
data.num_frames = 100
data.save = 'append'
data.label = None
data.auto_center = None
data.rebin = None
data.orientation = None

# create PtyScan instance
# The following call is equivalent to
# MF = ptypy.core.data.MoonFlowerScan(data)
# It uses data.source to find the proper PtyScan subclass
MF = experiment.makePtyScan(data)

MF.initialize()
for i in range(2):
    # autoprocess data
    msg = MF.auto(100)
    time.sleep(2)
    # logs the out put of .auto() to terminal prompt
    u.verbose.logger.info(u.verbose.report(msg), extra={'allprocesses': True})
