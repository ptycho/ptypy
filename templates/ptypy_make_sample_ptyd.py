"""
This script creates a sample *.ptyd data file using the built-in
test Scan `ptypy.core.data.MoonFlowerScan`
"""
import time
from ptypy import utils as u
from ptypy.core.data import MoonFlowerScan
# for verbose output
u.verbose.set_level("info")

# create data parameter branch
data = u.Param()
data.dfile = 'sample.ptyd'
data.num_frames = 200
data.save = 'append'
data.label = None
data.auto_center = None
data.rebin = None
data.orientation = None

# create PtyScan instance
MF = MoonFlowerScan(data)

MF.initialize()
for i in range(2):
    # autoprocess data
    msg = MF.auto(200)
    time.sleep(2)
    # logs the out put of .auto() to terminal prompt
    u.verbose.logger.info(u.verbose.report(msg), extra={'allprocesses': True})
