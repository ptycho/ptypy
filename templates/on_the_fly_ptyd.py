"""
This script creates a sample *.ptyd data file using the built-in
test Scan `ptypy.core.data.MoonFlowerScan`.

Instead of a monolithic file, this process while create data chunks in
separate and link to the main .ptyd file. The delay of 3 seconds
corresponds to a 0.3 seconds dwell time.

Tested only as single process.
"""
import sys
import time
import ptypy
from ptypy import utils as u

# for verbose output
u.verbose.set_level(3)

# create data parameter branch
data = u.Param()
data.dfile = '/tmp/ptypy/sample.ptyd'
data.shape = 256
data.num_frames = 200
data.min_frames = 10
data.save = 'link'
data.label=None
data.psize=172e-6
data.energy= 6.2
data.center='fftshift'
data.distance = 7
data.auto_center = None
data.rebin = None
data.orientation = None

# optionally validate the parameter tree
ptypy.defaults_tree['scandata.MoonFlowerScan'].validate(data)

# create PtyScan instance
MF = ptypy.core.data.MoonFlowerScan(data)
MF.initialize()
for i in range(20):
    # autoprocess data
    msg = MF.auto(10)
    time.sleep(3)
    # logs the out put of .auto() to terminal prompt
    u.verbose.logger.info(u.verbose.report(msg), extra={'allprocesses': True})
