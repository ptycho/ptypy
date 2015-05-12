
import sys
from ptypy.core import data
from ptypy import utils as u

import time
p = u.Param()
p.dfile = 'sample.ptyd'

if len(sys.argv)>1:
    p.dfile =sys.argv[1]

u.verbose.set_level(3)

p.shape = 128
p.num_frames = 100
p.save = 'append'

MF = data.MoonFlowerScan(p)
MF.initialize()
for i in range(2):
    msg = MF.auto(100)
    time.sleep(2)
    u.verbose.logger.info(u.verbose.report(msg), extra={'allprocesses': True})
