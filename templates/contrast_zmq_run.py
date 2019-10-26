"""
Loads data from a zmq stream published by Contrast,
https://github.com/alexbjorling/contrast
"""

from ptypy.core import Ptycho
from ptypy import utils as u
p = u.Param()

# for verbose output
p.verbose_level = 3

# set home path
p.io = u.Param()
p.io.home = "/tmp/ptypy/"
p.io.autosave = None

# max 200 frames (128x128px) of diffraction data
p.scans = u.Param()
p.scans.contrast = u.Param()
# now you have to specify which ScanModel to use with scans.XX.name,
# just as you have to give 'name' for engines and PtyScan subclasses.
p.scans.contrast.name = 'Full'
p.scans.contrast.data= u.Param()
p.scans.contrast.data.name = 'ContrastZmqScan'
p.scans.contrast.data.min_frames = 5
p.scans.contrast.data.detector = 'diff'
p.scans.contrast.data.xMotor = 'x'
p.scans.contrast.data.yMotor = 'y'
p.scans.contrast.data.host = 'localhost'
p.scans.contrast.data.port = 5556

# attach a reconstrucion engine
p.engines = u.Param()
p.engines.engine00 = u.Param()
p.engines.engine00.name = 'DM'
p.engines.engine00.numiter = 300
p.engines.engine00.numiter_contiguous = 10

# prepare and run
P = Ptycho(p,level=5)
