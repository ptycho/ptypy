"""
Loads data from a zmq stream published by Contrast at NanoMAX,
https://github.com/alexbjorling/contrast
"""
from ptypy.core import Ptycho
from ptypy import utils as u
import ptypy
ptypy.load_ptyscan_module("nanomax_streaming")

import tempfile
tmpdir = tempfile.gettempdir()

p = u.Param()

# for verbose output
p.verbose_level = "info"

# set home path
p.io = u.Param()
p.io.home = "/".join([tmpdir, "ptypy"])
p.io.autosave = u.Param(active=False)

# max 200 frames (128x128px) of diffraction data
p.scans = u.Param()
p.scans.contrast = u.Param()
# now you have to specify which ScanModel to use with scans.XX.name,
# just as you have to give 'name' for engines and PtyScan subclasses.
p.scans.contrast.name = 'Full'
p.scans.contrast.data= u.Param()
p.scans.contrast.data.name = 'NanomaxZmqScan'
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
p.engines.engine00.numiter_contiguous = 2

# prepare and run
if __name__ == "__main__":
    P = Ptycho(p,level=5)
