from ptypy.core import Ptycho
from ptypy import utils as u

p = u.Param()

# for verbose output
p.verbose_level = 3

# max 100 frames (128x128px) of diffraction data
p.scans = u.Param()
p.scans.scan01 = u.Param()
p.scans.scan01.name = 'Bragg3dModel'
p.scans.scan01.data= u.Param()
p.scans.scan01.data.name = 'Bragg3dSimScan'

# prepare and run
P = Ptycho(p,level=5)
