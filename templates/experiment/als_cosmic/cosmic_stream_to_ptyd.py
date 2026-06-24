from ptypy.experiment.cosmicstream import CosmicStreamLoader
import time
from ptypy import utils as u
u.verbose.set_level(4)
p = u.Param(dfile="test.ptyd", 
            save="append", 
            min_frames=50,
            rebin=2,
            shape=512,
            center=None,
            auto_center=True)
loader = CosmicStreamLoader(p) 
loader.initialize()
while True:
    msg = loader.auto(100)
    if msg == loader.WAIT:
        time.sleep(2)
    elif msg == loader.EOS:
        print("end")
        break
    else:
        print([it['index'] for it in msg['iterable']])