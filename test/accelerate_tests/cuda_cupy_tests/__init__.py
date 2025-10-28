import unittest
import numpy as np
import importlib

# shall we run performance tests?
perfrun = False

def have_cupy():
    if importlib.util.find_spec('cupy') is None:
        return False
    try:
        import cupy as cp
        cp.cuda.Device(0).compute_capability
        return True
    except cp.cuda.runtime.CUDARuntimeError:
        return False

if have_cupy():
    import cupy as cp

@unittest.skipIf(not have_cupy(), "no cupy available")
class CupyCudaTest(unittest.TestCase):
    
    def setUp(self):
        import sys
        np.set_printoptions(threshold=sys.maxsize, linewidth=np.inf)
        self.stream = cp.cuda.Stream()
        self.stream.use()
        
    def tearDown(self):
        np.set_printoptions()
        # back to default stream
        cp.cuda.Stream.null.use()