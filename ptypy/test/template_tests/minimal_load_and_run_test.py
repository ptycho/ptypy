"""
This script is a test for ptychographic reconstruction after an 
experiment has been carried out and the data is available in ptypy's
data file format in "/tmp/ptypy/sample.ptyd"
"""
import ptypy
from ptypy.core import Ptycho
from ptypy import utils as u

import unittest
import tempfile

class MinimalLoadAndRunTest(unittest.TestCase):
    @unittest.skip('skipping this for now. adp 23/09')
    def test_load_and_run(self):
        p = u.Param()
        p.verbose_level = 3
        p.io = u.Param()
        p.io.home = tempfile.gettempdir()+"/ptypy/"  
        p.autosave = None
        
        p.scans = u.Param()
        p.scans.MF = u.Param()
        p.scans.MF.data= u.Param()
        p.scans.MF.data.source = 'sample.ptyd'
        p.scans.MF.data.dfile = tempfile.gettempdir()+'/sample.ptyd'
        
        p.engine = u.Param()
        
        ## Common defaults for all engines
        p.engine.common = u.Param()
        # Total number of iterations
        p.engine.common.numiter = 100
        # Number of iterations to be executed in one go
        p.engine.common.numiter_contiguous = 1           
        # Fraction of valid probe area (circular) in probe frame
        p.engine.common.probe_support = None  
        # Number of iterations before probe update starts
        p.engine.common.probe_update_start = 2
        # Clip object amplitude into this intrervall
        p.engine.common.clip_object = None   # [0,1]
        
        ## DM default parameters
        p.engine.DM = u.Param()
        p.engine.DM.name = "DM"   
        # HIO parameter
        p.engine.DM.alpha = 1                            
        # Probe fraction kept from iteration to iteration
        p.engine.DM.probe_inertia = 0.01             
        # Object fraction kept from iteration to iteration
        p.engine.DM.object_inertia = 0.1             
        # If False: update object before probe
        p.engine.DM.update_object_first = True     
        # Gaussian smoothing (FWHM, pixel units) of object
        p.engine.DM.obj_smooth_std = 10
        # Loop the overlap constraint until probe changes lesser than this fraction
        p.engine.DM.overlap_converge_factor = 0.5  
        # Maximum iterations to be spent inoverlap constraint
        p.engine.DM.overlap_max_iterations = 100  
        # If rms of model vs diffraction data is smaller than this fraction, 
        # Fourier constraint is considered fullfilled
        p.engine.DM.fourier_relax_factor = 0.05    
        
        p.engines = u.Param()              
        p.engines.engine00 = u.Param()
        p.engines.engine00.name = 'DM'
        p.engines.engine00.numiter = 30
        p.engines.engine01 = u.Param()
        p.engines.engine01.name = 'ML'
        p.engines.engine01.numiter = 20
        
        P = Ptycho(p,level=5)

if __name__ == '__main__':
    unittest.main()