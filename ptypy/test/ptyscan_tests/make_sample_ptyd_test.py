"""
This script creates a sample *.ptyd data file using the built-in
test Scan `ptypy.core.data.MoonFlowerScan`
"""
import sys
import time
import ptypy
from ptypy import utils as u
import tempfile

u.verbose.set_level(3)

import unittest


class MakeSamplePtydTest(unittest.TestCase):
    DATA = u.Param(
        dfile = tempfile.gettempdir()+'/ptypytest/sample.ptyd',
        shape = 128,
        num_frames = 50,
        save = None,
        label=None,
        psize=1.0,
        energy=1.0,
        distance = 1.0,
        center='fftshift',
        auto_center = None ,
        rebin = None,
        orientation = None,
    )
    
    def _full_pipeline_with_three_calls_to_auto(self, save = None):
        # for verbose output
        u.verbose.set_level(3)
        
        # create data parameter branch
        data = self.DATA.copy()
        data.save = save
        
        # create PtyScan instance
        MF = ptypy.core.data.MoonFlowerScan(data)
        MF.initialize()
        
        msgs =[
            # 30 frames
            MF.auto(30, chunk_form='dp'),
            # shuld be 20 frames, but asked for hundred
            MF.auto(100, chunk_form='dp'),
            # source depleted
            MF.auto(100, chunk_form='dp')
        ]
        
        return MF,msgs
        
    def test_auto_make_sample(self):
        self._full_pipeline_with_three_calls_to_auto()
    
    def test_analyse_auto_data(self):
        S,msgs = self._full_pipeline_with_three_calls_to_auto()
        
        self.assertEqual(30,len(msgs[0]['iterable']), 
            "Scan did not prepare 30 frames as expected")
            
        self.assertEqual(20,len(msgs[1]['iterable']), 
            "Scan did not prepare 20 frames as expected")
            
        from ptypy.core.data import EOS
        self.assertEqual(msgs[2], EOS, 
            "Last auto call not identified as End of Scan (data.EOS)")
        
    def test_appended_ptyd(self):
        S,msgs = self._full_pipeline_with_three_calls_to_auto(save='append')
        
        from ptypy import io
        d = io.h5read(self.DATA['dfile'])
        
    def test_linked_ptyd(self):
        S,msgs = self._full_pipeline_with_three_calls_to_auto(save='link')
        
        from ptypy import io
        d = io.h5read(self.DATA['dfile'])



if __name__ == '__main__':
    unittest.main()
