"""
Import test
"""
import unittest

class AutoLoaderTest(unittest.TestCase):
        
    def test_load_engines_cupy(self):
        import ptypy
        ptypy.load_gpu_engines("cupy")
