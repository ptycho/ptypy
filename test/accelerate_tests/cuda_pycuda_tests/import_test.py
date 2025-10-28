"""
Import test
"""
import unittest

class AutoLoaderTest(unittest.TestCase):
        
    def test_load_engines_cuda(self):
        import ptypy
        ptypy.load_gpu_engines("cuda")
