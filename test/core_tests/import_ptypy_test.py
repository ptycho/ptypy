"""
So much stuff in the init files now, we better test the import.
"""

import unittest

class ImportPtypyTest(unittest.TestCase):
    def test_import(self):
        import ptypy

class PtypyLoaderTest(unittest.TestCase):
        
    def test_load_ptyscan_module(self):
        import ptypy
        ptypy.load_ptyscan_module("hdf5_loader")

    def test_load_all_ptyscan_modules(self):
        import ptypy
        ptypy.load_all_ptyscan_modules()
