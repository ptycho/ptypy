"""
So much stuff in the init files now, we better test the import.
"""


import unittest

class ImportPtypyTest(unittest.TestCase):
    def test_import(self):
        import ptypy

if __name__ == '__main__':
    unittest.main()