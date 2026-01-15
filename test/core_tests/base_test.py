'''
A test for the Base
'''

import unittest
from ptypy.core import Base


class BaseTest(unittest.TestCase):
    def test_base(self):
        a = Base()

if __name__ == '__main__':
    unittest.main()