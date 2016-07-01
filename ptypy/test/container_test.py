'''
A test for the Base
'''

import unittest
from ptypy.core import Container


class ContainerTest(unittest.TestCase):
    def test_container(self):
        a = Container()
        
if __name__ == '__main__':
    unittest.main()