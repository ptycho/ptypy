'''
A test for the Base
'''

import unittest
from ptypy.core import View, Container


class ViewTest(unittest.TestCase):
    def test_storage(self):
        cont = Container()
        a = View(cont)

if __name__ == '__main__':
    unittest.main()