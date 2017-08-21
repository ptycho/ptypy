'''
A test for the Base
'''

import unittest
from ptypy.core import Storage, Container


class StorageTest(unittest.TestCase):
    def test_storage(self):
        cont = Container()
        a = Storage(cont)

if __name__ == '__main__':
    unittest.main()