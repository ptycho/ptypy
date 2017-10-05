"""
Test parameters submodule
"""
import unittest

from ptypy.utils import Param


class ParamTest(unittest.TestCase):

    def test_dots_in_keys(self):
        """
        Test the setting and getting Param values by providing full
        paths as keys.
        """

        # single extra parent
        p = Param()
        p['x.y'] = 1
        assert p == Param({'x': Param({'y': 1})})
        assert p.x.y == 1

        # more implicit parents
        p = Param()
        p['a.b.c'] = 1
        assert p['a.b.c'] == 1
        assert p == Param({'a': Param({'b': Param({'c': 1})})})
        assert p.a.b.c == 1
        assert p.a.b is p['a.b'] 

        # many implicit parents
        p = Param()
        p['a.b.c.d.e.f.g'] = 1
        p['a.b.c.dd.ee'] = 2
        assert p['a.b.c.d.e.f.g'] == 1
        assert p.a.b.c.d.e.f.g == 1
        assert p.a.b.c.dd.ee == 2

if __name__ == "__main__":
    unittest.main()