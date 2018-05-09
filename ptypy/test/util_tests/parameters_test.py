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

        # Contains
        p = Param()
        p['a.b.c.d'] = 1
        assert 'a' in p
        assert 'a.b' in p
        assert 'a.b.c' in p
        assert 'a.b.c.d' in p
        assert 'b' not in p
        assert 'b.c.d' in p['a']
        assert 'c.d' in p['a.b']

    def test_from_dict_conversion_convert_false(self):
        '''
        Tests whether Param can convert correctly from dict. ref #90
        '''
        dct = {'one': {'one1': 1},
               'two': {'two1':
                           {'two1b': 'b'}}}
        p = Param()
        p.update(dct, Convert=False)
        exp_p = Param()
        exp_p = dct
        # print p
        self.assertEqual(exp_p, p, msg="The from dict method isn't working as expected")

    def test_from_dict_conversion_convert_true(self):

        dct = {'one': {'one1': 1},
               'two': {'two1':
                           {'two1b': 'b'}}}
        p = Param()
        p.update(dct, Convert=True)

        exp_p = Param()
        exp_p.one =Param()
        exp_p.one.one1 = 1
        exp_p.two = Param()
        exp_p.two.two1 = Param()
        exp_p.two.two1.two1b = 'b'
        # print p
        self.assertEqual(exp_p, p, msg='The from dict conversion has not worked as expected')


    def test_to_dict_conversion_convert_true(self):

        dct = {'one': {'one1': 1},
               'two': {'two1':
                           {'two1b': 'b'}}}
        p = Param()
        p.update(dct, Convert=True)

        out_dct = p._to_dict()

        self.assertDictEqual(dct, out_dct, msg='The to-dict method has not worked as expected.')


    def test_to_dict_conversion_convert_false(self):

        dct = {'one': {'one1': 1},
               'two': {'two1':
                           {'two1b': 'b'}}}
        p = Param()
        p.update(dct, Convert=False)

        out_dct = p._to_dict()

        self.assertDictEqual(dct, out_dct, msg='The to-dict method has not worked as expected.')


if __name__ == "__main__":
    unittest.main()