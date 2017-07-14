"""
Test validator submodule
"""
from ptypy.utils import validator
import unittest

#class ParameterTest(unittest.TestCase):


class EvalParameterTest(unittest.TestCase):

    def test_parse_doc_basic(self):
        """
        Test basic behaviour of the EvalParameter decorator.
        """
        root = validator.EvalParameter('')

        @root.parse_doc('engine')
        class FakeEngineClass(object):
            """
            Dummy documentation
            blabla, any text is allowed except
            a line that starts with "Parameters".

            Parameters:

            [name]
            default=DM
            type=str
            help=The name of the engine
            doc=The name of the engine can be DM or ML or ePIE or some others
                that will be implemented in the future.

            [numiter]
            default=1
            type=int
            lowlim=0
            help=Number of iterations
            """
            pass

        # A few checks
        assert root['engine.numiter'].limits == (0, None)
        assert root['engine.numiter'].options == {'default': '1', 'help': 'Number of iterations', 'lowlim': '0', 'type': 'int'}
        assert root['engine.name'].help == 'The name of the engine'
        assert root['engine'].implicit == True
        assert root['engine'].type == ['Param']
        assert FakeEngineClass.DEFAULTS == {'name': 'DM', 'numiter': 1}

    def test_parse_doc_order(self):
        """
        Test that implicit/explicit order is honored
        """
        root = validator.EvalParameter('')

        # Add the engine part
        @root.parse_doc('engine')
        class FakeEngineClass(object):
            """
            Blabla

            Parameters:

            [name]
            default=DM
            type=str
            help=The name of the engine
            doc=The name of the engine can be DM or ML or ePIE or some others
                that will be implemented in the future.

            [numiter]
            default=1
            type=int
            lowlim=0
            help=Number of iterations
            """
            pass

        # Add the io part
        @root.parse_doc('io')
        class FakeIOClass(object):
            """
            Blabla

            Parameters:

            [interaction.port]
            default=10005
            type=int
            help=The port to listen to

            [path]
            default='.'
            type=str
            help=The path
            """
            pass

        # Populate root - this enforces the proper order of parameters
        @root.parse_doc()
        class FakePtychoClass(object):
            """
            Dummy doc

            Parameters:

            [verbose_level]
            default=3
            type=int
            help=Verbose level

            [io]
            default=None
            type=Param
            help=Input/Output

            [scan]
            default=None
            type=Param
            help=Scan info

            [engine]
            default=None
            type=Param
            help=Engine info
            """

        descendant_name_list = [k for k, _ in root.descendants]
        assert descendant_name_list == ['verbose_level',
                                         'io',
                                         'io.interaction',
                                         'io.interaction.port',
                                         'io.path',
                                         'scan',
                                         'engine',
                                         'engine.name',
                                         'engine.numiter']


if __name__ == "__main__":
    unittest.main()