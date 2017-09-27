"""
Test descriptor submodule
"""
from ptypy.utils.descriptor import EvalDescriptor
import unittest


class EvalDescriptorTest(unittest.TestCase):

    def test_parse_doc_basic(self):
        """
        Test basic behaviour of the EvalDescriptor decorator.
        """
        root = EvalDescriptor('')

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
        root = EvalDescriptor('')

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

    def test_parse_doc_inheritance(self):
        """
        Test inheritance of EvalDescriptor decorator.
        """
        root = EvalDescriptor('')

        class FakeEngineBaseClass(object):
            """
            Dummy documentation
            blabla, any text is allowed except
            a line that starts with "Parameters".

            Parameters:

            [name]
            default=BaseEngineName
            type=str
            help=The name of the base engine
            doc=The name of the engine can be DM or ML or ePIE or some others
                that will be implemented in the future.

            [numiter]
            default=1
            type=int
            lowlim=0
            help=Number of iterations
            """
            pass

        @root.parse_doc('engine')
        class FakeEngineClass(FakeEngineBaseClass):
            """
            Engine-specific documentation

            Parameters:

            # It is possible to overwrite a base parameter
            [name]
            default=SubclassedEngineName
            type=str
            help=The name of the subclassed engine
            doc=The name of the engine can be DM or ML or ePIE or some others
                that will be implemented in the future.

            # New parameter
            [alpha]
            default=1.
            type=float
            lowlim=0
            help=Important parameter

            # New substructure
            [subengine.some_parameter]
            default=1.
            type=float
            lowlim=0.
            uplim=2.
            help=Another parameter
            """
            pass

        # A few checks
        assert root['engine.numiter'].limits == (0, None)
        assert root['engine.numiter'].options == {'default': '1', 'help': 'Number of iterations', 'lowlim': '0', 'type': 'int'}
        assert root['engine.name'].help == 'The name of the subclassed engine'
        assert root['engine'].implicit == True
        assert root['engine'].type == ['Param']
        assert FakeEngineClass.DEFAULTS == {'alpha': 1.0, 'name': 'SubclassedEngineName', 'numiter': 1, 'subengine': {'some_parameter': 1.0}}



if __name__ == "__main__":
    unittest.main()