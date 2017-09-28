"""
Test descriptor submodule
"""
import unittest

from ptypy.utils.descriptor import EvalDescriptor, CODES
from ptypy.utils import Param


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

            Defaults:

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

            Defaults:

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

            Defaults:

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

            Defaults:

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

            Defaults:

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

            Defaults:

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

    def test_parse_doc_wildcards(self):
        """
        Test that wildcards in the EvalDescriptor structure are handled
        properly, i e that they drop out of the DEFAULTS constants.
        """
        root = EvalDescriptor('')

        @root.parse_doc('scans.*')
        class FakeScanClass(object):
            """
            General info.

            Defaults:

            [energy]
            type = float
            default = 11.4
            help = Energy in keV
            lowlim = 1
            uplim = 20

            [comment]
            type = str
            default =
            help = Just some static parameter
            """
            pass

        @root.parse_doc()
        class FakePtychoClass(object):
            """

            General documentation.

            Defaults:

            [scans]
            type = Param
            default = None
            help = Engine container

            [run]
            type = str
            default = run
            help = Some parameter

            """
            pass

        assert dict(FakeScanClass.DEFAULTS._to_dict(True)) == {'comment': None, 'energy': 11.4}
        assert dict(FakePtychoClass.DEFAULTS._to_dict(True)) == {'run': 'run', 'scans': {'*': {'comment': None, 'energy': 11.4}}}

        # a correct param tree
        p = Param()
        p.run = 'my reconstruction run'
        p.scans = Param()
        p.scans.scan01 = Param()
        p.scans.scan01.energy = 3.14
        p.scans.scan01.comment = 'first scan'
        p.scans.scan02 = Param()
        p.scans.scan02.energy = 3.14 * 2
        p.scans.scan02.comment = 'second scan'
        root.validate(p, walk=True)

        # no scans entries
        p = Param()
        p.run = 'my reconstruction run'
        p.scans = Param()
        out = root.check(p, walk=True)
        assert out['scans.*']['*'] == CODES.MISSING

        # a bad scans entry
        p = Param()
        p.run = 'my reconstruction run'
        p.scans = Param()
        p.scans.scan01 = Param()
        p.scans.scan01.energy = 3.14
        p.scans.scan01.comment = 'first scan'
        p.scans.scan02 = 'not good'
        p.scans.scan03 = Param()
        p.scans.scan03.energy = 3.14 * 2
        p.scans.scan03.comment = 'second scan'
        out = root.check(p, walk=True)
        assert out['scans.scan02']['type'] == CODES.INVALID

        # a bad entry within a scan
        p = Param()
        p.run = 'my reconstruction run'
        p.scans = Param()
        p.scans.scan01 = Param()
        p.scans.scan01.energy = 3.14
        p.scans.scan01.comment = 'first scan'
        p.scans.scan02 = Param()
        p.scans.scan02.energy = 3.14 * 2
        p.scans.scan02.comment = 'second scan'
        p.scans.scan02.badparameter = 'not good'
        out = root.check(p, walk=True)
        assert out['scans.scan02']['badparameter'] == CODES.INVALID

    def test_parse_doc_symlinks(self):
        """
        Test that symlinks in the EvalDescriptor structure are handled
        properly.
        """
        root = EvalDescriptor('')

        @root.parse_doc('engine.DM')
        class FakeDMEngineClass(object):
            """
            Dummy documentation
            blabla, any text is allowed except
            a line that starts with "Defaults".

            Defaults:

            [name]
            default=DM
            type=str
            help=DM engine

            [numiter]
            default=1
            type=int
            lowlim=0
            help=Number of iterations
            """
            pass

        @root.parse_doc('engine.ML')
        class FakeMLEngineClass(object):
            """
            Dummy documentation

            Defaults:

            [name]
            default=ML
            type=str
            help=ML engine

            [numiter]
            default=1
            type=int
            lowlim=0
            help=Number of iterations
            """
            pass

        @root.parse_doc()
        class FakePtychoClass(object):
            """

            General documentation.

            Defaults:

            [engines]
            type = Param
            default =
            help = Container for all engines

            [engines.*]
            type = engine.DM, engine.ML
            default = engine.DM
            help = Engine wildcard. Defaults to DM
            """
            pass

        # a correct param tree
        p = Param()
        p.engines = Param()
        p.engines.engine01 = Param()
        p.engines.engine01.name = 'DM'
        p.engines.engine01.numiter = 10
        p.engines.engine02 = Param()
        p.engines.engine02.name = 'ML'
        p.engines.engine02.numiter = 10
        root.validate(p, walk=True)

        # no name
        p = Param()
        p.engines = Param()
        p.engines.engine01 = Param()
        p.engines.engine01.numiter = 10
        out = root.check(p, walk=True)
        assert out['engines.engine01']['symlink'] == CODES.INVALID

        # wrong name
        p = Param()
        p.engines = Param()
        p.engines.engine01 = Param()
        p.engines.engine01.name = 'ePIE'
        p.engines.engine01.numiter = 10
        out = root.check(p, walk=True)
        assert out['engines.engine01']['symlink'] == CODES.INVALID

    def test_parse_doc_wildcards_and_symlinks(self):
        """ 
        Tests the combination of wildcards and dynamic entries.
        """

        root = EvalDescriptor('')

        @root.parse_doc()
        class FakePtycho(object):
            """
            Docs...

            Defaults:

            # A template tree under scan
            [scan]
            type = Param
            default = 
            help = 

            [scan.data]
            type = Param
            default =
            help = Data preparation parameters

            [scan.data.load_parallel]
            type = bool
            default = True
            help = Load 

            [scan.data.source]
            type = str
            default = 'ptyd'
            help = Data source

            [scan.model]
            type = scanmodel.Vanilla, scanmodel.Full
            default = scanmodel.Vanilla
            help = Physical imaging model


            # The actual scans container
            [scans]
            type = Param
            default = None
            help = Container for scan instances

            [scans.*]
            type = scan
            default = scan
            help = Wildcard for scan instances

            """
            pass


        @root.parse_doc('scanmodel.Vanilla')
        class FakeVanillaScan(object):
            """
            Docs...

            Defaults:

            [name]
            type = str
            default = Vanilla
            help = Vanilla scan model

            [energy]
            type = float
            default = 9.3
            help = Photon energy
            """
            pass

        @root.parse_doc('scanmodel.Full')
        class FakeFullScan(FakeVanillaScan):
            """
            Docs...

            Defaults:

            [name]
            type = str
            default = Full
            help = Full scan model

            [probe_modes]
            type = int
            default = 1
            help = Number of mutually incoherent illumination modes
            """
            pass

        assert dict(FakeVanillaScan.DEFAULTS) == {'energy': 9.3, 'name': 'Vanilla'}
        assert dict(FakeFullScan.DEFAULTS) == {'energy': 9.3, 'name': 'Full', 'probe_modes': 1}

        # a minimal tree
        p = Param()
        root.validate(p, walk=True)

        # a hopefully correct tree with scans entry and model choice
        p = Param()
        p.scans = Param()
        p.scans.scan01 = Param()
        p.scans.scan01.model = Param()
        p.scans.scan01.model.name = 'Full'
        p.scans.scan01.model.probe_modes = 3
        root.validate(p, walk=True)



if __name__ == "__main__":
    unittest.main()