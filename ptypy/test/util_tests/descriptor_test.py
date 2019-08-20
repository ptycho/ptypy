"""
Test descriptor submodule
"""
import unittest

from ptypy import defaults_tree
from ptypy.utils.descriptor import EvalDescriptor, CODES
from ptypy.utils import Param


class SanityCheck(unittest.TestCase):

    def test_sanity(self):
        defaults_tree.sanity_check()


class EvalDescriptorTest(unittest.TestCase):

    def test_basic_functions(self):
        """
        Test EvalDescriptor behaviour
        """

        # Parameter declaration through formatted string
        x = EvalDescriptor('')
        x.from_string("""
        [param1]
        default = 0
        type = int
        help = A parameter
        uplim = 5
        lowlim = 0""")

        assert x['param1'].default == 0
        assert x['param1'].limits == (0, 5)
        assert x['param1'].type == ['int']

        # ConfigParser allows overwriting some properties
        x = EvalDescriptor('')
        x.from_string("""
        [param1]
        default = 0
        type = int
        help = A parameter

        [param2]
        default = a
        type = str
        help = Another parameter

        [param1]
        uplim = 5
        lowlim = 0""")

        assert x['param1'].limits == (0, 5)

        # Implicit branch creation
        x = EvalDescriptor('')
        x.from_string("""
        [category1.subcategory1.param1]
        default = 0
        type = int
        help = A parameter""")

        assert [k for k,v in x.descendants] == ['category1', 'category1.subcategory1', 'category1.subcategory1.param1']

        assert x['category1'].implicit == True

        x.from_string("""
        [category1]
        default =
        type = Param
        help = The first category""")

        assert x['category1'].implicit == False

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
        assert FakeEngineClass.DEFAULT == Param({'name': 'DM', 'numiter': 1})

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

        @root.parse_doc('haha')
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
        assert FakeEngineClass.DEFAULT == Param({'alpha': 1.0, 'name': 'SubclassedEngineName', 'numiter': 1, 'subengine': {'some_parameter': 1.0}})

    def test_parse_doc_wildcards(self):
        """
        Test that wildcards in the EvalDescriptor structure are handled
        properly.
        """
        root = EvalDescriptor('')

        @root.parse_doc('scan.FakeScan')
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
            default = {}
            help = Engine container

            [scans.*]
            type = @scan.*
            default = @scan.FakeScan
            help =

            [run]
            type = str
            default = run
            help = Some parameter

            """
            pass

        assert FakeScanClass.DEFAULT == Param({'comment': None, 'energy': 11.4})

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
        root.validate(p)

        # no scans entries
        p = Param()
        p.run = 'my reconstruction run'
        p.scans = Param()
        root.validate(p)

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
        out = root.check(p)
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
        out = root.check(p)
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
            type = @engine.*
            default = @engine.DM
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
        root.validate(p)

        # no name
        p = Param()
        p.engines = Param()
        p.engines.engine01 = Param()
        p.engines.engine01.numiter = 10
        out = root.check(p)
        assert out['engines.engine01']['symlink'] == CODES.INVALID

        # wrong name
        p = Param()
        p.engines = Param()
        p.engines.engine01 = Param()
        p.engines.engine01.name = 'ePIE'
        p.engines.engine01.numiter = 10
        out = root.check(p)
        assert out['engines.engine01']['symlink'] == CODES.INVALID


    def test_multiple_parameter_types_param(self):
        '''
        This tests that multiple possible parameter types are handled correctly
        '''
        root = EvalDescriptor('')

        # Add the io part
        @root.parse_doc()
        class FakeIOClass(object):
            """
            A fake IO class

            Defaults:
            
            [io]
            default = None
            type = Param
            help = Global parameters for I/O
            doc = Global parameter container for I/O settings.

            [io.autoplot]
            default = None
            type = Param, bool
            help = Plotting client parameters
            doc = In script you may set this parameter to ``None`` or ``False`` for no automatic plotting.
            
            [io.autoplot.imfile]
            default = "plots/%(run)s/%(run)s_%(engine)s_%(iterations)04d.png"
            type = str
            help = Plot images file name (or format string)
            doc = Plot images file name (or format string).
            userlevel = 1
        
            [io.autoplot.interval]
            default = 1
            type = int
            help = Number of iterations between plot updates
            doc = Requests to the server will happen with this iteration intervals. Note that this will work
              only if interaction.polling_interval is smaller or equal to this number. If ``interval
              =0`` plotting is disabled which should be used, when ptypy is run on a cluster.
            lowlim = -1
        
            [io.autoplot.threaded]
            default = True
            type = bool
            help = Live plotting switch
            doc = If ``True``, a plotting client will be spawned in a new thread and connected at
              initialization. If ``False``, the master node will carry out the plotting, pausing the
              reconstruction. This option should be set to ``True`` when ptypy is run on an isolated
              workstation.
        
            [io.autoplot.layout]
            default = None
            type = str, Param
            help = Options for default plotter or template name
            doc = Flexible layout for default plotter is not implemented yet. Please choose one of the
              templates ``'default'``,``'black_and_white'``,``'nearfield'``, ``'minimal'`` or ``'weak'``
            userlevel = 2
        
            [io.autoplot.dump]
            default = False
            type = bool
            help = Switch to dump plots as image files
            doc = Switch to dump plots as image files during reconstruction.
        
            [io.autoplot.make_movie]
            default = False
            type = bool
            help = Produce reconstruction movie after the reconstruction.
            doc = Switch to request the production of a movie from the dumped plots at the end of the
              reconstruction.
            
            """
            pass

        p = Param()
        p.io = Param()
        p.io.autoplot = True
        out = root.check(p)

    def test_float_validation(self):
        """
        Check float validation.
        """
        import numpy as np

        x = EvalDescriptor('')
        x.from_string("""
        [p]
        default = 0.
        type = float
        help = A parameter
        uplim = 5.
        lowlim = 0.""")

        # Basic test
        s = Param({'p': 1.})
        x.validate(s)

        # int to float
        s.p = 1
        x.validate(s)

        # numpy floats
        s.p = np.float16(1.)
        x.validate(s)
        s.p = np.float32(1.)
        x.validate(s)
        s.p = np.float64(1.)
        x.validate(s)
        s.p = np.float128(1.)
        x.validate(s)

    def test_save_json(self):

        pass

    def test_load_json(self):
        pass

if __name__ == "__main__":
    unittest.main()
