"""
Unittests for ptypy/core/data.py

This module contains basic unittests for default parameters, class
initialization, and method and function calls for the data.py module.

The order of the performed tests is:

- Default module parameters
- PtyScan class initialization (including default parameters) and methods
  calling; testing file creation requires dummy .ptyd file for comparison
- PtyScan class initialization with generic parameters and methods calling
- PtyScan class rebin test
- PtyScan class RuntimeError test
- PtydScan class initialization: just checking AssertionError, proper testing
  requires dummy .ptyd file
- MoonFlower class initialization with generic parameters and
  methods calling
- Data class initialization: proper testing requires dictionary of scan
  structures

The following methods on the left are called by the methods on the right and are
therefore not tested individually:
return_chunk_as --> auto
_mpi_pipeline_with_dictionaries --> get_data_chunk
load --> _mpi_pipeline_with_dictionaries
correct --> _mpi_pipeline_with_dictionaries
_mpi_autocenter --> get_data_chunk
end_of_scan property --> _mpi_check
frames_accessible property --> _mpi_check
abort property --> _mpi_check

Run python -m unittest discover in the ptypy root directory or
python test_data.py in the file's directory to perform the tests.

This file is part of the PTYPY package.
    :copyright: Copyright 2014 by the PTYPY team, see AUTHORS.
    :license: GPLv2, see LICENSE for details.
"""
# Python 2/3 compatibility
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Modules
import unittest
import numpy as np
import ptypy.core.data as d
import ptypy.utils as u


class TestDefaultParameters(unittest.TestCase):
    """Test default parameters"""

    def test_default_ptyd_dict(self):
        """Default ptyd dict unaltered"""
        self.assertEqual(d.PTYD['chunks'], {},
                         'Default value changed.')

        self.assertEqual(d.PTYD['meta'], {},
                         'Default value changed.')

        self.assertEqual(d.PTYD['info'], {},
                         'Default value changed.')

    def test_default_meta_dict(self):
        """Default meta dict unaltered"""
        self.assertEqual(d.META['label'], None,
                         'Default value changed.')

        self.assertEqual(d.META['experimentID'], None,
                         'Default value changed.')

        self.assertEqual(d.META['version'], '0.1',
                         'Default value changed.')

        self.assertEqual(d.META['shape'], None,
                         'Default value changed.')

        self.assertEqual(d.META['psize'], None,
                         'Default value changed.')

        self.assertEqual(d.META['energy'], None,
                         'Default value changed.')

        self.assertEqual(d.META['center'], None,
                         'Default value changed.')

        self.assertEqual(d.META['distance'], None,
                         'Default value changed.')

    def test_default_generic_dict(self):
        """Default generic dict unaltered"""
        self.assertEqual(d.GENERIC['dfile'], None,
                         'Default value changed.')

        self.assertEqual(d.GENERIC['chunk_format'], '.chunk%02d',
                         'Default value changed.')

        self.assertEqual(d.GENERIC['save'], None,
                         'Default value changed.')

        self.assertEqual(d.GENERIC['auto_center'], None,
                         'Default value changed.')

        self.assertEqual(d.GENERIC['load_parallel'], 'data',
                         'Default value changed.')

        self.assertEqual(d.GENERIC['rebin'], None,
                         'Default value changed.')

        self.assertEqual(d.GENERIC['orientation'], None,
                         'Default value changed.')

        self.assertEqual(d.GENERIC['min_frames'], 1,
                         'Default value changed.')

        self.assertEqual(d.GENERIC['positions_theory'], None,
                         'Default value changed.')

        self.assertEqual(d.GENERIC['num_frames'], None,
                         'Default value changed.')

        self.assertEqual(d.GENERIC['recipe'], {},
                         'Default value changed.')

    def test_default_constants(self):
        """Default constants dict unaltered"""
        # Note: Why is WAIT and EOS defined before dict Codes?
        self.assertEqual(d.WAIT, 'msg1',
                         'Default value changed.')

        self.assertEqual(d.EOS, 'msgEOS',
                         'Default value changed.')

        self.assertEqual(
            d.CODES['msg1'],
            'Scan unfinished. More frames available after a pause',
            msg='Default value changed.'
        )

        self.assertEqual(
            d.CODES['msgEOS'],
            'End of scan reached',
            msg='Default value changed.'
        )


class TestPtyScan(unittest.TestCase):
    """Test PtyScan class"""

    def setUp(self):
        """Set up PtyScan instance"""
        self.basic_ptyscan = d.PtyScan()

    def test_ptyscan_defaults(self):
        """PtyScan defaults unaltered"""
        self.assertEqual(
            self.basic_ptyscan.DEFAULT,
            d.GENERIC,
            'Default DEFAULT changed.'
        )

        self.assertEqual(
            self.basic_ptyscan.WAIT,
            d.WAIT,
            'Default WAIT changed.'
        )

        self.assertEqual(
            self.basic_ptyscan.EOS,
            d.EOS,
            'Default EOS changed.'
        )

    def test_init(self):
        """Assigning and updating init attributes"""
        # Only testing attributes that change during initialization.
        # Note: vanilla initialization changes rebin from None to 1, therefore
        # no testing of self.basic_ptyscan.info at the moment.

        self.assertDictEqual(
            self.basic_ptyscan.meta,
            d.META,
            'Updating self.basic_ptyscan.meta failed.'
        )

        self.assertIsNone(
            self.basic_ptyscan.num_frames,
            'Assigning instance attribute num_frames failed.'
        )

        # Note: Test valid for single core execution.
        self.assertEqual(
            self.basic_ptyscan.min_frames,
            1.,
            'Assigning instance attribute min_frames failed.'
        )

        self.assertEqual(
            self.basic_ptyscan.rebin,
            1.,
            'Assigning instance attribute rebin failed.'
        )

        self.assertFalse(
            self.basic_ptyscan.load_common_in_parallel,
            'Assigning instance attribute load_common_in_parallel failed.'
        )

        self.assertTrue(
            self.basic_ptyscan.load_in_parallel,
            'Assigning instance attribute load_in_parallel failed.'
        )

        # Note: post init method call at the end of initialization, not sure if
        # this should happen there

    def test_initialize(self):
        """Initializing PtyScan"""
        # Note: Not checking file creation process at the moment, would probably
        # require a comparison .ptyd file
        self.basic_ptyscan.initialize()

        # Check loading weight2d
        self.assertIsNone(
            self.basic_ptyscan.weight2d,
            'Loading weight2d failed.'
        )

        # Check loading common
        self.assertEqual(
            self.basic_ptyscan.common,
            u.Param(),
            'Loading common failed.'
        )

        # Check has_weight2d
        self.assertFalse(
            self.basic_ptyscan.has_weight2d,
            'Assigning has_weight2d failed.'
        )

        # Check has_positions
        self.assertFalse(
            self.basic_ptyscan.has_positions,
            'Assigning has_positions failed.'
        )

        # Assigning num_frames_actual
        self.assertIsNone(
            self.basic_ptyscan.info.num_frames_actual,
            'Assigning attribute num_frames_actual failed.'
        )

        # Check is_initialized
        self.assertTrue(
            self.basic_ptyscan.is_initialized,
            'Assigning is_initialized failed.'
        )

    @unittest.skip('Implementation not possible at the moment')
    def test_finalizing(self):
        """Finalizing PtyScan"""
        # Note: Not checking at the moment because saving is required

    @unittest.skip('Fix in self.checks required')
    def test_mpi_check(self):
        """Run _mpi_check() of PtyScan"""
        # Note: In vanilla initialization self.num_frames is None, therefore
        # computation of frames_accessible fails
        # Fix in self.checks: if self.num_frames is None: frames_accessible = 0

        # Check RuntimeError is raised
        self.assertRaises(
            RuntimeError,
            self.basic_ptyscan._mpi_check,
            chunksize=5
        )

    def test_mpi_indices(self):
        """Run _mpi_indices() of PtyScan"""
        self.assertDictEqual(
            self.basic_ptyscan._mpi_indices(1, 1),
            {'node': [1], 'chunk': [1], 'lm': [[0]]},
            msg='_mpi_indices calculation failed.'
        )

    def test_report(self):
        """Run report() of PtyScan"""
        # Note: this probably changes each time the code is run,
        # therefore just checking None case
        self.assertIsNone(
            self.basic_ptyscan.report(),
            msg='calling report() failed.'
        )

    @unittest.skip('Vanilla call leads to TypeError')
    def test_mpi_save_chunk(self):
        """Run _mpi_save_chunk of PtyScan"""
        # Note: Vanilla call leads to TypeError in todisk = dict(c) as c is None

    def tearDown(self):
        """Clean up"""
        # Note: Not sure if this required here


class TestPtyScanWithParam(unittest.TestCase):
    """Test PtyScan class with test parameters"""

    def setUp(self):
        """Set up PtyScan instance and test parameters"""
        pars = u.Param()
        pars.positions_theory = np.arange(3)
        pars.shape = 2
        pars.center = (1, 1)
        self.basic_ptyscan_wp = d.PtyScan(pars=pars)

    def test_init_wp(self):
        """Assigning and updating init attributes with parameters"""
        # Check updating num_frames via positions_theory
        self.assertEqual(
            self.basic_ptyscan_wp.num_frames,
            3,
            'Assigning num_frames failed.'
        )

    def test_initialize_wp(self):
        """Initializing PtyScan with parameters"""
        self.basic_ptyscan_wp.initialize()

        # Check loading weight2d
        self.assertTrue(
            np.array_equal(
                self.basic_ptyscan_wp.weight2d,
                np.ones([2, 2])
            ),
            'Loading weight2d failed.'
        )

    def test_mpi_check_wp(self):
        """Run _mpi_check() of PtyScan with parameters"""

        # Case1: Chunksize = 0, no data is processed
        self.assertEqual(
            self.basic_ptyscan_wp._mpi_check(0),
            'msg1',
            msg='_mpi_check() execution failed.'
        )

        # Case2: Chunksize = num_frames
        self.assertTupleEqual(
            self.basic_ptyscan_wp._mpi_check(3),
            (0, 3),
            msg='_mpi_check() execution failed.'
        )

        # Case3: Chunksize > num_frames
        self.assertEqual(
            self.basic_ptyscan_wp._mpi_check(4),
            'msgEOS',
            msg='_mpi_check() execution failed.'
        )

    def test_get_data_chunk_wp(self):
        """Run get_data_chunk() of PtyScan with parameters"""

        # Case1: Chunksize = 0, no data is processed
        self.assertEqual(
            self.basic_ptyscan_wp.get_data_chunk(0),
            'msg1',
            msg='get_data_chunk() execution failed.'
        )

        # Case2: Chunksize = num_frames
        # Note: checking values of returned Param individually
        c2 = self.basic_ptyscan_wp.get_data_chunk(3)

        self.assertTrue(
            np.array_equal(
                c2.positions,
                np.array([0, 1, 2])
            ),
            msg='get_data_chunk() execution failed.'
        )

        self.assertListEqual(
            c2.indices_node,
            [0, 1, 2],
            msg='get_data_chunk() execution failed.'
        )

        self.assertEqual(
            c2.num,
            0,
            msg='get_data_chunk() execution failed.'
        )

        self.assertDictEqual(
            c2.weights,
            {},
            msg='get_data_chunk() execution failed.'
        )

        self.assertListEqual(
            c2.indices,
            [0, 1, 2],
            msg='get_data_chunk() execution failed.'
        )

        self.assertTrue(
            np.array_equal(
                c2.data[0],
                np.array([[0, 0], [0, 0]])
            ),
            msg='get_data_chunk() execution failed.'
        )

        self.assertTrue(
            np.array_equal(
                c2.data[1],
                np.array([[1, 1], [1, 1]])
            ),
            msg='get_data_chunk() execution failed.'
        )

        self.assertTrue(
            np.array_equal(
                c2.data[2],
                np.array([[2, 2], [2, 2]])
            ),
            msg='get_data_chunk() execution failed.'
        )

        # Case3: Chunksize > num_frames
        self.assertEqual(
            self.basic_ptyscan_wp.get_data_chunk(4),
            'msgEOS',
            msg='get_data_chunk() execution failed.'
        )

    def test_auto_wp(self):
        """Run auto() of PtyScan with parameters"""

        # Case1: Chunksize = 0, no data is processed
        self.assertEqual(
            self.basic_ptyscan_wp.auto(0),
            'msg1',
            msg='auto() execution failed.'
        )

        # Case2: Chunksize = num_frames
        # Note: checking values of returned Param individually
        c2 = self.basic_ptyscan_wp.auto(3)

        self.assertIsNone(
            c2['common'].distance,
            msg='auto() execution failed.'
        )

        self.assertTrue(
            np.array_equal(
                c2['common'].center,
                np.array([1., 1.])
            ),
            msg='auto() execution failed.'
        )

        self.assertIsNone(
            c2['common'].energy,
            msg='auto() execution failed.'
        )

        self.assertIsNone(
            c2['common'].psize,
            msg='auto() execution failed.'
        )

        self.assertIsNone(
            c2['common'].label,
            msg='auto execution failed.'
        )

        self.assertTrue(
            np.array_equal(
                c2['common'].shape,
                np.array([2., 2.])
            ),
            msg='auto() execution failed.'
        )

        self.assertEqual(
            c2['common'].version,
            '0.1',
            msg='auto() execution failed.'
        )

        self.assertIsNone(
            c2['common'].experimentID,
            msg='auto() execution failed.'
        )

        self.assertTrue(
            np.array_equal(
                c2['common'].weight2d,
                np.array([[1., 1.], [1., 1.]])
            ),
            msg='auto() execution failed.'
        )

        # Note: only checking first item of entry iterable
        self.assertTrue(
            np.array_equal(
                c2['iterable'][0]['data'],
                np.array([[0., 0.], [0., 0.]])
            ),
            msg='auto() execution failed.'
        )

        self.assertEqual(
            c2['iterable'][0]['index'],
            0,
            msg='auto() execution failed.'
        )

        self.assertTrue(
            np.array_equal(
                c2['iterable'][0]['mask'],
                np.array([[1., 1.], [1., 1.]], dtype=bool)
            ),
            msg='auto() execution failed.'
        )

        self.assertEqual(
            c2['iterable'][0]['position'],
            0,
            msg='auto() execution failed.'
        )

        # Case3: Chunksize > num_frames
        self.assertEqual(
            self.basic_ptyscan_wp.auto(4),
            'msgEOS',
            msg='auto() execution failed.'
        )

    def test_return_chunk_as_wp(self):
        """Run return_chunk_as of PtyScan for RuntimeError test"""
        # return_chunk_as is called from auto, no separate testing,
        # only RuntimeError verification

        # Check RuntimeError is raised
        self.assertRaises(
            RuntimeError,
            self.basic_ptyscan_wp.return_chunk_as,
            chunk=3,
            kind='random_input'
        )

    def test_check_wp(self):
        """Run check() of PtyScan"""
        self.assertTupleEqual(
            self.basic_ptyscan_wp.check(),
            (1, None),
            msg='check() execution failed.'
        )

    def tearDown(self):
        """Clean up"""
        # Note: Not sure if this required here


class TestPtyScanForRebin(unittest.TestCase):
    """Test PtyScan class for rebinning in get_data_chunk"""

    def setUp(self):
        """Set up PtyScan instance for rebin test"""
        pars = u.Param()
        pars.positions_theory = np.arange(3)
        pars.shape = 4
        pars.center = (2, 2)
        pars.rebin = 2
        self.basic_ptyscan_rt = d.PtyScan(pars=pars)

    def test_get_data_chunk_rt(self):
        """Run get_data_chunk() of PtyScan for rebin test"""

        # Note: checking values of returned Param individually
        crt = self.basic_ptyscan_rt.get_data_chunk(3)
        self.assertTrue(
            np.array_equal(
                crt.positions,
                np.array([0, 1, 2])
            ),
            msg='get_data_chunk() execution failed.'
        )

        self.assertListEqual(
            crt.indices_node,
            [0, 1, 2],
            msg='get_data_chunk() execution failed.'
        )

        self.assertEqual(
            crt.num,
            0,
            msg='get_data_chunk() execution failed.'
        )

        self.assertDictEqual(
            crt.weights,
            {},
            msg='get_data_chunk() execution failed.'
        )

        self.assertListEqual(
            crt.indices,
            [0, 1, 2],
            msg='get_data_chunk() execution failed.'
        )

        self.assertTrue(
            np.array_equal(
                crt.data[0],
                np.array([[0, 0], [0, 0]])
            ),
            msg='get_data_chunk() execution failed.'
        )

        self.assertTrue(
            np.array_equal(
                crt.data[1],
                np.array([[1, 1], [1, 1]])
            ),
            msg='get_data_chunk() execution failed.'
        )

        self.assertTrue(
            np.array_equal(
                crt.data[2],
                np.array([[2, 2], [2, 2]])
            ),
            msg='get_data_chunk() execution failed.'
        )

    def tearDown(self):
        """Clean up"""
        # Note: Not sure if this required here


class TestPtyScanForRebinRuntimeError(unittest.TestCase):
    """Test PtyScan class for RuntimeError when rebinning in get_data_chunk"""

    def setUp(self):
        """Set up PtyScan instance for RuntimeError rebin test"""
        pars = u.Param()
        pars.positions_theory = np.arange(3)
        pars.shape = 4
        pars.center = (2, 2)
        pars.rebin = 10
        self.basic_ptyscan_rert = d.PtyScan(pars=pars)

    def test_get_data_chunk_rert(self):
        """Run get_data_chunk() of PtyScan for RuntimeError rebin test"""

        # Check RuntimeError is raised
        self.assertRaises(
            RuntimeError,
            self.basic_ptyscan_rert.get_data_chunk,
            chunksize=3
        )

    def tearDown(self):
        """Clean up"""
        # Note: Not sure if this required here


class TestPtydScan(unittest.TestCase):
    """Test PtydScan class"""
    # Note: vanilla initialization causes TypeError 'NoneType' at
    # source = pars['dfile'], therefore just testing assertion error
    # To test methods, fix vanilla initialization and/or introduce a test file
    # for loading

    def test_init_assertion_error(self):
        """Catching PtydScan init assertion error"""
        pars = u.Param()
        pars.dfile = 'test_path'

        # Source = None
        self.assertRaises(
            AssertionError,
            d.PtydScan,
            pars=pars
        )

        # Pars = None
        self.assertRaises(
            AssertionError,
            d.PtydScan,
            source='gibberish'
        )

        # Invalid source and pars
        self.assertRaises(
            AssertionError,
            d.PtydScan,
            source='gibberish',
            pars=pars
        )


class TestMoonFlowerScan(unittest.TestCase):
    """Test MoonFlower class"""

    def setUp(self):
        """Set up MoonFlower instance"""
        pars = u.Param()
        pars.positions_theory = np.arange(3)
        self.basic_mfs = d.MoonFlowerScan(pars=pars)

    def test_load_positions(self):
        """Run load_positions() of MoonFlower"""
        self.assertTrue(
            np.array_equal(
                self.basic_mfs.pos,
                self.basic_mfs.load_positions()
            ),
            msg='load_positions() of MoonFlower failed.'
        )

    def test_load_weight(self):
        """Run load_weight() of MoonFlower"""
        self.assertTrue(
            np.array_equal(
                self.basic_mfs.load_weight(),
                np.ones((256, 256))
            ),
            msg='load_weight() of MoonFlower failed.'
        )

    def test_load(self):
        """Run load() of MoonFlower"""
        indices = [1]
        load_return = self.basic_mfs.load(indices=indices)

        # Note: just checking shape of returned array
        self.assertTupleEqual(
            load_return[0][1].shape,
            (256, 256),
            msg='load() of MoonFlower failed.'
        )

        self.assertDictEqual(
            load_return[1],
            {},
            msg='load() of MoonFlower failed.'
        )

        self.assertDictEqual(
            load_return[2],
            {},
            msg='load() of MoonFlower failed.'
        )


class TestDataSource(unittest.TestCase):
    """Test DataSource class"""

    def setUp(self):
        """Set up DataSource instance"""
        # Note: just checking vanilla initialization,
        # proper testing requires dictionary of scan structures
        scans = u.Param()
        self.basic_datasource = d.DataSource(scans=scans)

    def test_scan_available_property(self):
        """Return scan_available property"""
        self.assertFalse(
            self.basic_datasource.scan_available,
            msg='Returning scan_available property failed.'
        )

    @unittest.skip('Testing requires dictionary of scan structures')
    def test_feed_data(self):
        """Run feed_data() of DataSource"""


if __name__ == '__main__':
    unittest.main()
