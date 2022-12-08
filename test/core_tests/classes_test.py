"""
Unittests for ptypy/core/classes.py

This module contains basic unittests for default parameters, class
initialization, and method and function calls for the classes.py module.

The order of the performed tests is:

- Default module parameters
- Base class initialization (including default parameters) and methods calling
- Module functions
- Container class initialization (including default parameters) and
  methods calling
- Storage class initialization (including default parameters) and
  methods calling
- View class initialization (including default parameters) and
  methods calling
- POD class initialization (including default parameters) and
  methods calling
- Freport class initialization (including default parameters) and
  methods calling

Functions or methods which cannot be tested at the moment are skipped.
Functional unittests (i.e. from the user's perspective) will be implemented in
the future.
Run python -m unittest discover in the ptypy root directory to perform the
tests.

This file is part of the PTYPY package.
    :copyright: Copyright 2014 by the PTYPY team, see AUTHORS.
    :license: see LICENSE for details.
"""

# Modules
import unittest
import numpy as np
import weakref as wr
import ptypy.core.classes as c
import ptypy.utils as u
from ptypy.core.ptycho import Ptycho
from ptypy.core.manager import ModelManager
from ptypy.core.geometry import Geo


# Unittests
class TestDefaultParameters(unittest.TestCase):
    """Test default parameters"""

    def test_default_parameters(self):
        """Default parameters unaltered"""
        self.assertEqual(c.DEFAULT_PSIZE, 1.,
                         'Default value changed.')

        self.assertEqual(c.DEFAULT_SHAPE, (1, 1, 1),
                         'Default value changed.')

        self.assertEqual(c.DEFAULT_ACCESSRULE.storageID, None,
                         'Default value changed.')

        self.assertEqual(c.DEFAULT_ACCESSRULE.shape, None,
                         'Default value changed.')

        self.assertEqual(c.DEFAULT_ACCESSRULE.coord, None,
                         'Default value changed.')

        self.assertEqual(c.DEFAULT_ACCESSRULE.psize, c.DEFAULT_PSIZE,
                         'Default value changed.')

        self.assertEqual(c.DEFAULT_ACCESSRULE.layer, 0,
                         'Default value changed.')

        self.assertTrue(c.DEFAULT_ACCESSRULE.active,
                        'Default value changed.')

    def test_prefixes(self):
        """Default prefixes unaltered"""
        self.assertEqual(c.BASE_PREFIX, 'B',
                         'Default prefix changed.')

        self.assertEqual(c.VIEW_PREFIX, 'V',
                         'Default prefix changed.')

        self.assertEqual(c.CONTAINER_PREFIX, 'C',
                         'Default prefix changed.')

        self.assertEqual(c.STORAGE_PREFIX, 'S',
                         'Default prefix changed.')

        self.assertEqual(c.POD_PREFIX, 'P',
                         'Default prefix changed.')

        self.assertEqual(c.MODEL_PREFIX, 'mod',
                         'Default prefix changed.')

        self.assertEqual(c.PTYCHO_PREFIX, 'pty',
                         'Default prefix changed.')

        self.assertEqual(c.GEO_PREFIX, 'G',
                         'Default prefix changed.')

        self.assertEqual(c.MEGAPIXEL_LIMIT, 50,
                         'Default MEGAPIXEL_LIMIT changed.')


class TestBase(unittest.TestCase):
    """Test Base class"""

    def setUp(self):
        """Set up Base instances"""
        self.basic_base = c.Base()
        self.test_owner = c.Base('test_owner', 'owner_id', BeOwner=True)
        self.test_member = c.Base(self.test_owner, 'member_id')

    def test_base_default_constants(self):
        """Default Base constants unaltered"""
        self.assertEqual(
            self.basic_base._CHILD_PREFIX,
            'ID',
            msg='Default _CHILD_PREFIX changed.'
        )

        self.assertEqual(
            self.basic_base._PREFIX,
            c.BASE_PREFIX,
            'Default _PREFIX changed.'
        )

    def test_init(self):
        """Assigning init attributes"""
        self.assertIsNone(
            self.basic_base.owner,
            'Assigning of instance attribute owner failed.'
        )

        self.assertIsNone(
            self.basic_base.ID,
            'Assigning of instance attribute ID failed.'
        )

        self.assertEqual(
            self.basic_base._pool,
            {},
            'Assigning of instance attribute _pool failed.'
        )

    def test_new_ptypy_object(self):
        """Registering of ptypy object to base"""
        # Note: just catching one possible case here
        self.assertTrue(
            self.test_owner._pool[
                self.test_member._PREFIX][
                self.test_member._PREFIX + 'member_id']
            is self.test_member,
            'Registering of ptypy object to base failed.'
        )

    def test_num_to_id(self):
        """Converting number to ID"""
        self.assertEqual(
            self.basic_base._num_to_id(24),
            '{0:04d}'.format(24),
            msg='Converting number to ID failed.'
        )

    def test_from_dict(self):
        """Create new base instance from dictionary"""
        self.assertIsInstance(
            self.basic_base._from_dict({'a': 1}),
            c.Base,
            'Creating new base instance from dictionary failed.'
        )

    @unittest.skip('Function not implemented in parent class.')
    def test_post_dict_import(self):
        """Changing of specific behavior of child classes after import"""
        # ToDo: test function for passing?

    def test_to_dict(self):
        """Extract information from container object and store in a dict"""
        self.assertListEqual(
            list(self.basic_base._to_dict().keys()),
            self.basic_base.__slots__,
            'Converting container object information to dictionary failed.'
        )

    def test_calc_mem_usage(self):
        """Calculating base memory usage"""
        # Note: just catching two possible case here

        # Test for basic class
        self.assertTupleEqual(
            self.basic_base.calc_mem_usage(),
            (64, 0, 0),
            'Calculating basic class memory usage failed.'
        )

        # Test for test_owner class
        self.assertTupleEqual(
            self.test_owner.calc_mem_usage(),
            (128, 64, 0),
            'Calculating test_owner class memory usage failed.'
        )

    def tearDown(self):
        """Clean up"""
        # Note: Not sure if this required here


class TestClassesFunctions(unittest.TestCase):
    """Test functions in classes"""

    def setUp(self):
        """Set up Base class instance with base ID"""
        self.base_with_id = c.Base('test_base_with_id', 'B', BeOwner=True)

    def test_get_class(self):
        """Determine ptypy class from unique `ID`"""
        self.assertIs(
            c.get_class(c.VIEW_PREFIX),
            c.View,
            'Getting ptypy class with VIEW_PREFIX failed.'
        )

        self.assertIs(
            c.get_class(c.PTYCHO_PREFIX),
            Ptycho,
            'Getting ptypy class with PTYCHO_PREFIX failed.'
        )

        self.assertIs(
            c.get_class(c.STORAGE_PREFIX),
            c.Storage,
            'Getting ptypy class with STORAGE_PREFIX failed.'
        )

        self.assertIs(
            c.get_class(c.CONTAINER_PREFIX),
            c.Container,
            'Getting ptypy class with CONTAINER_PREFIX failed.'
        )

        self.assertIs(
            c.get_class(c.BASE_PREFIX),
            c.Base,
            'Getting ptypy class with BASE_PREFIX failed.'
        )

        self.assertIs(
            c.get_class(c.POD_PREFIX),
            c.POD,
            'Getting ptypy class with POD_PREFIX failed.'
        )

        self.assertIs(
            c.get_class(c.PARAM_PREFIX),
            u.Param,
            'Getting ptypy class with PARAM_PREFIX failed.'
        )

        self.assertIs(
            c.get_class(c.MODEL_PREFIX),
            ModelManager,
            'Getting ptypy class with MODEL_PREFIX failed.'
        )

        self.assertIs(
            c.get_class(c.GEO_PREFIX),
            Geo,
            'Getting ptypy class with GEO_PREFIX failed.'
        )

    def test_valid_id(self):
        """ID of object `obj` compatible with the current format"""
        # Test valid case
        self.assertTrue(
            c.valid_ID(self.base_with_id),
            'ID of object `obj` incompatible with the current format'
        )

        # Test invalid case
        self.assertFalse(
            c.valid_ID(24),
            'ID of object `obj` incompatible with the current format'
        )

    @unittest.skip('Function not implemented yet.')
    def test_shift(self):
        """Placeholder for future subpixel shifting method"""

    def tearDown(self):
        """Clean up"""
        # Note: Not sure if this required here


class TestContainer(unittest.TestCase):
    """Test Container class"""

    def setUp(self):
        """Set up Container instances"""
        # Container for default parameters test
        self.basic_container_dpt = c.Container()
        self.basic_storage_dpt = c.Storage(self.basic_container_dpt)

        # Container for copies test
        self.basic_container_cpt = c.Container()
        self.basic_container_cpt_child = c.Container(self.basic_container_cpt)

        # Container for fill test
        self.basic_container_ft = c.Container()
        self.basic_storage_ft = c.Storage(self.basic_container_ft)

        # Container for View test
        self.basic_container_vt = c.Container()
        accessrule = u.Param()
        accessrule.shape = 1
        self.basic_view_vt = c.View(
            self.basic_container_vt, accessrule=accessrule
        )

        # Container for clear test
        self.basic_container_clt = c.Container()
        self.basic_storage_clt = c.Storage(
            self.basic_container_clt, shape=1, fill=24
        )

        # Container for __ifunction__ tests, empty and filled
        self.basic_container_ifte = c.Container()
        self.basic_storage_ifte = c.Storage(self.basic_container_ifte)

        self.basic_container_iftf = c.Container()
        self.basic_storage_iftf = c.Storage(
            self.basic_container_iftf, shape=1, fill=24
        )

        # Container for new storage test
        self.basic_container_nst = c.Container()

        # Maybe test empty container and also real container
                
    def test_container_default_constants(self):
        """Default container constants unaltered"""
        self.assertEqual(
            self.basic_container_dpt._PREFIX,
            c.CONTAINER_PREFIX,
            'Default _PREFIX changed.'
        )

    def test_init(self):
        """Calling super() and assigning init attributes"""
        self.assertIsInstance(
            self.basic_container_dpt,
            c.Base,
            'Container is not Base instance.'
        )

        self.assertEqual(
            self.basic_container_dpt.data_type,
            'complex',
            msg='Assigning of instance attribute data_type failed.'
        )

        self.assertIs(
            self.basic_container_dpt.original,
            self.basic_container_dpt,
            'Assigning of instance attribute original failed.'
        )

    def test_copies_property(self):
        """Property returning list of all copies"""
        self.assertListEqual(
            self.basic_container_cpt_child.copies,
            [],
            'Returning property list of all copies failed.'
        )

    @unittest.skip('Function does not work for default initialization.')
    def test_delete_copy(self):
        """Delete a copy or all copies of container from owner instance"""
        # ToDo: implement test case for function

    def test_dtype_property(self):
        """Numpy dtype of all internal data buffers"""
        # ToDo: define CType/FType as instance attribute in Base

        # Test basic_container
        self.assertEqual(
            self.basic_container_dpt.dtype,
            np.complex128,
            'Returning numpy dtype of all internal data buffers failed.'
        )

        # Test basic_storage
        self.assertEqual(
            self.basic_storage_dpt.dtype,
            np.complex128,
            'Returning numpy dtype of all internal data buffers failed.'
        )

    def test_S_property(self):
        """Internal dictionary of Storage instances in Container"""
        # ToDo: should be removed and replaced with storages to avoid confusion
        self.assertDictEqual(
            self.basic_container_dpt.S,
            {'S0000': self.basic_storage_dpt},
            'Returning internal dict of Storages in Container failed.'
        )

    def test_storages_property(self):
        """Internal dictionary of Storage instances in Container"""
        self.assertDictEqual(
            self.basic_container_dpt.storages,
            {'S0000': self.basic_storage_dpt},
            'Returning internal dict of Storages in Container failed.'
        )

    def test_Sp_property(self):
        """Internal dictionary of Storage instances in Container as Param"""
        # Test function returns u.Param()
        self.assertIs(
            type(self.basic_container_dpt.Sp),
            type(u.Param()),
            'Returning internal dict of Storages in Container as Param failed.'
        )

        # Test basic_storage is in returned u.Param()
        self.assertEqual(
            self.basic_container_dpt.Sp['S0000'],
            self.basic_storage_dpt,
            'Returning internal dict of Storages in Container as Param failed.'
        )

    def test_V_property(self):
        """Internal dictionary of View instances in Container"""
        # ToDo: should be removed and replaced with views to avoid confusion
        self.assertDictEqual(
            self.basic_container_vt.V,
            {'V0000': self.basic_view_vt},
            'Returning internal dict of Views in Container failed.'
        )

    def test_views_property(self):
        """Internal dictionary of View instances in Container"""
        self.assertDictEqual(
            self.basic_container_vt.views,
            {'V0000': self.basic_view_vt},
            'Returning internal dict of Views in Container failed.'
        )

    def test_Vp_property(self):
        """Internal dictionary of View instances in Container as Param"""
        # Test function returns u.Param()
        self.assertIs(
            type(self.basic_container_vt.Vp),
            type(u.Param()),
            'Returning internal dict of Views in Container as Param failed.'
        )

        # Test basic_view is in returned u.Param()
        self.assertEqual(
            self.basic_container_vt.Vp['V0000'],
            self.basic_view_vt,
            'Returning internal dict of Views in Container as Param failed.'
        )

    def test_size_property(self):
        """Return total number of pixels in container"""
        self.assertEqual(
            self.basic_container_dpt.size,
            1,
            'Returning total number of pixels in container failed.'
        )

    def test_nbytes_property(self):
        """Return total number of bytes of numpy array buffers in container"""
        self.assertEqual(
            self.basic_container_dpt.nbytes,
            16,
            'Returning total bytes number of array buffers in container failed.'
        )

    def test_views_in_storage(self):
        """Return a list of views on :any:`Storage` `s`"""
        # Note: just testing Container without any Views in it
        self.assertListEqual(
            self.basic_container_dpt.views_in_storage(self.basic_storage_dpt),
            [],
            'Returning list of views on :any:`Storage` `s` failed.'
        )

    @unittest.skip('Function fails during storage creation')
    def test_copy(self):
        """Create a new :any:`Container` matching self"""
        # Note: calling self.basic_container_dpt.copy() fails
        # ToDo: rewrite test or adjust function to work with basic container

    def test_fill(self):
        """Fill all storages with scalar value `fill`"""
        # Call function and set up verification array
        self.basic_container_ft.fill(24)
        self.verification_arr = np.zeros((1, 1, 1), dtype=complex)
        self.verification_arr[0] = 24

        # Compare data in storage with verification array
        self.assertTrue(
            np.array_equal(
                self.basic_container_ft.storages['S0000'].data,
                self.verification_arr
            ),
            'Filling all storages with scalar value failed.'
        )

    @unittest.skip('Not sure how to test this yet.')
    def test_allreduce(self):
        """MPI parallel ``allreduce`` with a sum as reduction"""
        # ToDo: implement assertion

    @unittest.skip('Function assigns np.empty arrays, asserting difficult.')
    def test_clear(self):
        """Reduce / delete all data in attached storages"""
        # ToDo: Function assigns np.empty arrays to data in storages,
        # asserting outcome difficult. Maybe use np.zeros instead.

    def test_new_storage(self):
        """Create and register a storage object"""
        self.basic_container_nst.new_storage()

        self.assertIsInstance(
            self.basic_container_nst.storages['S0000'],
            c.Storage,
            'Creating and registering a storage object failed.'
        )

    @unittest.skip('Not sure how to test this yet.')
    def test_reformat(self):
        """Reformat all storages in container"""
        # ToDo: implement assertion
        # Function does not return, simply modifies storages

    def test_report(self):
        """Return formatted report string on all storages in container"""
        # Note: can probably be done more gracefully
        self.assertEqual(
            self.basic_container_dpt.report(),
            'Containers ID: None\n'
            'Storage S0000\n'
            'Shape: (1, 1, 1)\n'
            'Pixel size (meters): 1 x 1\n'
            'Dimensions (meters): 1 x 1\n'
            'Number of views: 0\n',
            msg='Returning formatted report string on '
                'all storages in container failed.'
        )

    @unittest.skip('Testing string output is a bit greedy.')
    def test_formatted_report(self):
        """Return formatted string and a dict containing the information"""
        # Note: can probably be done more gracefully

        self.assertEqual(
            self.basic_container_dpt.formatted_report(),
            '(C)ontnr : '
            'Memory : '
            'Shape            : '
            'Pixel size      : '
            'Dimensions      : '
            'Views\n'
            '(S)torgs : '
            '(MB)   : '
            '(Pixel)          : '
            '(meters)        : '
            '(meters)        : '
            'act. \n'
            '----------------------------------------'
            '----------------------------------------\n'
            'None     :    0.0 : complex128\n'
            'S0000    :    0.0 :        1 * 1 * 1 : 1.00 * 1.00e+00 :'
            '   1.00*1.00e+00 :     0\n',
            msg='Returning formatted report string on '
                'all storages in container failed.'
        )

    def test__getitem__(self):
        """Access content through view"""
        # Test accessing content through view
        self.assertTrue(
            np.array_equal(
                self.basic_container_vt.__getitem__(self.basic_view_vt),
                np.zeros((1, 1), dtype=complex)
            ),
            'Accessing content through view failed.'
        )

        # Test accessing content through random input string
        self.assertRaises(
            ValueError,
            self.basic_container_dpt.__getitem__,
            view='random_input'
        )

    def test__setitem__(self):
        """Set content given by view"""
        # Test setting content through view
        self.basic_container_vt.__setitem__(self.basic_view_vt, 25)

        self.assertTrue(
            np.array_equal(
                self.basic_container_vt.__getitem__(self.basic_view_vt),
                np.array([[25. + 0.j]])
            ),
            'Setting content through view failed.'
        )

        # Test setting content through random input string
        self.assertRaises(
            ValueError,
            self.basic_storage_dpt.__setitem__,
            v='random_input',
            newdata='more_random_input'
        )

    @unittest.skip('String comparison not useful')
    def test_info(self):
        """Return container's total buffer space in bytes and storage info"""
        
        self.assertTupleEqual(
            self.basic_container_dpt.info(),
            (16,
             '          S0000 :'
             '    0.00 MB :'
             ':'
             ' data=(1, 1, 1) @complex128 psize=[ 1.  1.] center=[0 0]\n'),
            msg='Returning container buffer space (bytes) '
                'and storage info failed.'
        )

    def test__iadd__(self):
        """Testing __iadd__ function"""
        # Test container case
        self.basic_container_ifte.__iadd__(self.basic_container_iftf)

        self.assertTrue(
            np.array_equal(
                self.basic_container_ifte.storages['S0000'].data,
                np.array([[[24. + 0.j]]])
            ),
            'Testing __iadd__ function failed.'
        )

        # Test scalar case
        self.basic_container_dpt.__iadd__(25)

        self.assertTrue(
            np.array_equal(
                self.basic_container_dpt.storages['S0000'].data,
                np.array([[[25. + 0.j]]])
            ),
            'Testing __iadd__ function failed.'
        )

    def test__isub__(self):
        """Testing __isub__ function"""
        # Test container case
        self.basic_container_ifte.__isub__(self.basic_container_iftf)

        self.assertTrue(
            np.array_equal(
                self.basic_container_ifte.storages['S0000'].data,
                np.array([[[-24. + 0.j]]])
            ),
            'Testing __isub__ function failed.'
        )

        # Test scalar case
        self.basic_container_dpt.__isub__(25)

        self.assertTrue(
            np.array_equal(
                self.basic_container_dpt.storages['S0000'].data,
                np.array([[[-25. + 0.j]]])
            ),
            'Testing __isub__ function failed.'
        )

    def test__imul__(self):
        """Testing __imul__ function"""
        # Test container case
        # Not ideal, as value needs to be added first for reasonable check
        self.basic_container_ifte.__iadd__(1)
        self.basic_container_ifte.__imul__(self.basic_container_iftf)

        self.assertTrue(
            np.array_equal(
                self.basic_container_ifte.storages['S0000'].data,
                np.array([[[24. + 0.j]]])
            ),
            'Testing __imul__ function failed.'
        )

        # Test scalar case
        # Not ideal, as value needs to be added first for reasonable check
        self.basic_container_dpt.__iadd__(1)
        self.basic_container_dpt.__imul__(25)

        self.assertTrue(
            np.array_equal(
                self.basic_container_dpt.storages['S0000'].data,
                np.array([[[25. + 0.j]]])
            ),
            'Testing __imul__ function failed.'
        )

    def test__truediv__(self):
        """Testing __truediv__ function"""
        # Test container case
        # Not ideal, as value needs to be added first for reasonable check
        self.basic_container_ifte.__iadd__(24)
        self.basic_container_ifte.__truediv__(self.basic_container_iftf)

        self.assertTrue(
            np.array_equal(
                self.basic_container_ifte.storages['S0000'].data,
                np.array([[[1. + 0.j]]])
            ),
            'Testing __truediv__ function failed.'
        )

        # Test scalar case
        # Not ideal, as value needs to be added first for reasonable check
        self.basic_container_dpt.__iadd__(25)
        self.basic_container_dpt.__truediv__(25)

        self.assertTrue(
            np.array_equal(
                self.basic_container_dpt.storages['S0000'].data,
                np.array([[[1. + 0.j]]])
            ),
            'Testing __truediv__ function failed.'
        )

    def test__lshift__(self):
        """Testing __lshift__ function"""
        # Test container case
        self.basic_container_ifte.__lshift__(self.basic_container_iftf)

        self.assertTrue(
            np.array_equal(
                self.basic_container_ifte.storages['S0000'].data,
                np.array([[[24. + 0.j]]])
            ),
            'Testing __lshift__ function failed.'
        )

        # Test scalar case
        self.basic_container_dpt.__lshift__(25)

        self.assertTrue(
            np.array_equal(
                self.basic_container_dpt.storages['S0000'].data,
                np.array([[[25. + 0.j]]])
            ),
            'Testing __lshift__ function failed.'
        )

    def tearDown(self):
        """Clean up"""
        # Note: Not sure if this required here


class TestStorage(unittest.TestCase):
    """Test Storage class"""

    def setUp(self):
        """Set up Storage instances"""
        # Storage for default parameters test
        self.basic_container_dpt = c.Container()
        self.basic_storage_dpt = c.Storage(self.basic_container_dpt)

        # Storage for fill test
        self.basic_container_ft = c.Container()
        self.basic_storage_ft = c.Storage(self.basic_container_ft)

        # Storage for View test
        # Creating a view like this does not automatically link it to existing
        # storage. Shouldn't this be default behaviour?
        # self.basic_container_vt = c.Container()
        # self.basic_storage_vt = c.Storage(self.basic_container_vt)
        # self.basic_view_vt = c.View(self.basic_container_vt)
        # Adding View clashes with the np.empty test in test_init()

    def test_storage_default_constants(self):
        """Default container constants unaltered"""
        self.assertEqual(
            self.basic_storage_dpt._PREFIX,
            c.STORAGE_PREFIX,
            'Default _PREFIX changed.'
        )

    def test_init(self):
        """Calling super() and assigning init attributes"""
        self.assertIsInstance(
            self.basic_storage_dpt,
            c.Base,
            'Storage is not Base instance.'
        )

        self.assertEqual(
            self.basic_storage_dpt.shape,
            c.DEFAULT_SHAPE,
            'Assigning of instance attribute shape failed.'
        )

        self.assertEqual(
            self.basic_storage_dpt.data,
            np.empty(c.DEFAULT_SHAPE, np.complex128),
            'Assigning and filling of instance attribute data failed.'
        )

        self.assertEqual(
            self.basic_storage_dpt.layermap,
            [0],
            'Assigning of instance attribute layermap failed.'
        )

        self.assertEqual(
            self.basic_storage_dpt.nlayers,
            1,
            'Assigning of instance attribute nlayers failed.'
        )

        self.assertTrue(
            np.array_equal(
                self.basic_storage_dpt._center,
                np.zeros(2)
            ),
            'Assigning of instance attribute _center failed.'
        )

        self.assertTrue(
            np.array_equal(
                self.basic_storage_dpt.psize,
                np.ones(2)
            ),
            'Assigning of instance attribute psize failed.'
        )

        self.assertTrue(
            np.array_equal(
                self.basic_storage_dpt.origin,
                np.zeros(2)
            ),
            'Assigning of instance attribute origin failed.'
        )

        self.assertFalse(
            self.basic_storage_dpt.padonly,
            'Assigning of instance attribute padonly failed.'
        )

        self.assertFalse(
            self.basic_storage_dpt.model_initialized,
            'Assigning of instance attribute model_initialized failed.'
        )

    @unittest.skip("Doesn't work with slotted classes.")
    def test_to_dict(self):
        """Extract information from storage object and store in a dict"""
        # ToDo: specify Exception --> KeyError
        self.assertDictEqual(
            self.basic_storage_dpt._to_dict(),
            self.basic_storage_dpt.__dict__,
            'Converting storage object information to dictionary failed.'
        )

    @unittest.skip('Function not implemented.')
    def test_to_make_datalist(self):
        """Extract information from storage object and store in a dict"""
        # ToDo: check if required and implement assertion if needed

    def test_dtype(self):
        """Numpy dtype of all internal data buffers"""
        self.assertEqual(
            self.basic_storage_dpt.dtype,
            np.complex128,
            'Returning numpy dtype of all internal data buffers failed.'
        )

    @unittest.skip('Function fails during storage creation')
    def test_copy(self):
        """Return a copy of storage object"""
        # Note: calling self.basic_storage_dpt.copy() fails
        # ToDo: rewrite test or adjust function to work with basic storage

    def test_fill_data_set_to_None(self):
        """Fill all storages with scalar value `fill`"""
        # Call function and set up verification array
        self.basic_container_ft.storages['S0000'].data = None
        self.basic_storage_ft.fill()
        self.verification_arr = np.zeros((1, 1, 1), dtype=complex)

        # Compare data in storage with verification array
        self.assertTrue(
            np.array_equal(
                self.basic_container_ft.storages['S0000'].data,
                self.verification_arr
            ),
            'Filling all storages with scalar value failed.'
        )

    def test_fill_None_value(self):
        """Fill all storages with scalar value `fill`"""
        # Call function and set up verification array
        self.basic_storage_ft.fill(None)
        self.verification_arr = np.zeros((1, 1, 1), dtype=complex)

        # Compare data in storage with verification array
        self.assertTrue(
            np.array_equal(
                self.basic_container_ft.storages['S0000'].data,
                self.verification_arr
            ),
            'Filling all storages with scalar value failed.'
        )

    def test_fill_default_value(self):
        """Fill all storages with scalar value `fill`"""
        # Call function and set up verification array
        self.basic_storage_ft.fill()
        self.verification_arr = np.zeros((1, 1, 1), dtype=complex)

        # Compare data in storage with verification array
        self.assertTrue(
            np.array_equal(
                self.basic_container_ft.storages['S0000'].data,
                self.verification_arr
            ),
            'Filling all storages with scalar value failed.'
        )

    def test_fill_scalar_value(self):
        """Fill all storages with scalar value `fill`"""
        # Call function and set up verification array
        self.basic_storage_ft.fill(24)
        self.verification_arr = np.zeros((1, 1, 1), dtype=complex)
        self.verification_arr[0] = 24

        # Compare data in storage with verification array
        self.assertTrue(
            np.array_equal(
                self.basic_container_ft.storages['S0000'].data,
                self.verification_arr
            ),
            'Filling all storages with scalar value failed.'
        )

    def test_fill_1darray_value(self):
        """Fill all storages with scalar value `fill`"""
        # Assert ValueError for ndarray < 2D
        self.assertRaises(
            ValueError,
            self.basic_storage_ft.fill,
            fill=np.ones(1)
        )

    def test_fill_2darray_value(self):
        """Fill all storages with scalar value `fill`"""
        # Call function and set up verification array
        self.basic_storage_ft.fill(np.ones((1, 1)))
        self.verification_arr = np.ones((1, 1, 1), dtype=complex)

        # Compare data in storage with verification array
        self.assertTrue(
            np.array_equal(
                self.basic_container_ft.storages['S0000'].data,
                self.verification_arr
            ),
            'Filling all storages with scalar value failed.'
        )

    def test_fill_3darray_value(self):
        """Fill all storages with scalar value `fill`"""
        # Call function and set up verification array
        self.basic_storage_ft.fill(np.ones((1, 1, 1)))
        self.verification_arr = np.ones((1, 1, 1), dtype=complex)

        # Compare data in storage with verification array
        self.assertTrue(
            np.array_equal(
                self.basic_container_ft.storages['S0000'].data,
                self.verification_arr
            ),
            'Filling all storages with scalar value failed.'
        )

    def test_fill_4darray_value(self):
        """Fill all storages with scalar value `fill`"""
        # Assert ValueError for ndarray > 3D
        self.assertRaises(
            ValueError,
            self.basic_storage_ft.fill,
            fill=np.ones((1, 1, 1, 1))
        )

    @unittest.skip('Function simply calls Storage.update_views')
    def test_update(self):
        """Update internal state, including all views on this storage"""
        # ToDo: Storage.update() simply calls Storage.update_views()
        # remove and replace with Storage.update_views()

    @unittest.skip('Requires views for testing')
    def test_update_views(self):
        """Update internal state, including all views on this storage"""
        # ToDo: requires views in storage for testing

    def test_reformat(self):
        """Reformat all storages in container"""
        # Note: just testing storage without views on it
        # ToDo: proper function testing requires views
        self.assertIs(
            self.basic_storage_dpt.reformat(),
            self.basic_storage_dpt,
            'Reformatting all storages in container failed.'
        )

    @unittest.skip('Not sure how to test this yet')
    def test_update_distributed(self):
        """Update distributed"""
        # ToDo: implement assertion

    def test_to_pix(self):
        """Transform physical coordinates `coord` to pixel coordinates"""
        self.assertTrue(
            np.array_equal(
                self.basic_storage_dpt._to_pix((1, 1)),
                np.ones(2)
            ),
            'Transforming physical coordinates to pixel coordinates failed.'
        )

    def test_to_phys(self):
        """Transforms pixel coordinates `pix` to physical coordinates"""
        self.assertTrue(
            np.array_equal(
                self.basic_storage_dpt._to_phys((1, 1)),
                np.ones(2)
            ),
            'Transforming pixel coordinates to physical coordinates failed.'
        )

    def test_psize_property(self):
        """Return pixel size"""
        self.assertTrue(
            np.array_equal(
                self.basic_storage_dpt.psize,
                np.ones(2)
            ),
            'Returning pixel size failed.'
        )

    def test_psize_setter(self):
        """Set pixel size, and update all internal variables"""
        self.basic_storage_dpt.psize = (5, 5)
        self.assertTrue(
            np.array_equal(
                self.basic_storage_dpt.psize,
                np.array([5, 5])
            ),
            'Setting pixel size failed.'
        )

    def test_origin_property(self):
        """Return physical position of upper-left storage corner"""
        self.assertTrue(
            np.array_equal(
                self.basic_storage_dpt.origin,
                np.zeros(2)
            ),
            'Returning physical position of upper-left storage corner failed.'
        )

    def test_origin_setter(self):
        """Set origin and update all internal variables"""
        self.basic_storage_dpt.origin = (5, 5)
        self.assertTrue(
            np.array_equal(
                self.basic_storage_dpt.origin,
                np.array([5, 5])
            ),
            'Setting origin failed.'
        )

    def test_center_property(self):
        """Return origin position relative to upper-left storage corner"""
        self.assertTrue(
            np.array_equal(
                self.basic_storage_dpt.center,
                np.zeros(2)
            ),
            'Returning origin relative to upper-left storage corner failed.'
        )

    def test_center_setter(self):
        """Set center and update all internal variables"""
        self.basic_storage_dpt.center = (5, 5)
        self.assertTrue(
            np.array_equal(
                self.basic_storage_dpt.center,
                np.array([5, 5])
            ),
            'Setting center failed.'
        )

    def test_views_property(self):
        """Return all views referring to storage"""
        # Note: just testing storage without views on it
        # ToDo: proper function testing requires views
        self.assertListEqual(
            self.basic_storage_dpt.views,
            [],
            'Returning all views referring to storage failed.'
        )

    @unittest.skip('Not sure how to test this yet')
    def test_allreduce(self):
        """MPI parallel ``allreduce`` with default sum as reduction operation"""
        # ToDo: implement assertion

    @unittest.skip('Bug in function')
    def test_zoom_to_psize(self):
        """Change pixel size and zoom data buffer along last two axis"""
        # ToDo: implement assertion

    def test_grids(self):
        """Grids of internal buffer shape"""
        self.assertTupleEqual(
            self.basic_storage_dpt.grids(),
            (np.zeros((1, 1, 1)), np.zeros((1, 1, 1))),
            'Creating grids of internal buffer shape failed.'
        )

    def test_get_view_coverage(self):
        """Creating array representing view coverage per pixel"""
        self.assertTrue(
            np.array_equal(
                self.basic_storage_dpt.get_view_coverage(),
                np.zeros((1, 1, 1))
            ),
            'Creating array representing view coverage per pixel failed'
        )

    def test_report(self):
        """Return storage report as formatted string"""
        self.assertEqual(
            self.basic_storage_dpt.report(),
            'Shape: (1, 1, 1)\n'
            'Pixel size (meters): 1 x 1\n'
            'Dimensions (meters): 1 x 1\n'
            'Number of views: 0\n',
            msg='Returning storage report as formatted string failed'
        )

    @unittest.skip('Testing string output is a bit greedy.')
    def test_formatted_report(self):
        """Return formatted string and a dict containing the information"""
        # Test first part of report
        self.assertEqual(
            self.basic_storage_dpt.formatted_report()[0],
            '(C)ontnr : '
            'Memory : '
            'Shape            : '
            'Pixel size      : '
            'Dimensions      : '
            'Views\n'
            '(S)torgs : '
            '(MB)   : '
            '(Pixel)          : '
            '(meters)        : '
            '(meters)        : act. \n'
            '----------------------------------------'
            '----------------------------------------\n'
            'S0000    :    0.0 :        1 * 1 * 1 : '
            '1.00 * 1.00e+00 :   1.00*1.00e+00 :     0',
            msg='Returning formatted string and a dict containing '
                'the information of storage failed.'
        )

        # Test second part of report
        self.assertDictEqual(
            self.basic_storage_dpt.formatted_report()[1],
            {'dimension': (1.0, 1.0),
             'memory': 1.6e-05,
             'psize': (1.0, 1.0),
             'shape': (1, 1, 1),
             'views': 0},
            msg='Returning formatted string and a dict '
                'containing the information of storage failed.'
        )

    def test__getitem__(self):
        """Return view to internal data buffer"""
        # Note: just testing function raises error when called with argument
        # different to View
        # ToDo: proper function testing requires View
        self.assertRaises(
            ValueError,
            self.basic_storage_dpt.__getitem__,
            v='random_input'
        )

    def test__setitem__(self):
        """Set internal data buffer to `newdata` for the region of view `v`"""
        # Note: just testing function raises error when called with argument
        # different to View
        # ToDo: proper function testing requires View
        self.assertRaises(
            ValueError,
            self.basic_storage_dpt.__setitem__,
            v='random_input',
            newdata='more_random_input'
        )

    @unittest.skip('Testing string output is a bit greedy.')
    def test__str__(self):
        """Return __str__ of storage"""
        self.assertEqual(
            self.basic_storage_dpt.__str__(),
            '          S0000 :'
            '    0.00 MB :'
            ': data=(1, 1, 1) @complex128 psize=[ 1.  1.] center=[0 0]',
            msg='Returning __str__ of storage failed'
        )

    def tearDown(self):
        """Clean up"""
        # Note: Not sure if this required here


class TestView(unittest.TestCase):
    """Test View class"""

    def setUp(self):
        """Set up View instances"""
        # View for default parameters test
        self.basic_container_dpt = c.Container()
        accessrule = u.Param()
        accessrule.shape = 1
        self.basic_view_dpt = c.View(
            self.basic_container_dpt, accessrule=accessrule
        )

    def test_view_default_constants(self):
        """Default View constants unaltered"""
        self.assertEqual(
            self.basic_view_dpt.DEFAULT_ACCESSRULE,
            c.DEFAULT_ACCESSRULE,
            'Default DEFAULT_ACCESSRULE changed.'
        )

        self.assertEqual(
            self.basic_view_dpt._PREFIX,
            c.VIEW_PREFIX,
            'Default _PREFIX changed.'
        )

    def test_init(self):
        """Calling super() and assigning init attributes"""
        self.assertIsInstance(
            self.basic_view_dpt,
            c.Base,
            'View is not Base instance.'
        )

        self.assertIsNone(
            self.basic_view_dpt._pods,
            'Assigning of instance attribute _pods failed.'
        )

        self.assertIsNone(
            self.basic_view_dpt._pod,
            'Assigning of instance attribute _pod failed.'
        )

        self.assertTrue(
            self.basic_view_dpt.active,
            'Assigning of instance attribute active failed.'
        )

        # Originally set to None but then a Storage is created in self._set,
        # might be confusing behaviour
        self.assertIsInstance(
            self.basic_view_dpt.storage,
            c.Storage,
            'Assigning of instance attribute storage failed.'
        )

        self.assertIsNone(
            self.basic_view_dpt.storageID,
            'Assigning of instance attribute storageID failed.'
        )

        self.assertEqual(
            self.basic_view_dpt.dlayer,
            0,
            'Assigning of instance attribute dlayer failed.'
        )

        # Basically testing _set() called at the end of init() from now on

        self.assertTrue(
            np.array_equal(
                self.basic_view_dpt.psize,
                np.ones(2, dtype=float)
            ),
            'Assigning of instance attribute psize failed.'
        )

        self.assertTrue(
            np.array_equal(
                self.basic_view_dpt.shape,
                np.ones(2, dtype=int)
            ),
            'Assigning of instance attribute psize failed.'
        )

        self.assertTrue(
            np.array_equal(
                self.basic_view_dpt.coord,
                np.zeros(2, dtype=int)
            ),
            'Assigning of instance attribute psize failed.'
        )

    @unittest.skip('Testing string output is a bit greedy.')
    def test__str__(self):
        """Return __str__ of view"""
        self.assertEqual(
            self.basic_view_dpt.__str__(),
            'None -> S0000[V0000] : '
            'shape = [1 1] layer = 0 coord = [ 0.  0.]\n '
            'ACTIVE : slice = (0, slice(0, 1, None), slice(0, 1, None))',
            msg='Returning __str__ of view failed'
        )

    def test_slice_property(self):
        """Return a slice-tuple"""
        self.assertTupleEqual(
            self.basic_view_dpt.slice,
            (0, slice(0, 1, None), slice(0, 1, None)),
            'Returning a slice-tuple failed.'
        )

    @unittest.skip('Function not working correctly')
    def test_pod_property(self):
        """Return first :any:`POD` in the ``self.pods`` dict"""
        # pod -> NoneType not callable
        # Property cannot be tested in current state
        # as default value is not callable.
        # Does function become callable later on?
        # ToDO: fix self._pod() for None case

    def test_data_property(self):
        """Return view content in data buffer of associated storage"""
        self.assertTrue(
            np.array_equal(
                self.basic_view_dpt.data,
                np.zeros((1, 1), dtype=complex)
            ),
            'Returning view content in associated storage data buffer failed.'
        )

    def test_data_setter(self):
        """Set view content in data buffer of associated storage"""
        self.basic_view_dpt.data = 25

        self.assertTrue(
            np.array_equal(
                self.basic_view_dpt.data,
                np.array([[25. + 0.j]])
            ),
            'Setting view content in associated storage data buffer failed.'
        )

    def test_shape_property(self):
        """Return two dimensional shape of View"""
        self.assertTrue(
            np.array_equal(
                self.basic_view_dpt.shape,
                np.ones(2, dtype=int)
            ),
            'Returning two dimensional shape of View failed.'
        )

    def test_shape_setter(self):
        """Set two dimensional shape of View"""
        # Testing scalar case
        self.basic_view_dpt.shape = 5

        self.assertTrue(
            np.array_equal(
                self.basic_view_dpt.shape,
                np.array(([5, 5]))
            ),
            'Setting two dimensional shape of View failed.'
        )

        # Testing None case
        self.basic_view_dpt.shape = None

        self.assertIsNone(
            self.basic_view_dpt.shape,
            'Setting two dimensional shape of View failed.'
        )

    def test_dlow_property(self):
        """Return low side of the View's data range"""
        self.assertTrue(
            np.array_equal(
                self.basic_view_dpt.dlow,
                np.zeros(2, dtype=int)
            ),
            "Returning low side of View's data range failed."
        )

    def test_dlow_setter(self):
        """Set low side of the View's data range"""
        self.basic_view_dpt.dlow = 5, 10

        self.assertTrue(
            np.array_equal(
                self.basic_view_dpt.dlow,
                np.array([5, 10])
            ),
            "Setting low side of View's data range failed."
        )

    def test_dhigh_property(self):
        """Return high side of the View's data range"""
        self.assertTrue(
            np.array_equal(
                self.basic_view_dpt.dhigh,
                np.ones(2, dtype=int)
            ),
            "Returning high side of View's data range failed."
        )

    def test_dhigh_setter(self):
        """Set high side of the View's data range"""
        self.basic_view_dpt.dhigh = 10, 5

        self.assertTrue(
            np.array_equal(
                self.basic_view_dpt.dhigh,
                np.array([10, 5])
            ),
            "Setting high side of View's data range failed."
        )

    def test_dcoord_property(self):
        """Return center coordinate (index) in data buffer"""
        self.assertTrue(
            np.array_equal(
                self.basic_view_dpt.dcoord,
                np.zeros(2, dtype=int)
            ),
            "Returning center coordinate (index) in data buffer failed."
        )

    def test_dcoord_setter(self):
        """Set center coordinate (index) in data buffer"""
        self.basic_view_dpt.dcoord = 7, 8

        self.assertTrue(
            np.array_equal(
                self.basic_view_dpt.dcoord,
                np.array([7, 8])
            ),
            "Setting center coordinate (index) in data buffer failed."
        )

    def test_psize_property(self):
        """Return pixel size of the View"""
        self.assertTrue(
            np.array_equal(
                self.basic_view_dpt.psize,
                np.ones(2, dtype=float)
            ),
            "Returning pixel size of the View failed."
        )

    def test_psize_setter(self):
        """Set pixel size of the View"""
        # Testing scalar case
        self.basic_view_dpt.psize = 5

        self.assertTrue(
            np.array_equal(
                self.basic_view_dpt.psize,
                np.array(([5., 5.]))
            ),
            'Setting pixel size of the View failed.'
        )

        # Testing None case
        self.basic_view_dpt.psize = None

        self.assertIsNone(
            self.basic_view_dpt.psize,
            'Setting pixel size of the View failed.'
        )

    def test_coord_property(self):
        """Return the View's physical coordinate (meters)"""
        self.assertTrue(
            np.array_equal(
                self.basic_view_dpt.coord,
                np.zeros(2, dtype=float)
            ),
            "Returning the View's physical coordinate (meters) failed."
        )

    def test_coord_setter(self):
        """Set the View's physical coordinate (meters)"""
        # Testing scalar case
        self.basic_view_dpt.coord = 8, 9

        self.assertTrue(
            np.array_equal(
                self.basic_view_dpt.coord,
                np.array(([8., 9.]))
            ),
            "Setting the View's physical coordinate (meters) failed."
        )

        # Testing array case
        self.basic_view_dpt.coord = np.array([1., 2.])

        self.assertTrue(
            np.array_equal(
                self.basic_view_dpt.coord,
                np.array(([1., 2.]))
            ),
            "Setting the View's physical coordinate (meters) failed."
        )

        # Testing None case
        self.basic_view_dpt.coord = None

        self.assertTrue(
            np.array_equal(
                self.basic_view_dpt.coord,
                np.array(([0., 0.]))
            ),
            "Setting the View's physical coordinate (meters) failed."
        )

    def test_sp_property(self):
        """Return subpixel difference (m) between physical and data coord."""
        self.assertTrue(
            np.array_equal(
                self.basic_view_dpt.sp,
                np.zeros(2, dtype=float)
            ),
            "Returning subpixel difference of physical and data coord. failed."
        )

    def test_sp_setter(self):
        """Set subpixel difference (m) between physical and data coord."""
        # Testing scalar case
        self.basic_view_dpt.sp = 4, 5

        self.assertTrue(
            np.array_equal(
                self.basic_view_dpt.sp,
                np.array(([4., 5.]))
            ),
            "Setting subpixel difference of physical and data coord. failed."
        )

        # Testing array case
        self.basic_view_dpt.sp = np.array([6., 7.])

        self.assertTrue(
            np.array_equal(
                self.basic_view_dpt.sp,
                np.array(([6., 7.]))
            ),
            "Setting subpixel difference of physical and data coord. failed."
        )

        # Testing None case
        self.basic_view_dpt.sp = None

        self.assertTrue(
            np.array_equal(
                self.basic_view_dpt.sp,
                np.array(([0., 0.]))
            ),
            "Setting subpixel difference of physical and data coord. failed."
        )

    def tearDown(self):
        """Clean up"""
        # Note: Not sure if this required here


class TestPOD(unittest.TestCase):
    """Test POD class"""

    def setUp(self):
        """Set up POD instance"""
        # POD for default parameters test

        # Views for default parameter test
        accessrule = u.Param()
        accessrule.shape = 1
        self.basic_container_dpt = c.Container()
        self.basic_probe_view_dpt = c.View(
            self.basic_container_dpt, accessrule=accessrule
        )
        self.basic_obj_view_dpt = c.View(
            self.basic_container_dpt, accessrule=accessrule
        )
        self.basic_exit_view_dpt = c.View(
            self.basic_container_dpt, accessrule=accessrule
        )
        self.basic_diff_view_dpt = c.View(
            self.basic_container_dpt, accessrule=accessrule
        )
        self.basic_mask_view_dpt = c.View(
            self.basic_container_dpt, accessrule=accessrule
        )

        self.test_views = {
            'probe': self.basic_probe_view_dpt,
            'obj': self.basic_obj_view_dpt,
            'exit': self.basic_exit_view_dpt,
            'diff': self.basic_diff_view_dpt,
            'mask': self.basic_mask_view_dpt,
        }

        # Note: POD creation without view dict leads to several errors
        # Error in classes.py 1958: NoneType no attribute 'shape'
        # Error in classes.py 1959: NoneType no attribute 'CType'
        self.basic_pod_dpt = c.POD(views=self.test_views)

    def test_view_default_constants(self):
        """Default POD constants unaltered"""
        self.assertDictEqual(
            self.basic_pod_dpt.DEFAULT_VIEWS,
            {'probe': None,
             'obj': None,
             'exit': None,
             'diff': None,
             'mask': None, }
            ,
            'Default DEFAULT_ACCESSRULE changed.'
        )

        self.assertEqual(
            self.basic_pod_dpt._PREFIX,
            c.POD_PREFIX,
            'Default POD_PREFIX changed.'
        )

    def test_init(self):
        """Calling super() and assigning init attributes"""
        self.assertIsInstance(
            self.basic_pod_dpt,
            c.Base,
            'POD is not Base instance.'
        )

        self.assertFalse(
            self.basic_pod_dpt.is_empty,
            'Assigning of instance attribute is_empty failed.'
        )

        self.assertEqual(
            self.basic_pod_dpt.probe_weight,
            1.0,
            'Assigning of instance attribute probe_weight failed.'
        )

        self.assertEqual(
            self.basic_pod_dpt.object_weight,
            1.0,
            'Assigning of instance attribute object_weight failed.'
        )

        self.assertEqual(
            self.basic_pod_dpt.V,
            self.test_views,
            'Updating of instance attribute V (views) failed.'
        )

        self.assertIsNone(
            self.basic_pod_dpt.geometry,
            'Assigning of instance attribute geometry failed.'
        )

        self.assertEqual(
            self.basic_pod_dpt.ob_view,
            self.test_views['obj'],
            'Assigning of instance attribute ob_view failed.'
        )

        self.assertEqual(
            self.basic_pod_dpt.pr_view,
            self.test_views['probe'],
            'Assigning of instance attribute pr_view failed.'
        )

        self.assertEqual(
            self.basic_pod_dpt.di_view,
            self.test_views['diff'],
            'Assigning of instance attribute di_view failed.'
        )

        self.assertEqual(
            self.basic_pod_dpt.ex_view,
            self.test_views['exit'],
            'Assigning of instance attribute ex_view failed.'
        )

        self.assertEqual(
            self.basic_pod_dpt.ma_view,
            self.test_views['mask'],
            'Assigning of instance attribute ma_view failed.'
        )

    def test_active_property(self):
        """Return property describing pod activity"""
        self.assertTrue(
            self.basic_pod_dpt.active,
            "Returning convenience property describing pod activity failed."
        )

    @unittest.skip('Property not working for Default None case')
    def test_fw_property(self):
        """Return forward propagator of attached Geometry property"""
        # ToDo: Test with given Geometry and refactor Default case

    @unittest.skip('Property not working for Default None case')
    def test_bw_property(self):
        """Return backward propagator of attached Geometry property"""
        # ToDo: Test with given Geometry and refactor Default case

    def test_object_property(self):
        """Return property linking to slice of object"""
        self.assertTrue(
            np.array_equal(
                self.basic_pod_dpt.object,
                np.zeros((1, 1), dtype=complex)
            ),
            'Returning property linking to slice of object failed.'
        )

    def test_object_setter(self):
        """Set property linking to slice of object"""
        self.basic_pod_dpt.object = 25

        self.assertTrue(
            np.array_equal(
                self.basic_pod_dpt.object,
                np.array([[25. + 0.j]])
            ),
            'Setting property linking to slice of object failed.'
        )

    def test_probe_property(self):
        """Return property linking to slice of probe"""
        self.assertTrue(
            np.array_equal(
                self.basic_pod_dpt.probe,
                np.zeros((1, 1), dtype=complex)
            ),
            'Returning property linking to slice of probe failed.'
        )

    def test_probe_setter(self):
        """Set property linking to slice of probe"""
        self.basic_pod_dpt.probe = 26

        self.assertTrue(
            np.array_equal(
                self.basic_pod_dpt.probe,
                np.array([[26. + 0.j]])
            ),
            'Setting property linking to slice of probe failed.'
        )

    def test_exit_property(self):
        """Return property linking to slice of exit wave"""
        self.assertTrue(
            np.array_equal(
                self.basic_pod_dpt.exit,
                np.zeros((1, 1), dtype=complex)
            ),
            'Returning property linking to slice of exit wave failed.'
        )

    def test_exit_setter(self):
        """Set property linking to slice of exit wave"""
        self.basic_pod_dpt.exit = 27

        self.assertTrue(
            np.array_equal(
                self.basic_pod_dpt.exit,
                np.array([[27. + 0.j]])
            ),
            'Setting property linking to slice of exit wave failed.'
        )

    def test_diff_property(self):
        """Return property linking to slice of diffraction"""
        self.assertTrue(
            np.array_equal(
                self.basic_pod_dpt.diff,
                np.zeros((1, 1), dtype=complex)
            ),
            'Returning property linking to slice of diffraction failed.'
        )

    def test_diff_setter(self):
        """Set property linking to slice of diffraction"""
        self.basic_pod_dpt.diff = 28

        self.assertTrue(
            np.array_equal(
                self.basic_pod_dpt.diff,
                np.array([[28. + 0.j]])
            ),
            'Setting property linking to slice of diffraction failed.'
        )

    def test_mask_property(self):
        """Return property linking to slice of masking"""
        self.assertTrue(
            np.array_equal(
                self.basic_pod_dpt.mask,
                np.zeros((1, 1), dtype=complex)
            ),
            'Returning property linking to slice of masking failed.'
        )

    def test_mask_setter(self):
        """Set property linking to slice of masking"""
        self.basic_pod_dpt.mask = 29

        self.assertTrue(
            np.array_equal(
                self.basic_pod_dpt.mask,
                np.array([[29. + 0.j]])
            ),
            'Setting property linking to slice of masking failed.'
        )

    def tearDown(self):
        """Clean up"""
        # Note: Not sure if this required here


class TestFreport(unittest.TestCase):
    """Test _Freport class"""

    def setUp(self):
        """Set up _Freport instance"""
        # _Freport for default parameters test
        self.basic_Freport_dpt = c._Freport()

    def test_init(self):
        """Assigning init attributes"""

        self.assertEqual(
            self.basic_Freport_dpt.offset,
            8,
            'Assigning of instance attribute offset failed.'
        )

        self.assertDictEqual(
            self.basic_Freport_dpt.desc,
            dict([
                ('memory', 'Memory'),
                ('shape', 'Shape'),
                ('psize', 'Pixel size'),
                ('dimension', 'Dimensions'),
                ('views', 'Views'),
            ]),
            'Assigning of instance attribute desc failed.'
        )

        self.assertDictEqual(
            self.basic_Freport_dpt.units,
            dict([
                ('memory', '(MB)'),
                ('shape', '(Pixel)'),
                ('psize', '(meters)'),
                ('dimension', '(meters)'),
                ('views', 'act.'),
            ]),
            'Assigning of instance attribute units failed.'
        )

        self.assertListEqual(
            self.basic_Freport_dpt.table,
            [('memory', 6),
             ('shape', 16),
             ('psize', 15),
             ('dimension', 15),
             ('views', 5), ],
            'Assigning of instance attribute table failed.'
        )

        self.assertEqual(
            self.basic_Freport_dpt.h1,
            "(C)ontnr",
            msg='Assigning of instance attribute h1 failed.'
        )

        self.assertEqual(
            self.basic_Freport_dpt.h2,
            "(S)torgs",
            msg='Assigning of instance attribute h2 failed.'
        )

        self.assertEqual(
            self.basic_Freport_dpt.separator,
            " : ",
            msg='Assigning of instance attribute separator failed.'
        )

        self.assertEqual(
            self.basic_Freport_dpt.headline,
            "-",
            msg='Assigning of instance attribute headline failed.'
        )

    def test_header(self):
        """Create a header for formatted report method"""
        # Testing as_string=True case
        self.assertEqual(
            self.basic_Freport_dpt.header(as_string=True),
            '(C)ontnr : '
            'Memory : '
            'Shape            : '
            'Pixel size      : '
            'Dimensions      : '
            'Views\n'
            '(S)torgs : '
            '(MB)   : '
            '(Pixel)          : '
            '(meters)        : '
            '(meters)        : '
            'act. \n'
            '----------------------------------------'
            '----------------------------------------\n',
            msg='Creating a header for formatted report method failed.'
        )

        # Testing as_string=True case
        self.assertListEqual(
            self.basic_Freport_dpt.header(as_string=False),
            ['(C)ontnr : Memory : Shape            : '
             'Pixel size      : Dimensions      : Views',
             '(S)torgs : (MB)   : (Pixel)          : '
             '(meters)        : (meters)        : act. ',
             '-----------------------------------------'
             '---------------------------------------'],
            msg='Creating a header for formatted report method failed.'
        )

    def tearDown(self):
        """Clean up"""
        # Note: Not sure if this required here


class TestGenericContainerStorageViewSetup(unittest.TestCase):
    """Test Container/Storage/View setup as used in 'Concepts and Classes'"""

if __name__ == '__main__':
    unittest.main()
