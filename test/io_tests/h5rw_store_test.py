'''

This tests the functionality of h5rw and can be broken to include new types

'''

import unittest
import tempfile
import shutil
import h5py as h5
import ptypy.io as io
import ptypy.utils as u
import numpy as np
import collections

class owntype(object):
    pass

class H5rwStoreTest(unittest.TestCase):

    def setUp(self):
        io.h5options['UNSUPPORTED'] = 'fail'
        self.folder = tempfile.mkdtemp(suffix="H5rwTest")
        self.filepath = self.folder +'%s.h5'

    def tearDown(self):
        shutil.rmtree(self.folder)

    def test_store_str(self):
        data = 'test_string'
        content = {'string data': data}
        try:
            io.h5write(self.filepath % "store_string_test", content=content)
        except:
            self.fail(msg="Couldn't store a string type")

    def test_store_unicode(self):
        data = u'test_string \xe1'
        content = {'unicode data': data}
        try:
            io.h5write(self.filepath % "store_unicode_test", content=content)
        except:
            self.fail(msg="Couldn't store a unicode type")

    def test_store_dict(self):
        data = {'Monkey': "apple", "Moon" : 1, "flower": 2.0, 'an array' : np.ones((3,3))}
        content = {'dict data': data}
        try:
            io.h5write(self.filepath % "store_dict_test", content=content)
        except:
            self.fail(msg="Couldn't store a dict type")

    def test_store_ordered_dict(self):
        data = collections.OrderedDict()
        data['Monkey'] =  "apple"
        data["Moon"] =  "1"
        data["flower"] =  2.0
        data['an array'] = np.ones((3,3))
        content = {'ordered dict data': data}
        print(self.filepath % "store_ordered_dict_test")
        try:
            io.h5write(self.filepath % "store_ordered_dict_test", content=content)
        except:
            self.fail(msg="Couldn't store an OrderedDict type")

    def test_store_param(self):
        data = u.Param()
        data.monkey = "apple"
        data.moon = u.Param()
        data.moon.flower = 2.0

        content = {'string data': data}
        try:
            io.h5write(self.filepath % "store_param_test", content=content)
        except:
            self.fail(msg="Couldn't store a param type")

    def test_store_list(self):
        data = ['monkey', 'apple', 1.0, u'mo\xe1on']


        content = {'list data': data}
        try:
            io.h5write(self.filepath % "store_list_test", content=content)
        except:
            self.fail(msg="Couldn't store a list type")

    def test_store_tuple(self):
        data = ('monkey', 'apple', (1.0, u'mo\xe1on'))

        content = {'tuple data': data}
        try:
            io.h5write(self.filepath % "store_tuple_test", content=content)
        except:
            self.fail(msg="Couldn't store a tuple type")

    def test_store_ndarray(self):
        data = np.array([[1 ,2 ,3],[4, 5, 6], [7, 8, 9]])

        content = {'array data': data}
        try:
            io.h5write(self.filepath % "store_array_test", content=content)
        except:
            self.fail(msg="Couldn't store a array type")

    def test_store_numpy_record_array(self):
        data = np.recarray((8,), dtype=[('ID','<U16')])

        content = {'record array data': data}
        try:
            io.h5write(self.filepath % "store_record_array_test", content=content)
        except:
            self.fail(msg="Couldn't store a record array type")

    def test_store_scalar(self):
        data = 1.0

        content = {'scalar data': data}
        try:
            io.h5write(self.filepath % "store_scalar_test", content=content)
        except:
            self.fail(msg="Couldn't store a scalar type")

    def test_store_none(self):
        data = None

        content = {'Nonetype data': data}
        try:
            io.h5write(self.filepath % "store_None_test", content=content)
        except:
            self.fail(msg="Couldn't store a None type")

    @unittest.skip("I don't quite get what this does yet. To be filled in later.")
    def test_store_STR_CONVERT(self):
        '''
        What does this do?
        '''
        pass

    def test_fail_unsupported(self):

        def test_func():
            data = owntype()
            content = {'Owntype data': data}
            io.h5write(self.filepath % "store_dummytype_test", content=content)
        self.assertRaises(RuntimeError, test_func)


    def test_ignore_unsupported(self):
        io.h5options['UNSUPPORTED'] = 'ignore'

        def test_func():
            data = owntype()
            content = {'Owntype data': data}
            io.h5write(self.filepath % "store_dummytype_test", content=content)
        try:
            test_func()
        except:
            self.fail(msg="This should not have produced an exception!")

    def test_pickle_unsupported(self):
        io.h5options['UNSUPPORTED'] = 'pickle'

        def test_func():
            data = owntype()
            content = {'pickle data': data}
            io.h5write(self.filepath % "store_pickle_test", content=content)

        try:
            test_func()
        except:
            self.fail(msg="This should not have produced an exception!")

if __name__=='__main__':
    unittest.main()