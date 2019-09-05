'''

This tests the functionality of h5rw reading it's own data back in and can be broken to include new types
Since it's primary purpose is to read back it's own types, we create the test files using h5write. Lazy, but covers the use case.
'''

import unittest
import tempfile
import shutil
import h5py as h5
import ptypy.io as io
import ptypy.utils as u
import numpy as np
import collections


class H5rwLoadTest(unittest.TestCase):

    def setUp(self):
        io.h5options['UNSUPPORTED'] = 'fail'
        self.folder = tempfile.mkdtemp(suffix="H5rwTest")
        self.filepath = self.folder +'%s.h5'

    def tearDown(self):
        shutil.rmtree(self.folder)

    # @unittest.skip("Not yet implemented")
    def test_load_str(self):
        data = 'test_string'
        content = {'string data': data}
        io.h5write(self.filepath % "load_string_test", content=content)
        # and now read it back in...
        out = io.h5read(self.filepath % "load_string_test", "content")["content"]
        np.testing.assert_equal(content["string data"], out["string data"], err_msg="Can't read back in a string that we saved.")

    def test_load_unicode(self):
        data = u'test_string \xe1'
        content = {'unicode data': data}
        io.h5write(self.filepath % "load_unicode_test", content=content)
        out = io.h5read(self.filepath % "load_unicode_test", "content")["content"]
        np.testing.assert_equal(content["unicode data"], out["unicode data"], err_msg="Can't read back in a unicode string that we saved.")

    def test_load_dict(self):
        data = {'Monkey': "apple", "Moon" : 1, "flower": 2.0, 'an array' : np.ones((3,3))}
        content = {'dict data': data}
        io.h5write(self.filepath % "load_dict_test", content=content)
        out = io.h5read(self.filepath % "load_dict_test", "content")["content"]
        np.testing.assert_equal(content["dict data"], out["dict data"], err_msg="Can't read back in a dict that we saved.")

    def test_load_ordered_dict(self):
        data = collections.OrderedDict()
        data['Monkey'] =  "apple"
        data["Moon"] =  "1"
        data["flower"] =  2.0
        data['an array'] = np.ones((3,3))
        content = {'ordered dict data': data}
        io.h5write(self.filepath % "load_ordered_dict_test", content=content)
        out = io.h5read(self.filepath % "load_ordered_dict_test", "content")["content"]
        np.testing.assert_equal(content["ordered dict data"], out["ordered dict data"],
                                err_msg="Can't read back in an ordered dict that we saved.")

    def test_load_param(self):
        data = u.Param()
        data.monkey = "apple"
        data.moon = u.Param()
        data.moon.flower = 2.0

        content = {'param data': data}
        io.h5write(self.filepath % "load_param_test", content=content)
        out = io.h5read(self.filepath % "load_param_test", "content")["content"]
        np.testing.assert_equal(content["param data"], out["param data"],
                                err_msg="Can't read back in a param that we saved.")

    def test_load_list(self):
        data = ['monkey', 'apple', 1.0, 'moon']
        content = {'list data': data}
        io.h5write(self.filepath % "load_list_test", content=content)
        out = io.h5read(self.filepath % "load_list_test", "content")["content"]
        np.testing.assert_equal(content["list data"], out["list data"],
                                err_msg="Can't read back in a list that we saved.")

    def test_load_tuple(self):
        data = ('monkey', 'apple', (1.0, u'moon'))
        content = {'tuple data': data}
        io.h5write(self.filepath % "load_tuple_test", content=content)
        out = io.h5read(self.filepath % "load_tuple_test", "content")["content"]
        np.testing.assert_equal(content["tuple data"], out["tuple data"],
                                err_msg="Can't read back in a tuple that we saved.")

    def test_load_ndarray(self):
        data = np.array([[1 ,2 ,3],[4, 5, 6], [7, 8, 9]])
        content = {'array data': data}
        io.h5write(self.filepath % "load_array_test", content=content)
        out = io.h5read(self.filepath % "load_array_test", "content")["content"]
        np.testing.assert_array_equal(content["array data"], out["array data"],
                                err_msg="Can't read back in an array that we saved.")

    def test_load_numpy_record_array(self):
        data = np.recarray((8,), dtype=[('ID','<U16')])
        content = {'record array data': data}
        io.h5write(self.filepath % "load_record_array_test", content=content)
        out = io.h5read(self.filepath % "load_record_array_test", "content")["content"]
        np.testing.assert_array_equal(content["record array data"], out["record array data"],
                                err_msg="Can't read back in a record array that we saved.")

    def test_load_scalar(self):
        data = 1.0
        content = {'scalar data': data}
        io.h5write(self.filepath % "load_scalar_test", content=content)
        out = io.h5read(self.filepath % "load_scalar_test", "content")["content"]
        np.testing.assert_array_equal(content["scalar data"], out["scalar data"],
                                err_msg="Can't read back in a scalar that we saved.")

    def test_load_none(self):
        data = None
        content = {'Nonetype data': data}
        io.h5write(self.filepath % "load_None_test", content=content)
        out = io.h5read(self.filepath % "load_None_test", "content")["content"]
        np.testing.assert_array_equal(content["Nonetype data"], out["Nonetype data"],
                                err_msg="Can't read back in a None that we saved.")

    @unittest.skip("I don't quite get what this does yet. To be filled in later.")
    def test_load_STR_CONVERT(self):
        '''
        What does this do?
        '''
        pass

    def test_pickle_unsupported(self):
        io.h5options['UNSUPPORTED'] = 'pickle'
        from ptypy.core.data import PtyScan

        data = PtyScan()
        content = {'pickle data': data}
        io.h5write(self.filepath % "load_pickle_test", content=content)
        out = io.h5read(self.filepath % "load_pickle_test", "content")["content"]
        np.testing.assert_equal(type(out['pickle data']), type(content['pickle data']))
        np.testing.assert_equal(out['pickle data'].__dict__, content['pickle data'].__dict__)