'''
Tests whether the framework saves the output files
'''

import unittest
import tempfile
import h5py as h5

from test import utils as tu
import ptypy.utils as u

class FileSavingTest(unittest.TestCase):
    def test_output_file_saving_rfile_is_None(self):
        engine_params = u.Param()
        engine_params.name = 'DM'
        engine_params.numiter = 5
        engine_params.alpha =1
        engine_params.probe_update_start = 2
        engine_params.overlap_converge_factor = 0.05
        engine_params.overlap_max_iterations = 10
        engine_params.probe_inertia = 1e-3
        engine_params.object_inertia = 0.1
        engine_params.fourier_relax_factor = 0.01
        engine_params.obj_smooth_std = 20
        outpath = tempfile.mkdtemp(prefix='something')
        PtychoOutput = tu.EngineTestRunner(engine_params,propagator='farfield',output_path=outpath, output_file=None)
        # should have an assert to check no file is output. How do I do that?


    def test_output_file_saving_rfile_set(self):
        engine_params = u.Param()
        engine_params.name = 'DM'
        engine_params.numiter = 5
        engine_params.alpha =1
        engine_params.probe_update_start = 2
        engine_params.overlap_converge_factor = 0.05
        engine_params.overlap_max_iterations = 10
        engine_params.probe_inertia = 1e-3
        engine_params.object_inertia = 0.1
        engine_params.fourier_relax_factor = 0.01
        engine_params.obj_smooth_std = 20
        outpath = tempfile.mkdtemp(prefix='something')
        PtychoOutput = tu.EngineTestRunner(engine_params,propagator='farfield',output_path=outpath, output_file=outpath+'reconstruction')

        file_path = outpath + 'reconstruction.ptyr'

        expected_nodes = ['/content',
                          '/content/obj',
                          '/content/obj/SMFG00',
                          '/content/obj/SMFG00/data',
                          '/content',
                          '/content/probe',
                          '/content/probe/SMFG00',
                          '/content/probe/SMFG00/data']



        try:
            f = h5.File(file_path, 'r')
        except IOError:
            self.fail(msg="File does not exist")

        for node in expected_nodes:
            self.assertTrue(node in f, msg='No entry for %s in this file' % node)

        # and now explictly can I load the data?
        try:
            obj = f['/content/obj/SMFG00/data'][0]
        except KeyError:
            self.fail(msg="Couldn't load the object data")
        try:
            probe = f['/content/probe/SMFG00/data'][0]
        except KeyError:
            self.fail(msg="Couldn't load the probe data")


    def test_output_file_saving_rfile_set_kind_minimal(self):
        '''

        kin can be minimal, fullflat or dump
        '''
        engine_params = u.Param()
        engine_params.name = 'DM'
        engine_params.numiter = 5
        engine_params.alpha =1
        engine_params.probe_update_start = 2
        engine_params.overlap_converge_factor = 0.05
        engine_params.overlap_max_iterations = 10
        engine_params.probe_inertia = 1e-3
        engine_params.object_inertia = 0.1
        engine_params.fourier_relax_factor = 0.01
        engine_params.obj_smooth_std = 20
        outpath = tempfile.mkdtemp(prefix='something')
        PtychoOutput = tu.EngineTestRunner(engine_params,propagator='farfield', output_path=outpath, output_file=outpath+'reconstruction')

        file_path = outpath + 'reconstruction.ptyr'

        expected_nodes = ['/content',
                          '/content/obj',
                          '/content/obj/SMFG00',
                          '/content/obj/SMFG00/data',
                          '/content',
                          '/content/probe',
                          '/content/probe/SMFG00',
                          '/content/probe/SMFG00/data']



        try:
            f = h5.File(file_path, 'r')
        except IOError:
            self.fail(msg="File does not exist")

        for node in expected_nodes:
            self.assertTrue(node in f, msg='No entry for %s in this file' % node)

        # and now explictly can I load the data?
        try:
            obj = f['/content/obj/SMFG00/data'][0]
        except KeyError:
            self.fail(msg="Couldn't load the object data")
        try:
            probe = f['/content/probe/SMFG00/data'][0]
        except KeyError:
            self.fail(msg="Couldn't load the probe data")


    def test_output_file_saving_separate_save_run_kind_minimal(self):
        '''

        kin can be minimal, fullflat or dump
        '''
        engine_params = u.Param()
        engine_params.name = 'DM'
        engine_params.numiter = 5
        engine_params.alpha =1
        engine_params.probe_update_start = 2
        engine_params.overlap_converge_factor = 0.05
        engine_params.overlap_max_iterations = 10
        engine_params.probe_inertia = 1e-3
        engine_params.object_inertia = 0.1
        engine_params.fourier_relax_factor = 0.01
        engine_params.obj_smooth_std = 20
        outpath = tempfile.mkdtemp(prefix='something')
        PtychoOutput = tu.EngineTestRunner(engine_params,propagator='farfield', output_path=outpath, output_file=None)
        file_path = outpath + 'reconstruction.h5'

        print("now I am saving with save_run")
        PtychoOutput.save_run(file_path, kind='minimal')



        # object_string = '/content/obj'
        # probe_string = '/content/probe'
        expected_nodes = ['/content',
                          '/content/obj',
                          '/content/obj/SMFG00',
                          '/content/obj/SMFG00/data',
                          '/content',
                          '/content/probe',
                          '/content/probe/SMFG00',
                          '/content/probe/SMFG00/data']



        try:
            f = h5.File(file_path, 'r')
        except IOError:
            self.fail(msg="File does not exist")

        for node in expected_nodes:
            self.assertTrue(node in f, msg='No entry for %s in this file' % node)

        # and now explictly can I load the data?
        try:
            obj = f['/content/obj/SMFG00/data'][0]
        except KeyError:
            self.fail(msg="Couldn't load the object data")
        try:
            probe = f['/content/probe/SMFG00/data'][0]
        except KeyError:
            self.fail(msg="Couldn't load the probe data")
