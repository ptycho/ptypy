"""\
Utility functions and classes to support MPI computing.

This file is part of the PTYPY package.
    module:: test_utils
.. moduleauthor:: Aaron Parsons <scientificsoftware@diamond.ac.uk>
    :copyright: Copyright 2014 by the PTYPY team, see AUTHORS.
    :license: GPLv2, see LICENSE for details.
    :platform: Unix
    :synopsis: utilities for the test framework
"""
import inspect
import os
import tempfile
from .. import utils as u

def get_test_data_path(name):
    path = inspect.stack()[0][1]
    return '/'.join(os.path.split(path)[0].split(os.sep)[:-2] +
                    ['test_data/', name,'/'])


def TestRunner(ptyscan_instance,r):
        outdir = tempfile.mkdtemp()
        data = u.Param()
        data.recipe = r
        data.dfile = '%s/prep.ptyd' % outdir
        data.save = 'append'
        a = ptyscan_instance(data)
        a.initialize()
        msg = a.auto(20)


# def get_
# data = u.Param()
# 
# data.recipe = r                     ## (27) Data preparation recipe container
# data.source = 'dls'                ## (28) Describes where to get the data from.
# data.dfile = '/dls/mx-scratch/aaron/data/testdata/prepd%s.ptyd' % int(sys.argv[1])  ## (29) Prepared data file path
# data.shape = 128                    ## (31) Shape of the region of interest cropped from the raw data
# data.save = 'link'                ## (32) Saving mode
# data.psize = 55e-6                    ## (34) Detector pixel size before rebinning
# data.distance = 1.59                ## (35) Sample-to-detector distance
# data.orientation =  2                ## (37) Data frame orientation
# data.energy = 9.1                    ## (38) Photon energy of the incident radiation
# data.auto_center=True
# 
# def set_options(path, **kwargs):
#     options = {}
#     options['transport'] = kwargs.get('transport', 'hdf5')
#     options['process_names'] = kwargs.get('process_names', 'CPU0')
#     options['data_file'] = path
#     options['process_file'] = kwargs.get('process_file', '')
#     options['out_path'] = kwargs.get('out_path', tempfile.mkdtemp())
#     options['inter_path'] = options['out_path']
#     options['log_path'] = options['out_path']
#     options['run_type'] = 'test'
#     options['verbose'] = 'True'
#     return options


# class PtyScanTestCase(unittest.TestCase):
#     def __init__(self):
#         self.p = 
# 
#     def setUp(self):
#         self.output = tempfile.mkdtemp()
#     
#     def setRecipe(self,r):
#         self.r = r