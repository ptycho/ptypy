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

