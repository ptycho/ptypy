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


def TestRunner(ptyscan_instance,r=u.Param(),data=u.Param(),save_type='append', auto_frames=20, ncalls=1):
        u.verbose.set_level(3)
        out_dict = {}
        outdir = tempfile.mkdtemp()
        data.recipe = r
        data.dfile = '%s/prep.ptyd' % outdir
        out_dict['output_file'] = data.dfile
        data.save = save_type
        a = ptyscan_instance(data)
        a.initialize()
        out_dict['msgs'] = []
        i=0
        while i<ncalls:
            out_dict['msgs'].append(a.auto(auto_frames))
            i+=1
        return out_dict

