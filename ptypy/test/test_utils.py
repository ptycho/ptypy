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
import shutil
import os
import tempfile
from .. import utils as u
from ..core import Ptycho


def get_test_data_path(name):
    path = inspect.stack()[0][1]
    return '/'.join(os.path.split(path)[0].split(os.sep)[:-2] +
                    ['test_data/', name,'/'])


def PtyscanTestRunner(ptyscan_instance,r=u.Param(),data=u.Param(),save_type='append', auto_frames=20, ncalls=1, cleanup=True):
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
        if cleanup:
            shutil.rmtree(outdir)
        return out_dict

def EngineTestRunner(engine_params,propagator='farfield'):

    p = u.Param()
    p.verbose_level = 3
    p.io = u.Param()
    p.io.home = './'
    p.io.rfile = None
    p.io.autosave = u.Param(active=True)
    p.io.autoplot = u.Param(active=True)
#     p.io.autoplot.threaded = True
#     p.io.autoplot.imfile = 'plots/%(run)s/%(run)s_%(engine)s_%(iterations)04d.png'
#     p.io.autoplot.interval = 1
#     p.io.autoplot.layout = 'weak'
#     p.io.autoplot.dump = False
#     p.io.autoplot.make_movie = False
    p.ipython_kernel = False
    p.scans = u.Param()
    p.scans.MF = u.Param()
    p.scans.MF.name = 'Full'
    p.scans.MF.propagation = propagator
    p.scans.MF.data = u.Param()
    p.scans.MF.data.name = 'MoonFlowerScan'
    p.scans.MF.data.positions_theory = None
    p.scans.MF.data.auto_center = None
    p.scans.MF.data.min_frames = 1
    p.scans.MF.data.orientation = None
    p.scans.MF.data.num_frames = 100
    p.scans.MF.data.energy = 6.2
    p.scans.MF.data.shape = 256
    p.scans.MF.data.chunk_format = '.chunk%02d'
    p.scans.MF.data.rebin = None
    p.scans.MF.data.experimentID = None
    p.scans.MF.data.label = None
    p.scans.MF.data.version = 0.1
    p.scans.MF.data.dfile = None
    p.scans.MF.data.psize = 0.000172
    p.scans.MF.data.load_parallel = None
    p.scans.MF.data.distance = 7.0
    p.scans.MF.data.save = None
    p.scans.MF.data.center = 'fftshift'
    p.scans.MF.data.photons = 100000000.0
    p.scans.MF.data.psf = 0.0
    p.scans.MF.data.density = 0.2
    p.engines = u.Param()
    p.engines.engine00 = engine_params
    P = Ptycho(p, level=5)
    return P

