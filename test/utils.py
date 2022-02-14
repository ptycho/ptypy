"""\
Utility functions and classes to support MPI computing.

This file is part of the PTYPY package.
    module:: utils
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
from ptypy import utils as u
from ptypy.core import Ptycho


def get_test_data_path(name):
    path = inspect.stack()[0][1]
    return '/'.join(os.path.split(path)[0].split(os.sep)[:-2] +
                    ['test_data/', name,'/'])


def PtyscanTestRunner(ptyscan_instance, data_params, save_type='append', auto_frames=20, ncalls=1, cleanup=True):
        u.verbose.set_level(3)
        out_dict = {}
        outdir = tempfile.mkdtemp()
        data_params.dfile = '%s/prep.h5' % outdir
        out_dict['output_file'] = data_params.dfile
        data_params.save = save_type
        a = ptyscan_instance(data_params)
        a.initialize()
        out_dict['msgs'] = []
        i=0
        while i<ncalls:
            out_dict['msgs'].append(a.auto(auto_frames))
            i+=1
        if cleanup:
            shutil.rmtree(outdir)
        return out_dict


def EngineTestRunner(engine_params,propagator='farfield',output_path='./', output_file=None,
                    autosave=True, scanmodel="Full", verbose_level="info", init_correct_probe=False):


    p = u.Param()
    # p.data_type = "double"
    p.verbose_level = verbose_level
    p.io = u.Param()
    p.io.home = output_path
    p.io.rfile = "%s.ptyr" % output_file
    p.io.interaction = u.Param()
    p.io.interaction.active = False
    p.io.autosave = u.Param(active=autosave)
    p.io.autoplot = u.Param(active=False)
    p.scans = u.Param()
    p.scans.MF = u.Param()
    p.scans.MF.name = scanmodel
    p.scans.MF.propagation = propagator
    p.scans.MF.data = u.Param()
    p.scans.MF.data.name = 'MoonFlowerScan'
    p.scans.MF.data.num_frames = 200
    p.scans.MF.data.shape = 64
    p.scans.MF.data.save = None
    p.scans.MF.data.photons = 1e8
    p.scans.MF.data.psf = 0.0
    p.scans.MF.data.density = 0.2
    p.scans.MF.data.add_poisson_noise = False
    p.scans.MF.coherence = u.Param()
    p.scans.MF.coherence.num_probe_modes = 1
    p.engines = u.Param()
    p.engines.engine00 = engine_params
    P = Ptycho(p, level=4)
    if init_correct_probe:
        P.probe.S['SMFG00'].data[0] = P.model.scans['MF'].ptyscan.pr
    P.run()
    return P

