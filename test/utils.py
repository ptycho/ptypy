"""\
Utility functions and classes to support MPI computing.

This file is part of the PTYPY package.
    module:: utils
.. moduleauthor:: Aaron Parsons <scientificsoftware@diamond.ac.uk>
    :copyright: Copyright 2014 by the PTYPY team, see AUTHORS.
    :license: see LICENSE for details.
    :platform: Unix
    :synopsis: utilities for the test framework
"""
import inspect
import shutil
import os
import tempfile
import numpy as np
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


def EngineTestRunner2(engine_params,propagator='farfield',output_path='./', output_file=None,
                    autosave=True, scanmodel="Full", verbose_level="info", init_correct_probe=False):

    p = u.Param()
    p.verbose_level = verbose_level
    p.io = u.Param()
    p.io.home = output_path
    p.io.rfile = "%s.ptyr" % output_file
    p.io.interaction = u.Param()
    p.io.interaction.active = False
    p.io.autosave = u.Param(active=autosave)
    p.io.autoplot = u.Param(active=False)

    # Simulation parameters
    sim = u.Param()
    sim.energy = 17.0
    sim.distance = 2.886
    sim.psize = 51e-6
    sim.shape = 128
    sim.xy = u.Param()
    sim.xy.model = "round"
    sim.xy.spacing = 250e-9
    sim.xy.steps = 30
    sim.xy.extent = 4e-6

    sim.illumination = u.Param()
    sim.illumination.model = None
    sim.illumination.photons = 3e8
    sim.illumination.aperture = u.Param()
    sim.illumination.aperture.diffuser = None
    sim.illumination.aperture.form = "rect"
    sim.illumination.aperture.size = 35e-6
    sim.illumination.aperture.central_stop = None
    sim.illumination.propagation = u.Param()
    sim.illumination.propagation.focussed = 0.08
    sim.illumination.propagation.parallel = 0.0014
    sim.illumination.propagation.spot_size = None

    sim.sample = u.Param()
    sim.sample.model = u.xradia_star((1000,1000),minfeature=3,contrast=0.0)
    sim.sample.process = u.Param()
    sim.sample.process.offset = (100,100)
    sim.sample.process.zoom = 1.0
    sim.sample.process.formula = "Au"
    sim.sample.process.density = 19.3
    sim.sample.process.thickness = 2000e-9
    sim.sample.process.ref_index = None
    sim.sample.process.smoothing = None
    sim.sample.fill = 1.0+0.j

    sim.detector = 'GenericCCD32bit'
    sim.verbose_level = 1
    sim.psf = 1. # emulates partial coherence
    sim.plot = False

    # Scan model and initial value parameters
    p.scans = u.Param()
    p.scans.scan00 = u.Param()
    p.scans.scan00.name = scanmodel
    p.scans.scan00.coherence = u.Param()
    p.scans.scan00.coherence.num_probe_modes = 1
    p.scans.scan00.coherence.num_object_modes = 1
    p.scans.scan00.sample = u.Param()
    p.scans.scan00.sample.model = 'stxm'
    p.scans.scan00.sample.process =  None
    p.scans.scan00.propagation = propagator

    # (copy the simulation illumination and change specific things)
    p.scans.scan00.illumination = sim.illumination.copy(99)
    if not init_correct_probe:
        p.scans.scan00.illumination.aperture.form = 'circ'
        p.scans.scan00.illumination.propagation.focussed = 0.06
        p.scans.scan00.illumination.diversity = u.Param()
        p.scans.scan00.illumination.diversity.power = 0.1
        p.scans.scan00.illumination.diversity.noise = (np.pi,3.0)

    # Scan data (simulation) parameters
    p.scans.scan00.data = u.Param()
    p.scans.scan00.data.name = 'SimScan'
    p.scans.scan00.data.update(sim)
    p.scans.scan00.data.save = None
    p.engines = u.Param()
    p.engines.engine00 = engine_params
    P = Ptycho(p, level=4)
    P.run()
    return P
