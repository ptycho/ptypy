"""
Test for the threepie_serial engine.
"""
import sys

import numpy as np
import pytest

from ptypy import utils as u
from ptypy.core import Ptycho

from ptypy.custom.threepie import ThreePIE
from ptypy.custom.threepie_serial import ThreePIE_serial
#from ptypy.experiment.nanomax import NanomaxContrast  # this class have different defaults
#sys.path.insert(0, 'test/multislice/test/scripts') # did not work
from ptypy.experiment import nanomax_class


@pytest.fixture()
def threepie_serial_params():
    beamtime_basedir = "test/multislice"  # f'/home/litang/multislice'
    print(beamtime_basedir)
    sample = '0002_multislice'
    detector = 'eiger4m'

    scannr = 434
    distance_m = 4.150  # distance between the sample and the detector in meters
    defocus_um = -750  # distance between the focus and the sample plane in micro meters -> used for inital probe
    energy_keV = 8.0  # incident photon energy in keV ... now read from scan file
    cropping = 32
    binning = 2
    probe_modes = 2

    # 40 iterations: at 10 the stochastic view-shuffle noise floor of a
    # single engine (serial-vs-serial ncorr ~0.84) is as large as any
    # cross-backend difference, so equivalence cannot be shown.
    Niter = 40
    Nsave = 1

    # multiple object slices
    number_of_slices = 3
    slice_thickness = 1500.e-6

    # define the output directories
    out_dir = f'{beamtime_basedir}/process/{sample}/scan_{scannr:0>6}/ptycho_ptypy_Mslice_crop-{cropping}_bin-{binning}_slices-{number_of_slices}_pmodes-{probe_modes}/'
    out_dir_data = f'{out_dir}data/'
    out_dir_dumps = f'{out_dir}dumps/'
    out_dir_scripts = f'{out_dir}scripts/'
    out_dir_rec = f'{out_dir}rec/'

    # and what the files are supposed to be called
    path_data = f'{out_dir_data}data_scan_{scannr:0>6}.ptyd'  # the file with the prepared data
    path_dumps = f'{out_dir_dumps}dump_scan_{scannr:0>6}_%(engine)s_%(iterations)04d.ptyr'  # intermediate results
    path_rec = f'{out_dir_rec}rec_scan_{scannr:0>6}_%(engine)s_%(iterations)04d.ptyr'  # final reconstructions (of each engine)

    # multiple object slices
    number_of_slices = 3
    slice_thickness = 1500.e-6

    # General parameters
    p = u.Param()
    p.verbose_level = 3
    p.run = 'scan%d' % scannr

    # where to put the reconstructions
    p.io = u.Param()
    p.io.home = out_dir_rec  # where to save the final reconstructions
    p.io.rfile = path_rec  # how to name those files for the final reconstructions
    p.io.autosave = u.Param()
    p.io.autosave.rfile = path_dumps  # where to save the intermediate reconstructions and how to name them
    p.io.autoplot = u.Param(active=False)
    p.io.interaction = u.Param(active=False)

    # Scan parameters
    p.scans = u.Param()
    p.scans.scan00 = u.Param()
    p.scans.scan00.name = 'Full'
    p.scans.scan00.coherence = u.Param()
    p.scans.scan00.coherence.num_probe_modes = probe_modes  # number of probe modes
    p.scans.scan00.coherence.num_object_modes = 1  # number of object modes

    p.scans.scan00.data = u.Param()
    p.scans.scan00.data.name = 'NanomaxContrast'
    p.scans.scan00.data.path = beamtime_basedir + '/raw/' + sample + '/'
    p.scans.scan00.data.detector = detector
    p.scans.scan00.data.maskfile = \
    {'merlin': '/data/visitors/nanomax/common/masks/merlin/latest.h5',
     'pilatus': None,
     'eiger': '/data/visitors/nanomax/common/masks/eiger/eiger_4M_blinking_pixels.h5',
     # legacy
     'eiger1m': None,
     # /data/visitors/nanomax/20211244/2022062908/notebooks/eiger_4M_blinking_pixels_plus.h5
     'eiger4m': None}[detector]
    # '/data/visitors/nanomax/20220550/2023121308/macros/ptycho/masks/20231208_mask_eiger4m.h5'}
    p.scans.scan00.data.scanNumber = scannr
    p.scans.scan00.data.xMotor = 'pseudo/x'
    p.scans.scan00.data.yMotor = 'pseudo/y'
    p.scans.scan00.data.zDetectorAngle = 0.0  # rotation of the detector around the beam axis in [deg]
    p.scans.scan00.data.xyAxisSkewOffset = 0.0
    p.scans.scan00.data.shape = cropping  # size of the window of the diffraction patterns to be used in pixel
    p.scans.scan00.data.save = 'append'
    p.scans.scan00.data.dfile = path_data  # once all data is collected, save it as .ptyd file
    p.scans.scan00.data.center = (1281,
                                  772)  # center of the diffraction pattern (y,x) in pixel or None -> auto
    p.scans.scan00.data.cropOnLoad = True  # only load used part of detector frames -> save memory
    # requires center to be set explicitly
    {'merlin': (False, False, True),  # (do_transpose, do_flipud, do_fliplr)
     'pilatus': (False, True, False),
     'eiger': (False, True, False),  # legacy
     'eiger1m': (False, True, False),
     'eiger4m': (False, False, True)}[
        detector]  # when mounted 180 degrees rotated
    # 'eiger4m': (False, True, False)}[detector] #old version when mounted the right way around
    p.scans.scan00.data.distance = distance_m  # distance between sample and detector in [m]
    p.scans.scan00.data.psize = {'pilatus': 172e-6,
                                 'merlin': 55e-6,
                                 'eiger': 75e-6,  # legacy
                                 'eiger1m': 75e-6,
                                 'eiger4m': 75e-6}[detector]
    p.scans.scan00.data.rebin = binning
    p.scans.scan00.data.energy = energy_keV  # incident photon energy in [keV], now read from file
    p.scans.scan00.data.I0 = None  # can be like 'alba2/1'
    p.scans.scan00.data.min_frames = 10
    p.scans.scan00.data.load_parallel = 'all'

    # scan parameters: illumination
    p.scans.scan00.illumination = u.Param()
    p.scans.scan00.illumination.model = None  # option 1: probe is initialized from a guess
    p.scans.scan00.illumination.aperture = u.Param()
    p.scans.scan00.illumination.aperture.form = 'circ'  # initial probe is a rectangle (KB focus)
    p.scans.scan00.illumination.aperture.size = 100e-9  # of this size in [m] the focus
    p.scans.scan00.illumination.propagation = u.Param()
    p.scans.scan00.illumination.propagation.parallel = 1. * defocus_um * 1e-6  # propagate the inital guess -> gives phase curvature
    p.scans.scan00.illumination.diversity = u.Param()
    p.scans.scan00.illumination.diversity.power = (1)

    p.engines = u.Param()
    ############################################################################
    #
    ############################################################################

    # general
    p.engines.engine00 = u.Param()
    p.engines.engine00.name = 'ThreePIE_serial'  # 'ePIE_multislice'
    p.engines.engine00.numiter = Niter  # number of iterations
    p.engines.engine00.numiter_contiguous = Nsave  # save a dump file every x iterations
    p.engines.engine00.probe_support = 3  # non-zero probe area as fraction of the probe frame
    # p.engines.engine00.probe_update_start = 50          # number of iterations before probe update starts
    p.engines.engine00.probe_center_tol = 1

    # p.engines.engine00.obj_smooth_std = 10.
    # p.engines.engine00.clip_object = (0,1)
    p.engines.engine00.number_of_slices = number_of_slices
    p.engines.engine00.slice_thickness = slice_thickness
    p.engines.engine00.fslices = f'{out_dir_rec}rec_crop-{cropping}_slices-{number_of_slices}_iter-{Niter:0>4}.h5'
    yield p



def test_reconstruction_equvalence(threepie_serial_params):
    """
    """

    p = threepie_serial_params
    P = Ptycho(p, level=5)
    obj = P.obj.storages['Sscan00G00'].data[0, :, :]
    probe = P.probe.storages['Sscan00G00'].data[0, :, :]

    # Replace the serialized engine with the CPU pod/view reference; rerun.
    p.engines.engine00.name = "ThreePIE"
    P_basic = Ptycho(p, level=5)

    for key, storage in P.obj.storages.items():
        storage_basic = P_basic.obj.storages[key]
        np.testing.assert_array_almost_equal(
            storage_basic.origin, storage.origin)

    for key, view in P.obj.views.items():
        view_basic = P_basic.obj.views[key]
        np.testing.assert_array_almost_equal(
            view_basic.psize, view.psize)
        np.testing.assert_array_almost_equal(
            view_basic.coord, view.coord)
        np.testing.assert_array_almost_equal(
            view_basic.pcoord, view.pcoord)
        np.testing.assert_array_almost_equal(
            view_basic.dcoord, view.dcoord)
        np.testing.assert_array_almost_equal(
            view_basic.dlow, view.dlow)
        np.testing.assert_array_almost_equal(
            view_basic.dhigh, view.dhigh)

    # asserts: both engines shuffle views independently, so elementwise
    # equality is not attainable. Compare with the same phase/scale-
    # invariant correlation used by threepie_serial_test.py.
    # Measured same-engine (serial-vs-serial) reproducibility at this
    # crop-32 / 3-slice / real-data configuration spans ncorr 0.70-0.93
    # for object and probe, so this is a smoke-level equivalence check;
    # the tight cross-backend checks are the moonflower tests.
    def ncorr(a, b):
        a = a.ravel(); b = b.ravel()
        a = a - a.mean(); b = b - b.mean()
        num = np.abs(np.vdot(a, b))
        den = np.linalg.norm(a) * np.linalg.norm(b)
        return float(num / den) if den else 0.0

    obj_basic = P_basic.obj.storages['Sscan00G00'].data[0, :, :]
    probe_basic = P_basic.probe.storages['Sscan00G00'].data[0, :, :]
    c_obj = ncorr(obj, obj_basic)
    c_probe = ncorr(probe, probe_basic)
    print(f"object correlation ThreePIE_serial vs ThreePIE: {c_obj:.4f}")
    print(f"probe  correlation ThreePIE_serial vs ThreePIE: {c_probe:.4f}")
    assert c_obj > 0.6, f"object correlation too low: {c_obj:.4f}"
    assert c_probe > 0.6, f"probe correlation too low: {c_probe:.4f}"

