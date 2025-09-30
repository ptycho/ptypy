import pytest

import os

import ptypy.utils as u
from ptypy import defaults_tree
from ptypy.io.json_rw import jwrite
from ptypy.experiment.p06 import *
import numpy as np


@pytest.fixture()
def scan_00039_paths():
    experiment_path = 'test/experiment/p06_test_data'
    scan_id = 39
    detector_name = "eiger_4m_01"
    rec_name = "ptypy_out"
    scan_path_raw = os.path.join(experiment_path, "raw", "alignment", f"scan_{scan_id:05.0f}")
    scan_path_processed = os.path.join(experiment_path, "processed", "alignment", f"scan_{scan_id:05.0f}")
    session_path_processed = os.path.join(experiment_path, "processed", "alignment")
    session_path_raw = os.path.join(experiment_path, "raw", "alignment")
    detector_data_dir = os.path.join(scan_path_raw, detector_name)
    raw_processed_data_path = os.path.join(scan_path_processed, "raw_processed_data.h5")
    mask_path = os.path.join(experiment_path, 'shared/analysis_config/masks/eiger_4m_01_mask.tiff')
    parameters_path = os.path.join(experiment_path, f'scan_{scan_id:05.0f}_parameters.json')
    nexus_path = os.path.join(session_path_raw, f'scan_{scan_id:05.0f}.nxs')
    home_path = os.path.join(scan_path_processed, rec_name)
    rfile = os.path.join("rec", f"rec_scan_{scan_id:05.0f}_%(engine)s_%(iterations)04d.ptyr")
    autosave_rfile = os.path.join("dumps", f"dump_scan_{scan_id:05.0f}_%(engine)s_%(iterations)04d.ptyr")
    dfile = os.path.join(scan_path_processed, rec_name, 'data', f"data_scan_{scan_id:05.0f}.ptyd")

    return {
        "detector_data_dir": detector_data_dir,
        "raw_processed_data_path": raw_processed_data_path,
        "mask_path": mask_path,
        "scan_path_raw": scan_path_raw,
        "parameters_path": parameters_path,
        "nexus_path": nexus_path,
        "home_path": home_path,
        "rfile": rfile,
        "dfile": dfile,
        "autosave_rfile": autosave_rfile,
    }

@pytest.fixture()
def scan_00039_parameters(scan_00039_paths):
    """
    Defines the desired parameter tree structure.
    """

    template_path = "test/experiment/p06_test_data/scan_00039_parameters_old.json"
    #template_path = "test/experiment/p06_test_data/little_star_v1.json"
    p = u.param_from_json(template_path)
    meta = p.pop('__meta__')  # Remove '__meta__' as it will fail parameter validation.

    #p = u.Param()
    #p.scans = u.Param()
    #p.scans.scan00 = u.Param()
    #p.scans.scan00.name = "BlockFull"
    p.scans.scan00.data = u.Param()
    p.scans.scan00.data.name = "P06Scan"
    p.scans.scan00.data.scan_path_raw = scan_00039_paths["scan_path_raw"]
    p.scans.scan00.data.maskfile = scan_00039_paths["mask_path"]
    p.scans.scan00.data.nexus_path = scan_00039_paths["nexus_path"]
    p.scans.scan00.data.positions_path = scan_00039_paths["raw_processed_data_path"]
    p.scans.scan00.data.dfile = scan_00039_paths["dfile"]
    p.scans.scan00.data.xMotor = "scany"
    p.scans.scan00.data.yMotor = "scanz"
    p.scans.scan00.data.xMotorAngle = 0
    p.scans.scan00.data.yMotorAngle = 0
    p.scans.scan00.data.xMotorFlipped = True
    p.scans.scan00.data.yMotorFlipped = True
    p.scans.scan00.data.zDetectorAngle = 0
    p.scans.scan00.data.xyAxisSkewOffset = 0
    p.scans.scan00.data.center = (1, 1)
    p.scans.scan00.data.shape = (2, 2)
    p.scans.scan00.data.energy = None
    p.scans.scan00.data.detector = "eiger_4m_01"
    p.scans.scan00.data.psize = 75e-6
    p.scans.scan00.data.distance = 3.58
    p.scans.scan00.data.rebin = 1
    p.scans.scan00.data.save = "append"
    p.scans.scan00.data.cropOnLoad = True
    p.scans.scan00.data.I0 = None
    p.scans.scan00.data.min_frames = 10
    p.scans.scan00.data.load_parallel = "all"
    p.scans.scan00.data.orientation = [False, False, False]
    p.scans.scan00.data.position_bounds = ((-1, 1), (1e-6, 5e-6))

    #p.io = u.Param()
    p.io.home = scan_00039_paths["home_path"]
    p.io.rfile = scan_00039_paths["rfile"]
    p.io.autosave.rfile = scan_00039_paths["autosave_rfile"]

    defaults_tree['ptycho'].validate(p)

    jwrite("test/experiment/p06_test_data/scan_00039_parameters_tmp.json", p)
    #jwrite("test/experiment/p06_test_data/little_star_v2.json", p)

    return p


def test_load_positions(scan_00039_parameters):
    p = scan_00039_parameters
    p06scan = P06Scan(p.scans.scan00.data)
    positions = p06scan.load_positions()
    assert positions.shape == (1681, 2)
    assert not np.any(np.isnan(positions))


def test_load(scan_00039_parameters):
    p = scan_00039_parameters
    p06scan = P06Scan(p.scans.scan00.data)
    indices = np.arange(15)
    raw, positions, weights = p06scan.load(indices)
    assert len(raw) == len(indices)
    assert len(positions) == len(indices)
    assert len(weights) == len(indices)

    # Test exception raised if no frames were selected.
    with pytest.raises(IOError):
        p.scans.scan00.data.position_bounds = [[1, -1], [1, -1]]
        p06scan = P06Scan(p.scans.scan00.data)

    p.scans.scan00.data.position_bounds = [[None, None], [None, None]]  # no valid positions
    p06scan = P06Scan(p.scans.scan00.data)
    #raw, positions, weights = p06scan.load(indices)




