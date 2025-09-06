import pytest

import os

import ptypy.utils as u
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
    parameters_path = os.path.join(experiment_path, 'scan_00039_parameters.json')
    nexus_path = os.path.join(session_path_raw, 'scan_00039.nxs')
    home_path = os.path.join(scan_path_processed, rec_name, 'rec')
    rfile = os.path.join(home_path, "rec_scan_00039_%(engine)s_%(iterations)04d.ptyr")
    autosave_rfile = os.path.join("rec_scan_00039_%(engine)s_%(iterations)04d.ptyr")
    dfile = os.path.join(scan_path_processed, rec_name, 'data', "data_scan_00039.ptyd")

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
    p = u.param_from_json(template_path)
    meta = p.pop('__meta__')  # Remove '__meta__' as it will fail parameter validation.

    #p = u.Param()
    p.scans = u.Param()
    p.scans.scan00 = u.Param()
    p.scans.scan00.name = "BlockFull"
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
    p.scans.scan00.data.zDetectorAngle = 0
    p.scans.scan00.data.xyAxisSkewOffset = 0
    p.scans.scan00.data.center = (1000, 1000)
    p.scans.scan00.data.shape = (8, 8)
    p.scans.scan00.data.energy = None
    p.scans.scan00.data.detector = "eiger_4m_01"


    #p.io = u.Param()
    p.io.home = scan_00039_paths["home_path"]
    p.io.rfile = scan_00039_paths["rfile"]
    p.io.autosave.rfile = "autosave_"+scan_00039_paths["rfile"]




    jwrite("test/experiment/p06_test_data/scan_00039_parameters.json", p)

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
    indices = np.arange(10)
    raw, positions, weights = p06scan.load(indices)
    for key, item in positions.items():
        assert item.shape == (1681, )
        assert not np.any(np.isnan(item))




