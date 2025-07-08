import pytest

import os

import ptypy.utils as u
from ptypy.experiment.p06 import *


@pytest.fixture()
def scan_00039_paths():
    experiment_path = 'test/experiment/p06_test_data'
    scan_id = 39
    detector_name = "eiger_4m_01"
    scan_path_raw = os.path.join(experiment_path, "raw", f"scan_{scan_id:5.0f}")
    scan_path_processed = os.path.join(experiment_path, "processed", f"scan_{scan_id:5.0f}")
    detector_data_dir = os.path.join(scan_path_raw, detector_name)
    raw_processed_data_path = os.path.join(scan_path_raw, "raw_processed_data.h5")
    mask_path = os.path.join(experiment_path, 'eiger_')

    return {
        "detector_data_dir": detector_data_dir,
        "raw_processed_data_path": raw_processed_data_path,
        "mask_path": mask_path
    }


def test_load_positions(scan_00039_paths):
    p = u.param_from_json("test/experiment/p06_test_data/processed/alignment/scan_00039/params_00039/json")
    # p.scans.scan00.data = u.Param()
    # p.scans.scan00.data.name = 'ScanningMirrorMoonFlower'
    # p.scans.scan00.data.shift_pos_factor = 0.0  # 1e-6 / 0.1 * 0.1
    # p.scans.scan00.data.num_frames = 5
    # p.scans.scan00.data.save = None
    # p.scans.scan00.data.density = 0.01
    # p.scans.scan00.data.photons = 1e10
    # p.scans.scan00.data.add_poisson_noise = False
    # p.scans.scan00.data.psf = 0
    # p.engines = u.Param()
    # p.engines.engine00 = u.Param()
    # p.engines.engine00.name = "DM_scanning_mirror"
    # p.engines.engine00.numiter = 4
    # p.engines.engine00.numiter_contiguous = 2
    #
    # p.engines.engine00.position_refinement = u.Param()
    # p.engines.engine00.position_refinement.method = "GridSearch"
    # p.engines.engine00.position_refinement.start = 1
    # p.engines.engine00.position_refinement.stop = 200
    # p.engines.engine00.position_refinement.interval = 10
    # p.engines.engine00.position_refinement.nshifts = 8
    # p.engines.engine00.position_refinement.amplitude = 50.0e-9
    # p.engines.engine00.position_refinement.max_shift = 100.0e-9
    # p.engines.engine00.position_refinement.record = False
    #P = Ptycho(p, level=5)
    p06scan = P06Scan(p)
