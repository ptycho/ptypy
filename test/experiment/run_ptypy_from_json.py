"""
Offline ptychographic data preparation and reconstruction using ptypy at P06.

This script is adapted for and requires ptypy 0.5.
CPU engines have been tested on the Maxwell cluster.
GPU engines still need some fix in the conda env.
"""

import os
import sys
import time

import h5py
import numpy as np
from mpi4py import MPI

#sys.path.insert(0, '/asap3/petra3/gpfs/common/p06/maxwell/ptypy/build/lib/')
import ptypy
from ptypy.core import Ptycho
from ptypy import utils as u

ptypy.load_ptyscan_module("p06")

#from distutils.version import LooseVersion
# if LooseVersion(ptypy.version) >= LooseVersion('0.5.0'):
#     #ptypy.load_ptyscan_module("p06micro_new_edit")
#     ptypy.load_ptyscan_module("p06")
#     #ptypy.load_gpu_engines(arch="cupy")



#config_path = "test/experiment/p06_test_data/scan_00039_parameters.json"
config_path = "test/experiment/p06_test_data/little_star_v2.json"
p = u.param_from_json(config_path)
meta = p.pop('__meta__')  # Remove '__meta__' as it will fail parameter validation.

# stuff to only do once
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
if rank==0:
    # create output directories if it does not already exist
    os.makedirs(os.path.dirname(os.path.join(p.io.home, p.io.autosave.rfile)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.join(p.io.home, p.io.rfile)), exist_ok=True)
    os.makedirs(os.path.dirname(p.scans.scan00.data.dfile), exist_ok=True)
    
ptypy.io.json_rw.jwrite(os.path.join(p.io.home, 'ptypy_parameters.json'), p)

#if LooseVersion(ptypy.version) < LooseVersion('0.5.0'):
#    raise Exception('Use ptypy 0.5.0 or better!')


P = Ptycho(p, level=4)


### run the reconstructions
print('about to P.run()')
P.run()
P.finalize()
