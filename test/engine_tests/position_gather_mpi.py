"""
Helper for position_gather_test.py, meant to run under MPI:

    mpiexec -n 2 python position_gather_mpi.py ENGINE [ENGINE ...]

For every engine, a small reconstruction with position refinement
(record = True) is run and saved. The positions in the saved file must be
the refined positions of every view, including the views that were
reconstructed on other MPI ranks than the master. Exits with 1 on the master
if they are not.

This file is part of the PTYPY package.
    :copyright: Copyright 2014 by the PTYPY team, see AUTHORS.
    :license: see LICENSE for details.
"""
import shutil
import sys
import tempfile

import h5py
import numpy as np

import ptypy
from ptypy import utils as u
from ptypy.core import Ptycho
from ptypy.utils import parallel

ptypy.load_gpu_engines("serial")


def run(engine, outdir):
    p = u.Param()
    p.verbose_level = "critical"
    p.frames_per_block = 20
    p.io = u.Param()
    p.io.home = outdir
    p.io.rfile = "%s.ptyr" % engine
    p.io.interaction = u.Param(active=False)
    p.io.autosave = u.Param(active=False)
    p.io.autoplot = u.Param(active=False)
    p.scans = u.Param()
    p.scans.MF = u.Param()
    p.scans.MF.name = "BlockFull" if engine.endswith("_serial") else "Full"
    p.scans.MF.propagation = "farfield"
    p.scans.MF.data = u.Param()
    p.scans.MF.data.name = "MoonFlowerScan"
    p.scans.MF.data.num_frames = 100
    p.scans.MF.data.shape = 64
    p.scans.MF.data.save = None
    p.scans.MF.data.photons = 1e8
    p.scans.MF.data.density = 0.2
    p.scans.MF.data.add_poisson_noise = False
    p.engines = u.Param()
    p.engines.engine00 = u.Param()
    p.engines.engine00.name = engine
    p.engines.engine00.numiter = 4
    p.engines.engine00.position_refinement = u.Param()
    p.engines.engine00.position_refinement.start = 1
    p.engines.engine00.position_refinement.interval = 1
    p.engines.engine00.position_refinement.record = True
    P = Ptycho(p, level=4)
    initial = {v.ID: np.array(v.coord) for v in P.obj.S["SMFG00"].views}
    P.run()

    # the refined position of every view, from the rank that reconstructed it
    owned = {}
    for dv in P.diff.views.values():
        if dv.active:
            owned[dv.pod.ob_view.ID] = (np.array(dv.pod.ob_view.coord), parallel.rank)
    owned = parallel.gather_dict(owned)

    path = P.save_run(kind="minimal")
    parallel.loadmanager.reset()
    if not parallel.master:
        return True
    with h5py.File(path, "r") as h:
        if "content/positions/SMFG00" not in h:
            print("%-12s no positions in the saved file" % engine, flush=True)
            return False
        saved = h["content/positions/SMFG00"][()]
    views = [v for v in P.obj.S["SMFG00"].views if v.pod.pr_view.layer == 0]
    expected = np.array([owned[v.ID][0] for v in views])
    elsewhere = np.array([owned[v.ID][1] != 0 for v in views])
    moved = np.array([np.abs(owned[v.ID][0] - initial[v.ID]).max() > 0 for v in views])
    bad = np.abs(saved - expected).max(1) > 1e-12
    print("%-12s %d views, %d refined on other ranks than the master, %d saved positions differ from the refined ones"
          % (engine, len(views), (elsewhere & moved).sum(), bad.sum()), flush=True)
    if not (elsewhere & moved).any():
        print("%-12s no view moved on the other ranks, the test cannot tell" % engine, flush=True)
        return False
    return not bad.any()


def main():
    ok = True
    outdir = parallel.bcast(tempfile.mkdtemp() if parallel.master else None)
    for engine in sys.argv[1:]:
        ok = run(engine, outdir) and ok
    parallel.barrier()
    if parallel.master:
        shutil.rmtree(outdir, ignore_errors=True)
        if not ok:
            sys.exit(1)


if __name__ == "__main__":
    main()
