"""
This script is a test for ptychographic reconstruction in the absence
of actual data. It uses the test Scan class
`ptypy.core.data.MoonFlowerScan` to provide "data".
"""
import numpy as np
from ptypy.core import Ptycho
from ptypy import utils as u

import tempfile
tmpdir = tempfile.gettempdir()

p = u.Param()

# for verbose output
p.verbose_level = "info"

# set home path
p.io = u.Param()
p.io.home = "/".join([tmpdir, "ptypy"])
p.io.autosave = u.Param(active=False)
p.io.interaction = u.Param(active=False)

# max 200 frames (128x128px) of diffraction data
p.scans = u.Param()
p.scans.MF = u.Param()
# now you have to specify which ScanModel to use with scans.XX.name,
# just as you have to give 'name' for engines and PtyScan subclasses.
p.scans.MF.name = 'BlockFull'
p.scans.MF.data= u.Param()
p.scans.MF.data.name = 'MoonFlowerScan'
p.scans.MF.data.shape = 128
p.scans.MF.data.num_frames = 200
p.scans.MF.data.save = None

# position distance in fraction of illumination frame
p.scans.MF.data.density = 0.2
# total number of photon in empty beam
p.scans.MF.data.photons = 1e8
# Gaussian FWHM of possible detector blurring
p.scans.MF.data.psf = 0.

# attach a reconstrucion engine
p.engines = u.Param()
p.engines.engine00 = u.Param()
p.engines.engine00.name = 'EPIE'
p.engines.engine00.probe_support = 1
p.engines.engine00.numiter = 1000
p.engines.engine00.numiter_contiguous = 10
p.engines.engine00.position_refinement = u.Param()
p.engines.engine00.position_refinement.start = 50
p.engines.engine00.position_refinement.stop = 950
p.engines.engine00.position_refinement.interval = 10
p.engines.engine00.position_refinement.nshifts = 32
p.engines.engine00.position_refinement.amplitude = 5e-7
p.engines.engine00.position_refinement.max_shift = 1e-6
p.engines.engine00.position_refinement.method = "GridSearch"
p.engines.engine00.position_refinement.metric = "photon"
p.engines.engine00.position_refinement.record = True

# prepare and run
if __name__ == "__main__":
    P = Ptycho(p, level=4)

    # Mess up the positions in a predictible way (for MPI)
    a = 0.

    coords = []
    coords_start = []
    for pname, pod in P.pods.items():

        # Save real position
        coords.append(np.copy(pod.ob_view.coord))
        before = pod.ob_view.coord
        psize = pod.pr_view.psize
        perturbation = psize * ((3e-7 * np.array([np.sin(a), np.cos(a)])) // psize)
        new_coord = before + perturbation # make sure integer number of pixels shift
        pod.ob_view.coord = new_coord
        coords_start.append(np.copy(pod.ob_view.coord))
        #pod.diff *= np.random.uniform(0.1,1)
        a += 4.
    coords = np.array(coords)
    coords_start = np.array(coords_start)

    P.obj.reformat()

    # Run
    P.run()
    P.finalize()

    coords_new = []
    for pname, pod in P.pods.items():
        coords_new.append(np.copy(pod.ob_view.coord))
    coords_new = np.array(coords_new)

    import matplotlib.pyplot as plt
    plt.figure(figsize=(10,10), dpi=60)
    plt.title("RMSE = %.2f um" %(np.sqrt(np.sum((coords_new-coords)**2,axis=1)).mean()*1e6))
    plt.plot(coords[:,0], coords[:,1], marker='.', color='k', lw=0, label='original')
    plt.plot(coords_start[:,0], coords_start[:,1], marker='x', color='r', lw=0, label='start')
    plt.plot(coords_new[:,0], coords_new[:,1], marker='.', color='r', lw=0, label='end')
    plt.legend()
    plt.savefig("/".join([tmpdir, "ptypy", "posref_eval_epie.pdf"]), bbox_inches='tight')
    plt.show()