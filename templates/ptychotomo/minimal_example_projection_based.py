"""
This script is a test for ptycho-tomographic reconstructions.
"""
from ptypy.core import Ptycho
from ptypy import utils as u
import ptypy.simulations as sim
import ptypy.utils.tomo as tu
from ptypy.custom import DM_ptycho_tomo

import astra
import matplotlib.pyplot as plt
import pathlib
import numpy as np
import tempfile
tmpdir = tempfile.gettempdir()

### PTYCHO PARAMETERS
p = u.Param()
p.verbose_level = "info"
p.data_type = "single"

p.run = None
p.io = u.Param()
p.io.home = "/".join([tmpdir, "ptypy_letizia"])
p.io.autosave = u.Param(active=False)
p.io.autoplot = u.Param(active=False)
p.io.autoplot.layout='minimal'

# Simulation parameters
sim = u.Param()
sim.energy = u.keV2m(1.0)/6.32e-7
sim.distance = 15e-2
sim.psize = 24e-6
sim.shape = 32
sim.xy = u.Param()
sim.xy.model = "round"
sim.xy.spacing = 0.3e-3
sim.xy.steps = 9
sim.xy.extent = (5e-3,5e-3)

sim.illumination = u.Param()
sim.illumination.model = None
sim.illumination.photons = int(1e9)
sim.illumination.aperture = u.Param()
sim.illumination.aperture.diffuser = None
sim.illumination.aperture.form = "circ"
sim.illumination.aperture.size = 1.0e-3
sim.illumination.aperture.edge = 10
sim.illumination.aperture.central_stop = None
sim.illumination.propagation = u.Param()
sim.illumination.propagation.focussed = None
sim.illumination.propagation.parallel = 0.13
sim.illumination.propagation.spot_size = None

nangles = 19
pshape = 56
angles = np.linspace(0, np.pi, nangles, endpoint=True)
pgeom = astra.create_proj_geom("parallel3d", 1.0, 1.0, pshape, pshape, angles)
vgeom = astra.create_vol_geom(pshape, pshape, pshape)
rmap = tu.refractive_index_map(pshape)#.ravel()
proj_real_id, proj_real = astra.create_sino3d_gpu(rmap.real, pgeom, vgeom)
proj_imag_id, proj_imag = astra.create_sino3d_gpu(rmap.imag, pgeom, vgeom)
proj = np.moveaxis(proj_real + 1j * proj_imag, 1,0)

sim.sample = u.Param()
#sim.sample.model = proj[0]
sim.sample.process = u.Param()
sim.sample.process.offset = (0,0)
sim.sample.process.formula = None
sim.sample.process.density = None
sim.sample.process.thickness = None
sim.sample.process.ref_index = None
sim.sample.process.smoothing = None
sim.sample.fill = 1.0+0.j
sim.plot=False
sim.detector = u.Param(dtype=np.uint32,full_well=2**32-1,psf=None)


# Scan model
scan = u.Param()
scan.name = 'BlockFull'

scan.coherence = u.Param()
scan.coherence.num_probe_modes=1

scan.illumination = u.Param()
scan.illumination.model=None
scan.illumination.aperture = u.Param()
scan.illumination.aperture.diffuser = None
scan.illumination.aperture.form = "circ"
scan.illumination.aperture.size = 1.0e-3
scan.illumination.aperture.edge = 15
scan.illumination.propagation = u.Param()
scan.illumination.propagation.focussed = None
scan.illumination.propagation.parallel = 0.03
scan.illumination.propagation.spot_size = None

# Scan data (simulation) parameters
scan.data = u.Param()
scan.data.name = 'SimScan'
#scan.data.update(sim)

# Iterate over nr. of tomographic angles
print('##########################')
p.scans = u.Param()
for i in range(nangles):
    simi = sim.copy(depth=99)
    simi.sample.model = np.exp(1j * proj[i])
    scani = scan.copy(depth=99)
    scani.data.update(simi)
    setattr(p.scans, f"scan{i}", scani)

# Reconstruction parameters
p.engines = u.Param()
p.engines.engine00 = u.Param()
p.engines.engine00.name = 'DMPtychoTomo'
p.engines.engine00.numiter = 35
p.engines.engine00.fourier_relax_factor = 0.05
p.engines.engine00.probe_center_tol = 1

u.verbose.set_level("info")

if __name__ == "__main__":
    P = Ptycho(p,level=5)

    # # Tomography
    # angles_dict = {}
    # for i,k in enumerate(P.obj.S):
    #     angles_dict[k] = angles[i]

    # vol = np.zeros((pshape, pshape, pshape), dtype=np.complex64)
    # T = tu.AstraTomoWrapperViewBased(P.obj, vol, angles_dict, obj_is_refractive_index=False, mask_threshold=35)
    # T.backward(type="SIRT3D_CUDA", iter=100)

    # # Plotting
    pshape = P._vol.shape[0]
    rmap = tu.refractive_index_map(pshape)
    X = rmap.reshape(pshape, pshape, pshape)
    R = np.real(P._vol)
    I = np.imag(P._vol)
    
    pos_limit = max([np.max(X.real), np.max(R)])
    neg_limit = min([np.min(X.real), np.min(R)])

    fig, axes = plt.subplots(ncols=3, nrows=2, figsize=(6,4), dpi=100)
    for i in range(3):
        for j in range(2):
            ax = axes[j,i]
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xticklabels([])
            ax.set_yticklabels([])
    axes[0,0].set_title("slice(Z)")
    axes[0,1].set_title("slice(Y)")
    axes[0,2].set_title("slice(X)")
    axes[0,0].set_ylabel("Original")
    axes[0,0].imshow((X.real)[pshape//2], vmin=neg_limit, vmax=pos_limit)
    axes[0,1].imshow((X.real)[:,pshape//2], vmin=neg_limit, vmax=pos_limit)
    axes[0,2].imshow((X.real)[:,:,pshape//2], vmin=neg_limit, vmax=pos_limit)
    axes[1,0].set_ylabel("Recons")
    axes[1,0].imshow((R)[pshape//2], vmin=neg_limit, vmax=pos_limit)
    axes[1,1].imshow((R)[:,pshape//2], vmin=neg_limit, vmax=pos_limit)
    im1 = axes[1,2].imshow((R)[:,:,pshape//2], vmin=neg_limit, vmax=pos_limit)
    fig.suptitle('Real part, final vol')
    fig.colorbar(im1, ax=axes.ravel().tolist())
    plt.show()

    pos_limit = max([np.max(X.imag), np.max(I)])
    neg_limit = min([np.min(X.imag), np.min(I)])
    fig, axes = plt.subplots(ncols=3, nrows=2, figsize=(6,4), dpi=100)
    for i in range(3):
        for j in range(2):
            ax = axes[j,i]
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xticklabels([])
            ax.set_yticklabels([])
    axes[0,0].set_title("slice(Z)")
    axes[0,1].set_title("slice(Y)")
    axes[0,2].set_title("slice(X)")
    axes[0,0].set_ylabel("Original")
    axes[0,0].imshow((X.imag)[pshape//2], vmin=neg_limit, vmax=pos_limit)
    axes[0,1].imshow((X.imag)[:,pshape//2], vmin=neg_limit, vmax=pos_limit)
    axes[0,2].imshow((X.imag)[:,:,pshape//2], vmin=neg_limit, vmax=pos_limit)
    axes[1,0].set_ylabel("Recons")
    axes[1,0].imshow((I)[pshape//2], vmin=neg_limit, vmax=pos_limit)
    axes[1,1].imshow((I)[:,pshape//2], vmin=neg_limit, vmax=pos_limit)
    im1 = axes[1,2].imshow((I)[:,:,pshape//2], vmin=neg_limit, vmax=pos_limit)

    fig.suptitle('Imag part, final vol')
    fig.colorbar(im1, ax=axes.ravel().tolist())
    plt.show()
