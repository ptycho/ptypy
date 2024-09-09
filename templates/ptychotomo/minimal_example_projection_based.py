"""
This script is a test for ptycho-tomographic reconstructions.
"""
from ptypy.core import Ptycho
from ptypy import utils as u
import ptypy.simulations as sim
import ptypy.utils.tomo as tu
from ptypy.custom import ML_separate_grads_ptychotomo
import random

import astra
import numpy as np
import tempfile
tmpdir = tempfile.gettempdir()


### PTYCHO PARAMETERS
p = u.Param()
p.verbose_level = "info"
p.data_type = "single"

p.run = None
p.io = u.Param()
p.io.home = "/".join([tmpdir, "ptypy_jari"])
p.io.autosave = u.Param(active=True)
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
scan.name = 'BlockFull' #'Full'

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

all_shifts = [
    (-1, 0), (-1, 1), (0, 1), (0, -1), (1, -1), (1, 0), (1, 1)
]

shift_probes = False
if shift_probes:
    # Repeatable random shifts
    random.seed(0)
    # For saving shifts to file
    # Make sure file is empty
    if u.parallel.master:
        with open("probe_shifts.txt", "w") as myfile:
            myfile.write('\n')

# Iterate over nr. of tomographic angles
print('##########################')
p.scans = u.Param()
for i in range(nangles):
    simi = sim.copy(depth=99)
    proj_new = proj[i]

    if shift_probes:
        selected_shift = random.choice(all_shifts)

        # Save the shifts applied to file
        if u.parallel.master:
            with open("probe_shifts.txt", "a") as myfile:
                myfile.write('probe '+ str(i) + ':    (' +str(selected_shift[0]) +  ', ' + str(selected_shift[1]) + ') \n')
        shifted_proj_1 = np.roll(proj_new, selected_shift[0], axis=0)   # up or down (neg = up, pos = down)
        proj_new = np.roll(shifted_proj_1, selected_shift[1], axis=1)   # right or left (neg = left, pos = right)

    simi.sample.model = np.exp(1j * proj_new)
    scani = scan.copy(depth=99)
    scani.data.update(simi)
    setattr(p.scans, f"scan{i}", scani)

# Write out angles
np.save("simulated_angles.npy", angles)

# Reconstruction parameters
p.engines = u.Param()
p.engines.engine = u.Param()
p.engines.engine.name = 'MLPtychoTomo'
p.engines.engine.angles = 'simulated_angles.npy'
p.engines.engine.init_vol_zero = True
#p.engines.engine.init_vol_real = 'starting_vol_for_ML_simulated/real_vol_35it.npy'
#p.engines.engine.init_vol_imag = 'starting_vol_for_ML_simulated/imag_vol_35it.npy'
#p.engines.engine.init_vol_blur = False
#p.engines.engine.init_vol_blur_sigma = 2.5
p.engines.engine.numiter = 200
p.engines.engine.numiter_contiguous = 10
p.engines.engine.probe_support = None
p.engines.engine.probe_fourier_support = None
#p.engines.engine.rescale_vol_gradient = True
#p.engines.engine.rescale_vol_gradient_factor = 0.57
#p.engines.engine.weight_gradient = True
#p.engines.engine.reg_del2 = True
#p.engines.engine.reg_del2_amplitude = 1e8
#p.engines.engine.smooth_gradient = 2.5
#p.engines.engine.smooth_gradient_decay = 0.75
#p.engines.engine.OPR = True
#p.engines.engine.OPR_modes = 15
#p.engines.engine.OPR_method = "second"
#p.engines.engine.probe_update_start = 0 # is the default
#p.engines.engine.poly_line_coeffs = "quadratic"

u.verbose.set_level("info")

if __name__ == "__main__":
    P = Ptycho(p,level=5)

    ## Modifying probes  #############################
#     P = Ptycho(p,level=3)

#     storage_list = list(P.probe.storages.values())

#     # shift storages and Transfer views
#     i = 0
#     for storage in storage_list:
#         # storage.center = (20,20)       # changed from (16, 16)
#         storage.data[:] = gaussian_filter(storage.data, sigma=0.8)

#         for v in storage.views:
#             v.storage = storage
#             v.storageID = storage.ID

#         i += 1

#     # # Update probe
#     P.probe.reformat()

#     # # Unforunately we need to delete the storage here due to DM being unable
#     # # to ignore unused storages. This is due to the /=nrm division in the
#     # # probe update
#     # for s in storage_list[1:]:
#     #     P.probe.storages.pop(s.ID)

#     # Finish level 4
#     P.print_stats()
#     P.init_engine()
#     P.run()
#     P.finalize()
    # ##############################
