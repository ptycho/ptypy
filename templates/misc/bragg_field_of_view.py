"""
Example script which uses the 3d Bragg ptycho code to calculate and
plot the 3d field of view as compared to the incoming probe.
"""
from ptypy.core import Ptycho
from ptypy import utils as u
import ptypy
ptypy.load_ptyscan_module("Bragg3dSim")
import matplotlib.pyplot as plt
import numpy as np

# Set up a parameter tree
p = u.Param()

p.verbose_level = "info"

# illumination for data simulation and pods
illumination = u.Param()
illumination.aperture = u.Param()
illumination.aperture.size = (1e-6, 3e-6)
illumination.aperture.form = 'rect'

# these parameters determine the whole geometry (rocking steps, theta, energy, ...)
p.scans = u.Param()
p.scans.scan01 = u.Param()
p.scans.scan01.name = 'Bragg3dModel'            # 3d Bragg
p.scans.scan01.illumination = illumination
p.scans.scan01.data= u.Param()
p.scans.scan01.data.name = 'Bragg3dSimScan'     # PtyScan which provides simulated data
p.scans.scan01.data.illumination = illumination
p.scans.scan01.data.theta_bragg = 20.0          # the central Bragg angle
p.scans.scan01.data.shape = 512
p.scans.scan01.data.psize = 40e-6
p.scans.scan01.data.n_rocking_positions = 40    # 40 rocking positions per scanning position
p.scans.scan01.data.dry_run = True              # Don't actually calculate diff patterns

# Create a Ptycho instance, this creates a numerical sample and simulates
# the diffraction experiment
P = Ptycho(p,level=2)

# This particular PtyScan also exports the object used for simulation as an attribute
S_true = P.model.scans['scan01'].ptyscan.simulated_object

# We can grab the object storage from the Ptycho instance
S = list(P.obj.storages.values())[0]

# Similarly, we can find a view of the probe
probeView = list(P.probe.views.values())[0]

# Let's define an object view to study
objView = S.views[1]

# In order to visualize the field of view, we'll create an empty copy of
# the object and set its value to 1 where covered by the chosen view.
S_display = S.copy(owner=S.owner, ID='Sdisplay')
S_display.fill(0.0)
S_display[objView] = 1

# Then, to see how the probe is contained by this field of view, we add
# the probe and the numerical sample itself to the above view.
S_display[objView][np.where(np.abs(probeView.data) > .1)] = 2
S_display.data[np.where(S_true.data)] = 3

# Until now, we've been operating in the non-orthogonal 'natural'
# coordinate system, which is good but hard to understand. We can
# convert to orthogonal (z, x, y) space by using a method on the
# geometry object, found from any of the pods.
geo = list(P.pods.values())[0].geometry
S_display_cart = geo.coordinate_shift(S_display, input_system='natural', input_space='real', keep_dims=True)

# Plot some slices
fig, ax = plt.subplots(nrows=1, ncols=3)
x, z, y = S_display_cart.grids()

# all Bragg storages are (r3, r1, r2) or (x, z, y), so...

cmap = plt.get_cmap('viridis', lut=4)

arr = np.abs(S_display_cart.data[0][:,:,objView.dcoord[2]]).T # (z, x) from top left
arr = np.flipud(arr)                                          # (z, x) from bottom left
ax[0].imshow(arr, extent=[x.min(), x.max(), z.min(), z.max()],
    interpolation='none', vmin=0, vmax=3, cmap=cmap)
plt.setp(ax[0], ylabel='z', xlabel='x', title='side view')

arr = np.abs(S_display_cart.data[0][:,objView.dcoord[1],:]).T # (y, x) from top left
ax[1].imshow(arr, extent=[x.min(), x.max(), y.max(), y.min()],
    interpolation='none', vmin=0, vmax=3, cmap=cmap)
plt.setp(ax[1], ylabel='y', xlabel='x', title='top view')

arr = np.abs(S_display_cart.data[0][objView.dcoord[0],:,:]) # (z, y) from top left
arr = np.flipud(arr)                                        # (z, y) from bottom left
im = ax[2].imshow(arr, extent=[y.min(), y.max(), z.min(), z.max()],
    interpolation='none', vmin=0, vmax=3, cmap=cmap)
plt.setp(ax[2], ylabel='z', xlabel='y', title='front view')

pc = plt.colorbar(im, ax=list(ax))
pc.set_ticks([1, 2, 3])
pc.set_ticklabels(['field of view', 'probe', 'object'])

plt.show()
