# This script is a demonstration of how the new container and geometry
# classes can be used to do 3d Bragg ptychography. Everything is done
# manually on the View and Geo_Bragg instances, no PtyScan objects or
# model managers or pods are involved yet.

# I've taken the sample and geometry from the numerical experiment in
# Berenguer et al., PRB 2013. Unlike that example, the probe here is
# just a flat rectangle.

# The imports. I've found it useful to visualize 3d data in mayavi, but
# here we just look at projections with matplotlib.
import ptypy
import numpy as np
import matplotlib.pyplot as plt

# Set up a 3D geometry and a scan
# -------------------------------

# The following geometry corresponds to an oversampling ratio of 1 along
# the rocking curve, which means that the numerical field of view
# tightly contains the exit wave.
g = ptypy.core.geometry_bragg.Geo_Bragg(psize=(13e-6, 13e-6, 0.01/4), shape=(128, 128, 9*4), energy=8.5, distance=2.0, theta_bragg=22.32)

# Set up scan positions along y, perpendicular to the incoming beam and
# to the thin layer stripes.
Npos = 11
positions = np.zeros((Npos,3))
positions[:, 1] = np.arange(Npos) - Npos/2.0
positions *= .5e-6

# Set up the object and its views
# -------------------------------

# Create a container for the object array, which will represent the
# object in the non-orthogonal coordinate system conjugate to the
# q-space measurement frame.
C = ptypy.core.Container(data_type=np.complex128, data_dims=3)

# For each scan position in the orthogonal coordinate system, find the
# natural coordinates and create a View instance there.
views = []
for pos in positions:
    pos_ = g._r1r2r3(pos)
    views.append(ptypy.core.View(C, storageID='S0000', psize=g.resolution, coord=pos_, shape=g.shape))
S = C.storages['S0000']
C.reformat()

# Define the test sample based on the orthogonal position of each voxel.
# First, the cartesian grid is obtained from the geometry object, then
# this grid is used as a condition for the sample's magnitude.
zz, yy, xx = g.transformed_grid(S, input_space='real', input_system='natural')
S.fill(0.0)
S.data[(zz >= -90e-9) & (zz < 90e-9) & (yy >= 1e-6) & (yy < 2e-6) & (xx < 1e-6)] = 1
S.data[(zz >= -90e-9) & (zz < 90e-9) & (yy >= -2e-6) & (yy < -1e-6)] = 1

# Set up the probe and calculate diffraction patterns
# ---------------------------------------------------

# First set up a two-dimensional representation of the probe, with
# arbitrary pixel spacing. The probe here is defined as a 1.5 um by 3 um
# flat square, but this container will typically come from a 2d
# transmission ptycho scan of an easy test object.
Cprobe = ptypy.core.Container(data_dims=2, data_type='float')
Sprobe = Cprobe.new_storage(psize=10e-9, shape=500)
z, y = Sprobe.grids()
Sprobe.data[(z > -.75e-6) & (z < .75e-6) & (y > -1.5e-6) & (y < 1.5e-6)] = 1

# The Bragg geometry has a method to prepare a 3d Storage by extruding
# the 2d probe and interpolating to the right grid. The returned storage
# contains a single view compatible with the object views.
Sprobe_3d = g.prepare_3d_probe(Sprobe, system='natural')
probeView = Sprobe_3d.views[0]

# Calculate diffraction patterns by using the geometry's propagator.
diff = []
for v in views:
    diff.append(np.abs(g.propagator.fw(v.data * probeView.data))**2)

# Visualize a single field of view with probe and object
# ------------------------------------------------------

# A risk with 3d Bragg ptycho is that undersampling along theta leads to
# a situation where the actual exit wave is not contained within the
# field of view. This plot shows the situation for this combination of
# object, probe, and geometry. Note that the thin sample is what makes
# the current test experiment possible.

# In order to visualize the field of view, we'll create a copy of the
# object storage and set its value equal to 1 where covered by the first
# view.
S_display = S.copy(owner=C)
S_display.fill(0.0)
S_display[S.views[0]] = 1

# Then, to see how the probe is contained by this field of view, we add
# the probe and the object itself to the above view.
S_display[S.views[0]] += probeView.data
S_display.data += S.data * 2

# To visualize how this looks in cartesian real space, make a shifted
# (nearest-neighbor interpolated) copy of the object Storage.
S_display_cart = g.coordinate_shift(S_display, input_system='natural', input_space='real', keep_dims=False)

# Plot that
fig, ax = plt.subplots(nrows=1, ncols=2)
z, y, x = S_display_cart.grids()
ax[0].imshow(np.mean(np.abs(S_display_cart.data[0]), axis=1), extent=[x.min(), x.max(), z.min(), z.max()], interpolation='none', origin='lower')
plt.setp(ax[0], ylabel='z', xlabel='x', title='side view')
ax[1].imshow(np.mean(np.abs(S_display_cart.data[0]), axis=0), extent=[x.min(), x.max(), y.min(), y.max()], interpolation='none', origin='lower')
plt.setp(ax[1], ylabel='y', xlabel='x', title='top view')

# Visualize the probe positions along the scan
# --------------------------------------------

# beam/sample overlap in non-orthogonal coordinates
import matplotlib.gridspec as gridspec
plt.figure()
gs = gridspec.GridSpec(Npos, 2, width_ratios=[3,1], wspace=.0)
ax, ax2 = [], []
r1, r2, r3 = Sprobe_3d.grids()
for i in range(len(S.views)):
    # overlap
    ax.append(plt.subplot(gs[i, 0]))
    ax[-1].imshow(np.mean(np.abs(views[i].data + probeView.data), axis=0).T, vmin=0, vmax=.07, extent=[r2.min(), r2.max(), r3.min(), r3.max()])
    plt.setp(ax[-1], xlabel='r2', ylabel='r3', xlim=[r2.min(), r2.max()], ylim=[r3.min(), r3.max()], yticks=[])
    # diffraction
    ax2.append(plt.subplot(gs[i, 1]))
    ax2[-1].imshow(diff[i][:,:,18])
    plt.setp(ax2[-1], ylabel='q1', xlabel='q2', xticks=[], yticks=[])
plt.suptitle('Probe, sample, and slices of 3d diffraction peaks along the scan')
plt.draw()

# Reconstruct the numerical data
# ------------------------------

# Reconstruction should be done considering the algorithms developed
# (for example using good regularization and considering noise models
# etc), but here I've just used a naive PIE generalization which seems
# to work OK for numerical noise-free data.

# Keep a copy of the object storage, and fill the actual one with an
# initial guess (like zeros everywhere).
S_true = S.copy(owner=C)
S.fill(0.0)

# Then run this very temporary and inefficient reconstruction loop.
fig, ax = plt.subplots(ncols=3)
errors = []
ferrors = []
r1, r2, r3 = S.grids()
for i in range(100):
    print i
    ferrors_ = []
    for j in range(len(views)):
        exit_ = views[j].data * probeView.data
        prop = g.propagator.fw(exit_)
        ferrors_.append(np.abs(prop)**2 - diff[j])
        prop[:] = np.sqrt(diff[j]) * np.exp(1j * np.angle(prop))
        exit = g.propagator.bw(prop)
        views[j].data += 1.0 * np.conj(probeView.data) / (np.abs(probeView.data).max())**2 * (exit - exit_)
    errors.append(np.abs(S.data - S_true.data).sum())
    ferrors.append(np.mean(ferrors_))

    if not (i % 5):
        ax[0].clear()
        ax[0].plot(errors/errors[0])
        ax[0].plot(ferrors/ferrors[0])
        ax[1].clear()
        S_cart = g.coordinate_shift(S, input_space='real', input_system='natural', keep_dims=False)
        z, y, x = S_cart.grids()
        ax[1].imshow(np.mean(np.abs(S_cart.data[0]), axis=0), extent=[x.min(), x.max(), y.min(), y.max()], interpolation='none', origin='lower')
        plt.setp(ax[1], ylabel='y', xlabel='x', title='top view')
        ax[2].clear()
        ax[2].imshow(np.mean(np.abs(S_cart.data[0]), axis=1), extent=[x.min(), x.max(), z.min(), z.max()], interpolation='none', origin='lower')
        plt.setp(ax[2], ylabel='z', xlabel='x', title='side view')
        plt.draw()
        plt.pause(.01)
