# In this tutorial, we want to provide the information
# needed to create an engine compatible with the state mixture
# expansion of ptychogrpahy as described in Thibault et. al 2013 [#modes]_ .

# First we import ptypy and the utility module
import ptypy
from ptypy import utils as u
import numpy as np

# Preparing a managing Ptycho instance
# ------------------------------------

# We need to prepare a managing :any:`Ptycho` instance.
# It requires a parameter tree, as specified in :ref:`parameters`

# First, we create a most basic input paramater tree. There
# are many default values, but we specify manually only a more verbose
# output and single precision for the data type.
p = u.Param()
p.verbose_level = 3
p.data_type = "single"

# Now, we need to create a set of scans that we wish to reconstruct
# in the reconstruction run. We will use a single scan that we call 'MF' and
# mark the data source as 'test' to use the |ptypy| internal
# :py:class:`MoonFlowerScan`
p.scans = u.Param()
p.scans.MF = u.Param()
p.scans.MF.name = 'Full'
p.scans.MF.data = u.Param()
p.scans.MF.data.name = 'MoonFlowerScan'
p.scans.MF.data.shape = 128
p.scans.MF.data.num_frames = 400

# This bare parameter tree will be the input for the :any:`Ptycho`
# class which is constructed at ``level=2``. It means that it creates
# all necessary basic :any:`Container` instances like *probe*, *object*
# *diff* , etc. It also loads the first chunk of data and creates all
# :any:`View` and :any:`POD` instances, as the verbose output will tell.
P = ptypy.core.Ptycho(p, level=2)

# A quick look at the diffraction data
diff_storage = list(P.diff.storages.values())[0]
fig = u.plot_storage(diff_storage, 0, slices=(slice(2), slice(None), slice(None)), modulus='log')
fig.savefig('ownengine_%d.png' % fig.number, dpi=300)
# Plot of simulated diffraction data for the first two positions.

# We don't need to use |ptypy|'s :any:`Ptycho` class to arrive at this
# point. The structure ``P`` that we arrive with at the end of
# :ref:`simupod` suffices completely.

# Probe and object are not so exciting to look at for now. As default,
# probes are initialized with an aperture like support.
probe_storage = list(P.probe.storages.values())[0]
fig = u.plot_storage(list(P.probe.S.values())[0], 1)
fig.savefig('ownengine_%d.png' % fig.number, dpi=300)
# Plot of the starting guess for the probe.

# .. _basic_algorithm:

# A basic Difference-Map implementation
# -------------------------------------

# Now we can start implementing a simple DM algorithm. We need three basic
# functions, one is the ``fourier_update`` that implements the Fourier
# modulus constraint.

# .. math::
#    \psi_{d,\lambda,k} = \mathcal{D}_{\lambda,z}^{-1}\left\{\sqrt{I_{d}}\frac{\mathcal{D}_{\lambda,z} \{\psi_{d,\lambda,k}\}}{\sum_{\lambda,k} |\mathcal{D}_{\lambda,z} \{\psi_{d,\lambda,k}\} |^2}\right\}

def fourier_update(pods):
    import numpy as np
    pod = list(pods.values())[0]
    # Get Magnitude and Mask
    mask = pod.mask
    modulus = np.sqrt(np.abs(pod.diff))
    # Create temporary buffers
    Imodel = np.zeros_like(pod.diff)
    err = 0.
    Dphi = {}
    # Propagate the exit waves
    for gamma, pod in pods.items():
        Dphi[gamma] = pod.fw(2*pod.probe*pod.object - pod.exit)
        Imodel += np.abs(Dphi[gamma] * Dphi[gamma].conj())
    # Calculate common correction factor
    factor = (1-mask) + mask * modulus / (np.sqrt(Imodel) + 1e-10)
    # Apply correction and propagate back
    for gamma, pod in pods.items():
        df = pod.bw(factor*Dphi[gamma]) - pod.probe*pod.object
        pod.exit += df
        err += np.mean(np.abs(df*df.conj()))
    # Return difference map error on exit waves
    return err

def probe_update(probe, norm, pods, fill=0.):
    """
    Updates `probe`. A portion `fill` of the probe is kept from
    iteration to iteration. Requires `norm` buffer and pod dictionary
    """
    probe *= fill
    norm << fill + 1e-10
    for name, pod in pods.items():
        if not pod.active: continue
        probe[pod.pr_view] += pod.object.conj() * pod.exit
        norm[pod.pr_view] += pod.object * pod.object.conj()
    # For parallel usage (MPI) we have to communicate the buffer arrays
    probe.allreduce()
    norm.allreduce()
    probe /= norm

def object_update(obj, norm, pods, fill=0.):
    """
    Updates `object`. A portion `fill` of the object is kept from
    iteration to iteration. Requires `norm` buffer and pod dictionary
    """
    obj *= fill
    norm << fill + 1e-10
    for pod in pods.values():
        if not pod.active: continue
        pod.object += pod.probe.conj() * pod.exit
        norm[pod.ob_view] += pod.probe * pod.probe.conj()
    obj.allreduce()
    norm.allreduce()
    obj /= norm

def iterate(Ptycho, num):
    # generate container copies
    obj_norm = P.obj.copy(fill=0.)
    probe_norm = P.probe.copy(fill=0.)
    errors = []
    for i in range(num):
        err = 0
        # fourier update
        for di_view in Ptycho.diff.V.values():
            if not di_view.active: continue
            err += fourier_update(di_view.pods)
        # probe update
        probe_update(Ptycho.probe, probe_norm, Ptycho.pods)
        # object update
        object_update(Ptycho.obj, obj_norm, Ptycho.pods)
        # print error
        errors.append(err)
        if i % 3==0: print(err)
    # cleanup
    P.obj.delete_copy()
    P.probe.delete_copy()
    #return error
    return errors

# We start off with a small number of iterations.
iterate(P, 9)

# We note that the error (here only displayed for 3 iterations) is
# already declining. That is a good sign.
# Let us have a look how the probe has developed.
fig = u.plot_storage(list(P.probe.S.values())[0], 2)
fig.savefig('ownengine_%d.png' % fig.number, dpi=300)
# Plot of the reconstructed probe after 9 iterations. We observe that
# the actaul illumination of the sample must be larger than the initial
# guess.

# Looks like the probe is on a good way. How about the object?
fig = u.plot_storage(list(P.obj.S.values())[0], 3, slices='0,120:-120,120:-120')
fig.savefig('ownengine_%d.png' % fig.number, dpi=300)
# Plot of the reconstructed object after 9 iterations. It is not quite
# clear what object is reconstructed

# Ok, let us do some more iterations. 36 will do.
iterate(P, 36)

# Error is still on a steady descent. Let us look at the final
# reconstructed probe and object.
fig = u.plot_storage(list(P.probe.S.values())[0], 4)
fig.savefig('ownengine_%d.png' % fig.number, dpi=300)
# Plot of the reconstructed probe after a total of 45 iterations.
# It's a moon !


fig = u.plot_storage(list(P.obj.S.values())[0], 5, slices='0,120:-120,120:-120')
fig.savefig('ownengine_%d.png' % fig.number, dpi=300)
# Plot of the reconstructed object after a total of 45 iterations.
# It's a bunch of flowers !


# .. [#modes] P. Thibault and A. Menzel, **Nature** 494, 68 (2013)
