import numpy as np
from ptypy.core import geometry, Ptycho, View, POD, Container, Base, illumination
from ptypy import utils as u, resources

do_recon = True

class Base2(Base):
    pass


def exit_wave_from_di_view(di_view):
    sorted_podnames = sorted(di_view.pods.keys())
    for ix, pname in enumerate(sorted_podnames):
        if ix == 0:
            pod = di_view.pods[pname]
            pod.exit = pod.object * pod.probe
        else:
            previous_pod = di_view.pods[sorted_podnames[ix - 1]]
            current_pod = di_view.pods[pname]
            current_pod.probe = previous_pod.fw(previous_pod.exit)
            current_pod.exit = current_pod.probe * current_pod.object
    return current_pod


def basic_fourier_update(diff_view, alpha=1.):
    # Prepare dict for storing propagated waves


    # Get measured data
    I = diff_view.data

    # Get the mask
    fmask = diff_view.pod.mask

    sorted_podnames = sorted(di_view.pods.keys())

    upstream_pod = di_view.pods[sorted_podnames[0]]
    upstream_pod.exit = upstream_pod.object * upstream_pod.probe
    downstream_pod = upstream_pod

    # final_podname = sorted_podnames[1]
    # downsteam_pod = di_view.pods[final_podname]
    # downsteam_pod.probe = upstream_pod.fw(upstream_pod.exit)
    f = downstream_pod.fw((1 + alpha) * downstream_pod.probe * downstream_pod.object - alpha * downstream_pod.exit)

    # would need some adding here for modes
    af2 = u.abs2(f)

    fmag = np.sqrt(np.abs(I))
    af = np.sqrt(af2)

    # Fourier magnitudes deviations
    fdev = af - fmag
    err_fmag = np.sum(fmask * fdev ** 2) / fmask.sum()
    err_exit = 0.


    # No power bound
    fm = (1 - fmask) + fmask * fmag / (af + 1e-10)

    df = downstream_pod.bw(fm * f) - downstream_pod.probe * downstream_pod.object
    downstream_pod.exit += df
    err_exit += np.mean(u.abs2(df))


    return np.array([err_fmag, 0.0, err_exit])


P = Base2()
P.CType = np.complex128
P.FType = np.float64
ptycho = Ptycho()

P.obj = Container(P, ID='Cobj', data_type='complex')
P.probe = Container(P, ID='Cprobe', data_type='complex')

frame_shape = 128

num_probe_modes = 1
num_obj_modes = 1

g = u.Param()
g.energy = 0.700
g.distance = 55e-3
g.psize = 64e-6
g.shape = frame_shape
g.propagation = "farfield"
G = geometry.Geo(P, "G00", pars=g)

slice_separation = 5e-6
num_slices = 2

from ptypy.core import xy

density = 0.2
num_frames = 200
ppos = u.Param()
ppos.spacing = G.resolution * G.shape * density
ppos.steps = np.int(np.round(np.sqrt(num_frames))) + 1
ppos.extent = ppos.steps * ppos.spacing
ppos.model = 'round'
ppos.count = num_frames
positions = xy.from_pars(ppos)

P.exit = Container(P, 'Cexit', data_type='complex')
P.diff = Container(P, 'Cdiff', data_type='real')
P.mask = Container(P, 'Cmask', data_type='real')

new_pods = []
new_probe_ids = {}
new_object_ids = {}

oar = View.DEFAULT_ACCESSRULE.copy()
oar.storageID = 'S00'
oar.psize = G.psize
oar.layer = 0
oar.shape = G.shape
oar.active = True

di_views = []
ma_views = []

for idx, pos in enumerate(positions):
    # the rule
    r = oar.copy()
    r.coord = pos
    oar.layer = idx
    di_views.append(View(P.diff, None, r))
    ma_views.append(View(P.mask, None, r))

P.diff.reformat()
P.mask.reformat()
di_views_saved = di_views
ma_views_saved = ma_views

farfield_geo = G

nearfield_g = u.Param()
nearfield_g.energy = 0.700
nearfield_g.distance = slice_separation
nearfield_g.psize = G.resolution[0]
nearfield_g.shape = frame_shape
nearfield_g.propagation = "nearfield"

nearfield_geo = geometry.Geo(P, "G01", pars=nearfield_g)

geometries = {0: [nearfield_geo],
              1: [farfield_geo]}
# geometries = {0: [farfield_geo]}


# Loop through diffraction patterns
for i in range(len(di_views)):
    dv, mv = di_views[i], ma_views[i]

    index = dv.layer
    # print i
    # Object and probe position
    pos_pr = u.expect2(0.0)
    pos_obj = positions[i]
    for slice_idx, geo in geometries.iteritems():
        # print("The geometries are:%s" % geo)
        for geometry_index, geometry in enumerate(geo):
            # Make new IDs and keep them in record
            # sharing_rules is not aware of IDs with suffix

            # print("slice idx:%s, geometry index:%s" % (slice_idx, geometry_index))

            # No modes required to start with
            pm = 1
            om = 1


            obj_layer = om
            exit_index = index

            # contents
            if slice_idx == 0:
                pv = View(container=P.probe,
                          accessrule={'shape': geometry.shape,
                                      'psize': geometry.resolution,
                                      'coord': pos_pr,
                                      'storageID': 'SMF00',
                                      'layer': pm,
                                      'active': True})
            else:
                pv = View(container=P.probe,
                          accessrule={'shape': geometry.shape,
                                      'psize': geometry.resolution,
                                      'coord': pos_pr,
                                      'storageID': 'SMF%02d' % (i+1),
                                      'layer': pm,
                                      'active': True})


            ov = View(container=P.obj,
                      accessrule={'shape': geometry.shape,
                                  'psize': geometry.resolution,
                                  'coord': pos_obj,
                                  'storageID': 'SMF%02d' % slice_idx,
                                  'layer': obj_layer,
                                  'active': True})

            ev = View(container=P.exit,
                      accessrule={'shape': geometry.shape,
                                  'psize': geometry.resolution,
                                  'coord': pos_pr,
                                  'storageID': 'S000G%02d' % slice_idx,
                                  'layer': exit_index,
                                  'active': True})

            views = {'probe': pv,
                     'obj': ov,
                     'diff': dv,
                     'mask': mv,
                     'exit': ev}

            pod = POD(ptycho=P,
                      ID=None,
                      views=views,
                      geometry=geometry)  # , meta=meta)
            pod.probe_weight = 1
            pod.object_weight = 1

            new_pods.append(pod)

P.obj.reformat()
P.probe.reformat()
P.exit.reformat()
#

ill = u.Param()
ill.model = None
ill.photons = 1e11
ill.aperture = u.Param()
ill.aperture.form = "circ"
ill.aperture.size = 500e-6
ill.propagation = u.Param()
ill.propagation.focussed = 7.058e-3
ill.propagation.parallel = 15e-6

illumination.init_storage(P.probe.storages['SMF00'], ill)

storage1 = P.obj.storages['SMF00']
storage1.fill(resources.flower_obj(storage1.shape[-2:]))

storage2 = P.obj.storages['SMF01']

trees = resources.tree_obj(u.expect2(storage2.shape[-2:]))
obj2 = trees * np.exp(1j * (trees - 0.5) * np.pi / 4)
storage2.fill(obj2)


for dname, di_view in enumerate(di_views):
    sorted_podnames = sorted(di_view.pods.keys())
    upstream_pod = di_view.pods[sorted_podnames[0]]
    downstream_pod = di_view.pods[sorted_podnames[1]]
    upstream_pod.exit = upstream_pod.object * upstream_pod.probe
    downstream_pod.probe = upstream_pod.fw(upstream_pod.exit)
    downstream_pod.exit = downstream_pod.probe * downstream_pod.object
    downstream_pod.diff = u.abs2(downstream_pod.fw(downstream_pod.exit))
    downstream_pod.mask = np.ones(downstream_pod.diff.shape)


# now blat the object storages so we have something to go in there. This means we at least have a consistent model.
#
np.random.seed(1) # for consistency
storage1.fill(np.random.rand(*storage1.shape) + 1j*np.random.rand(*storage1.shape))
storage2.fill(np.random.rand(*storage2.shape) + 1j*np.random.rand(*storage2.shape))

# # now prime the exit waves




def equation10c(object_slice, ob_nrm, ob_nrm_storage):
    for vidx, view in enumerate(object_slice.views):
        if not view.pod.active:
            continue
        view.pod.object += view.pod.probe.conj() * view.pod.exit
        ob_nrm[view.pod.ob_view] += u.cabs2(view.pod.probe)
    object_slice.data[:] /= ob_nrm_storage.data


def equation10b(object_slice):
    for vidx, view in enumerate(object_slice.views):
        if not view.pod.active:
            continue
        # no sum required, as it's just independent
        # eq 10b
        view.pod.probe = view.pod.object.conj() * view.pod.exit / u.cabs2(view.pod.object)


def equation10a(object_slice, pr_storage,  nrm, pr_nrm_storage):
    for vidx, view in enumerate(object_slice.views):
        if not view.pod.active:
            continue
        # equation 10a
        view.pod.probe += view.pod.object.conj() * view.pod.exit
        nrm[view.pod.pr_view] += u.cabs2(view.pod.object)
    pr_storage.data[:] /= pr_nrm_storage.data


if do_recon:
    num_its = 20
    update_probe = True
    alpha = 1.0
    fourier_relax_factor = 0.05
    overlap_max_iterations = 1
    object_inertia = 1e-4
    probe_inertia = 1e-9
    P.error = []

    # Generate container copies
    P.ob_buf = P.obj.copy(P.obj.ID + '_alt', fill=0.)
    P.ob_nrm = P.obj.copy(P.obj.ID + '_nrm', fill=0.)

    P.pr_buf = P.probe.copy(P.probe.ID + '_alt', fill=0.)
    P.pr_nrm = P.probe.copy(P.probe.ID + '_nrm', fill=0.)

    for name, s in P.probe.storages.iteritems():
        sh = s.data.shape
        ll, xx, yy = u.grids(sh, FFTlike=False)
        support = (np.pi * (xx ** 2 + yy ** 2) < 0.7 * sh[1] * sh[2])
        probe_support = support

    max_frame = np.zeros((frame_shape, frame_shape))
    min_frame = np.zeros_like(max_frame)
    mean_frame = np.zeros_like(max_frame)
    norm = np.zeros_like(max_frame)

    Itotal = []
    for maview, diview in zip(P.mask.views.values(), P.diff.views.values()):
        if not diview.active:
            continue
        dv = diview.data
        m = maview.data
        v = m * dv
        Itotal.append(np.sum(v))
        max_frame[max_frame < v] = v[max_frame < v]
        min_frame[min_frame > v] = v[min_frame > v]
        mean_frame += v
        norm += m

    mean_frame /= (norm + (norm == 0))

    P.diff.S['S00'].norm = norm
    P.diff.S['S00'].max_power = np.max(Itotal)
    P.diff.S['S00'].tot_power = np.sum(Itotal)
    P.diff.S['S00'].mean_power = P.diff.S['S00'].tot_power / (len(Itotal) * mean_frame.shape[-1] ** 2)

    mean_power = 0.
    for name, s in P.diff.storages.iteritems():
        mean_power += s.mean_power
    P.mean_power = mean_power / len(P.diff.storages)

    for it in range(num_its):
        print("Iteration: %s" % str(it))

        for dname, di_view in P.diff.views.iteritems():
            basic_fourier_update(di_view, alpha=alpha)

        object_cfact = object_inertia * P.mean_power / 2
        object_storage_names = P.obj.storages.keys()
        downstream_slice = P.obj.storages[object_storage_names[1]]
        upstream_slice = P.obj.storages[object_storage_names[0]]

        #start with the slice closest to the detector

        for inner in range(overlap_max_iterations):
            #  start with the object update first
            # Preconditioning helps with sensitivity around 0

            downstream_slice.data[:] = downstream_slice.data * object_cfact
            ob_nrm = P.ob_nrm
            ob_nrm_storage = ob_nrm.storages[downstream_slice.ID]
            ob_nrm_storage.fill(object_cfact)

            # the following is 10c
            equation10c(downstream_slice, ob_nrm, ob_nrm_storage)

            # now update the "probe". This is different (weaker?) to the usual update also.
            # I haven't done any preconditioning here, I think I should...
            if update_probe:
                equation10b(downstream_slice)

            # for now this loop will exit based on the number of iterations. Should insert a change monitor here...

        # We've exited the lok for the downstream slice
        # now to update the exit wave for the downstream slice with this new guess for object and probe. We blat the
        # probe guess for the downstream slice, but thats ok.
        downstream_slice_views = downstream_slice.views
        for vidx, view in enumerate(upstream_slice.views):
            if not view.pod.active:
                continue

            view.pod.exit = view.pod.bw(downstream_slice_views[vidx].pod.probe)

        # now for the upstream (sample boundary slice), this should just be the regular update

        for inner in range(overlap_max_iterations):
            # first lets update the object
            # Preconditioning helps with sensitivity around 0
            upstream_slice.data[:] = upstream_slice.data[:] * object_cfact
            ob_nrm = P.ob_nrm
            ob_nrm_storage = ob_nrm.storages[upstream_slice.ID]
            ob_nrm_storage.fill(object_cfact)
            equation10c(upstream_slice, ob_nrm, ob_nrm_storage)


            if update_probe:
                # now to update the probe. This is actually the probe
                pr_nrm = P.pr_nrm
                pr_nrm_storage = pr_nrm.storages[upstream_slice.ID]
                probe_storage = P.probe.storages[upstream_slice.ID]
                probe_cfact = probe_inertia * len(probe_storage.views) / probe_storage.data.shape[0]

                probe_storage.data[:] = probe_cfact * probe_storage.data
                pr_nrm_storage.fill(probe_cfact)
                equation10a(upstream_slice, probe_storage, pr_nrm, pr_nrm_storage)
                # apply probe support. This seems particularly important for multislice, I guess due to the propagation artefacts.
                probe_storage.data[:] = probe_storage.data * probe_support
                # for now this loop will exit based on the number of iterations. Should insert a change monitor here...