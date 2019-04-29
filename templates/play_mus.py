'''
Playing wiht the model to see if I can find the bug

'''

from collections import OrderedDict
from ptypy import utils as u
from ptypy.core import geometry, Ptycho, View, POD
import numpy as np



self.Cdiff = Container(self, ID='Cdiff', data_type='real')
self.Cmask = Container(self, ID='Cmask', data_type='bool')

diff = Cdiff.new_storage(shape=sh, psize=self.psize, padonly=True,
                                   layermap=None)

ptycho = Ptycho()
label = 'MF'
main_geo_pars = u.Param()
main_geo_pars.energy = 6.2
main_geo_pars.distance = 7.0
main_geo_pars.psize = 172e-6
main_geo_pars.shape = 256
main_geo_pars.center = 'fftshift'
main_geo_pars.propagation = 'farfield'

print('initialising geometries')
num_energies = 1
num_slices = 2
num_probe_modes = 1
num_object_modes = 1
geometries = OrderedDict()  # the keys will be the slice number

for ix in range(num_slices):
    geometries[ix] = []

# get an initial geometry estimate of the resolution
# Extract necessary info from the received data package
get_keys = ['distance', 'center', 'energy', 'psize', 'shape']
dummy_geo_pars = u.Param({key: main_geo_pars[key] for key in get_keys})
# Add propagation info from this scan model
dummy_geo_pars.propagation = main_geo_pars.propagation
geoID = geometry.Geo._PREFIX + 'dummy'
dummy_geo = geometry.Geo(ptycho, geoID, pars=dummy_geo_pars)
# now we fix the sample pixel size, This will make the frame size adapt
dummy_geo.p.resolution_is_fix = True
psize = dummy_geo.resolution
shape = np.array(dummy_geo.shape)

for slice_idx in range(num_slices):
    if slice_idx == (num_slices - 1):
        geo_pars = u.Param()
        geo_pars.propagation = 'farfield'
        geo_pars.psize = main_geo_pars['psize']
        geo_pars.distance = main_geo_pars['distance']
        geo_pars.energy = main_geo_pars['energy']
        geo_pars.shape = main_geo_pars['shape']
        geo_pars.center = main_geo_pars['center']
    else:
        geo_pars = u.Param()
        geo_pars.propagation = 'nearfield'
        geo_pars.psize = psize
        geo_pars.distance = p.thickness.slice_separation
        geo_pars.energy = main_geo_pars['energy']
        geo_pars.shape = main_geo_pars['shape']
        geo_pars.center = main_geo_pars['center']

    # The multispectral case will have multiple geometries
    for energy_idx, fac in enumerate([main_geo_pars.energy]):
        print("slice idx:%s, geo_idx:%s" % (slice_idx, energy_idx))
        geoID = '%s_%s _energy_%02d_slice_%02d' % (geometry.Geo._PREFIX, label, energy_idx, slice_idx)
        g = geometry.Geo(ptycho, geoID, pars=geo_pars)
        # now we fix the sample pixel size, This will make the frame size adapt
        g.p.resolution_is_fix = True
        # save old energy value:
        g.p.energy_orig = g.energy
        # change energy
        g.energy *= fac
        # append the geometry
        geometries[slice_idx].append(g)

print('\n' + ('Creating PODS', 'l'))
new_pods = []
new_probe_ids = {}
new_object_ids = {}

label = label

# Get a list of probe and object that already exist
existing_probes = ptycho.probe.storages.keys()
existing_objects = ptycho.obj.storages.keys()
print('Found these probes : ' + ', '.join(existing_probes))
print('Found these objects: ' + ', '.join(existing_objects))

object_id = 'S' + label
probe_id = 'S' + label

positions = new_positions
di_views = new_diff_views
ma_views = new_mask_views

# Loop through diffraction patterns
for i in range(len(di_views)):
    dv, mv = di_views.pop(0), ma_views.pop(0)

    index = dv.layer

    # Object and probe position
    pos_pr = u.expect2(0.0)
    pos_obj = positions[i] if 'empty' not in p.tags else 0.0
    print("The geometry keys are:%s" % geometries.keys())
    for slice_idx, geometries in geometries.iteritems():
        print("The geometries are:%s" % geometries)
        for geometry_index, geometry in enumerate(geometries):
            # Make new IDs and keep them in record
            # sharing_rules is not aware of IDs with suffix

            pdis = p.coherence.probe_dispersion

            if pdis is None or str(pdis) == 'achromatic':
                gind = 0
            else:
                gind = num_slices * geometry_index + slice_idx

            probe_id_suf = probe_id + 'G%02d' % gind
            if (probe_id_suf not in new_probe_ids.keys() and probe_id_suf not in existing_probes):
                new_probe_ids[probe_id_suf] = True

            odis = None

            if odis is None or str(odis) == 'achromatic':
                gind = slice_idx
            else:
                gind = p.thickness.num_slices * geometry_index + slice_idx

            object_id_suf = object_id + 'G%02d' % gind
            print("slice idx:%s, geometry index:%s" % (slice_idx, geometry_index))
            print("The object id suffix is %s" % object_id_suf)
            if (object_id_suf not in new_object_ids.keys() and object_id_suf not in existing_objects):
                new_object_ids[object_id_suf] = True

            # Loop through modes
            for pm in range(p.coherence.num_probe_modes):
                for om in range(p.coherence.num_object_modes):
                    # Make a unique layer index for exit view
                    # The actual number does not matter due to the
                    # layermap access
                    obj_layer = om
                    exit_index = index * 10000 + pm * 100 + p.thickness.num_slices * om + obj_layer

                    # Create views
                    # Please note that mostly references are passed,
                    # i.e. the views do mostly not own the accessrule
                    # contents
                    pv = View(container=ptycho.probe,
                              accessrule={'shape': geometry.shape,
                                          'psize': geometry.resolution,
                                          'coord': pos_pr,
                                          'storageID': probe_id_suf,
                                          'layer': pm,
                                          'active': True})

                    ov = View(container=ptycho.obj,
                              accessrule={'shape': geometry.shape,
                                          'psize': geometry.resolution,
                                          'coord': pos_obj,
                                          'storageID': object_id_suf,
                                          'layer': obj_layer,
                                          'active': True})

                    ev = View(container=ptycho.exit,
                              accessrule={'shape': geometry.shape,
                                          'psize': geometry.resolution,
                                          'coord': pos_pr,
                                          'storageID': (dv.storageID +
                                                        'G%02d' % geometry_index),
                                          'layer': exit_index,
                                          'active': dv.active})

                    views = {'probe': pv,
                             'obj': ov,
                             'diff': dv,
                             'mask': mv,
                             'exit': ev}

                    pod = POD(ptycho=ptycho,
                              ID=None,
                              views=views,
                              geometry=geometry)  # , meta=meta)

                    new_pods.append(pod)

                    pod.probe_weight = 1
                    pod.object_weight = 1