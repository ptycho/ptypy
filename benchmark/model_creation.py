from ptypy.core.classes import View, Storage, POD, Container, Base
from ptypy.utils import parallel, expect2, logger
from ptypy.core.xy import raster_scan
import numpy as np
import time
import json
import sys

# logger.setLevel(3)
data_type = "single"

class Ptycho(Base):
    def __init__(self, *args, data_type="single", **kwargs):
        super().__init__(*args, **kwargs)
        self.FType = np.dtype('f' + str(np.dtype(np.sctypeDict[data_type]).itemsize)).type
        self.CType = np.dtype('c' + str(2 * np.dtype(np.sctypeDict[data_type]).itemsize)).type

t0 = time.time()

base = Ptycho()

Cdiff = Container(owner=base, ID='Cdiff', data_type='real')
Cmask = Container(owner=base, ID='Cmask', data_type='bool')
Cobj = Container(owner=base, ID='Cobj', data_type='complex')
Cprb = Container(owner=base, ID='Cprb', data_type='complex')
Cexit = Container(owner=base, ID='Cexit', data_type='complex')

scale = int(sys.argv[1])
nx = 32 * (2**scale)
ny = 32 
nframes = nx*ny
psize = 1
bsize = 1024
nsize = 128
nprobemodes = 2
# sh = (bsize,nsize,nsize)

allpositions = raster_scan(nx =nx, ny = ny,
                        dx = nsize//8,
                        dy = nsize//8)

for bindex in range(nframes // bsize):

    parallel.barrier()
    if parallel.master:
        print(f"Starting Block {bindex}")

    istart, iend = bindex * bsize, (bindex + 1) * bsize
    indices = range(istart, iend)
    indices_node = parallel.loadmanager.assign(indices)[parallel.rank]

    diff = Cdiff.new_storage(shape=(len(indices_node),nsize,nsize), psize=psize, padonly=True,
                                    fill=0.0, layermap=indices_node)
    mask = Cmask.new_storage(shape=(len(indices_node), nsize, nsize), psize=psize, padonly=True,
                                    fill=1.0, layermap=indices_node)

    diff_views = []
    mask_views = []
    positions = []

    for k, index in enumerate(indices):

        dv = View(Cdiff, storageID=diff.ID, shape=nsize, coord=0.0, psize=psize)  # maybe use index here
        mv = View(Cmask, storageID=mask.ID, shape=nsize, coord=0.0, psize=psize)
        active = k in indices_node

        dv.active = active
        mv.active = active
        dv.layer = index
        mv.layer = index

        diff_views.append(dv)
        mask_views.append(mv)

        if active:
            l = indices_node.index(k)
            dv.dlayer = l
            mv.dlayer = l
            dv.data[:] = 0
            mv.data[:] = 1

    positions = allpositions[istart:iend]

    diff.nlayers = parallel.MPImax(diff.layermap) + 1
    mask.nlayers = parallel.MPImax(mask.layermap) + 1

    new_diff_views = diff_views
    new_mask_views = mask_views

    new_pods = []
    new_probe_ids = {}
    new_object_ids = {}

    label = "scan_00"
    object_id = 'S' + label
    probe_id = 'S' + label

    # Loop through diffraction patterns
    for i in range(len(new_diff_views)):
        dv, mv = new_diff_views.pop(0), new_mask_views.pop(0)
        index = dv.layer

        # Object and probe position
        pos_pr = expect2(0.0)
        pos_obj = positions[i]

        gind = 0
        probe_id_suf = probe_id + 'G%02d' % gind
        new_probe_ids[probe_id_suf] = True

        gind = 0
        object_id_suf = object_id + 'G%02d' % gind
        new_object_ids[object_id_suf] = True

        # Loop through modes
        for pm in range(nprobemodes):
            for om in range(1):
                # Make a unique layer index for exit view
                # The actual number does not matter due to the
                # layermap access
                exit_index = index * 10000 + pm * 100 + om

                # Create views
                # Please note that mostly references are passed,
                # i.e. the views do mostly not own the accessrule
                # contents
                pv = View(container=Cprb,
                            accessrule={'shape': (nsize,nsize),
                                        'psize': psize,
                                        'coord': pos_pr,
                                        'storageID': probe_id_suf,
                                        'layer': pm,
                                        'active': True})

                ov = View(container=Cobj,
                            accessrule={'shape': (nsize,nsize),
                                        'psize': psize,
                                        'coord': pos_obj,
                                        'storageID': object_id_suf,
                                        'layer': om,
                                        'active': True})

                ev = View(container=Cexit,
                            accessrule={'shape': (nsize,nsize),
                                        'psize': psize,
                                        'coord': pos_pr,
                                        'storageID': (dv.storageID +
                                                    'G%02d' % gind),
                                        'layer': exit_index,
                                        'active': dv.active})

                views = {'probe': pv,
                            'obj': ov,
                            'diff': dv,
                            'mask': mv,
                            'exit': ev}

                pod = POD(ptycho=base,
                            ID=None,
                            views=views,
                            geometry=None)

                new_pods.append(pod)

                pod.probe_weight = 1.0
                pod.object_weight = 1.0

    logger.info('Process %d created %d new PODs, %d new probes and %d new objects.' % (
        parallel.rank, len(new_pods), len(new_probe_ids), len(new_object_ids)), extra={'allprocesses': True})

t1 = time.time()

# Reformatting
Cprb.reformat(True)
Cobj.reformat(True)
Cexit.reformat(True)

# print(Cexit.formatted_report())

output = {}
output["reformat_time"] = time.time() - t1
output["model_time"] = t1 - t0
output["nframes"] = nframes
output["blocksize"] = bsize
output["nmodes"] = nprobemodes
output["shape"] = nsize
output["objsize"] = [int(sh) for sh in pod.ob_view.storage.shape]

# print(output)

if parallel.master:
    with open(f"./model_creation_{scale}.json", "w") as f:
        json.dump(output, f)