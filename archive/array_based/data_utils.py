'''
Created on 4 Jan 2018

@author: clb02321
'''

import numpy as np
from ptypy.accelerate.array_based import FLOAT_TYPE


def _vectorise_array_access(diff_storage):
    # Sort views according to layer in diffraction stack
    views = diff_storage.views
    dlayers = [view.dlayer for view in views]
    views = [views[i] for i in np.argsort(dlayers)]
    view_IDs = [view.ID for view in views]

    # Master pod
    mpod = views[0].pod

    # Determine linked storages for probe, object and exit waves
    pr = mpod.pr_view.storage
    ob = mpod.ob_view.storage
    ex = mpod.ex_view.storage

    poe_ID = (pr.ID, ob.ID, ex.ID)
    probe_weights = []
    object_weights = []
    addr = []
    for view in views:
        address = []
        for _pname, pod in view.pods.items():
            # store them for each pod
            # create addresses
            probe_weights.append(pod.probe_weight)
            object_weights.append(pod.object_weight)

            a = np.array([
                (pod.pr_view.dlayer, pod.pr_view.dlow[0], pod.pr_view.dlow[1]),
                (pod.ob_view.dlayer, pod.ob_view.dlow[0], pod.ob_view.dlow[1]),
                (pod.ex_view.dlayer, pod.ex_view.dlow[0], pod.ex_view.dlow[1]),
                (pod.di_view.dlayer, pod.di_view.dlow[0], pod.di_view.dlow[1]),
                (pod.ma_view.dlayer, pod.ma_view.dlow[0], pod.ma_view.dlow[1])])
            addr.append(a)

    addr_out = np.array(addr).astype(np.int32)


    return view_IDs, poe_ID, addr_out, np.array(probe_weights, dtype=np.float32), np.array(object_weights, dtype=np.float32)

def pod_to_arrays(P, storage_id, scan_model='Full'):
    '''
    :param P. A ptycho instance
    :param: storage_id The storage ID for this scan.
    
    :return
    a dictionary containing:
        diffraction: The diffraction data
        probe: the probe from the FIRST POD
        obj: The object buffer
        exit wave: The exit wave buffer
        mask: The diffraction masks
        meta: The meta data, containing an 'addr' array for the addresses
    '''
    if scan_model == 'Full':
        diffraction_storages_to_iterate = P.di.storages[storage_id]
        mask_storages = P.ma.storages[storage_id]
        view_IDs, poe_IDs, addr, probe_weights, object_weights = _vectorise_array_access(diffraction_storages_to_iterate)
        meta = {'view_IDs': view_IDs,
                'poe_IDs': poe_IDs,
                'addr': addr}
        main_pod = P.di.V[view_IDs[0]].pod # we will use this to get all the information
        probe_array = main_pod.pr_view.storage.data
        obj_array = main_pod.ob_view.storage.data
        obj_viewcover = main_pod.ob_view.storage.get_view_coverage()
        exit_wave_array = main_pod.ex_view.storage.data
        mask_array = mask_storages.data # can we have booleans?
        diff_array = diffraction_storages_to_iterate.data


    return {'diffraction': diff_array,
            'probe': probe_array,
            'probe weights': probe_weights,
            'obj': obj_array,
            'object viewcover': obj_viewcover,
            'object weights': object_weights,
            'exit wave': exit_wave_array,
            'mask': mask_array,
            'meta': meta}

