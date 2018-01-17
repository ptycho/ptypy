'''
Created on 4 Jan 2018

@author: clb02321
'''

import numpy as np
from ..utils.verbose import log


def serialize_array_access(diff_storage):
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

    addr = []
    for view in views:
        address = []
        for _pname, pod in view.pods.iteritems():
            # store them for each pod
            # create addresses
            a = np.array([
                (pod.pr_view.dlayer, pod.pr_view.dlow[0], pod.pr_view.dlow[1]),
                (pod.ob_view.dlayer, pod.ob_view.dlow[0], pod.ob_view.dlow[1]),
                (pod.ex_view.dlayer, pod.ex_view.dlow[0], pod.ex_view.dlow[1]),
                (pod.di_view.dlayer, pod.di_view.dlow[0], pod.di_view.dlow[1]),
                (pod.ma_view.dlayer, pod.ma_view.dlow[0], pod.ma_view.dlow[1])])
            address.append(a)
        # store data for each view
        # addresses
        addr.append(address)
    # store them for each storage
    return view_IDs, poe_ID, np.array(addr).astype(np.int32)


def pod_to_numpy(diff_storage):
    '''
    converts between the pod structure and a series of numpy arrays and a look-up table
    :param diff_storage: The diffraction storage.
    :return: dictionary containing the probe, mask, diffraction data, exit wave and object buffers and a LUT of the metadata.
    '''
    


