'''
object_probe_interaction

Contains things pertinent to the probe and object interaction.
Should have all the engine updates
'''

import numpy as np


def difference_map_realspace_constraint(obj, probe, exit_wave, addr, alpha):
    '''
    in theory this can just be called in ptypy instead of get_exit_wave
    '''
    view_dlayer = 0 # what is this?
    addr_info = addr[:,(view_dlayer)] # addresses, object references
    probe_and_object = scan_and_multiply(probe, obj, exit_wave.shape, addr_info)
    return np.subtract(np.multiply(np.add(1.0, alpha), probe_and_object), np.multiply(alpha, exit_wave))


def scan_and_multiply(probe, obj, exit_shape, addresses):
    sh = exit_shape
    po = np.zeros((sh[0], sh[1], sh[2]), dtype=obj.dtype)
    for pa, oa, ea, _da, _ma in addresses:
        po[ea] = np.multiply(probe[pa[0], pa[1]:(pa[1] + sh[1]), pa[2]:(pa[2] + sh[2])],
                             obj[oa[0], oa[1]:(oa[1] + sh[1]), oa[2]:(oa[2] + sh[2])])
    return po