'''
object_probe_interaction

Contains things pertinent to the probe and object interaction.
Should have all the engine updates
'''

import numpy as np
from . import COMPLEX_TYPE

def difference_map_realspace_constraint(probe_and_object, exit_wave, alpha):
    '''
    in theory this can just be called in ptypy instead of get_exit_wave
    '''
    return (1.0 + alpha) * probe_and_object - alpha*exit_wave


def scan_and_multiply(probe, obj, exit_shape, addresses):
    sh = exit_shape
    po = np.zeros((sh[0], sh[1], sh[2]), dtype=COMPLEX_TYPE)
    for pa, oa, ea, _da, _ma in addresses:
        po[ea[0]] = np.multiply(probe[pa[0], pa[1]:(pa[1] + sh[1]), pa[2]:(pa[2] + sh[2])],
                             obj[oa[0], oa[1]:(oa[1] + sh[1]), oa[2]:(oa[2] + sh[2])])
    return po