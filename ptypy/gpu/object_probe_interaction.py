'''
object_probe_interaction

Contains things pertinent to the probe and object interaction.
Should have all the engine updates
'''

import numpy as np

def get_exit_wave(serialized_scan):
    '''
    probably useful for Darren
    '''
    meta = serialized_scan['meta'] # probably want to extract these at a later date, but just to get stuff going...
    probe = serialized_scan['probe']
    obj = serialized_scan['obj']
    exit_wave = serialized_scan['exit wave']
    sh = probe.shape
    view_dlayer = 0 # what is this?
    addr_info = meta['addr'][:,(view_dlayer)] # addresses, object references
    for pa, oa, ea,  _da, _ma in addr_info:
#         print("pa:%s, oa:%s, ea:%s, da:%s, ma:%s" % (pa, oa, ea,  da, ma))
        exit_wave[ea[0], ea[1]:(ea[1]+sh[1]), ea[2]:(ea[2]+sh[2])] = \
            np.multiply(probe[pa[0], pa[1]:(pa[1]+sh[1]), pa[2]:(pa[2]+sh[2])], 
                        obj[oa[0], oa[1]:(oa[1]+sh[1]), oa[2]:(oa[2]+sh[2])])
    serialized_scan['exit wave'] = exit_wave
    return serialized_scan

def difference_map_realspace_constraint(serialized_scan, alpha):
    '''
    in theory this can just be called in ptypy instead of get_exit_wave
    '''
    meta = serialized_scan['meta'] # probably want to extract these at a later date, but just to get stuff going...
    probe = serialized_scan['probe']
    sh = probe.shape

    obj = serialized_scan['obj']
    exit_wave = serialized_scan['exit wave']
    view_dlayer = 0 # what is this?
    addr_info = meta['addr'][:,(view_dlayer)] # addresses, object references
    for pa, oa, ea,  _da, _ma in addr_info:
        foo = np.multiply(probe[pa[0], pa[1]:(pa[1]+sh[1]), pa[2]:(pa[2]+sh[2])], 
                          obj[oa[0], oa[1]:(oa[1]+sh[1]), oa[2]:(oa[2]+sh[2])])
        exit_wave[ea[0], ea[1]:(ea[1]+sh[1]), ea[2]:(ea[2]+sh[2])] = \
        (1.0 + alpha) * foo - alpha * exit_wave[ea[0], ea[1]:(ea[1]+sh[1]), ea[2]:(ea[2]+sh[2])]
    serialized_scan['exit wave'] = exit_wave
    return serialized_scan


