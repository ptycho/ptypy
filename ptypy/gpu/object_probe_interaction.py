'''
object_probe_interaction

Contains things pertinent to the probe and object interaction.
Should have all the engine updates
'''

import numpy as np

def get_exit_wave(obj, probe, exit_wave, addr):
    '''
    probably useful for Darren
    '''
    sh = probe.shape
    view_dlayer = 0 # what is this?
    addr_info = addr[:,(view_dlayer)] # addresses, object references
    for pa, oa, ea,  _da, _ma in addr_info:
#         print("pa:%s, oa:%s, ea:%s, da:%s, ma:%s" % (pa, oa, ea,  da, ma))
        exit_wave[ea[0], ea[1]:(ea[1]+sh[1]), ea[2]:(ea[2]+sh[2])] = \
            np.multiply(probe[pa[0], pa[1]:(pa[1]+sh[1]), pa[2]:(pa[2]+sh[2])], 
                        obj[oa[0], oa[1]:(oa[1]+sh[1]), oa[2]:(oa[2]+sh[2])])
    return exit_wave

def difference_map_realspace_constraint(obj, probe, exit_wave, addr, alpha):
    '''
    in theory this can just be called in ptypy instead of get_exit_wave
    '''
    sh = probe.shape
    view_dlayer = 0 # what is this?
    addr_info = addr[:,(view_dlayer)] # addresses, object references
    for pa, oa, ea,  _da, _ma in addr_info:
        foo = np.multiply(probe[pa[0], pa[1]:(pa[1]+sh[1]), pa[2]:(pa[2]+sh[2])], 
                          obj[oa[0], oa[1]:(oa[1]+sh[1]), oa[2]:(oa[2]+sh[2])])
        exit_wave[ea[0], ea[1]:(ea[1]+sh[1]), ea[2]:(ea[2]+sh[2])] = \
        (1.0 + alpha) * foo - alpha * exit_wave[ea[0], ea[1]:(ea[1]+sh[1]), ea[2]:(ea[2]+sh[2])]
    return exit_wave


