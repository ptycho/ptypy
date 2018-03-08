'''
object_probe_interaction

Contains things pertinent to the probe and object interaction.
Should have all the engine updates
'''

import numpy as np
from .. import utils as u
from . import COMPLEX_TYPE


def difference_map_realspace_constraint(obj, probe, exit_wave, addr, alpha):
    '''
    in theory this can just be called in ptypy instead of get_exit_wave
    '''
    view_dlayer = 0 # what is this?
    addr_info = addr[:,(view_dlayer)] # addresses, object references
    probe_and_object = scan_and_multiply(probe, obj, exit_wave.shape, addr_info)
    return (1.0 + alpha) * probe_and_object - alpha*exit_wave


def scan_and_multiply(probe, obj, exit_shape, addresses):
    sh = exit_shape
    po = np.zeros((sh[0], sh[1], sh[2]), dtype=COMPLEX_TYPE)
    for pa, oa, ea, _da, _ma in addresses:
        po[ea[0]] = np.multiply(probe[pa[0], pa[1]:(pa[1] + sh[1]), pa[2]:(pa[2] + sh[2])],
                             obj[oa[0], oa[1]:(oa[1] + sh[1]), oa[2]:(oa[2] + sh[2])])
    return po


def difference_map_update_object(ob, object_weights, probe, exit_wave, ob_viewcover, addr_info, mean_power, ob_inertia, ob_smooth_std=None, clip_object=None):

    sh = exit_wave.shape

    cfact = ob_inertia * mean_power  *  (ob_viewcover + 1.)
    ob_nrm  = cfact
    print('average cfact is %s' % np.mean(ob_nrm).real)

    if ob_smooth_std is not None:
        smooth_mfs = [ob_smooth_std, ob_smooth_std]
        ob = ob_nrm * u.c_gf(ob, smooth_mfs)
    else:
        ob *= ob_nrm

    for pa, oa, ea, _da, _ma in addr_info:
        probe_mode = probe[pa[0]]
        ob[oa[0], oa[1]:(oa[1] + sh[1]), oa[2]:(oa[2] + sh[2])] += probe_mode.conj() * exit_wave[ea[0]] * object_weights[oa[0]]
        ob_nrm[oa[0], oa[1]:(oa[1] + sh[1]), oa[2]:(oa[2] + sh[2])] += probe_mode.conj() * probe_mode * object_weights[oa[0]]


    ob/= ob_nrm

    if clip_object is not None:
        clip_min, clip_max = clip_object
        ampl_obj = np.abs(ob)
        phase_obj = np.exp(1j * np.angle(ob))
        too_high = (ampl_obj > clip_max)
        too_low = (ampl_obj < clip_min)
        ob[too_high] = clip_max * phase_obj[too_high]
        ob[too_low] = clip_min * phase_obj[too_low]