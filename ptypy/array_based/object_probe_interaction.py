'''
object_probe_interaction

Contains things pertinent to the probe and object interaction.
Should have all the engine updates
'''

import numpy as np
from array_utils import norm2, complex_gaussian_filter, abs2, mass_center, shift_zoom, clip_complex_magnitudes_to_range, sum_to_buffer_inplace
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


def difference_map_update_object(ob, object_weights, ob_nrm, probe, exit_wave, addr_info, cfact_object, ob_smooth_std=None, clip_object=None):

    ob_nrm  *= cfact_object

    if ob_smooth_std is not None:
        smooth_mfs = [ob_smooth_std, ob_smooth_std]
        ob = ob_nrm * complex_gaussian_filter(ob, smooth_mfs)
    else:
        ob *= ob_nrm

    extract_object_from_exit_wave(exit_wave, ob, ob_nrm, object_weights, probe, addr_info)

    if clip_object is not None:
        clip_min, clip_max = clip_object
        clip_complex_magnitudes_to_range(ob, clip_min, clip_max)


def extract_object_from_exit_wave(exit_wave, ob, ob_nrm, object_weights, probe, addr_info):
    pa, oa, ea, _da, _ma = zip(*addr_info)
    divided_exit_wave = np.zeros_like(exit_wave)
    normalisation = np.zeros_like(exit_wave)
    factor_out_exit_wave(divided_exit_wave, exit_wave, probe, object_weights, ea, pa, oa)
    get_object_probe_normalisation(normalisation, probe, object_weights, ea, pa, oa)
    sum_to_buffer_inplace(divided_exit_wave, ob, ea, oa)
    sum_to_buffer_inplace(normalisation, ob_nrm, ea, oa)
    ob /= ob_nrm
    return ob


def difference_map_update_probe(ob, probe_weights, probe, pr_nrm, exit_wave, addr_info, cfact_probe, probe_support=None):

    old_probe = probe
    probe *= cfact_probe
    pr_nrm *=cfact_probe

    extract_probe_from_exit_wave(exit_wave, ob, pr_nrm, probe, probe_weights, addr_info)

    # Distribute result with MPI
    if probe_support is not None:
        probe *= probe_support

    # Compute relative change in probe

    change = norm2(probe - old_probe) /norm2(probe)

    return np.sqrt(change / probe.shape[0])


def extract_probe_from_exit_wave(exit_wave, ob, pr_nrm, probe, probe_weights, addr_info):
    pa, oa, ea, _da, _ma = zip(*addr_info)
    divided_exit_wave = np.zeros_like(exit_wave)
    normalisation = np.zeros_like(exit_wave)
    factor_out_exit_wave(divided_exit_wave, exit_wave, ob, probe_weights, ea, oa, pa)
    get_object_probe_normalisation(normalisation, ob, probe_weights, ea, oa, pa)
    sum_to_buffer_inplace(divided_exit_wave, probe, ea, pa)
    sum_to_buffer_inplace(normalisation, pr_nrm, ea, pa)
    probe /= pr_nrm
    return probe


def get_object_probe_normalisation(normalisation, ob, probe_weights, norm_addr, ob_addr, pw_addr):
    outshape = normalisation.shape
    for ea, oa, pa in zip(norm_addr, ob_addr, pw_addr):
        normalisation[ea[0]] = ob[oa[0], oa[1]:(oa[1] + outshape[1]), oa[2]:(oa[2] + outshape[2])] * \
                               ob[oa[0], oa[1]:(oa[1] + outshape[1]), oa[2]:(oa[2] + outshape[2])].conj() * \
                               probe_weights[pa[0]]


def factor_out_exit_wave(divided_exit_wave, exit_wave, ob, probe_weights, ew_addr, ob_addr, pr_addr):
    sh = exit_wave.shape
    for ea, oa, pa in zip(ew_addr, ob_addr, pr_addr):
        divided_exit_wave[ea[0]] = ob[oa[0], oa[1]:(oa[1] + sh[1]), oa[2]:(oa[2] + sh[2])].conj() * exit_wave[ea[0]] * \
                                   probe_weights[pa[0]]


def center_probe(probe, center_tolerance):
    for idx in range(probe.shape[0]):
        c1 = mass_center(abs2(probe[idx]).sum(0))
        c2 = np.asarray(probe[idx].shape[-2:]) // 2
        if np.sqrt(norm2(c1 - c2)) < center_tolerance:
            break

        probe[idx] = shift_zoom(probe[idx],
                                 (1.,) * 3,
                                 (0, c1[0], c1[1]),
                                 (0, c2[0], c2[1]))


