'''
A module of the relevant error metrics
'''

from propagation import farfield_propagator
from object_probe_interaction import scan_and_multiply
import numpy as np


def log_likelihood(probe, obj, mask, exit_wave, Idata, prefilter, postfilter, addr):
    view_dlayer=0
    addr_info = addr[:, view_dlayer]
    LLerror = np.zeros(Idata.shape[0], dtype=np.float64)
    probe_and_object = scan_and_multiply(probe, obj, exit_wave.shape, addr_info)
    ft = farfield_propagator(probe_and_object, prefilter, postfilter, direction='forward')
    abs2 = (np.multiply(ft, ft.conj())).real
    LL = np.zeros_like(Idata)
    for pa, oa, ea, da, ma in addr_info:
        LL[da[0]] += abs2[ea[0]]

    for pa, oa, ea, da, ma in addr_info:
        LLerror[da[0]] = np.divide(np.sum(np.power(np.multiply(mask[ma[0]], (np.subtract(LL[da[0]], Idata[da[0]]))), 2) / np.add(Idata[da[0]], 1.)), np.prod(LL[da[0]].shape))
    return LLerror

def far_field_error(current_solution, pbound, measured_solution, mask, addr):
    addr_info = addr[:, 0]
    fdev = np.subtract(current_solution, measured_solution)
    summed_mask = np.sum(mask, axis=(-2, -1))
    fdev2 = np.power(fdev, 2)
    masked_fdev2 = np.multiply(mask, fdev2)
    summed_masked_fdev2 = np.sum(masked_fdev2, axis=(-2, -1))
    err_fmag = summed_masked_fdev2 / summed_mask
    for _pa, _oa, ea, da, ma in addr_info:
        if pbound[da[0]] is not None:
            err_fmag[da[0]] /= pbound[da[0]]
    return err_fmag

def realspace_error(difference_in_exitwave):
    return np.mean(np.power(np.abs(difference_in_exitwave), 2), axis=(-2,-1))
