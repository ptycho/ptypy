'''
A module of the relevant error metrics
'''

from propagation import farfield_propagator
from array_utils import sum_to_buffer, abs2
from . import FLOAT_TYPE, COMPLEX_TYPE
import numpy as np


def log_likelihood(probe_and_obj, mask, Idata, prefilter, postfilter, addr_info):
    _pa, _oa, ea, da, ma = zip(*addr_info)
    LLerror = np.zeros(Idata.shape[0], dtype=FLOAT_TYPE)
    ft = farfield_propagator(probe_and_obj, prefilter, postfilter, direction='forward')
    abs2_ft = abs2(ft)
    LL = sum_to_buffer(abs2_ft, Idata.shape, ea, da, dtype=Idata.dtype)
    max_da = np.max(da)+1
    for dai, mai in zip(da[:max_da], ma[:max_da]):
        LLerror[dai[0]] = np.divide(np.sum(np.power(np.multiply(mask[mai[0]], (np.subtract(LL[dai[0]], Idata[dai[0]]))), 2) / np.add(Idata[dai[0]], 1.)), np.prod(LL[dai[0]].shape))
    return LLerror

def far_field_error(current_solution, measured_solution, mask):
    fdev = np.subtract(current_solution, measured_solution)
    summed_mask = np.sum(mask, axis=(-2, -1))
    fdev2 = np.power(fdev, 2)
    masked_fdev2 = np.multiply(mask, fdev2)
    summed_masked_fdev2 = np.sum(masked_fdev2, axis=(-2, -1))
    err_fmag = summed_masked_fdev2 / summed_mask
    return err_fmag

def realspace_error(difference_in_exitwave, ea_first_column, da_first_column, out_length):
    errors = np.mean(abs2(difference_in_exitwave), axis=(-2, -1))
    collapsed_errors = np.zeros((out_length,), dtype=FLOAT_TYPE)
    for ea_idx, da_idx in zip(ea_first_column, da_first_column):
        collapsed_errors[da_idx] += errors[ea_idx]
    return collapsed_errors
