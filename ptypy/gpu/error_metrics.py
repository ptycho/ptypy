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
    abs2 = (ft * ft.conj()).real
    LL = np.zeros_like(Idata)
    for pa, oa, ea, da, ma in addr_info:
        LL[da[0]] += abs2[ea[0]]

    for pa, oa, ea, da, ma in addr_info:
        LLerror[da[0]] = (np.sum(mask[ma[0]] * (LL[da[0]] - Idata[da[0]]) ** 2 / (Idata[da[0]] + 1.))
                        / np.prod(LL[da[0]].shape))

    return LLerror