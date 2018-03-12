'''
a module to holds the constraints
'''

import numpy as np

from error_metrics import log_likelihood, far_field_error, realspace_error
from object_probe_interaction import difference_map_realspace_constraint, scan_and_multiply
from propagation import farfield_propagator
import array_utils as au
from . import COMPLEX_TYPE, FLOAT_TYPE

def renormalise_fourier_magnitudes(f, af, fmag, mask, err_fmag, addr_info, pbound):
    renormed_f = np.zeros(f.shape, dtype=np.complex128)
    for _pa, _oa, ea, da, ma in addr_info:
        m = mask[ma[0]]
        magnitudes = fmag[da[0]]
        absolute_magnitudes = af[da[0]]
        fourier_space_solution = f[ea[0]]
        fourier_error = err_fmag[da[0]]
        if pbound is None:
            fm = (1 - m) + m * magnitudes / (absolute_magnitudes + 1e-10)
            renormed_f[ea[0]] = np.multiply(fm, fourier_space_solution)
        elif (fourier_error > pbound):
            # Power bound is applied
            fdev = absolute_magnitudes - magnitudes
            renorm = np.sqrt(pbound / fourier_error)
            fm = (1 - m) + m * (magnitudes + fdev * renorm) / (absolute_magnitudes + 1e-10)
            renormed_f[ea[0]] = np.multiply(fm, fourier_space_solution)
        else:
            renormed_f[ea[0]] = np.zeros_like(fourier_space_solution)
    return renormed_f

def get_difference(addr_info, alpha, backpropagated_solution, err_fmag, exit_wave, pbound, probe_object):
    df = np.zeros(exit_wave.shape, dtype=np.complex128)
    for _pa, _oa, ea, da, ma in addr_info:
        if (pbound is None) or (err_fmag[da[0]] > pbound):
            df[ea[0]] = np.subtract(backpropagated_solution[ea[0]], probe_object[ea[0]])
        else:
            df[ea[0]] = alpha * np.subtract(probe_object[ea[0]], exit_wave[ea[0]])
    return df

def difference_map_fourier_constraint(mask, Idata, obj, probe, exit_wave, addr, prefilter, postfilter, pbound=None, alpha=1.0, LL_error=True, do_realspace_error=True):
    '''
    This kernel just performs the fourier renormalisation.
    :param mask. The nd mask array
    :param diffraction. The nd diffraction data
    :param farfield_stack. The current iterant.
    :param addr. The addresses of the stacks.
    :return: The updated iterant
            : fourier errors
    '''
    view_dlayer = 0 # what is this?
    addr_info = addr[:,(view_dlayer)] # addresses, object references
    probe_object = scan_and_multiply(probe, obj, exit_wave.shape, addr_info)

    # Buffer for accumulated photons
    # For log likelihood error # need to double check this adp
    if LL_error is True:
        err_phot = log_likelihood(probe_object, mask, Idata, prefilter, postfilter, addr)
    else:
        err_phot = np.zeros(Idata.shape[0], dtype=FLOAT_TYPE)
    
    
    constrained = difference_map_realspace_constraint(probe_object, exit_wave, alpha)
    f = farfield_propagator(constrained, prefilter, postfilter, direction='forward')
    pa, oa, ea, da, ma = zip(*addr_info)
    af2 = au.sum_to_buffer(au.abs2(f), Idata.shape, ea, da, dtype=FLOAT_TYPE)

    fmag = np.sqrt(np.abs(Idata))
    af = np.sqrt(af2)
    # # Fourier magnitudes deviations(current_solution, pbound, measured_solution, mask, addr)
    err_fmag = far_field_error(af, fmag, mask)

    vectorised_rfm = renormalise_fourier_magnitudes(f, af, fmag, mask, err_fmag, addr_info, pbound)

    backpropagated_solution = farfield_propagator(vectorised_rfm,
                                                  postfilter.conj(),
                                                  prefilter.conj(),
                                                  direction='backward')

    df = get_difference(addr_info, alpha, backpropagated_solution, err_fmag, exit_wave, pbound, probe_object)

    exit_wave += df

    if do_realspace_error:
        err_exit = realspace_error(df)
    else:
        err_exit = np.zeros((Idata.shape[0]))

    if pbound is not None:
        err_fmag /= pbound

    return np.array([err_fmag, err_phot, err_exit])


