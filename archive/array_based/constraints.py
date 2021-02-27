'''
a module to holds the constraints
'''

import numpy as np

from .error_metrics import log_likelihood, far_field_error, realspace_error
from .object_probe_interaction import difference_map_realspace_constraint, scan_and_multiply, difference_map_overlap_update
from .propagation import farfield_propagator
from ptypy.accelerate.array_based import array_utils as au
from ptypy.accelerate.array_based import COMPLEX_TYPE, FLOAT_TYPE

def renormalise_fourier_magnitudes(f, af, fmag, mask, err_fmag, addr_info, pbound):
    renormed_f = np.zeros(f.shape, dtype=COMPLEX_TYPE)
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
    df = np.zeros(exit_wave.shape, dtype=COMPLEX_TYPE)
    for _pa, _oa, ea, da, ma in addr_info:
        if (pbound is None) or (err_fmag[da[0]] > pbound):
            df[ea[0]] = np.subtract(backpropagated_solution[ea[0]], probe_object[ea[0]])
        else:
            df[ea[0]] = alpha * np.subtract(probe_object[ea[0]], exit_wave[ea[0]])
    return df

def difference_map_fourier_constraint(mask, Idata, obj, probe, exit_wave, addr_info, prefilter, postfilter, pbound=None, alpha=1.0, LL_error=True, do_realspace_error=True):
    '''
    This kernel just performs the fourier renormalisation.
    :param mask. The nd mask array
    :param diffraction. The nd diffraction data
    :param farfield_stack. The current iterant.
    :param addr. The addresses of the stacks.
    :return: The updated iterant
            : fourier errors
    '''

    probe_object = scan_and_multiply(probe, obj, exit_wave.shape, addr_info)

    # Buffer for accumulated photons
    # For log likelihood error # need to double check this adp
    if LL_error is True:
        err_phot = log_likelihood(probe_object, mask, Idata, prefilter, postfilter, addr_info)
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
        ea_first_column = np.array(ea)[:, 0]
        da_first_column = np.array(da)[:, 0]
        err_exit = realspace_error(df, ea_first_column, da_first_column, Idata.shape[0])
    else:
        err_exit = np.zeros((Idata.shape[0]))

    if pbound is not None:
        err_fmag /= pbound

    return np.array([err_fmag, err_phot, err_exit])


def difference_map_iterator(diffraction, obj, object_weights, cfact_object, mask, probe, cfact_probe, probe_support,
                            probe_weights, exit_wave, addr, pre_fft, post_fft, pbound, overlap_max_iterations, update_object_first,
                            obj_smooth_std, overlap_converge_factor, probe_center_tol, probe_update_start, alpha=1,
                            clip_object=None, LL_error=False, num_iterations=1):
    curiter = 0

    errors = np.zeros((num_iterations, 3, len(diffraction)), dtype=FLOAT_TYPE)
    for it in range(num_iterations):
        if (((it+1) % 10) == 0) and (it>0):
            print("iteration:%s" % (it+1)) # it's probably a good idea to print this if possible for some idea of progress
        # numpy dump here for 64x64 and 4096x4096

        errors[it] = difference_map_fourier_constraint(mask,
                                                   diffraction,
                                                   obj,
                                                   probe,
                                                   exit_wave,
                                                   addr,
                                                   prefilter=pre_fft,
                                                   postfilter=post_fft,
                                                   pbound=pbound,
                                                   alpha=alpha,
                                                   LL_error=LL_error)

        do_update_probe = (probe_update_start <= curiter)
        difference_map_overlap_update(addr,
                                      cfact_object,
                                      cfact_probe,
                                      do_update_probe,
                                      exit_wave,
                                      obj,
                                      object_weights,
                                      probe,
                                      probe_support,
                                      probe_weights,
                                      overlap_max_iterations,
                                      update_object_first,
                                      obj_smooth_std,
                                      overlap_converge_factor,
                                      probe_center_tol,
                                      clip_object=clip_object)
        curiter += 1
    return errors