'''
a module to holds the constraints
'''

import numpy as np

import array_utils as au
from error_metrics import log_likelihood, far_field_error, realspace_error
from object_probe_interaction import difference_map_realspace_constraint, scan_and_multiply
from propagation import farfield_propagator
from . import FLOAT_TYPE


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


def ML_grad_fourier_update(mask, Idata, obj, probe, exit_wave, addr, prefilter, postfilter, ob_grad, pr_grad, weights,
                           floating_intensities):
    error_out = []
    floating_intensity_coefficient = []
    LL = np.array([0.0])
    # Outer loop: through diffraction patterns
    for dname, diff_view in self.di.views.iteritems():


        # Weights and intensities for this view
        w = weights[diff_view]
        I = diff_view.data

        Imodel = np.zeros_like(I)
        f = {}

        # First pod loop: compute total intensity
        for name, pod in diff_view.pods.iteritems():

            f[name] = pod.fw(pod.probe * pod.object)
            Imodel += au.abs2(f[name])

        # Floating intensity option
        if floating_intensities:
            float_intensity = ((w * Imodel * I).sum()
                               / (w * Imodel ** 2).sum())
            floating_intensity_coefficient.append(float_intensity)
            Imodel *= float_intensity

        DI = Imodel - I

        # Second pod loop: gradients computation
        LLL = np.sum(w * DI ** 2)
        for name, pod in diff_view.pods.iteritems():
            xi = pod.bw(w * DI * f[name])
            ob_grad[pod.ob_view] += 2. * xi * pod.probe.conj()
            pr_grad[pod.pr_view] += 2. * xi * pod.object.conj()

        diff_view.error = LLL
        LL += LLL
        error_out.append(np.array([0, LLL / np.prod(DI.shape), 0]))

    error_out = np.array(error_out)
    if floating_intensities:
        floating_intensity_coefficient = np.array(floating_intensity_coefficient)
    return LL, error_out, floating_intensity_coefficient


def ML_poly_line_fourier_update(mask, Idata, obj, probe, exit_wave, addr, prefilter, postfilter, LL, ob_h, pr_h,
                                weights, floating_intensities):
    B = np.zeros((3,), dtype=np.longdouble)
    Brenorm = 1. / LL[0] ** 2

    # Outer loop: through diffraction patterns
    k = 0
    for dname, diff_view in self.di.views.iteritems():
        if not diff_view.active:
            continue

        # Weights and intensities for this view
        w = weights[diff_view]
        I = diff_view.data

        A0 = None
        A1 = None
        A2 = None

        for name, pod in diff_view.pods.iteritems():
            if not pod.active:
                continue
            f = pod.fw(pod.probe * pod.object)
            a = pod.fw(pod.probe * ob_h[pod.ob_view]
                       + pr_h[pod.pr_view] * pod.object)
            b = pod.fw(pr_h[pod.pr_view] * ob_h[pod.ob_view])

            if A0 is None:
                A0 = au.abs2(f)
                A1 = 2 * np.real(f * a.conj())
                A2 = (2 * np.real(f * b.conj())
                      + au.abs2(a))
            else:
                A0 += au.abs2(f)
                A1 += 2 * np.real(f * a.conj())
                A2 += 2 * np.real(f * b.conj()) + au.abs2(a)

        if floating_intensities:
            A0 *= floating_intensities[k]
            A1 *= floating_intensities[k]
            A2 *= floating_intensities[k]
            k += 1
        A0 -= I

        B[0] += np.dot(w.flat, (A0 ** 2).flat) * Brenorm
        B[1] += np.dot(w.flat, (2 * A0 * A1).flat) * Brenorm
        B[2] += np.dot(w.flat, (A1 ** 2 + 2 * A0 * A2).flat) * Brenorm
    return B, Brenorm


def regul_del2_grad(x, amplitude,  LLL, delxy, axes):
    del_xf = au.delxf(x, axis=axes[0])
    del_yf = au.delxf(x, axis=axes[1])
    del_xb = au.delxb(x, axis=axes[0])
    del_yb = au.delxb(x, axis=axes[1])

    delxy[:] = [del_xf, del_yf, del_xb, del_yb]
    grad = 2. * amplitude * (del_xb + del_yb - del_xf - del_yf)

    LLL[:] = amplitude * (au.norm2(del_xf)
                          + au.norm2(del_yf)
                          + au.norm2(del_xb)
                          + au.norm2(del_yb))

    return grad
