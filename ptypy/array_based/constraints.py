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

def ML_Gaussian_grad(ob_grad, pr_grad, weights, floating_intensities):

    # We need an array for MPI
    LL = np.array([0.])


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
            Imodel += u.abs2(f[name])

        # Floating intensity option
        if floating_intensities:
            diff_view.float_intens_coeff = ((w * Imodel * I).sum()
                                            / (w * Imodel ** 2).sum())
            Imodel *= diff_view.float_intens_coeff

        DI = Imodel - I

        # Second pod loop: gradients computation
        LLL = np.sum((w * DI ** 2))
        for name, pod in diff_view.pods.iteritems():

            xi = pod.bw(w * DI * f[name])
            ob_grad[pod.ob_view] += 2. * xi * pod.probe.conj()
            pr_grad[pod.pr_view] += 2. * xi * pod.object.conj()

            # Negative log-likelihood term
            # LLL += (w * DI**2).sum()

        # LLL
        diff_view.error = LLL
        error_dct[dname] = np.array([0, LLL / np.prod(DI.shape), 0])
        LL += LLL

    # Object regularizer
    if self.regularizer:
        for name, s in self.ob.storages.iteritems():
            self.ob_grad.storages[name].data += self.regularizer.grad(
                s.data)
            LL += self.regularizer.LL

    self.LL = LL / self.tot_measpts

    return ob_grad, pr_grad, error_dct

def ML_gaussian_grad(probe, obj, exit_wave, Idata, ob_grad, pr_grad, weights, floating_intensities, prefilter, postfilter, addr, regularizer='regul_del2'):
    view_dlayer = 0
    addr_info = addr[:, (view_dlayer)]
    sh = exit_wave.shape
    pa, oa, ea, da, ma  = zip(*addr_info)
    probe_object = scan_and_multiply(probe, obj, sh, addr_info)
    f = farfield_propagator(probe_object, prefilter, postfilter, direction='forward')
    Imodel = au.sum_to_buffer(au.abs2(f), Idata.shape, ea, da, dtype=FLOAT_TYPE)
    DI = Imodel - Idata
    weights_DI = weights * DI
    renormed_f = np.zeros_like(f)
    for pa, oa, ea, da, ma  in addr_info:
        renormed_f[ea[0]] = np.multiply(weights_DI[da[0]], f[ea[0]])

    xi = farfield_propagator(renormed_f, postfilter.conj(), prefilter.conj(), direction='backward')

    for pa, oa, ea, da, ma  in addr_info:
        ob_grad[oa[0],oa[1]:(oa[1]+sh[1]), oa[2]:(oa[2]+sh[2])] += 2.0 * xi[ea[0]] * probe[pa[0]].conj()
        pr_grad[pa[0]] += 2.0 * xi[ea[0]] * obj[oa[0],oa[1]:(oa[1]+sh[1]), oa[2]:(oa[2]+sh[2])].conj()

    LLL = (np.sum((weights_DI * DI), axis=(-2, -1)))/ Idata.size
    if regularizer is 'regul_del2':
        for pa, oa, ea, da, ma in addr_info:
            ob_grad[oa[0],oa[1]:(oa[1]+sh[1]), oa[2]:(oa[2]+sh[2])] += regul_del2_grad(obj[oa[0],oa[1]:(oa[1]+sh[1]), oa[2]:(oa[2]+sh[2])], LLL, delxy, axes)
    else:
        raise NotImplementedError("Don't recognize regularizer: %s. Only 'regul_del2' is supported" % regularizer)

    error = np.array([0.0, LLL/ np.probe(DI.shape[-2:]), 0.0])

    return ob_grad, pr_grad, error

def regul_del2_grad(x, amplitude,  LLL, delxy, axes):
    del_xf = u.delxf(x, axis=axes[0])
    del_yf = u.delxf(x, axis=axes[1])
    del_xb = u.delxb(x, axis=axes[0])
    del_yb = u.delxb(x, axis=axes[1])

    delxy[:] = [del_xf, del_yf, del_xb, del_yb]
    grad = 2. * amplitude * (del_xb + del_yb - del_xf - del_yf)

    LLL[:] = amplitude * (u.norm2(del_xf)
                                + u.norm2(del_yf)
                                + u.norm2(del_xb)
                                + u.norm2(del_yb))

    return grad